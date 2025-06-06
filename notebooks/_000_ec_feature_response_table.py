import contextlib
import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import toml
from scipy import stats
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import artifact
from artifact.constants import CONFIG_DIR, RESPONSE_PREFIX
from artifact.datasets import rename_columns_using_cypher
from artifact.plotting import plot_training_responses_with_bounds

# Load configuration
config = toml.load(CONFIG_DIR / f"{Path(__file__).stem}.toml")
pca_features_file_path = Path(config["paths"]["pca_features_file_path"])
knee_response_data_directory = Path(config["paths"]["knee_response_data_directory"])
history_output_filename = config["paths"]["history_output_filename"]
output_path = Path(config["paths"]["output_path"])
clean_dir = Path(config["paths"]["cleaned_data_directory"])
MAXIMUM_FRAMES_IN_SIMULATION = config["simulation"]["number_of_frames"]

val_size = None
test_size = 0.2


def load_response_data(directory: Path) -> pd.DataFrame:
    """Load response data from the specified directory.

    Args:
        directory (Path): The directory containing the response data files.

    Returns:
        pd.DataFrame: The loaded response data.
    """
    response_dataframe = pd.DataFrame()
    for dir in tqdm(directory.iterdir()):
        if dir.is_dir():
            patient_id = dir.name
            file_path = dir / history_output_filename
            raw_dataframe = pd.read_csv(file_path, on_bad_lines="skip").T
            raw_dataframe.columns = raw_dataframe.iloc[0]
            raw_dataframe = (
                raw_dataframe[1:]
                .reset_index(names="TIME")
                .drop(index=len(raw_dataframe) - 2)
            )
            raw_dataframe.columns.name = None

            columns = raw_dataframe.columns.insert(0, "subject").insert(1, "frame")
            patient_id_col = pd.Series(
                [patient_id] * len(raw_dataframe), name="subject"
            )
            frame_col = pd.Series(range(len(raw_dataframe)), name="frame")
            raw_dataframe = pd.concat(
                [patient_id_col, frame_col, raw_dataframe], axis=1, ignore_index=True
            )
            raw_dataframe.columns = columns

            response_dataframe = pd.concat(
                [response_dataframe, raw_dataframe], ignore_index=True
            )
            response_dataframe.columns = columns
    response_dataframe.set_index("subject", inplace=True)
    return response_dataframe


def load_feature_data(file_path: Path) -> pd.DataFrame:
    """Load feature data from the specified file path.

    Args:
        file_path (Path): The file path to the feature data.

    Returns:
        pd.DataFrame: The loaded feature data.
    """
    feature_dataframe = pd.read_csv(file_path)
    return feature_dataframe.rename(columns={"Unnamed: 0": "subject"})


def stack_data(data: pd.DataFrame, df_to_stack_to: pd.DataFrame) -> pd.DataFrame:
    """Stack data to match the number of frames in the response data.

    Args:
        data (pd.DataFrame): The data to be stacked.
        df_to_stack_to (pd.DataFrame): The response data to match the frames.

    Returns:
        pd.DataFrame: The stacked data.
    """
    new_data = pd.DataFrame()
    row_length = len(data)
    column_names = data.columns

    for i in tqdm(range(row_length)):
        row = data.iloc[i]
        num_frames = df_to_stack_to.loc[row["subject"]].shape[0]
        row = row.to_numpy().reshape(1, -1)
        row = np.tile(row, (num_frames, 1))
        row = pd.DataFrame(row, columns=column_names)
        new_data = pd.concat([new_data, row], ignore_index=True)

    return new_data


def import_pca_data(pickle_path, compression="gzip", abaqus_cypher=None):
    """Import PCA data from a pickle file into a pandas DataFrame."""
    df = pd.read_pickle(pickle_path, compression=compression)
    df = rename_columns_using_cypher(df, abaqus_cypher)
    return df


def drop_columns(data_df, regex_list, invert: bool = False):
    """Remove columns using regular expressions."""
    negate = not invert
    return artifact.select_by_regex(data_df, regex_list, axis=1, negate=negate)


def remove_extraneous_rows(*dataframes: pd.DataFrame, max_sim_frames: int):
    """Remove extraneous rows from the data."""
    idx = pd.IndexSlice
    updated_dfs = [None] * len(dataframes)
    for df_idx, df in enumerate(dataframes):
        updated_dfs[df_idx] = df.loc[idx[:, : max_sim_frames + 1], :]
    return updated_dfs


def drop_failed_simulations(features, response, *, max_sim_frames):
    """Drop failed simulations from the features and response data."""
    success_ids = response.loc[
        response.index.get_level_values(1) == max_sim_frames
    ].index.get_level_values(0)
    response = response.loc[success_ids].reindex(success_ids, level=0)
    features.index.set_names(response.index.names, inplace=True)
    features = features.reindex(response.index, axis=0, method="ffill")
    return features, response


def convert_data_types(*dataframes: pd.DataFrame):
    """Convert data types of the features and response data."""
    new_dataframes = [None] * len(dataframes)
    for idx, df in enumerate(dataframes):
        new_dataframes[idx] = df.convert_dtypes()
    return new_dataframes


def filter_and_cast_types(features, response, *, max_sim_frames):
    """Filter and cast types of the features and response data."""
    features, response = remove_extraneous_rows(
        features, response, max_sim_frames=max_sim_frames
    )
    features, response = drop_failed_simulations(
        features, response, max_sim_frames=max_sim_frames
    )
    features, response = convert_data_types(features, response)
    return features, response


def combine_features_and_response(features, response, prepend_response=True):
    """Combine features and response data into a single DataFrame."""
    if prepend_response:
        response = response.add_prefix(RESPONSE_PREFIX)
    data_table = pd.concat([features, response], axis=1)

    # Check if there are two time columns and ensure they match exactly
    time_columns = [
        col for col in data_table.columns if re.match(r"time", col, re.IGNORECASE)
    ]
    if len(time_columns) == 2:
        assert data_table[time_columns[0]].equals(data_table[time_columns[1]]), (
            "Time columns do not match"
        )
        # Drop the time column from the features
        data_table.drop(columns=time_columns[0], inplace=True)

    return data_table


def get_train_test_subject_ids(
    response_df, val_size=None, test_size=0.2, random_state=None
):
    """Get train, test, and optionally validation subject IDs.

    Args:
        response_df (pd.DataFrame): The response DataFrame.
        test_size (float): The proportion of the dataset to include in the test split.
        val_size (float, optional): The proportion of the dataset to include in the
            validation split.
        random_state (int): Random state for reproducibility.

    Returns:
        tuple: Train, test, and optionally validation subject IDs.
    """
    unique_subject_ids = response_df.index.get_level_values(0).unique()
    unique_subject_ids_seven_digits = unique_subject_ids.str[:7].unique()

    if val_size is not None:
        train_ids, temp_ids = train_test_split(
            unique_subject_ids_seven_digits,
            test_size=(test_size + val_size),
            random_state=random_state,
        )
        val_ids, test_ids = train_test_split(
            temp_ids,
            test_size=(test_size / (test_size + val_size)),
            random_state=random_state,
        )
        return train_ids, val_ids, test_ids
    else:
        train_ids, test_ids = train_test_split(
            unique_subject_ids_seven_digits,
            test_size=test_size,
            random_state=random_state,
        )
        return train_ids, test_ids


def filter_dataframes_by_ids(dataframe, *id_sets):
    """Filter the DataFrame based on training and testing IDs."""
    multiindex = dataframe.index
    subject_id_format = r"(\d{7})(M\d{2})([RL])"
    subject_id_index = multiindex.get_level_values(0)
    id_df = subject_id_index.str.extract(subject_id_format)
    id_df.set_index(multiindex, inplace=True)
    id_df.columns = ["subject", "frame", "side"]

    return_dfs = []
    for id_set in id_sets:
        pattern = "|".join(id_set)
        is_subject = id_df["subject"].str.contains(pattern)
        return_dfs.append(dataframe[is_subject])

    return return_dfs


def update_pca_features_tables(multi_indexed_features, subject_indexed_features):
    """Updates the PCA features table with real CSV data."""
    # subjects = original_pca_features.set_index("subject").index.unique()
    subjects = multi_indexed_features.index.get_level_values(0).unique()
    updated_pca_features = subject_indexed_features.loc[subjects]
    updated_pca_features.loc[:, "frame"] = 0
    updated_pca_features.set_index([updated_pca_features.index, "frame"], inplace=True)
    updated_pca_features.index.names = ["subject_id", "frame"]
    updated_pca_features = updated_pca_features.reindex(
        multi_indexed_features.index, axis=0, method="ffill"
    )
    updated_pca_features["time"] = updated_pca_features.index.get_level_values(1)
    return updated_pca_features


def find_outliers_by_mad(
    data: pd.DataFrame, num_mad: float = 3.0, scale_mad: float = 1.0
) -> (pd.Index, pd.DataFrame):
    """Find outliers based on the median absolute deviation (MAD) of the feature columns.

    Args:
        data (pd.DataFrame): The data to analyze for outliers.
        num_mad (float): The number of median absolute deviations to use as the threshold.

    Returns:
        tuple: The indices of the outliers and a DataFrame indicating which features were outliers.
    """
    # index data to show only the first frame
    data = data.reset_index().set_index("subject")
    mask = data["frame"] == 0
    data = data[mask].drop(columns=["frame"])
    with contextlib.suppress(KeyError):
        data.drop(columns="time", inplace=True)

    median = data.median()
    mad = stats.median_abs_deviation(data, scale=scale_mad, axis=0)
    # mad = (data - median).abs().median()
    threshold = num_mad * mad
    outlier_df = (data - median).abs() > threshold
    outlier_subject_ids = data[outlier_df.any(axis=1)].index
    outlier_df = outlier_df.loc[outlier_subject_ids]
    return outlier_subject_ids, outlier_df


def load_additional_subject_features(file_path: Path) -> pd.DataFrame:
    """Load subject-weight-hkangle-bone-lengths data and return indexed by 'subject'."""
    df = pd.read_csv(file_path)

    df.rename(columns={"identifier": "subject"}, inplace=True)
    df.set_index("subject", inplace=True)
    return df


def load_and_reformat_vahid_hka_data(file_path: Path) -> pd.DataFrame:
    """Load and reformat Vahid HKA data for merging."""
    df = pd.read_csv(file_path)
    # Example: rename columns or cast types if needed
    df.rename(columns={"identifier": "subject"}, inplace=True)
    df = (
        df.reset_index()
        .set_index(["subject", "variable"])
        .drop(columns="index", axis=1)
    )
    # new_level_1_values = df.index.get_level_values(1).unique().append(pd.Index(["frame"]))
    # new_index = pd.MultiIndex.from_product(
    #     [df.index.get_level_values(0).unique(), new_level_1_values],
    #     names=["subject", "variable"],
    # )
    # df = df.reindex(new_index, fill_value=np.nan)
    new_df = pd.DataFrame()
    for subject in df.index.get_level_values(0).unique():
        subject_df = df.loc[subject].T
        subject_df.loc[:, "frame"] = np.arange(len(subject_df.index))
        subject_df.loc[:, "subject"] = subject
        subject_df = (
            subject_df.reset_index()
            .set_index(["subject", "frame"])
            .drop("index", axis=1)
        )
        new_df = pd.concat([new_df, subject_df])

    return new_df


def main():
    """Main function to load, process, and save the features and response data."""
    response_dataframe = load_response_data(knee_response_data_directory)
    # Load Vahid HKA data
    vahid_hka_file_path = Path(config["paths"]["additional_response_data_path"])
    vahid_hka_df = load_and_reformat_vahid_hka_data(vahid_hka_file_path)
    # Merge dataframes
    vahid_hka_df = vahid_hka_df.reset_index().set_index(["subject"])
    merged_response_df = pd.merge(
        response_dataframe,
        vahid_hka_df,
        on=["subject", "frame"],
        how="outer",
        suffixes=("_raw", "_adjusted"),
    )

    feature_dataframe = load_feature_data(pca_features_file_path)

    feature_dataframe = feature_dataframe.set_index("subject")
    feature_columns = feature_dataframe.columns
    try:
        feature_columns = feature_columns.astype(int)
        feature_columns = [f"PC{col:02d}" for col in feature_columns]
    except ValueError:
        pass
    feature_dataframe.columns = feature_columns

    additional_feature_file = Path(config["paths"]["additional_features_file_path"])
    additional_feature_df = load_additional_subject_features(additional_feature_file)

    def correct_subject_identifiers(additional_feature_df, feature_dataframe):
        r"""Correct subject identifiers in the additional feature dataframe.

        Args:
            additional_feature_df (pd.DataFrame): The additional feature dataframe with missing M\d{2}.
            feature_dataframe (pd.DataFrame): The feature dataframe with correct subject identifiers.

        Returns:
            pd.DataFrame: The additional feature dataframe with corrected subject identifiers.
        """
        corrected_subjects = []
        for subject in additional_feature_df.index:
            lr = subject[-1]
            subject = subject[:-1]
            pattern = re.compile(rf"^{subject}M\d{{2}}{lr}$")
            match = feature_dataframe.index[feature_dataframe.index.str.match(pattern)]
            if not match.empty:
                corrected_subjects.append(match[0])
            else:
                corrected_subjects.append(subject)
        additional_feature_df.index = corrected_subjects
        return additional_feature_df

    additional_feature_df = correct_subject_identifiers(
        additional_feature_df, feature_dataframe
    )
    feature_dataframe = feature_dataframe.merge(
        additional_feature_df, how="left", left_index=True, right_index=True
    )

    stacked_feature_dataframe = stack_data(
        feature_dataframe.reset_index(), response_dataframe
    )
    stacked_feature_dataframe.set_index(["subject"], inplace=True)

    merged_features_response_data = (
        pd.concat([stacked_feature_dataframe, merged_response_df], axis=1)
        .reset_index()
        .set_index(["subject", "frame"])
    )

    updated_feature_dataframe = update_pca_features_tables(
        multi_indexed_features=merged_features_response_data,
        subject_indexed_features=feature_dataframe,
    )
    updated_response_dataframe = merged_response_df.reset_index().set_index(
        ["subject", "frame"]
    )
    updated_response_dataframe.columns = updated_response_dataframe.columns.str.lower()

    # Filter and cast types
    frame_levels = updated_response_dataframe.index.get_level_values(1)
    subject_levels = updated_response_dataframe.index.get_level_values(0)
    pca_subjects = feature_dataframe.index
    absoulte_max_frames = frame_levels.max()
    count_with_max = np.count_nonzero(frame_levels == absoulte_max_frames)
    unique_subject_count = subject_levels.nunique()
    pca_unique_subject_count = pca_subjects.nunique()

    assert pca_unique_subject_count == unique_subject_count
    n_original_subjects = unique_subject_count

    updated_feature_dataframe, updated_response_dataframe = filter_and_cast_types(
        updated_feature_dataframe,
        updated_response_dataframe,
        max_sim_frames=MAXIMUM_FRAMES_IN_SIMULATION,
    )
    n_failed_simulations = (
        n_original_subjects
        - updated_response_dataframe.index.get_level_values(0).nunique()
    )

    failed_simulation_subjects = list(
        set(subject_levels) - set(updated_response_dataframe.index.get_level_values(0))
    )
    print(f"Failed simulation subjects: {failed_simulation_subjects}")

    for col in response_dataframe.columns:
        col = col.lower()
        if f"{col}_adjusted" in updated_response_dataframe.columns and np.allclose(
            updated_response_dataframe[f"{col}_adjusted"].astype(float),
            updated_response_dataframe[f"{col}_raw"].astype(float),
            rtol=1e-5,
            atol=1e-8,
        ):
            updated_response_dataframe.rename(columns={f"{col}_raw": col}, inplace=True)
            updated_response_dataframe.drop(columns=[f"{col}_adjusted"], inplace=True)
    # Combine feature and response data
    merged_features_response_data = combine_features_and_response(
        updated_feature_dataframe, updated_response_dataframe
    )

    # Find outliers based on MAD of the feature columns
    feature_columns = updated_feature_dataframe.columns
    outlier_subjects, outlier_features = find_outliers_by_mad(
        merged_features_response_data[feature_columns],
        num_mad=9,
        scale_mad=1,
        # merged_features_response_data[feature_columns[1:4]], num_mad=3
    )
    outlier_subjects, outlier_features = find_outliers_by_mad(
        merged_features_response_data[feature_columns[:3]],
        num_mad=2.7,
        scale_mad="normal",
        # merged_features_response_data[feature_columns[1:4]], num_mad=3
    )
    n_feature_outliers = len(outlier_subjects)

    # Reshape the passed series to be dataframes with frame as the index
    time_series = (
        merged_features_response_data["response_time"].unstack(level=0).iloc[:, 0]
    )
    response_outliers = set()
    for response_str in [
        "fem_ap",
        "knee_fe",
        "knee_ie",
        "knee_ml",
        "knee_ap",
        "knee_vv",
    ]:  # , "tib_vv", "tib_ie"]:
        df = merged_features_response_data[f"response_{response_str}"].unstack(level=0)
        fig, ax = plt.subplots(figsize=(10, 6))
        outlier_subjects_response = plot_training_responses_with_bounds(
            time_series.astype(float),
            df.T,
            num_mad=4,
            scale_mad=1,
            ax=ax,
            outlier_threshold=0.3,
            plot_all_trials=False,
        )
        ax.set_title(f"{response_str}")
        response_outliers.update(outlier_subjects_response)
    plt.show()

    # Plot training responses with bounds and detect outliers
    # fig, ax = plt.subplots(figsize=(10, 6))
    # outlier_subjects_response = plot_training_responses_with_bounds(
    #     time_series.astype(float),
    #     knee_fe_dataframe.T,
    #     num_mad=4,
    #     scale_mad=1,
    #     ax=ax,
    #     outlier_threshold=0.3,
    # )
    # print(len(outlier_subjects_response))
    # outlier_subjects_response = plot_training_responses_with_bounds(
    #     time_series.astype(float),
    #     knee_fe_dataframe.T,
    #     num_mad=2.7,
    #     scale_mad="normal",
    #     ax=ax,
    #     outlier_threshold=0.3,
    # )
    print(len(response_outliers))
    # plt.close()

    # Combine outliers from features and response
    outlier_subjects = outlier_subjects.union(response_outliers)
    n_fe_profile_outliers = len(response_outliers)

    # Display subject identifiers corresponding to the outliers
    print(f"\nOutlier subjects ({len(outlier_subjects)}): {outlier_subjects}")
    print(f"\nOutlier features:\n{outlier_features}")

    # Drop outlier subjects from the dataset
    merged_features_response_data = merged_features_response_data.drop(
        outlier_subjects, level=0
    )

    # Drop subjects with PC00 less than 700
    pc00_outliers = (
        merged_features_response_data[merged_features_response_data["PC00"] < -700]
        .index.get_level_values(0)
        .unique()
    )
    merged_features_response_data = merged_features_response_data.drop(
        pc00_outliers, level=0
    )
    # Get train, test, and optionally validation subject IDs
    num_valid_subjects = merged_features_response_data.index.get_level_values(
        0
    ).nunique()

    if val_size is not None:
        train_ids, val_ids, test_ids = get_train_test_subject_ids(
            merged_features_response_data,
            val_size=val_size,
            test_size=test_size,
            random_state=42,
        )

        # Filter dataframes by train and test IDs
        train_data, val_data, test_data = filter_dataframes_by_ids(
            merged_features_response_data, train_ids, val_ids, test_ids
        )
    else:
        train_ids, test_ids = get_train_test_subject_ids(
            merged_features_response_data, test_size=test_size, random_state=42
        )

        # Filter dataframes by train and test IDs
        train_data, test_data = filter_dataframes_by_ids(
            merged_features_response_data, train_ids, test_ids
        )
        val_data = pd.DataFrame()

    # Save the data
    train_data.to_parquet(clean_dir.joinpath("train.parquet"))
    test_data.to_parquet(clean_dir.joinpath("test.parquet"))
    if val_size is not None:
        val_data.to_parquet(clean_dir.joinpath("validation.parquet"))
    else:
        clean_dir.joinpath("validation.parquet").unlink(missing_ok=True)

    merged_features_response_data.to_csv(output_path.with_suffix(".csv"))
    merged_features_response_data.to_pickle(output_path)
    # Assert that the first 7 digits of each entry in test_data do not appear in train_data
    train_subjects = train_data.index.get_level_values(0).str[:7].unique()
    test_subjects = test_data.index.get_level_values(0).str[:7].unique()
    assert not any(subject in train_subjects for subject in test_subjects), (
        "Test subjects overlap with train subjects"
    )
    # Check how many of the outlier subjects came from train and test
    outlier_subjects_seven_digits = pd.Series(outlier_subjects).str[:7].unique()
    train_outliers = [
        subject
        for subject in outlier_subjects_seven_digits
        if subject in train_subjects
    ]
    test_outliers = [
        subject for subject in outlier_subjects_seven_digits if subject in test_subjects
    ]

    print(f"Number of original subjects: {n_original_subjects}")
    print(f"Number of failed simulations: {n_failed_simulations}")
    print(f"Number of feature outliers: {n_feature_outliers}")
    print(f"Number of FE profile outliers: {n_fe_profile_outliers}")
    print(f"Number of filtered subjects: {num_valid_subjects}")
    print(
        f"Number of outlier subjects with other knee in train data: {len(train_outliers)}"
    )
    print(
        f"Number of outlier subjects with other knee in test data: {len(test_outliers)}"
    )
    print("Done!")

    response_outliers = set()

    for response_str in [
        "fem_ap",
        "knee_fe",
        "knee_ie",
        "knee_ml",
        "knee_ap",
        "knee_vv",
    ]:  # , "tib_vv", "tib_ie"]:
        df = train_data[f"response_{response_str}"].unstack(level=0)
        fig, ax = plt.subplots(figsize=(10, 6))
        outlier_subjects_response = plot_training_responses_with_bounds(
            time_series.astype(float),
            df.T,
            num_mad=4,
            scale_mad=1,
            ax=ax,
            outlier_threshold=0.3,
            plot_all_trials=True,
        )
        ax.set_title(f"train_{response_str}")
        response_outliers.update(outlier_subjects_response)
    plt.show()
    response_outliers = set()
    for response_str in [
        "fem_ap",
        "knee_fe",
        "knee_ie",
        "knee_ml",
        "knee_ap",
        "knee_vv",
    ]:  # , "tib_vv", "tib_ie"]:
        df = test_data[f"response_{response_str}"].unstack(level=0)
        fig, ax = plt.subplots(figsize=(10, 6))
        outlier_subjects_response = plot_training_responses_with_bounds(
            time_series.astype(float),
            df.T,
            num_mad=4,
            scale_mad=1,
            ax=ax,
            outlier_threshold=0.3,
            plot_all_trials=True,
        )
        ax.set_title(f"test_{response_str}")
        response_outliers.update(outlier_subjects_response)
    plt.show()


if __name__ == "__main__":
    main()
