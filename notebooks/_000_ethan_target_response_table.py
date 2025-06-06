from pathlib import Path

import numpy as np
import pandas as pd
import toml
from tqdm import tqdm

from artifact.constants import CONFIG_DIR

# Load configuration
config = toml.load(CONFIG_DIR / f"{Path(__file__).stem}.toml")
pca_features_file_path = Path(config["paths"]["pca_features_file_path"])
knee_response_data_directory = Path(config["paths"]["knee_response_data_directory"])
history_output_filename = config["paths"]["history_output_filename"]
output_path = Path(config["paths"]["output_path"])


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
                .reset_index(names="TIME") .drop(index=len(raw_dataframe) - 2)
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


def main():
    """Main function to load, process, and save the merged features and response data."""
    response_dataframe = load_response_data(knee_response_data_directory)
    feature_dataframe = load_feature_data(pca_features_file_path)
    stacked_feature_dataframe = stack_data(feature_dataframe, response_dataframe)
    stacked_feature_dataframe.set_index(["subject"], inplace=True)

    merged_features_response_data = (
        pd.concat([stacked_feature_dataframe, response_dataframe], axis=1)
        .reset_index()
        .set_index(["subject", "frame"])
    )

    merged_features_response_data.to_pickle(output_path)
    print("Done!")


if __name__ == "__main__":
    main()
