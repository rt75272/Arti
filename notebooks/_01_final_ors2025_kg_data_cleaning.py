# %% [markdown]
#
# # Comprehensive Exam
#
# ## ORS Abstract
#
# Kalin Gibbons
#
# Nov 10 2024

# ### Data Description
#

# ### Data cleaning

# %%
import logging
import math
import os
import sys
from pathlib import Path

# !%load_ext autoreload
# !%autoreload 2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import scipy.io as spio

# !%matplotlib inline
# !%config InlineBackend.figure_format = 'retina'
import seaborn as sns
import sklearn
import statsmodels.api as sm
from IPython.display import display
from sklearn.model_selection import train_test_split
from statsmodels.formula.api import ols

import artifact
from artifact.constants import DATA_DIR, RESPONSE_PREFIX
from artifact.core_api import select_by_regex

# %%
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["font.size"] = 14
mpl.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.figsize"] = (9, 5.5)

sns.set_context("poster")
sns.set(rc={"figure.figsize": (16, 9.0)})
sns.set_style("whitegrid")

pd.set_option("display.max_rows", 120)
pd.set_option("display.max_columns", 120)

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

# %% [markdown]
#
# ## Data Cleaning
#
# ---
#
# The data is contained within `MATLAB` binary files, which are easily imported into a
# `pandas` `DataFrame`. Some of the sensors implemented within the FEA model that
# generated the data do not consistently activate during a deep-knee-bend, so those
# columns need to be filtered out. Finally, if the uniform feature draw resulted in a
# particularly infeasible implant geometry, the simulation would fail, producing no
# results. These failed simulations will need to be removed.
#
# ### Controlling variable and function definitions


# %%

drop_regex = [
    # r'^time$',
    r"(femfe|patthick|patml|patsi)",  # features held constant
    r"^\w{3}_[xyz]_\w{3,4}$",
    r"^post\w+",
    r"^v[ilm][2-6]_(disp|force)$",
    r"^v[lm]1_(disp|force)$",
    r"^vert_(disp|force)$",
    r"^flex_(force|rot)$",
    r"^ap_force$",
    r"^(vv|ie)_torque",
    r"^(ml|pcm|pcl|pol)_force$",  # Always zero
    r"^(lclp|lcl|pmc|lcla)_force$",  # Often zero and bad predict
    r"^(pom|alc|mcl|mclp)_force$",  # Often zero and fairly bad predict.
]


def drop_columns(data_df, regex_list, invert: bool = False):
    """Remove columns using regular expressions."""
    negate = not invert
    return artifact.select_by_regex(data_df, regex_list, axis=1, negate=negate)


def import_pca_data(pickle_path, compression="gzip"):
    """Import PCA data from a pickle file into a panda DataFrame."""
    df = pd.read_pickle(pickle_path, compression=compression)
    columns = df.columns
    abaqus_cypher = {
        "fem": "femur",
        "pat": "patella",
        "tib": "tibia",
        "fib": "fibula",
        "cof": "friction coefficient",
        "cop": "center of pressure",
        "ml": "medial-lateral",
        "ap": "anterior-posterior",
        "si": "superior-inferior",
    }
    abaqus_cypher = {}
    for o, n in abaqus_cypher.items():
        columns = list(map(lambda x: x.replace(o, n), columns))

    df.columns = columns
    return df


def remove_extraneous_rows(*dataframes, max_sim_frames: int):
    r"""Remove extraneous rows from the data.

    Vahid's data had an extra frame on only some of the simulations, due to frame/output
    timing issues. This function removes the extraneous rows from the data.

    Args:
        *dataframes: dataframes to filter.
        max_sim_frames (int): The maximum number of frames specified to be output in the
            simulation.

    Returns:
        tuple: A tuple containing the filtered data.
    """
    idx = pd.IndexSlice
    updated_dfs = [None] * len(dataframes)
    for df_idx, df in enumerate(dataframes):
        updated_dfs[df_idx] = df.loc[idx[:, :max_sim_frames], :]
    return updated_dfs


def drop_failed_simulations(features, response, *, max_sim_frames):
    """Drop failed simulations from the features and response data.

    Args:
        features (pd.DataFrame): The features data.
        response (pd.DataFrame): The response data.
        max_sim_frames (int): The maximum number of frames.

    Returns:
        tuple: A tuple containing the features and response data with failed simulations
        dropped.
    """
    success_ids = response.loc[
        response.index.get_level_values(1) == max_sim_frames
    ].index.get_level_values(0)
    response = response.loc[success_ids].reindex(success_ids, level=0)
    features.index.set_names(response.index.names, inplace=True)
    features = features.reindex(response.index, axis=0, method="ffill")
    return features, response


def convert_data_types(features, response):
    """Convert data types of the features and response data.

    Args:
        features (pd.DataFrame): The features data.
        response (pd.DataFrame): The response data.

    Returns:
        tuple: A tuple containing the features and response data with converted types.
    """
    features = features.convert_dtypes()
    response = response.convert_dtypes()
    return features, response


def filter_and_cast_types(features, response, *, max_sim_frames):
    """Filter and cast types of the features and response data.

    Drop the extraneous rows, drop failed simulations, and convert data types.

    Args:
        features (pd.DataFrame): The features data.
        response (pd.DataFrame): The response data.
        max_sim_frames (int): The maximum number of frames.

    Returns:
        tuple: A tuple containing the filtered and casted features and response data.
    """
    features, response = remove_extraneous_rows(
        features, response, max_sim_frames=max_sim_frames
    )
    features, response = drop_failed_simulations(
        features, response, max_sim_frames=max_sim_frames
    )
    features, response = convert_data_types(features, response)
    return features, response


def combine_features_and_response(features, response, prepend_response=True):
    """Combine features and response data into a single DataFrame.

    Args:
        features (pd.DataFrame): The features data.
        response (pd.DataFrame): The response data.

    Returns:
        pd.DataFrame: The combined data.
    """
    if prepend_response:
        response = response.add_prefix(RESPONSE_PREFIX)
    data_table = pd.concat([features, response], axis=1)
    return data_table


def get_train_test_subject_ids(response_df, test_size=0.2, random_state=42):
    """Get train and test subject IDs.

    Args:
        response_df (pd.DataFrame): The response DataFrame containing subject IDs.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: A tuple containing the training and testing subject IDs.
    """
    # Extract unique subject IDs from the response DataFrame
    unique_subject_ids = response_df.index.get_level_values(0).unique()

    # Extract only the seven digits at the beginning of each subject ID
    unique_subject_ids_seven_digits = unique_subject_ids.str[:7].unique()

    # Randomly sample the unique seven-digit IDs into training (80%) and testing (20%) groups
    train_ids, test_ids = train_test_split(
        unique_subject_ids_seven_digits, test_size=test_size, random_state=random_state
    )

    return train_ids, test_ids


def filter_dataframes_by_ids(dataframe, *id_sets):
    """Filter the DataFrame based on training and testing IDs.

    Args:
        dataframe (pd.DataFrame): The DataFrame to be filtered.
        *id_sets (list): List of ID sets to filter the DataFrame (train ids, test ids).

    Returns:
        tuple: A tuple containing the filtered DataFrames.
    """
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
    """Updates the PCA features table with real CSV data.

    This function takes the original PCA features and updates them with the
    corresponding data from a real CSV file. It ensures that the updated PCA
    features have the same index structure as the original PCA features.

    Args:
        original_pca_features (pd.DataFrame): The original PCA features DataFrame
            with a multi-level index (subject_id, frame).
        real_csv (pd.DataFrame): The real CSV data containing updated PCA features
            with subject_id as the index.

    Returns:
        pd.DataFrame: The updated PCA features DataFrame with the same index
        structure as the original PCA features.
    """
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


# %% [markdown]
# ### Locating the data
#
# The `MATLAB` MAT files are stored in the `data/interim` folder because the raw data
# was stored in plaintext CSV files after being extracted from the FEA simulations.
# Once cleaned, we'll store the cleaned data in `data/preprocessed`.

# %%
# Source paths
dirty_data_dir = DATA_DIR / "interim"

manual_pca_features_fname = (
    Path("D:") / "git-data" / "kneemesher" / "raw" / "target_response_data_aa.pkl"
)
auto_pca_features_fname = (
    Path("D:") / "git-data" / "kneemesher" / "raw" / "target_response_data_as.pkl"
)
assert manual_pca_features_fname.is_file(), (
    f"File {manual_pca_features_fname} not found."
)

manual_response_fname = (
    Path("D:") / "git-data" / "kneemesher" / "raw" / "amanda-approved-knees.pkl.tar.gz"
)
assert manual_response_fname.is_file(), f"File {manual_response_fname} not found."

auto_response_fname = (
    Path("D:")
    / "git-data"
    / "kneemesher"
    / "raw"
    / "auto-segment-dilated-bones.pkl.tar.gz"
)
assert auto_response_fname.is_file(), f"File {auto_response_fname} not found."

# Destination paths
cleaned_dir = dirty_data_dir.parent / "preprocessed"
cleaned_test_path = cleaned_dir / "test.parquet"
cleaned_train_path = cleaned_dir / "train.parquet"

# %% [markdown]
# ### Data import and cleaning
#
# Reading the PKL file tables into memory, and outputting part of the dataframes to
# take a look at the data, then dropping the extraneous columns and taking a look at
# the results. We'll only look at the results from the testing set, but run identical
# operations on the training set, as well.

# %%
# All
manual_pca_features = import_pca_data(manual_pca_features_fname)
real_csv = pd.read_csv(
    Path("D:") / "git-data" / "kneemesher" / "raw" / "features.csv", index_col=0
)


# %%

manual_pca_features = update_pca_features_tables(manual_pca_features, real_csv)


# %%
drop_regex = [
    r"femur_\d+_[xyz]$",
    r"patella_\d+_[xyz]$",
    r"tibia_\d+_[xyz]$",
    r"fibula_\d+_[xyz]$",
    r"TIME",
]

# %%

manual_pca_features = drop_columns(manual_pca_features, drop_regex, invert=True)[0]
manual_response = pd.read_pickle(manual_response_fname, compression="gzip")


auto_pca_features = import_pca_data(auto_pca_features_fname)
auto_pca_features = update_pca_features_tables(auto_pca_features, real_csv)
auto_pca_features = drop_columns(auto_pca_features, drop_regex, invert=True)[0]
auto_response = pd.read_pickle(auto_response_fname, compression="gzip")


# %%
MAX_FRAMES = manual_response.index.get_level_values(1).max() - 1
manual_pca_features, manual_response = filter_and_cast_types(
    manual_pca_features, manual_response, max_sim_frames=MAX_FRAMES
)
manual_response.columns

# %%

manual_pca_df = combine_features_and_response(manual_pca_features, manual_response)
manual_pca_df

# %%

# Create the auto PCA DataFrame
auto_pca_features, auto_response = filter_and_cast_types(
    auto_pca_features, auto_response, max_sim_frames=MAX_FRAMES
)
auto_pca_features.columns
auto_pca_df = combine_features_and_response(auto_pca_features, auto_response)
auto_pca_df

# Split the manual and auto DataFrames based on training and testing IDs
# %%

train_ids, test_ids = get_train_test_subject_ids(manual_response)
manual_train_df, manual_test_df = filter_dataframes_by_ids(
    manual_pca_df, train_ids, test_ids
)

# %%

auto_train_df, auto_test_df = filter_dataframes_by_ids(auto_pca_df, train_ids, test_ids)

# %%
clean_train = pd.concat([manual_train_df, auto_train_df], axis=0)
clean_test = pd.concat([manual_test_df, auto_test_df], axis=0)

# %%


# %% [markdown]
# ## Save the cleaned data

# Everything looked great, so we can save the cleaned data.

# %%
clean_test.to_parquet(cleaned_test_path)
clean_train.to_parquet(cleaned_train_path)

# %%
