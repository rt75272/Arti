"""Configures sample datasets and defines sample dataset loading functions."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import artifact.constants
from artifact.core_api import select_by_regex

# warnings.filterwarnings('ignore', category=UserWarning, module=pd)

vahid_drop_regex = [
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

vahid_processed_history_output_cypher = {
    "_le_": "_strain_",
    "_s_": "_stress_",
    "tibia": "tib",
    "femur": "fem",
    "patella": "pat",
    "fimula": "fib",
}
vahid_raw_history_output_cypher = {
    # Joints
    "knee": "tibiofemoral",
    "patfem": "patellofemoral",
    # Directions
    "med": "medial",
    "lat": "lateral",
    # Bones
    "fem": "femur",
    "pat": "patella",
    "tib": "tibia",
    "fib": "fibula",
    # Abaqus terms
    "cof": "friction coefficient ",
    "cop": "center of pressure ",
    "cfn": "contact force ",
    # Axes
    "ap": "anterior-posterior",
    "ml": "medial-lateral",
    "si": "superior-inferior",
    # Rotations
    "ie": "internal-external",
    "fe": "flexion-extension",
    "vv": "varus-valgus",
}

"""A dictionary of regular expressions for selecting functional groups."""
tkr_group_lut = dict(
    contact_mechanics=r"^(?!pat).+(_area|_press|_cop_\d)$",
    joint_loads=r"^(?!pat).+(_force_\d|_torque_\d)$",
    kinematics=r"^(?!pat).+(_lat|_ant|_inf|_valgus|_external)$",
    ligaments=r"^(?!ml|pl).+(_force|_disp)$",
    patella=r"(pl|pat).*",
)

kl_group_lut = dict(
    contact_mechanics=r"^(?!pat).+(_CoF1|_CoF2|_CoF3|_area)$",
    joint_loads=r"^(?!pat).+(_force_|_torque_)$",
    kinematics=r"^(?!pat).+(_ml|_fe|_ap|_cc|_si|_ie)$",
    patella=r"(pat).*",
)

ors_group_lut = dict(
    contact_mechanics=r"^(?!pat).+(_area|_press|_cop_\d|_cof\d|_friction\scoefficient|cpress)$",
    joint_loads=r"^((?!pat).+(_cfn\d|_force_\d|_torque_\d|_force)$)|(^output_(?:le|s)_cart-(?!patella).+)",
    kinematics=r"^(?!pat).+(_ml|_ap|_si|_vv|_ie)$",
    ligaments=r"^(?!ml|pl|quad|hams).+(_force|_disp)$",
    # patfem_contact_mechanics=r"^(pat).+(_area|_press|_cop_\d|_cof\d|_friction\scoefficient|cpress)$",
    patfem_joint_loads=r"(^(pat).+(_cfn\d|_force_\d|_torque_\d|_force)$)|(^output_(?:le|s)_cart-(patella).+)",
    patfem_kinematics=r"^(pat).+(_ml|_ap|_si|_vv|_ie)$",
    # patfem_ligaments=r"^(ml|pl).+(_force|_disp)$",
    muscles=r"^(quad|hams)_force$",
)

ors_group_lut_with_fe = dict(
    contact_mechanics=r"^(?!pat).+(_area|_press|_cop_\d|_cof\d|_friction\scoefficient|cpress)$",
    joint_loads=r"^(?!pat).+(_cfn\d|_force_\d|_torque_\d|_force)$|^(Output_(?:le|s)_cart-(?!patella).+)",
    kinematics=r"^(?!pat).+(_ml|_ap|_si|_vv|_ie|_fe)$",
    ligaments=r"^(?!ml|pl|quad|hams).+(_force|_disp)$",
    patfem_contact_mechanics=r"^(pat).+(_area|_press|_cop_\d|_cof\d|_friction\scoefficient|cpress)$",
    patfem_joint_loads=r"^(pat).+(_cfn\d|_force_\d|_torque_\d|_force)$|^(Output_(?:le|s)_cart-(patella).+)",
    patfem_kinematics=r"^(pat).+(_ml|_ap|_si|_vv|_ie)$",
    patfem_ligaments=r"^(ml|pl).+(_force|_disp)$",
    muscles=r"^(quad|hams)_force$",
)


def split_df(df, predictor_index):
    """Split dataframe into predictor and response columns.

    Uses the difference between two sets of indices to split, where the user
    passes the predictor columns as a function parameter.

    Args:
        df (DataFrame): The dataframe to be split.
        predictor_index ([list] int): A list of numeric indices for the
            predictor columns.

    Returns:
        [DataFrame]: The predictor and response dataframes.
    """
    every_index = np.arange(df.shape[1])
    response_index = np.setdiff1d(every_index, predictor_index)
    try:
        pred_df = df.iloc[:, predictor_index].drop(columns=["cam_rad"])  # constant
    except KeyError:
        pred_df = df.iloc[:, predictor_index]

    resp_df = df.iloc[:, response_index]
    # pred_df = pred_df.loc[pd.IndexSlice[:, 0], :]
    pred_df = pred_df.xs(0, level="frame")
    return pred_df.astype(float), resp_df


def drop_columns(data_df, regex_list, inplace=False):
    """Drop DataFrame columns by a list of regular expressions.

    Args:
        data_df (DataFrame): The dataframe with columns to be searched.
        regex_list ([list] str): A list of regular expressions for pattern
            matching.
        inplace (bool, Optional): Allow mutating of data_df. Defaults to False.

    Returns:
        DataFrame: A dataframe with any matching columns removed.
    """
    cols = data_df.columns
    needs_drop = np.any([cols.str.contains(x) for x in regex_list], axis=0)
    return data_df.drop(cols[needs_drop], axis="columns", inplace=inplace)


def identify_predictor_columns(df):
    columns = df.columns
    is_response = np.array(
        [col.startswith(artifact.constants.RESPONSE_PREFIX) for col in columns]
    )
    return np.where(~is_response)[0]


def select_group(
    df, functional_groups, pred_idx, include_time_response, group_lut, **kwargs
):
    if functional_groups is not None:
        functional_groups = pd.Series(functional_groups)
        patterns = list()
        if include_time_response:
            patterns.append("TIME")
            patterns.append("time")
        for functional_group in functional_groups:
            if functional_group in group_lut:
                patterns.append(group_lut[functional_group])
            else:
                warnings.warn(
                    f"{functional_group} not found. Choose: {group_lut.keys()}",
                    stacklevel=2,
                )
        selected_df, _ = select_by_regex(df, patterns, axis=1, **kwargs)
        selected_df.columns = selected_df.columns.str.lower()
        if "time" not in selected_df.columns and include_time_response:
            selected_df["time"] = df.index.get_level_values(1)
        return pd.concat([df.iloc[:, pred_idx], selected_df], axis=1)


def prepare_dataframe_for_analysis(
    data,
    functional_groups,
    remove_response_prefix,
    include_time_response,
    group_lut,
    **kwargs,
):
    pred_idx = identify_predictor_columns(data)
    if remove_response_prefix:
        data.columns = data.columns.str.removeprefix(artifact.constants.RESPONSE_PREFIX)
    data = select_group(
        data, functional_groups, pred_idx, include_time_response, group_lut, **kwargs
    )
    return split_df(data, pred_idx)


def load_tkr(
    functional_groups=None,
    subset=None,
    include_time_response: bool = True,
    *,
    exclude_validation: bool = False,
    **kwargs,
):
    """Reader function for the 2018 total knee replacement dataset.

    Able to load only a subset of the data (train, test), as well as only a
    subset of the variables using functional group names as the selector:

        * contact_mechanics - Tibiofemoral contact areas and pressures
        * joint_loads - Tibiofemoral muscle forces and moments
        * kinematics - Joint coordinate system tibiofemoral kinematics
        * ligaments - Tibiofemoral ligament elongations and developed forces
        * patella - All of the above for the patellofemoral joint

    Args:
        functional_groups ([str], optional): A list of functional groups to load.
            Defaults to None.
        subset (str, optional): Either the train or test subset. Defaults to
            None.

    Returns:
        [DataFrame]: If a subset is selected, a pair of dataframes for the
            features or response variables. If no subset of passed, than a
            tuple of pairs of dataframes.
    """
    pred_idx = np.arange(0, 14)

    data_dir = artifact.constants.DATA_DIR / "preprocessed"

    if (subset is None) or (subset.lower() == "test"):
        test_data = pd.read_parquet(data_dir / "test.parquet")
        test_feat, test_resp = prepare_dataframe_for_analysis(
            test_data,
            functional_groups,
            False,
            include_time_response,
            tkr_group_lut,
            **kwargs,
        )

    if (subset is None) or (subset.lower() == "train"):
        train_data = pd.read_parquet(data_dir / "train.parquet")
        train_feat, train_resp = prepare_dataframe_for_analysis(
            train_data,
            functional_groups,
            False,
            include_time_response,
            tkr_group_lut,
            **kwargs,
        )

    if not exclude_validation and (
        (subset is None) or (subset.lower() in ["valid", "validate", "validation"])
    ):
        val_file = data_dir / "validate.parquet"
        try:
            val_data = pd.read_parquet(val_file)
            val_feat, val_resp = prepare_dataframe_for_analysis(
                val_data,
                functional_groups,
                False,
                include_time_response,
                tkr_group_lut,
                **kwargs,
            )
        except FileNotFoundError as e:
            warnings.warn(f"Validation dataset not found: {e}", stacklevel=2)
            val_feat, val_resp = pd.DataFrame(), pd.DataFrame()
        except Exception as e:
            warnings.warn(f"Error loading validation data: {e}", stacklevel=2)
            val_feat, val_resp = pd.DataFrame(), pd.DataFrame()
    else:
        val_feat, val_resp = pd.DataFrame(), pd.DataFrame()

    if subset is None:
        return (train_feat, train_resp), (test_feat, test_resp), (val_feat, val_resp)
    if subset.lower() == "train":
        return train_feat, train_resp
    if subset.lower() == "test":
        return test_feat, test_resp
    if subset.lower() in ["valid", "validate", "validation"]:
        return val_feat, val_resp


def load_imorphics(
    functional_groups=None,
    subset=None,
    include_time_response: bool = True,
    remove_response_prefix: bool = True,
    update_feature_names: bool = False,
    *,
    exclude_validation: bool = False,
    **kwargs,
):
    """Reader function for the 2025 OAI Imoprphics dataset.

    Able to load only a subset of the data (train, test), as well as only a
    subset of the variables using functional group names as the selector:

        * contact_mechanics - Tibiofemoral contact areas and pressures
        * joint_loads - Tibiofemoral muscle forces and moments
        * kinematics - Joint coordinate system tibiofemoral kinematics
        * ligaments - Tibiofemoral ligament elongations and developed forces
        * patella - All of the above for the patellofemoral joint

    Args:
        functional_groups ([str], optional): A list of functional groups to load.
            Defaults to None.
        subset (str, optional): Either the train or test subset. Defaults to
            None.
        include_time_response (bool, optional): Include the time response variable.
            Defaults to True.
        remove_response_prefix (bool, optional): Remove the response prefix from the
            response variables. Defaults to True.

    Returns:
        [DataFrame]: If a subset is selected, a pair of dataframes for the
            features or response variables. If no subset of passed, than a
            tuple of pairs of dataframes.
    """
    data_dir = artifact.constants.DATA_DIR / "preprocessed"

    if (subset is None) or (subset.lower() == "test"):
        test_data = pd.read_parquet(data_dir / "test.parquet")
        test_feat, test_resp = prepare_dataframe_for_analysis(
            test_data,
            functional_groups,
            remove_response_prefix,
            include_time_response,
            ors_group_lut,
            **kwargs,
        )
        if update_feature_names:
            test_resp = rename_columns_using_cypher(
                test_resp, vahid_processed_history_output_cypher
            )
            test_resp = rename_columns_using_cypher(
                test_resp, vahid_raw_history_output_cypher
            )

    if (subset is None) or (subset.lower() == "train"):
        train_data = pd.read_parquet(data_dir / "train.parquet")
        train_feat, train_resp = prepare_dataframe_for_analysis(
            train_data,
            functional_groups,
            remove_response_prefix,
            include_time_response,
            ors_group_lut,
            **kwargs,
        )
        if update_feature_names:
            train_resp = rename_columns_using_cypher(
                train_resp, vahid_processed_history_output_cypher
            )
            train_resp = rename_columns_using_cypher(
                train_resp, vahid_raw_history_output_cypher
            )

    if not exclude_validation and (
        (subset is None) or (subset.lower() in ["valid", "validate", "validation"])
    ):
        val_file = data_dir / "validation.parquet"
        try:
            val_data = pd.read_parquet(val_file)
            val_feat, val_resp = prepare_dataframe_for_analysis(
                val_data,
                functional_groups,
                remove_response_prefix,
                include_time_response,
                ors_group_lut,
                **kwargs,
            )
            if update_feature_names:
                val_resp = rename_columns_using_cypher(
                    val_resp, vahid_processed_history_output_cypher
                )
                val_resp = rename_columns_using_cypher(
                    val_resp, vahid_raw_history_output_cypher
                )
        except FileNotFoundError as e:
            warnings.warn(f"Validation dataset not found: {e}", stacklevel=2)
            val_feat, val_resp = pd.DataFrame(), pd.DataFrame()
        except Exception as e:
            warnings.warn(f"Error loading validation data: {e}", stacklevel=2)
            val_feat, val_resp = pd.DataFrame(), pd.DataFrame()
    else:
        val_feat, val_resp = pd.DataFrame(), pd.DataFrame()

    if subset is None and exclude_validation:
        return (train_feat, train_resp), (test_feat, test_resp)
    elif subset is None and not exclude_validation:
        return (train_feat, train_resp), (test_feat, test_resp), (val_feat, val_resp)
    elif subset.lower() == "train":
        return train_feat, train_resp
    elif subset.lower() == "test":
        return test_feat, test_resp
    elif subset.lower() in ["valid", "validate", "validation"]:
        return val_feat, val_resp
    else:
        raise ValueError(f"Unknown subset: {subset}")


def rename_columns_using_cypher(df, abaqus_cypher):
    columns = df.columns
    if abaqus_cypher is None:
        abaqus_cypher = {}
    for o, n in abaqus_cypher.items():
        columns = list(map(lambda x: x.replace(o, n), columns))

    df.columns = columns
    return df
