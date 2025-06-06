# %% [markdown]
#
# # Comprehensive Exam
#
# ## Coding Artifact
#
# Kalin Gibbons
#
# Nov 20, 2020

# ## Model Selection
#
# Base selection of regressors is performed by fitting multiple regressors without
# performing any parameter tuning, then comparing the resulting errors across
# functional groups. Models with lower errors will be marked for further investigation.

# %%
import logging
import sys

# !%load_ext autoreload
# !%autoreload 2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# !%matplotlib inline
# !%config InlineBackend.figure_format = 'retina'
# import seaborn as sns
import pandas as pd

# from catboost import CatBoostRegressor
from IPython.display import display
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm


import artifact
from artifact.constants import REGRESSION_PROFILE_PATH
from artifact.datasets import load_imorphics, ors_group_lut
from artifact.helpers import RegressionProfile

# %%
plt.rcParams["figure.figsize"] = (9, 5.5)
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["font.size"] = 14
mpl.rcParams["font.family"] = "Times New Roman"

# sns.set_context("poster")
# sns.set(rc={'figure.figsize': (16, 9.)})
# sns.set_style("whitegrid")

pd.set_option("display.max_rows", 120)
pd.set_option("display.max_columns", 120)

logging.basicConfig(level=logging.INFO, stream=sys.stdout)


# %% [markdown]
# ## Profiling the regressors
#
# First, we'll choose potential regressors to investigate. Early choices are linear,
# decision trees, as well as boosting and forest ensemble methods.

# ### Learner Selection
LUT = ors_group_lut
# LUT = tkr_group_lut

READER = load_imorphics
# READER = load_tkr

# %%
poly_degree = 2
learners = (
    Ridge(),  # L2 regularization.
    Lasso(),  # L1 regularization.
    LinearRegression(),
    ElasticNet(),
    Pipeline(
        [
            ("poly", PolynomialFeatures(degree=poly_degree, include_bias=False)),
            ("scaler", StandardScaler()),
            ("linreg", LinearRegression()),
        ]
    ),  # Polynomial + linear regression with scaling.
    Pipeline(
        [
            ("poly", PolynomialFeatures(degree=poly_degree, include_bias=False)),
            ("scaler", StandardScaler()),
            ("lasso", Lasso(max_iter=10000)),
        ]
    ),  # Polynomial + Lasso regression with scaling.
    Pipeline(
        [
            ("poly", PolynomialFeatures(degree=poly_degree, include_bias=False)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge()),
        ]
    ),  # Polynomial + Ridge regression with scaling.
)
# %% [markdown]

# Next, we'll select a functional group to examine, and only load the necessary
# data.

# ### Functional group selection

# %%
func_groups = list(LUT.keys())
func_groups

# %%
# func_groups = func_groups[0]

# %% [markdown]
# ### Fitting and profiling

# If the profiling results from the selected functional group have been loaded,
# then the `force_search` flag will need to be set to `True` to overwrite the
# previous profiling session.

# %%
force_search = True

# %%
for group in func_groups:
    shared_kwargs = dict(
        results_reader=READER,
        functional_groups=group,
        remove_response_prefix=True,
        include_time_response=False,
        include_time_predictor=True,
    )
    tkr_train = artifact.Results(**shared_kwargs, subset="train")
    tkr_valid = artifact.Results(**shared_kwargs, subset="test")
    display(tkr_train.response_names[1:])

    reg_prof = RegressionProfile(load_path=REGRESSION_PROFILE_PATH)

    learner_names = [x.__str__().replace("()", "") for x in learners]
    scaler = StandardScaler()
    regr = artifact.Regressor(tkr_train, tkr_valid, learners[0], scaler=scaler)
    err_df = pd.DataFrame(index=learner_names)

    saved_keys = reg_prof.error_dataframes.keys()
    if (force_search) or (group not in saved_keys):
        resp_pbar = tqdm(regr.train_results.response_names, desc="Processing...")
        for resp in resp_pbar:
            if resp == "time":
                continue
            resp_pbar.set_description(f"Processing {resp}")
            errs = np.zeros_like(learner_names, dtype=float)
            lrn_pbar = tqdm(learners, desc="Fitting...", leave=False)
            for idx, lrn in enumerate(lrn_pbar):
                desc = f"{learner_names[idx].replace('base_estimator=', '')}"
                lrn_pbar.set_description(desc)
                regr.learner = MultiOutputRegressor(lrn)
                y_pred = regr.fit(resp).predict()
                errs[idx] = regr.prediction_error
            err_df[resp] = errs
            lrn_pbar.close()
        resp_pbar.close()

        reg_prof.add_results(group, err_df)
        reg_prof.save(REGRESSION_PROFILE_PATH)

# %% [markdown]
# ## Results

# %%
# reg_prof.summarize(group)
for key in reg_prof.error_dataframes:
    reg_prof.describe(key)

# %%
