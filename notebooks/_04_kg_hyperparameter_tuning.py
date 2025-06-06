# %% [markdown]
"""This script performs hyperparameter tuning for various regression models using the Optuna library.

It includes the following steps:

1. Import necessary libraries and modules.
2. Load and preprocess the dataset.
3. Define an objective function for Optuna to optimize.
4. Create and run an Optuna study to find the best hyperparameters.
5. Plot the optimization history and display the best trial results.
6. Apply the best hyperparameters to the selected model and generate plots for the predictions.

The script supports multiple regression models, including:
- RandomForestRegressor
- ElasticNet
- SVR
- GradientBoostingRegressor
- AdaBoostRegressor
- DecisionTreeRegressor
- Ridge
- Lasso
- XGBRegressor
- CatBoostRegressor

The CatBoostRegressor is contained in the `catboost` package.
"""
#  # Comprehensive Exam
#
#  ## Coding Artifact
#
#  Kalin Gibbons
#
#  Nov 20, 2020
#
#  > Note: A hyperparameter is a numerical or other measurable factor
#  responsible for some aspect of training a machine learning model, whose value
#  cannot be estimated from the data, unlike regular parameters which represent
#  inherent properties of the natural processes which generated data.
#
#  ## Hyperparameter Optimization
#
#  There are several python packages with automatic hyperparameter selection
#  algorithms. A relatively recent contribution which I find particularly easy
#  to use is [optuna](https://optuna.org/), which is detailed in this
#  [2019 paper](https://arxiv.org/abs/1907.10902). Optuna allows the user to
#  suggest ranges of values for parameters of various types, then utilizes a
#  parameter sampling algorithms to find an optimal set of hyperparameters. Some
#  of the sampling schemes available are:
#
#  * Grid Search
#  * Random
#  * Bayesian
#  * Evolutionary
#
# While the parameter suggestion schemes available are:
#
#  * Integers
#    * Linear step
#    * Logarithmic step
#  * Floats
#    * Logarithmic
#    * Uniform
#  * Categorical
#    * List
#
#  This notebook uses Optuna to implement hyperparameter tuning on a number of
#  ensemble algorithms.
#
#  ## Imports

# %%
import contextlib
import logging
import math
import os
import pickle
import sys
from pathlib import Path

# !%load_ext autoreload
# !%autoreload 2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import optuna

# !%matplotlib inline
# !%config InlineBackend.figure_format = 'retina'
# import seaborn as sns
import pandas as pd
import scipy as sp
import scipy.io as spio
import sklearn
import statsmodels.api as sm
from catboost import CatBoostRegressor
from colorama import Fore, Style
from IPython.display import clear_output, display
from optuna.visualization import plot_optimization_history
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from statsmodels.formula.api import ols
from tqdm.auto import tqdm
from xgboost import XGBRegressor

import artifact
from artifact.constants import MODEL_DIR, REGRESSION_PROFILE_PATH
from artifact.datasets import load_imorphics, load_tkr, ors_group_lut, tkr_group_lut
from artifact.helpers import RegressionProfile

# %%

plt.rcParams["figure.figsize"] = (9, 5.5)
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["font.size"] = 14
mpl.rcParams["font.family"] = "Times New Roman"
plt.rcParams["svg.fonttype"] = "none"

# sns.set_context("poster")
# sns.set(rc={'figure.figsize': (16, 9.)})
# sns.set_style("whitegrid")

pd.set_option("display.max_rows", 120)
pd.set_option("display.max_columns", 120)

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

READER = load_imorphics
LUT = ors_group_lut

# READER = load_tkr
# LUT = tkr_group_lut

MODEL_DIR.joinpath("learners").mkdir(exist_ok=True, parents=True)
# %% [markdown]
#  Next, we'll select a functional group to examine, and only load the necessary
#  data.
#  ### Functional group selection

# %%
func_groups = list(LUT.keys())
func_groups


# %%
# group = "contact_mechanics"
# group = "joint_loads"
# group = "kinematics"
# group = "ligaments"
# group = "patella"


# %% [markdown]
#  ### Loading the data
#
#  We'll load a subset of the data containing the responses making up the chosen
#  functional group.

# %%
regressors = dict()
for group in func_groups:
    shared_kwargs = dict(
        results_reader=READER,
        functional_groups=group,
        include_time_response=False,
        include_time_predictor=True,
        remove_response_prefix=True,
        exclude_validation=False,
    )
    tkr_train = artifact.Results(**shared_kwargs, subset="train")
    # tkr_valid = artifact.Results(**shared_kwargs, subset="validation")
    tkr_test = artifact.Results(**shared_kwargs, subset="test")
    display(tkr_train.response_names[1:])

    try:
        reg_prof = RegressionProfile(load_path=REGRESSION_PROFILE_PATH)
        reg_prof.describe(group)
    except FileNotFoundError:
        pass

    # ### Creating the optimization study
    #
    # First we must define an objective function, which suggests the ranges of
    # hyperparameters to be sampled. We can use switch-cases to optimize the machine
    # learning algorithm itself, in addition to the hyperparameters.

    learners = (
        # GradientBoostingRegressor(),
        # Ridge(),
        Lasso(),
        ElasticNet(),
        # SVR(),
        # XGBRegressor(),
        # CatBoostRegressor(verbose=0),
        # AdaBoostRegressor(DecisionTreeRegressor()),
        # AdaBoostRegressor(LinearRegression()),
        # DecisionTreeRegressor(),
        # RandomForestRegressor(),
        # AdaBoostRegressor()
    )

    def objective(trial, train, test, regressors):
        reg_strs = [r.__repr__() for r in regressors]
        regressor_name = trial.suggest_categorical("classifier", reg_strs)

        if regressor_name == "GradientBoostingRegressor()":
            n_estimators = trial.suggest_int("n_estimators", 50, 200)
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)
            max_depth = trial.suggest_int("max_depth", 3, 10)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
            learner_obj = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
            )
            cv = 7

        elif regressor_name == "RandomForestRegressor()":
            n_estimators = trial.suggest_int("n_estimators", 100, 500)
            max_depth = trial.suggest_int("max_depth", 3, 20)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
            learner_obj = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
            )
            cv = 7

        elif (
            regressor_name
            == "AdaBoostRegressor(base_estimator=DecisionTreeRegressor())"
        ):
            criterion = trial.suggest_categorical(
                "criterion",
                ["absolute_error", "friedman_mse", "squared_error"],  # , "poisson"]
            )
            splitter = trial.suggest_categorical("splitter", ["best", "random"])
            max_depth = trial.suggest_categorical("max_depth", [3, 4, 5])
            min_samples_split = trial.suggest_categorical(
                "min_samples_split",
                [
                    2,
                ],
            )
            min_samples_leaf = trial.suggest_float("min_samples_leaf", 0, 0.5)
            estimator = DecisionTreeRegressor(
                criterion=criterion,
                splitter=splitter,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
            )

            loss = trial.suggest_categorical(
                "loss", ["linear", "square", "exponential"]
            )
            n_estimators = trial.suggest_categorical("n_estimators", [100])
            learner_obj = AdaBoostRegressor(
                estimator, n_estimators=n_estimators, loss=loss
            )
            cv = 7

        elif regressor_name == "AdaBoostRegressor(base_estimator=LinearRegression())":
            loss = trial.suggest_categorical(
                "loss", ["linear", "square", "exponential"]
            )
            n_estimators = trial.suggest_categorical("n_estimators", [100])
            learner_obj = AdaBoostRegressor(
                LinearRegression(), n_estimators=n_estimators, loss=loss
            )
            cv = 7

        elif regressor_name == "DecisionTreeRegressor()":
            criterion = trial.suggest_categorical(
                "criterion",
                ["absolute_error", "friedman_mse", "squared_error"],  # , "poisson"]
            )
            splitter = trial.suggest_categorical("splitter", ["best", "random"])
            max_depth = trial.suggest_categorical("max_depth", [3, 4, 5])
            min_samples_split = trial.suggest_categorical(
                "min_samples_split",
                [
                    2,
                ],
            )
            min_samples_leaf = trial.suggest_float("min_samples_leaf", 0, 0.5)
            learner_obj = DecisionTreeRegressor(
                criterion=criterion,
                splitter=splitter,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
            )
            cv = 7

        elif regressor_name == "Ridge()":
            # alpha = trial.suggest_loguniform('alpha', 1e-5, 10)
            alpha = trial.suggest_float("alpha_ridge", 4, 6)
            learner_obj = Ridge(alpha=alpha)
            cv = 7

        elif regressor_name == "Lasso()":
            alpha = trial.suggest_float("alpha_lasso", 1e-5, 1.0)
            learner_obj = Lasso(alpha=alpha)
            cv = 7

        elif regressor_name == "ElasticNet()":
            alpha = trial.suggest_float("alpha_elastic", 1e-5, 1.0, log=True)
            l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
            learner_obj = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
            cv = 7

        elif regressor_name == "SVR()":
            C = trial.suggest_float("C", 1e-4, 1e-3, log=True)
            kernel = trial.suggest_categorical(
                "kernel", ["linear", "poly", "rbf", "sigmoid"]
            )
            learner_obj = SVR(C=C, kernel=kernel)
            cv = 7

        elif regressor_name == "XGBRegressor()":
            n_estimators = trial.suggest_int("n_estimators", 50, 500)
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)
            max_depth = trial.suggest_int("max_depth", 3, 10)
            learner_obj = XGBRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                tree_method="auto",
                eval_metric="rmse",
            )
            cv = 7

        elif regressor_name == "CatBoostRegressor(verbose=0)":
            depth = trial.suggest_int("depth", 4, 8)
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)
            iterations = trial.suggest_int("iterations", 50, 500)
            learner_obj = CatBoostRegressor(
                depth=depth,
                learning_rate=learning_rate,
                iterations=iterations,
                verbose=0,
            )
            cv = 7

        elif regressor_name == "AdaBoostRegressor()":
            pass

        else:
            pass

        regressor = artifact.Regressor(
            train, test, learner_obj, scaler=StandardScaler()
        )
        scores = regressor.cross_val_score(n_jobs=-1, cv=cv)

        return scores.mean() * 100

    # ### Running the optimization
    #
    # Optuna will sample the parameters automatically, for a maximum number of trials
    # specified.

    study = optuna.create_study(direction="minimize")

    # Adjust n_trials based on the regressor
    # if any(
    #     lr.__class__.__name__ in ["RandomForestRegressor", "GradientBoostingRegressor"]
    #     for lr in learners
    # ):
    #     n_trials = 5
    # else:
    #     n_trials = 50
    n_trials = 50
    make_plots = True

    study.optimize(
        lambda t: objective(t, tkr_train, tkr_test, learners),
        n_trials=n_trials,
        n_jobs=-1,
        # timeout=60 * 60 * 1.5,
    )

    # plot_optimization_history(study).show()
    print(study.best_trial)
    print(
        Fore.YELLOW
        + f"\nBest trial\n  RMSE% = {study.best_value} \n  {study.best_params}"
    )
    print(Style.RESET_ALL)

    # ### Plotting the results from the optimization
    #
    # We can assign the hyperparameters selected by optuna, and plot the resulting joint mechanics.

    learner_strs = [lrn.__repr__() for lrn in learners]
    learner_dict = dict(zip(learner_strs, learners))
    learner_kwargs = study.best_params.copy()
    learner = learner_dict[learner_kwargs["classifier"]]
    learner_kwargs.pop("classifier")
    learner_kwargs_updated = {}
    for k, v in learner_kwargs.items():
        # update the alpha_x keys to just alpha
        if "alpha" in k:
            learner_kwargs_updated["alpha"] = v
        else:
            learner_kwargs_updated[k] = v
    learner.set_params(**learner_kwargs_updated)

    # Save the best study trials and learner keywords
    with open(MODEL_DIR / "learners" / f"best_study_trials_{group}.txt", "w") as f:
        f.write(f"Best trial\n  RMSE% = {study.best_value} \n  {study.best_params}\n")
        f.write(f"Updated learner keywords: {learner_kwargs_updated}\n")

    # Pickle the individual learner
    with open(MODEL_DIR / "learners" / f"learner_{group}.pkl", "wb") as f:
        pickle.dump(learner, f)

    lrn_name = type(learner).__name__
    with contextlib.suppress(AttributeError):
        lrn_name = "-".join((lrn_name, type(learner.base_estimator).__name__))

    top_fig_dir = MODEL_DIR / "predictions"
    save_dir = top_fig_dir / group / lrn_name
    n_rows, n_cols = 4, 3
    try:
        tim = tkr_train.time
    except KeyError:
        tim = np.arange(0, tkr_train.response.shape[0])
    scaler = StandardScaler()
    regr = artifact.Regressor(tkr_train, tkr_test, learner, scaler=scaler)
    regressors[group] = regr
    # regr.fit().predict()
    # clear_output(wait
    if make_plots:
        for resp_name in tkr_train.response_names:
            if resp_name == "time":
                continue
            artifact.create_plots(
                n_rows,
                n_cols,
                regr,
                resp_name,
                save_dir,
                random_combined_plots=True,
                save_subplots=True,
                skip_plotting=not make_plots,
                force_clobber=True,
            )
            # clear_output(wait=True)
    results_file = MODEL_DIR / "learners" / f"results_log.txt"
    print(results_file)
    with results_file.open("a") as f:
        f.write(f"Group: {group}\n")
        f.write(f"Best trial\n  RMSE% = {study.best_value} \n  {study.best_params}\n")
        f.write(f"Updated learner keywords: {learner_kwargs_updated}\n")
        f.write(f"{group} test set RMSE%: {regr.prediction_error * 100}%\n")
        print(f"{group} test set RMSE%: {regr.prediction_error * 100}%")
        for rname in tkr_train.response_names:
            if "time" in rname:
                continue
            scores = regr.score_test_set(rname)
            f.write(f"\t{rname}: {scores * 100}%\n")

    display(f"Scores for {rname}: {scores * 100}%")
    clear_output(wait=True)

# Save the final learners in a dictionary
final_learners = {group: regr.learner for group, regr in regressors.items()}
with open(MODEL_DIR / "learners" / "final_learners.pkl", "wb") as f:
    pickle.dump(final_learners, f)

# %%

results_file = MODEL_DIR / "learners" / "results.txt"
with open(results_file, "w") as f:
    for group, regr in regressors.items():
        mean_rmse = np.mean(regr.prediction_error)
        print(f"group: {group}, mean RMSE%: {100 * mean_rmse:.2f}%")
        f.write(f"group: {group}, mean RMSE%: {100 * mean_rmse}%\n")
        for rname in regr.test_results.response_names:
            if "time" in rname:
                continue
            scores = regr.score_test_set(rname)
            print(f"\t{rname}: {scores * 100}%")
            f.write(f"\t{rname}: {scores * 100}%\n")
# %%
top_fig_dir = MODEL_DIR / "predictions"
view = artifact.plotting.ImageViewer(top_fig_dir)
view.show()

# %%
