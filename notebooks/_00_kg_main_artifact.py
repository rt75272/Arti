# %% [markdown]
# # Comprehensive Exam
#
# ## Coding Artifact
#
# Kalin Gibbons
#
# Nov 20, 2020

# ## Introduction
#
# ---
#
# Outcomes of total knee arthroplasty (TKA) are dependent on surgical technique,
# patient variability, and implant design. Poor surgical or design choices can lead to
# undesirable contact mechanics and joint kinematics, including poor joint alignment,
# instability, and reduced range of motion. Of these three factors, implant design and
# surgical alignment are within our control, and there is a need for robust implant
# designs that can accommodate variability within the patient population. One of the
# tools used to evaluate implant designs is simulation through finite element analysis
# (FEA), which offers considerable early design-stage speedups when compared to
# traditional prototyping and mechanical testing. Nevertheless, the usage of FEA
# predicates a considerable amount of software and engineering knowledge, and it can
# take a great deal of time, compute or otherwise, to generate and analyze results.
# Currently used hardware and software combinations can take anywhere from 4 to 24
# hours to run a single simulation of one daily-living task, for a moderately complex
# knee model. A possible solution to this problem is to use the FEA results to train
# predictive machine learning regression models capable of quickly iterating over a
# number of potential designs. Such models could be used to hone in on a subset of
# potential designs worthy of further investigation.

# ### Data Description
#
# This _training_ dataset is generated from simplified finite element models of a
# cruciate-sacrificing, post and cam driven knee implant performing a deep-knee-bend.
# The implant geometries and surgical alignments are parameterized by 13 predictor
# variables which were drawn using Latin hypercube sampling from a range of currently
# used manufacturer dimensions, and angles performed during successful surgeries. There
# were originally 15 predictors for this dataset, but two were fixed at average values
# for this particular batch of simulations. For the test dataset, the same predictors
# were uniformly drawn across the ranges of potential values.


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
from IPython.display import clear_output, display
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from statsmodels.formula.api import ols
from tqdm.auto import tqdm

import artifact
from artifact.datasets import kl_group_lut, load_tkr, tkr_group_lut

# %%
plt.rcParams["figure.figsize"] = (9, 5.5)
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["font.size"] = 14
mpl.rcParams["font.family"] = "Times New Roman"

sns.set_context("poster")
sns.set(rc={"figure.figsize": (16, 9.0)})
sns.set_style("whitegrid")


pd.set_option("display.max_rows", 120)
pd.set_option("display.max_columns", 120)

logging.basicConfig(level=logging.INFO, stream=sys.stdout)


# %% [markdown]
# #
#
# ## Data [Cleaning](01-kg-data-cleaning.py) and [Exploratory Analysis](02-kg-exploratory-analysis.py)
#
# ---
#
# We can begin by loading the datasets from MATLAB binary files. Following that we can
# take a look at the predictor distributions within the training and test sets, and
# confirm that the Latin hypercube and uniform sampling successfully covered the design
# space.


# %%
# Load and describe the training data
tkr_train = artifact.Results(results_reader=load_tkr, subset="train")
tkr_train.describe_features()
plt.show()


# %%
# Load and describe the test space
tkr_test = artifact.Results(results_reader=load_tkr, subset="test")
tkr_test.describe_features()
plt.show()


# %% [markdown]
#
# It looks like the LHS did a good job of covering all of the design space for our
# training set, but our histograms are looking less uniform for the testing set. The
# parameter for the offset of the trochlear groove in particular seem right-skewed. We
# can lean on domain knowledge to guess that a positive offset lead to more simulation
# failures, which we have already removed. This makes sense because your trochlear
# groove is a smooth asymmetrical surface that your patella (kneecap) rides in, and
# moving this in a way that compounds the effect of asymmetrical vasti muscle
# distribution would increase the likelihood of patellar dislocation during, making
# simulated patient fail to complete their deep-knee-bend. A negative offset most
# likely reduced the risk of these dislocations, while a positive offset had the
# opposite effect. We'll skip going back and checking the histograms before removing
# the empty rows; that's a problem for the FEA modeler.

# ### Ranking Feature Importances
#
# This dataset began with 15 features, but cam radius and femoral flexion-extension
# were removed after being found to be the least important. Let's check how the other
# features rank.


# %%
sns.set_theme("poster", "whitegrid", font="Times New Roman")
tkr_train.plot_feature_importances()
plt.gca().grid(which="major", axis="x")  # seaborn bug makes this opposite?
plt.show()


# %% [markdown]
#
# This dataset has been truncated to only include contact mechanics and joint loads
# response data, and we're seeing posterior femoral radius and tibial conformity ratios
# as being the most important, followed by internal-external alignment of the tibial
# insert. This makes sense because the majority of our contact mechanics response
# variables concern the center of pressure ordinates, and the area of contact between
# the femoral component and the plastic tibial spacer.
#
# Some of those features don't seem very important. Let's try running principal
# component analysis on this dataset to see how quickly we're capturing variance.

# %%

sns.reset_orig()
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["font.size"] = 32
mpl.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.figsize"] = (16, 9)
plt.rcParams["lines.linewidth"] = 3.5
plt.rcParams["lines.markersize"] = 15

# %%
tkr_train.plot_feature_pca()
plt.show()


# %%
tkr_train.plot_feature_importances(use_pareto=True)
plt.show()


# %% [markdown]
#
# Looks like each variable is contributing about the same amount to the overall
# variance of this feature data, but we've picked up about 80% of the importance by the
# sixth feature. We're performing regression instead of classification, so the
# importance plots holds more weight for our use case. We'll leave them in because I
# want to make a comparison to some earlier results from looking at this problem in
# 2017.

# ## Regression [Model Selection](./03-kg-model-selection.py)
#
# ---
#
# Now that we understand our data, let's see if we can predict the output values. We're
# going to skip splitting into a develop-validate-test set for this example project,
# again so I can compare to earlier results.


# %% Choose the response of interest

learners = [
    # GradientBoostingRegressor(n_estimators=100),
    # RandomForestRegressor(n_estimators=100),
    # AdaBoostRegressor(DecisionTreeRegressor(), n_estimators=100),
    # AdaBoostRegressor(LinearRegression(), n_estimators=100),
    DecisionTreeRegressor(),
    Ridge(),
]
learner_names = [x.__str__().replace("()", "") for x in learners]
scaler = StandardScaler()
regr = artifact.Regressor(tkr_train, tkr_test, learners[0], scaler=scaler)
err_df = pd.DataFrame(index=learner_names)

resp_pbar = tqdm(regr.train_results.response_names, desc="Processing...")
for resp in resp_pbar:
    if resp == "time":
        continue
    resp_pbar.set_description(f"Processing {resp}")
    errs = np.zeros_like(learner_names, dtype=float)
    lrn_pbar = tqdm(learners, desc="Fitting...", leave=False)
    for idx, lrn in enumerate(lrn_pbar):
        desc = f'{learner_names[idx].replace("base_estimator=", "")}'
        lrn_pbar.set_description(desc)
        regr.learner = MultiOutputRegressor(lrn)
        y_pred = regr.fit(resp).predict()
        errs[idx] = regr.prediction_error
    lrn_pbar.close()
    err_df[resp] = errs

resp_pbar.close()
clear_output(wait=True)
best_learners = err_df.idxmin()

print("Best learner counts:")
display(best_learners.value_counts(), best_learners.sort_values())


# %% [markdown]
#
# Looks like the ordinary linear regression had the smallest errors. Let's take a look
# at the plots and see how it does.


# %%

print("Exporting can take a long time. Would you like to export plots? Type 'yes'")
user_input = input()
export_plots = True if user_input.lower() == "yes" else False
if export_plots:
    sns.reset_orig()

    sns.set_context("poster")
    sns.set(rc={"figure.figsize": (16, 20)})
    sns.set_theme(style="white", font_scale=1.6, font="Times New Roman")

    plt.rcParams["lines.linewidth"] = 4.0

    func_groups = list(kl_group_lut.keys())
    regr.learner = MultiOutputRegressor(LinearRegression())
    lrn_name = type(regr.learner.estimator).__name__
    top_fig_dir = Path.cwd().parent / "models" / "final_plots"
    n_rows, n_cols = 4, 3
    tim = tkr_train.response["time"][0]
    for group in func_groups:
        save_dir = top_fig_dir / group / lrn_name
        shared_kwargs = dict(results_reader=load_tkr, functional_groups=group)
        tkr_train = artifact.Results(**shared_kwargs, subset="train")
        tkr_test = artifact.Results(**shared_kwargs, subset="test")
        for resp_name in tkr_train.response_names:
            if resp_name == "time":
                continue
            artifact.create_plots(n_rows, n_cols, regr, resp_name, save_dir)
        clear_output(wait=True)


# %%
view = artifact.plotting.ImageViewer(top_fig_dir)
view.show()

# %%
