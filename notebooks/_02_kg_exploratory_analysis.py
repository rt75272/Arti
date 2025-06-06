# %% [markdown]
#
# # Comprehensive Exam
#
# ## Coding Artifact
#
# Kalin Gibbons
#
# Nov 20, 2020

# ## Exploratory Analysis

# %%
import logging
import math
import os
import re
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
from statsmodels.formula.api import ols

import artifact
from artifact.datasets import load_imorphics, load_tkr  # noqa: F401

# %%
sns.set_context("poster")
sns.set(rc={"figure.figsize": (16, 9.0)})
sns.set_style("whitegrid")


pd.set_option("display.max_rows", 120)
pd.set_option("display.max_columns", 120)

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

# %% [markdown]
#
# ## Data Exploration
#
# ---
#
# We can begin by loading the datasets from MATLAB binary files. Following that we can
# take a look at the predictor distributions within the training and test sets, and
# confirm that the Latin hypercube and uniform sampling successfully covered the design
# space.


RESULTS_READER = load_imorphics

# %%
# Load and describe the training data
tkr_train = artifact.Results(
    results_reader=RESULTS_READER,
    subset="train",
    functional_groups=["contact_mechanics", "joint_loads"],
    # flags=re.IGNORECASE,
    remove_response_prefix=True,
    include_time_response=True,
    include_time_predictor=False,
)
# %%

tkr_train.describe_features()
plt.show()


# %%
# Load and describe the test space
tkr_test = artifact.Results(
    results_reader=RESULTS_READER,
    subset="test",
    functional_groups=["contact_mechanics", "joint_loads"],
    # flags=re.IGNORECASE,
    remove_response_prefix=True,
    include_time_response=True,
    include_time_predictor=False,
)
tkr_test.describe_features()
plt.show()

# %%
# Optionally load and describe the validation data
try:
    tkr_validation = artifact.Results(
        results_reader=RESULTS_READER,
        subset="validation",
        functional_groups=["contact_mechanics", "joint_loads"],
        # flags=re.IGNORECASE,
        remove_response_prefix=True,
        include_time_response=True,
        include_time_predictor=False,
    )
    tkr_validation.describe_features()
    plt.show()
except Exception as e:
    logging.info(f"No validation dataset found: {e}")

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
plt.figure()
tkr_train.plot_feature_importances()
plt.gca().grid(which="major", axis="x")  # seaborn bug?
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

plt.figure()
tkr_train.plot_feature_pca()
plt.show()


# %%
plt.figure()
tkr_train.plot_feature_importances(use_pareto=True)
plt.show()


# %% [markdown]
#
# Looks like each principal component is contributing about the same amount to the
# overall variance of this feature data, but we've picked up about 80% of the
# importance by the sixth feature. We're performing regression instead of
# classification, so the importance plots holds more weight for our use case. We'll
# leave them in because I want to make a comparison to some earlier results from
# looking at this problem in 2017.

# %%
