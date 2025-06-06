from pathlib import Path

from artifact import __file__ as artifact_file

PACKAGE_DIR = Path(artifact_file).parents[2]
DATA_DIR = PACKAGE_DIR / "data"
MODEL_DIR = PACKAGE_DIR / "models"
CONFIG_DIR = PACKAGE_DIR / "configs"
LOG_DIR = PACKAGE_DIR / "logs"
# %%
REGRESSION_PROFILE_PATH = (
    Path.cwd().parent / "models" / "selection" / "learner_profiles.pkl"
)
RESPONSE_PREFIX = "response_"
