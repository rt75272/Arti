"""Helper classes and functions."""

# %%
import contextlib
import pickle
from pathlib import Path

from colorama import Fore, Style
from IPython.display import display


class RegressionProfile:
    """Stores and summarizes the results of regression profiling."""

    def __init__(self, load_path=None):
        """Instantiate or load a RegressionProfile object.

        Args:
            load_path (Path|str, optional): The filepath locating a saved
                RegressionProfile object. Defaults to None.
        """
        self.error_dataframes = dict()
        if load_path:
            with contextlib.suppress(FileNotFoundError):
                self.load(load_path)

    def __repr__(self):
        """Represent RegressionProfile object as a string.

        Returns:
            str: A list of all the keys within the RegressionProfile.
        """
        keys = self.error_dataframes.keys()
        if keys is not None:
            return f"RegressionProfile object with keys {list(keys)}"
        else:
            return "Uninitialized RegressionProfile object"

    def add_results(self, name, error_dataframe):
        """Add profiling error dataframe under an identifying name.

        Args:
            name (str): The uniquely identifying name for these error results.
            error_dataframe (DataFrame): A table of errors from cross
                validated profiling.
        """
        self.error_dataframes[name] = error_dataframe

    def save(self, save_path):
        """Save profiling results to disk.

        Args:
            save_path (Path|str): The path to the profiling results to be
                saved.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(exist_ok=True, parents=True)
        with open(save_path, "wb") as file:
            pickle.dump(self.error_dataframes, file, pickle.HIGHEST_PROTOCOL)

    def load(self, load_path):
        """Load profiling results from disk.

        Args:
            load_path (Path|str): The path to the profiling results to be
                loaded.
        """
        with open(load_path, "rb") as file:
            self.error_dataframes = pickle.load(file)

    def describe(self, name):
        """Print profiling results by regressor.

        Calculates summary statistics across all observations for each
            regressor, as well as aggregating the best regressors for each
            response in a ranked list.

        Args:
            name (str): Name of the functional group to be examined.
        """
        try:
            df = self.error_dataframes[name]
        except KeyError:
            print(f"No error summary with key {name} was found.")
            return
        best_learners = df.idxmin()
        print(1 * "\n")
        print(Fore.YELLOW + name + "\n" + len(name) * "-")
        # print(len(name) * '-')
        print(Style.RESET_ALL)
        print("Best learners total by response:")
        display(best_learners.value_counts(), best_learners.sort_values())
        print("\n\nSorted by median RMS error (smallest to largest):")
        display(df.T.describe().T.sort_values(by=["50%"]))
        print("\n\nRMS Errors:")
        display(df)
        print(2 * "\n")
