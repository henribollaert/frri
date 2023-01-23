import pandas as pd
from typing import Optional, Protocol
from pathlib import Path
from re import search
import os
import numpy as np


def get_dataset(folder_path, keyword, remove_cat=True):
    """
    Returns a dataset from a specified folder with a given keyword in the name, possibly after removing categorical
    features.

    Parameters
    ----------
    folder_path path to folder of the dataset
    keyword     keyword [d*]tra or [d*]tst
    remove_cat  remove categorical features, yes or no?

    Returns
    -------
    tuple containing numpy array containing x values, numpy array containing y values
    """
    set_list = [_ for _ in folder_path.iterdir() if keyword in _.name]
    assert len(set_list) == 1, f'{ len(set_list)} files with {keyword} in their name.'

    dataset = pd.read_csv(set_list[0], header=None, comment='@')
    if remove_cat:
        nums = [t != 'object' for t in dataset.dtypes]
        nums[-1] = False
        x_dataset = dataset.loc[:, nums]
    else:
        x_dataset = dataset.iloc[:, :-1]
    y_dataset = dataset.iloc[:, -1]
    return x_dataset.values, y_dataset.values


class RuleInductionModel(Protocol):
    def fit(self, data, labels):
        ...

    def get_info(self) -> str:
        ...

    def predict(self, datax: np.ndarray) -> np.ndarray:
        ...

    def get_rules_as_string(self) -> list[str]:
        ...


# todo finish, also print info and rules!
def test_save(model: RuleInductionModel,
              datasets_folder: Path,
              results_folder: Path,
              get_rules: bool = True,
              print_info: bool = False,
              exclude: Optional[list[str]] = None,
              include: Optional[list[str]] = None,
              verbose: bool = False,
              nr_of_folds: int = 10
              ):
    """
    This method runs a given model on a collection of data sets and saves the predictions
    in the given path.

    :param model:
    :param datasets_folder:
    :param results_folder:
    :param get_rules:
    :param exclude:
    :param include:
    :param verbose:
    :param nr_of_folds:
    :return:
    """
    for dataset_dir in datasets_folder.iterdir():
        # skip .gitignore files
        if dataset_dir.name[0] == ".":
            continue

        # get short name
        short_name = dataset_dir.name[:search(r'\d', dataset_dir.name).start()][:-1]

        # skip files if we're using excluded or must_include
        if (exclude is not None and short_name in exclude) or \
                (include is not None and short_name not in include):
            continue

        if verbose:
            print(short_name)

        # create the folder for the results on this dataset
        dataset_result_path = results_folder / short_name
        if not os.path.exists(dataset_result_path):
            os.makedirs(dataset_result_path)

        for fold in range(nr_of_folds):
            # create the folder for the results on this fold of the dataset
            fold_result_path = dataset_result_path / f"fold{fold + 1}"
            if not os.path.exists(fold_result_path):
                os.makedirs(fold_result_path)

            # get the train and test sets
            x_train, y_train = get_dataset(dataset_dir, f"{fold + 1}tra")
            x_test, y_test = get_dataset(dataset_dir, f"{fold + 1}tst")

            # skip if we already have results for these parameters
            if (fold_result_path / f"fold{fold + 1}.dat").is_file():
                continue

            # create the rules
            model.fit(x_train, y_train)

            # query on the test set
            predictions = model.predict(x_test)

            # save the predictions
            with open(fold_result_path / f"fold{fold + 1}.dat", 'w') as f:
                if print_info:
                    f.write(model.get_info())
                for item in predictions:
                    f.write(f"{item}\n")

            if get_rules:
                with open(fold_result_path / f"rules_fold{fold + 1}.dat", 'w') as f:
                    for item in model.get_rules_as_string():
                        f.write(f"{item}\n")


def calculate_score(data_folder,
                    results_folder,
                    metric,
                    exclude=None,
                    nr_of_folds=10,
                    verbose=False):
    if exclude is None:
        exclude = ['abalone']
    scores = {}
    for dataset_dir in data_folder.iterdir():
        # skip .gitignore files
        if dataset_dir.name[0] == ".":
            continue
        # very dirty, search depends on the fact that the names don't contain underscores
        short_name = dataset_dir.name[:search(r'\d', dataset_dir.name).start()][:-1]
        if short_name in exclude:
            continue

        if verbose:
            print(short_name)

        dataset_result_path = results_folder / short_name

        # create dictionaries to save results of this dataset
        sum_of_metrics = 0
        for fold in range(nr_of_folds):

            # create the folder for the results on this fold of the dataset
            fold_result_path = dataset_result_path / f"fold{fold + 1}"
            if fold_result_path.exists():
                _, y_test = get_dataset(dataset_dir, f"{fold + 1}tst")
                predictions = pd.read_csv(fold_result_path / f"{fold + 1}.dat",
                                          comment='@', header=None)
                sum_of_metrics += metric(y_test, predictions)

        # add scores to the dictionary
        scores[short_name] = sum_of_metrics / nr_of_folds

    return scores
