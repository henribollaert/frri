import pandas as pd
from typing import Optional, Protocol
from pathlib import Path
from re import search
import os
import numpy as np
from timeit import default_timer as timer

from hhelper.data_loader import get_dataset


class RuleInductionModel(Protocol):
    def fit(self, data, labels, types: Optional):
        ...

    def get_info(self) -> str:
        ...

    def predict(self, datax: np.ndarray) -> np.ndarray:
        ...

    def predict_proba(self, datax: np.ndarray) -> np.ndarray:
        ...

    def get_rules_as_string(self) -> list[str]:
        ...


def skip(name: str, include: Optional[list[str]], exclude: Optional[list[str]]):
    """
    Returns true if the data set should be skipped.
    :param name: name of the data set
    :param include: collection of names of data sets that should be included, or None
    :param exclude: collection of names of data sets that should be excluded, or None
    :return: True if the data set should be skipped
    """
    return (exclude is not None and name in exclude) or (include is not None and name not in include)


def test_save(
        model: RuleInductionModel,
        datasets_folder: Path,
        results_folder: Path,
        get_rules: bool = True,
        print_info: bool = False,
        exclude: Optional[list[str]] = None,
        include: Optional[list[str]] = None,
        verbose: bool = False,
        nr_of_folds: int = 10,
        encode_labels: bool = False,
        use_data_types: bool = True,  # todo this can be handled better (* operator?)
        save_probas: bool = False,
        timing: bool = False  # todo implement timing
) -> None:
    """
    This method runs a given model on a collection of data sets and saves the predictions
    in the given path.

    :param model: model to test
    :param datasets_folder: folder containing the data sets
    :param results_folder: folder where the results should be saved
    :param get_rules: should we save the rules?
    :param exclude: list of data sets to exclude
    :param include: list of data sets to include
    :param verbose: should we print the name of the data set we are testing on?
    :param nr_of_folds: number of folds for the cross-validation
    :param encode_labels: should we encode the labels as ints and save the dict to a file?
    :param save_probas: save probabilities instead of just the productions
    :param timing: should we track the time each run takes?
    :return: Nothing
    """
    for dataset_dir in datasets_folder.iterdir():
        # skip .gitignore files
        if dataset_dir.name[0] == ".":
            continue

        # get short name
        short_name = dataset_dir.name[:search(r'\d', dataset_dir.name).start()][:-1]

        # skip files if we're using excluded or must_include
        if skip(short_name, include, exclude):
            continue

        if verbose:
            print(short_name)

        # create the folder for the results on this dataset
        dataset_result_path = results_folder / short_name
        if not os.path.exists(dataset_result_path):
            os.makedirs(dataset_result_path)

        for fold in range(nr_of_folds):
            # PRELIMINARIES

            # create the folder for the results on this fold of the dataset
            fold_result_path = dataset_result_path / f"fold{fold + 1}"
            if not os.path.exists(fold_result_path):
                os.makedirs(fold_result_path)

            # skip if we already have results for these parameters
            if (fold_result_path / f"fold{fold + 1}.dat").is_file():
                continue

            # get the train and test sets
            x_train, y_train, t_train = get_dataset(dataset_dir, f"{fold + 1}tra", get_datatypes=True)
            x_test, y_test = get_dataset(dataset_dir, f"{fold + 1}tst", get_datatypes=False)

            if encode_labels:
                # encode the labels to ints
                classes = list(np.unique(np.append(y_train, y_test)))
                y_train = np.array([classes.index(label) for label in y_train])

                # save the encoding
                with open(fold_result_path / f"label_encoding_fold{fold + 1}.npy", 'wb') as f:
                    np.save(f, classes)

            # TRAINING AND PREDICTION
            lines = []
            predictions: Optional[np.ndarray] = None
            try:
                # fit to the training set
                if use_data_types:
                    if timing:
                        start = timer()
                    model.fit(x_train, y_train, t_train)
                    if timing:
                        end = timer()
                else:
                    if timing:
                        start = timer()
                    model.fit(x_train, y_train)
                    if timing:
                        end = timer()
                if print_info:
                    lines.append(model.get_info())
                if timing:
                    lines.append(f"@execution_time: {end - start}s")
            except Exception as err:
                lines.extend([f"Error while training on fold {fold + 1}.", str(err)])
                if verbose:
                    print(lines)
            else:
                try:
                    # query on the test set
                    if save_probas:
                        predictions = model.predict_proba(x_test)
                    else:
                        predictions = model.predict(x_test)
                except Exception as err:
                    lines.extend([f"Error while predicting on fold {fold + 1}.", str(err)])
                    if verbose:
                        print(lines)
                finally:
                    # save the rules
                    if get_rules:
                        with open(fold_result_path / f"rules_fold{fold + 1}.dat", 'w') as f:
                            for item in model.get_rules_as_string():
                                f.write(f"{item}\n")
            finally:
                # save the predictions
                with open(fold_result_path / f"fold{fold + 1}.dat", 'w') as f:
                    for item in lines:
                        f.write(f"{item}\n")
                    if predictions is not None:  # this was one tab further
                        if save_probas:
                            np.savetxt(fname=f, X=predictions, delimiter=",", fmt='%.7f')
                        else:
                            np.savetxt(fname=f, X=predictions, delimiter=",", fmt='%i')


def calculate_score(data_folder: Path,
                    results_folder: Path,
                    metric,
                    aggregation_function=np.mean,
                    label_encoding: bool = False,
                    exclude: Optional[list[str]] = None,
                    include: Optional[list[str]] = None,
                    nr_of_folds: int = 10,
                    verbose: bool = False
                    ) -> dict[str, float]:
    """
    This method returns the average value of the metric on each data set in the data folder.
    :param label_encoding: do we need to look up the label encoding?
    :param data_folder: folder containing the data sets
    :param results_folder: folder containing the results
    :param metric: metric to use
    :param aggregation_function: function used to aggregate the metric scores on different folds
    :param exclude: data sets to exclude
    :param include: data sets to include
    :param nr_of_folds: number of folds of the cross-validation
    :param verbose: should we print the name of the data set on which we are calculating the score?
    :return: dictionary containing the average score on each data set
    """
    if exclude is None:
        exclude = ['abalone']
    scores = {}
    for dataset_dir in data_folder.iterdir():
        # skip .gitignore files
        if dataset_dir.name[0] == ".":
            continue
        # very dirty, search depends on the fact that the names don't contain underscores
        short_name = dataset_dir.name[:search(r'\d', dataset_dir.name).start()][:-1]
        if skip(short_name, include, exclude):
            continue

        if verbose:
            print(short_name)

        dataset_result_path = results_folder / short_name

        # create dictionaries to save results of this dataset
        metrics = []
        successful_folds = 0
        for fold in range(nr_of_folds):

            # create the folder for the results on this fold of the dataset
            fold_result_path = dataset_result_path / f"fold{fold + 1}"
            if fold_result_path.exists():
                _, y_test = get_dataset(dataset_dir, f"{fold + 1}tst")

                if label_encoding:
                    classes = list(np.load(str(fold_result_path / f"label_encoding_fold{fold + 1}.npy")))
                    y_test = np.array([classes.index(label) for label in y_test])

                predictions = pd.read_csv(fold_result_path / f"fold{fold + 1}.dat",
                                          comment='@', header=None).values
                try:
                    metrics.append(metric(y_test, predictions))
                except TypeError:
                    print(f"Unsuccessful for fold{fold + 1} on {short_name}.")
                else:
                    successful_folds += 1

        # add scores to the dictionary
        scores[short_name] = aggregation_function(metrics) if successful_folds > 0 else np.NaN

    return scores


def count_all_rules(results_folder: Path,
                    exclude: Optional[list[str]] = None,
                    include: Optional[list[str]] = None,
                    nr_of_folds: int = 10,
                    verbose: bool = False) -> dict[str, float]:
    """
    Counts the rules generated for achieving the results in the results folder
    :param results_folder: path to the folder containing the results and the rules
    :param exclude: data sets to exclude
    :param include: data sets to include
    :param nr_of_folds: number of folds used in cross-validation
    :param verbose: should we print the name of the data set on which we are counting the rules?
    :return: dictionary of the average number of rules for each data set
    """
    if exclude is None:
        exclude = ['abalone']
    amount_of_rules = {}
    for dataset_dir in results_folder.iterdir():
        if dataset_dir.name[0] == "." or skip(dataset_dir.name, include, exclude):
            continue

        if verbose:
            print(dataset_dir.name)

        dataset_result_path = results_folder / dataset_dir.name

        # create dictionaries to save results of this dataset
        sum_of_rules = 0
        for fold in range(nr_of_folds):
            # look up the folder for the results on this fold of the dataset
            sum_of_rules += count_rules(
                dataset_result_path / f"fold{fold + 1}" / f"rules_fold{fold + 1}.dat"
            )

        # add scores to the dictionary
        amount_of_rules[dataset_dir.name] = sum_of_rules / nr_of_folds

    return amount_of_rules


def count_rules(file: Path) -> int:
    """
    Counts the number of rules (i.e. lines) in a file
    :param file: path to the file containing the rules
    :return: number of rules
    """
    with open(file, 'r') as f:
        amount = len(f.readlines())
    return amount


def count_all_attributes(results_folder: Path,
                         exclude: Optional[list[str]] = None,
                         include: Optional[list[str]] = None,
                         metric=np.average,
                         counter: str = ',',
                         nr_of_folds: int = 10,
                         verbose: bool = False) -> dict[str, float]:
    """
    Counts the average metric(number of attributes) in the rules generated
    for achieving the results in the results folder.
    SO if metric is median, we calculate the median on each fold and then return the average median
    rule length.
    :param metric: summary metric to apply to the list of lengths
    :param counter: str that gets counted on every line todo def make prettier
    :param results_folder: path to the folder containing the results and the rules
    :param exclude: data sets to exclude
    :param include: data sets to include
    :param nr_of_folds: number of folds used in cross-validation
    :param verbose: should we print the name of the data set on which we are counting the rules?
    :return: dictionary of the average number of attributes in the rules for each data set
    """
    if exclude is None:
        exclude = ['abalone']
    amount_of_attributes = {}
    for dataset_dir in results_folder.iterdir():
        if dataset_dir.name[0] == "." or skip(dataset_dir.name, include, exclude):
            continue

        if verbose:
            print(dataset_dir.name)

        dataset_result_path = results_folder / dataset_dir.name

        # create dictionaries to save results of this dataset
        sum_of_attributes = 0
        for fold in range(nr_of_folds):
            # look up the folder for the results on this fold of the dataset
            sum_of_attributes += metric(count_attributes(
                dataset_result_path / f"fold{fold + 1}" / f"rules_fold{fold + 1}.dat",
                counter=counter
            ))

        # add scores to the dictionary
        amount_of_attributes[dataset_dir.name] = sum_of_attributes / nr_of_folds
        if verbose:
            print(amount_of_attributes[dataset_dir.name])

    return amount_of_attributes


def count_attributes(file: Path, counter: str = ',') -> list[int]:
    """
    Counts the number of conditional attributes of each rule in a file.
    :param counter: which string to count
    :param file: path to the file containing the rules
    :return: list containing the number of conditional attributes
    """
    with open(file, 'r') as f:
        nrs = []
        for line in f.readlines():
            nrs.append(line.count(counter))
    return nrs
