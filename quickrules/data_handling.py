import pandas as pd
from typing import Optional, Protocol, Union
from pathlib import Path
from re import search
import os
import numpy as np
from sklearn.base import BaseEstimator


def get_dataset(
        folder_path: Path,
        keyword: str,
        remove_cat: bool = False,
        get_datatypes: bool = False) -> tuple[np.ndarray, ...]:
    """
    Returns the unique dataset in the folder indicated by folder_path that contains the keyword.
    Categorical features can be removed, and the datatypes can also be returned
    :param folder_path: path where the data is located
    :param keyword: keyword contained in the name of the dataset
    :param remove_cat: should categorical features be removed (except for the decision attribute)
    :param get_datatypes: should we return the
    :return: values of conditional features, values of decision attribute,
    possibly the datatypes of the conditional features
    """
    set_list = [_ for _ in folder_path.iterdir() if keyword in _.name]
    assert len(set_list) == 1, f'{len(set_list)} files with {keyword} in their name.'

    dataset = pd.read_csv(set_list[0], header=None, comment='@')
    if remove_cat:
        nums = [t != 'object' for t in dataset.dtypes]
        nums[-1] = False
        x_dataset = dataset.loc[:, nums]
    else:
        x_dataset = dataset.iloc[:, :-1]
    if get_datatypes:
        result = x_dataset.values, dataset.iloc[:, -1].values, x_dataset.dtypes.values
    else:
        result = x_dataset.values, dataset.iloc[:, -1].values
    return result


class RuleInductionModel(Protocol):
    def fit(self, data, labels, types):
        ...

    def get_info(self) -> str:
        ...

    def predict(self, datax: np.ndarray) -> np.ndarray:
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

            # get the train and test sets
            x_train, y_train, t_train = get_dataset(dataset_dir, f"{fold + 1}tra", get_datatypes=True)
            x_test, y_test = get_dataset(dataset_dir, f"{fold + 1}tst", get_datatypes=False)

            # skip if we already have results for these parameters
            if (fold_result_path / f"fold{fold + 1}.dat").is_file():
                continue

            if encode_labels:
                # encode the labels to ints
                classes = list(np.unique(np.append(y_train, y_test)))
                y_train = np.array([classes.index(label) for label in y_train])

                # save the encoding
                with open(fold_result_path / f"label_encoding_fold{fold + 1}.npy", 'wb') as f:
                    np.save(f, classes)

            # TRAINING AND PREDICTION
            lines = []
            if print_info:
                lines.append(model.get_info())
            try:
                # fit to the training set
                if use_data_types:
                    model.fit(x_train, y_train, t_train)
                else:
                    model.fit(x_train, y_train)
            except Exception as err:
                lines.extend([f"Error while training on fold {fold}.", str(err)])
                if verbose:
                    print(lines)
            else:
                try:
                    # query on the test set
                    lines = model.predict(x_test)
                except Exception as err:
                    lines = [f"Error while predicting on fold {fold}.", str(err)]
                    if verbose:
                        print(lines)
                else:
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


def calculate_score(data_folder: Path,
                    results_folder: Path,
                    metric,
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
        sum_of_metrics = 0
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
                    sum_of_metrics += metric(y_test, predictions)
                except TypeError:
                    print(f"Unsuccessful for fold{fold + 1} on {short_name}.")
                else:
                    successful_folds += 1

        # add scores to the dictionary
        scores[short_name] = sum_of_metrics / successful_folds if successful_folds > 0 else np.NaN

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
                         counter: str = ',',
                         nr_of_folds: int = 10,
                         verbose: bool = False) -> dict[str, float]:
    """
    Counts the average number of attributes in the rules generated for achieving the results in the results folder
    :param counter: todo def make prettier
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
            sum_of_attributes += np.average(count_attributes(
                dataset_result_path / f"fold{fold + 1}" / f"rules_fold{fold + 1}.dat"
            ))

        # add scores to the dictionary
        amount_of_attributes[dataset_dir.name] = sum_of_attributes / nr_of_folds

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


def balanced_accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64 | np.ndarray:
    """
    Calculate the balanced accuracy score given the true labels and predicted labels.
    :param y_true: The true labels.
    :param y_pred: The predicted labels.

    :return: The balanced accuracy score.
    """
    classes = list(np.unique(np.append(y_true, y_pred)))

    y_true_prime = np.array([classes.index(label) for label in y_true])
    y_pred_prime = np.array([classes.index(label) for label in y_pred])

    # Count the number of occurrences of each label in y_true
    label_counts = np.bincount(y_true_prime)

    # Calculate the number of occurrences of each label in y_pred for each true label
    tp_counts = np.bincount(y_true_prime[y_pred_prime == y_true_prime], minlength=len(label_counts))

    # Calculate the balanced accuracy
    balanced_acc = np.mean(
        np.divide(tp_counts, label_counts, out=np.zeros_like(tp_counts, dtype='float64'), where=label_counts != 0),
        where=label_counts != 0
    )

    return balanced_acc


def bold(data, optimum='max', format_string="%.3f"):
    """
    Returns a pandas dataframe with formatted strings and bolded maximaL values.
    :param data:
    :param optimum:
    :param format_string:

    :return: data with bolded specified optimum
    """
    if optimum == 'max':
        optima = data != data.max()
    else:
        optima = data != data.min()
    bolded = data.apply(lambda x: "\\textbf{%s}" % format_string % x)
    formatted = data.apply(lambda x: format_string % x)
    return formatted.where(optima, bolded)
