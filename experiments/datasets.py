"""
Datasets for the federated experiments that do not need files: scikit-learn built-in datasets, synthetic data, and
public OpenML datasets, which are downloaded into the scikit-learn cache and never stored in the repository.

Every loader returns the features and the labels, not normalised.
"""
from typing import Callable

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine, make_classification


def _sklearn(load: Callable) -> Callable[[], tuple[np.ndarray, np.ndarray]]:
    return lambda: load(return_X_y=True)


def synthetic(n_samples: int = 1000, n_features: int = 10, n_classes: int = 4, seed: int = 0):
    """
    Synthetic data with half of the features informative. The seed is that of the data, not of the splits.
    """
    return make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, n_features // 2),
        n_redundant=0,
        n_classes=n_classes,
        n_clusters_per_class=2,
        random_state=seed,
    )


def openml(name: str, version: int = 1) -> Callable[[], tuple[np.ndarray, np.ndarray]]:
    """
    Needs network access the first time. Only numeric features are supported. Not tested yet: the machine where this
    was written could not reach OpenML.
    """
    def load():
        from sklearn.datasets import fetch_openml
        X, y = fetch_openml(name, version=version, return_X_y=True, as_frame=False, parser="auto")
        return X.astype(float), y
    return load


DATASETS: dict[str, Callable[[], tuple[np.ndarray, np.ndarray]]] = {
    "iris": _sklearn(load_iris),
    "wine": _sklearn(load_wine),
    "breast_cancer": _sklearn(load_breast_cancer),
    "digits": _sklearn(load_digits),
    "synthetic": synthetic,
    "yeast": openml("yeast", 1),
}
