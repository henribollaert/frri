"""
Data handling for horizontal federated learning: all clients have the same features, but different objects.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from hhelper.data_loader import get_dataset

PROTOCOL_SEEDS = range(5)


def load_keel_dataset(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads a complete dataset from its KEEL folder. KEEL only provides 10 folds, and the train and test file of a fold
    together contain all objects. The split into folds is not used: the protocol makes its own splits.
    :param dataset_dir: folder containing the fold files, e.g. keel-data/yeast-10-fold
    :return: values of conditional features, values of decision attribute
    """
    x_train, y_train = get_dataset(dataset_dir, "1tra")
    x_test, y_test = get_dataset(dataset_dir, "1tst")
    return np.vstack([x_train, x_test]), np.concatenate([y_train, y_test])


@dataclass
class ClientData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


@dataclass
class HorizontalSplit:
    X_test: np.ndarray
    y_test: np.ndarray
    clients: list[ClientData]


def _hold_out(
        indices: np.ndarray,
        y: np.ndarray,
        fraction: float,
        rng: np.random.Generator,
        stratify: bool
) -> tuple[np.ndarray, np.ndarray]:
    """
    Splits the indices in a remaining part and a held out part. When stratifying, a class that is too small to give
    an object to the held out part stays in the remaining part.
    """
    labels = y[indices] if stratify else np.zeros(len(indices))
    remaining, held_out = [], []
    for label in np.unique(labels):
        group = rng.permutation(indices[labels == label])
        n_held_out = int(round(fraction * len(group)))
        held_out.append(group[:n_held_out])
        remaining.append(group[n_held_out:])
    return rng.permutation(np.concatenate(remaining)), np.concatenate(held_out)


def _deal(indices: np.ndarray, y: np.ndarray, n_clients: int, stratify: bool) -> list[np.ndarray]:
    """
    Deals the indices out to the clients, so client sizes differ by at most 1. Sorting by class first means that this
    also holds for the number of objects of each class, even for classes with fewer objects than clients.
    """
    ordered = indices[np.argsort(y[indices], kind="stable")] if stratify else indices
    return [ordered[client::n_clients] for client in range(n_clients)]


def horizontal_split(
        X: np.ndarray,
        y: np.ndarray,
        n_clients: int,
        seed: int,
        stratify: bool = True
) -> HorizontalSplit:
    """
    Beata's protocol: 80/20 global train/test split, equal and disjoint split of the training objects over the
    clients, 80/20 train_local/test_local split on each client.
    :param X: conditional features
    :param y: decision attribute
    :param n_clients: number of clients
    :param seed: seed of the random splits
    :param stratify: do we keep the class distribution in each split. The protocol does not specify this.
    :return: the global test set and the data of each client
    """
    rng = np.random.default_rng(seed)
    train, test = _hold_out(np.arange(len(X)), y, 0.2, rng, stratify)

    clients = []
    for indices in _deal(train, y, n_clients, stratify):
        train_local, test_local = _hold_out(indices, y, 0.2, rng, stratify)
        clients.append(ClientData(X[train_local], y[train_local], X[test_local], y[test_local]))
    return HorizontalSplit(X[test], y[test], clients)


def global_min_max(client_ranges: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """
    :param client_ranges: (per-feature minimum, per-feature maximum) of the train_local data of each client
    :return: per-feature minimum and maximum over all clients
    """
    mins, maxs = zip(*client_ranges)
    return np.min(mins, axis=0), np.max(maxs, axis=0)


def global_classes(client_labels: list[np.ndarray]) -> np.ndarray:
    """
    :param client_labels: labels in the train_local data of each client
    :return: sorted labels over all clients. Rule decisions in global models are indices in this array.
    """
    return np.unique(np.concatenate(client_labels))


class FixedRangeScaler:
    """
    Min-max scaler for a range that is known beforehand. Fitting does nothing, so RuleGenerator.fit can not replace
    the global range by the range of the data of a client.
    """

    def __init__(self, data_min: np.ndarray, data_max: np.ndarray):
        self.scaler = MinMaxScaler().fit(np.vstack([data_min, data_max]))
        self.n_features_in_ = len(data_min)

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.transform(X)

    def fit_transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        return self.transform(X)
