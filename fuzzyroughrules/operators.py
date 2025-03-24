import numpy as np
from enum import Enum
from quickrules.weights import Weights, LinearWeights, TruncatedWeights

def lukasiewicz_implicator(x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray | float:
    return np.minimum(1 - x + y, 1)


class ImplicatorInclusion:
    """
    Fuzzy sets are modelled as numpy arrays that contain the membership of each element of the dataset (incl 0's).
    """
    def __init__(
            self,
            implicator=lukasiewicz_implicator,  # should use numpy vectorisation
    ):
        self.implicator = implicator

    def inclusion(self, A: np.ndarray, B: np.ndarray) -> float:
        A = np.squeeze(A)
        B = np.squeeze(B)
        return np.min(self.implicator(A, B))

class StrictInclusion:
    """
    Fuzzy sets are modelled as numpy arrays that contain the membership of each element of the dataset (incl 0's).

    this class models the strict inclusion of fuzzy sets, i.e. A subset of B if A(x) \leq B(x) for all x.
    """

    def inclusion(self, A: np.ndarray, B: np.ndarray) -> float:
        return np.all(np.less_equal(np.squeeze(A), np.squeeze(B)))



class OWAImplicatorInclusion:
    """
    Fuzzy sets are modelled as numpy arrays that contain the membership of each element of the dataset (incl 0's).
    """
    def __init__(
            self,
            implicator=lukasiewicz_implicator,  # should use numpy vectorisation
            weight_function: Weights = LinearWeights(),
    ):
        self.implicator = implicator
        self.weight_function = weight_function

    def inclusion(self, A: np.ndarray, B: np.ndarray) -> float:
        A = np.squeeze(A)
        B = np.squeeze(B)
        weights = self.weight_function(len(A))
        implicator_values = self.implicator(A, B)
        implicator_values.sort()  # memberships are sorted ascending -> we want to emulate "min"

        weighted_memberships = [w * m for w, m in zip(weights, implicator_values)]
        return sum(weighted_memberships)


class RelationTypes(Enum):
    UNUSED = 0
    DOMINATED = 1
    DOMINANT = -1
    INDISCERNIBLE = 2


def triangular_relation(
        x: np.ndarray,
        y: np.ndarray,
        slopes: np.ndarray[float] = None,  # 1 slope per attribute, not per sample
        rel_types: np.ndarray[RelationTypes] = None
):
    # setting default arguments
    if slopes is None:
        slopes = np.ones(x.shape[1], dtype=float)
    if rel_types is None:
        rel_types = np.array(x.shape[1] * [RelationTypes.INDISCERNIBLE])

    x, y = np.atleast_2d(x), np.atleast_2d(y)
    type_divisions = {rel_type: np.where(rel_types == rel_type)[0] for rel_type in
                      [RelationTypes.DOMINATED, RelationTypes.DOMINANT, RelationTypes.INDISCERNIBLE]}

    comparisons = []
    for rel_type, div in type_divisions.items():
        if div.size > 0:
            comparisons.append(triangular_dominance(x[:, div], y[:, div], slopes[div], rel_type))

    if len(comparisons) == 0:
        return np.ones((x.shape[0], y.shape[0]))

    return np.min(np.concatenate(comparisons, -1), -1)


def triangular_dominance(x, y, slope=1., rel_type=RelationTypes.INDISCERNIBLE):
    ext_dim_x = y.shape[0]
    ext_dim_y = x.shape[0]
    x_ext = np.repeat(x[:, np.newaxis, :], ext_dim_x, axis=1)
    y_ext = np.repeat(y[np.newaxis, :, :], ext_dim_y, axis=0)

    res = 1
    if rel_type == RelationTypes.DOMINATED:
        res = np.maximum(np.minimum(1 - (y_ext - x_ext) / slope, 1), 0)
    elif rel_type == RelationTypes.DOMINANT:
        res = np.maximum(np.minimum(1 - (x_ext - y_ext) / slope, 1), 0)
    elif rel_type == RelationTypes.INDISCERNIBLE:
        res = np.maximum(np.minimum(1 - np.abs(y_ext - x_ext) / slope, 1), 0)

    if res.shape[-1] > 0:
        return np.min(res, -1)
    else:
        return 1 + np.sum(res, -1)
