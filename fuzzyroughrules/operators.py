import numpy as np


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
        return np.min(self.implicator(A, B))
