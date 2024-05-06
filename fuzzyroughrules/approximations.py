import fuzzyroughrules.fuzzy_operators as fo
import numpy as np
from typing import Optional
from dataclasses import dataclass


class LowerApproximation:
    def __init__(self, implicator=None):
        self.implicator = implicator

    def get_approximation(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        rel_matrix_x = fo.triangular_similarity(X, X)
        rel_matrix_y = fo.discernibility_matrix(y, y)
        return fo.get_ind_low_apr(rel_matrix_x, rel_matrix_y)

    def __str__(self):
        return "LowerApproximation()"


@dataclass
class MulticlassMSECVXApproximation:
    weights: Optional[np.ndarray] = None
    nn_approx: int = -1
    n_jobs: Optional[int] = None

    def get_approximation(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        rel_matrix_x = fo.triangular_similarity(X, X)
        rel_matrix_y = fo.discernibility_matrix(y, y)
        return fo.get_multiclass_granular_approx_mse_cvxopt(
            relation_matrix_x=rel_matrix_x,
            relation_matrix_y=rel_matrix_y,
            weights=self.weights,
            nn_approx=self.nn_approx,
            n_jobs=self.n_jobs
        )
