import fuzzyroughrules.fuzzy_operators as fo
import numpy as np


class LowerApproximation:
    def get_approximation(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        rel_matrix_x = fo.triangular_similarity(X, X)
        rel_matrix_y = fo.discernibility_matrix(y, y)
        return fo.get_ind_low_apr(rel_matrix_x, rel_matrix_y)

    def __str__(self):
        return "LowerApproximation()"

# class