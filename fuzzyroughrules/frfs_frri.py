from dataclasses import dataclass

import numpy as np
from sklearn.utils.validation import check_X_y
from fuzzyroughrules.rule_induction_base import RuleGenerator
from frlearn.feature_preprocessors import FRFS
from sklearn.utils.validation import check_is_fitted
from frlearn.statistics.feature_preprocessors import Standardiser


class OFRFS(FRFS):
    """
    Conserves the order in which attributes are selected by FRFS.
    """

    def _construct(self, X, y):
        model = super()._construct(X, y)
        X_scaled = Standardiser()(X)(X)
        R_a = np.minimum(np.maximum(1 - np.abs(X_scaled[:, None, :] - X_scaled), 0), y[:, None, None] != y[:, None])
        POS_A_size = self._POS_size(R_a)
        selected_attributes = np.full(X.shape[-1], False)
        ordered_selected_attributes = []
        remaining_attributes = set(range(X.shape[-1]))
        best_size = 0
        condition = (lambda: np.sum(selected_attributes) < self.n_features) if self.n_features else (
            lambda: best_size < POS_A_size)
        while condition():
            best_size = 0
            for i in remaining_attributes:
                candidate = selected_attributes.copy()
                candidate[i] = True
                candidate_size = self._POS_size(R_a[..., candidate])
                if candidate_size > best_size:
                    best_size = candidate_size
                    new_attribute = i
            selected_attributes[new_attribute] = True
            ordered_selected_attributes.append(new_attribute)
            remaining_attributes.remove(new_attribute)
        model.selection = ordered_selected_attributes
        return model


@dataclass
class FRFSRuleGenerator(RuleGenerator):
    attribute_ratio: float = 1
    frfs_type: type(FRFS) = FRFS

    def fit(self, X: np.ndarray, y: np.ndarray, types: np.ndarray = None):
        X, y = check_X_y(X, y)
        n_features = X.shape[1]
        frfs = self.frfs_type(n_features=np.rint(n_features*self.attribute_ratio))
        self.preprocessor_ = frfs(X, y)
        X = self.preprocessor_(X)

        RuleGenerator.fit(self, X, y, types)

    def predict_proba(self, X: np.ndarray, normalized: bool = True) -> np.ndarray:
        check_is_fitted(self)
        X = self.preprocessor_(X)
        return RuleGenerator.predict_proba(self, X, normalized)


    def __str__(self) -> str:
        if hasattr(self, "approximation_"):
            return f"@frfs-frri\n@approximation: {self.approximation_}\n@scaler: {self.scaler}"
        else:
            return f"@frfs-frri-NOT-YET-FITTED"
