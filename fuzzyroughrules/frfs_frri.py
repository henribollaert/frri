import numpy as np
from sklearn.utils.validation import check_X_y
from fuzzyroughrules.rule_induction_base import RuleGenerator
from frlearn.feature_preprocessors import FRFS


class FRFSRuleGenerator(RuleGenerator):

    def fit(self, X: np.ndarray, y: np.ndarray, types: np.ndarray = None):
        X, y = check_X_y(X, y)
        n_features = X.shape[1]
        preprocessor = FRFS(n_features=n_features)
        model = preprocessor(X, y)
        X = model(X)

        super().fit(X, y, types)
