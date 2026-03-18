import numpy as np
from sklearn.utils.validation import check_X_y
from sklearn.model_selection import GridSearchCV
from fuzzyroughrules.rule_induction_base import RuleGenerator
from dataclasses import dataclass
from typing import Optional


@dataclass
class CVInclusionRuleGenerator(RuleGenerator):
    threshold_options: Optional[list] = None
    scoring = None

    def fit(self, X: np.ndarray, y: np.ndarray, types: np.ndarray = None):
        X, y = check_X_y(X, y)
        threshold_options = \
            [1 - 1e-6, 0.9999, 0.999, 0.99, 0.95, 0.9] \
            if self.threshold_options is None else self.threshold_options
        scoring = 'balanced_accuracy' if self.scoring is None else self.scoring
        gs = GridSearchCV(
            estimator=RuleGenerator(),
            param_grid={'inclusion_threshold': threshold_options},
            scoring=scoring,
            n_jobs=-1
        )

        gs.fit(X, y)

        self.set_params(**gs.best_params_)

        super().fit(X, y, types)

