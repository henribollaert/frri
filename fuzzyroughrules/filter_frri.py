# TODO implement simple filter and sort preprocessor for frri using infomration gain and pcc metrics from sklearn
# find inspiration in the code from OFRFS

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.utils.validation import check_X_y
from fuzzyroughrules.rule_induction_base import RuleGenerator
from frlearn.feature_preprocessors import FRFS
from sklearn.utils.validation import check_is_fitted
from frlearn.statistics.feature_preprocessors import Standardiser
from sklearn.feature_selection import mutual_info_classif, r_regression

class Metric(Protocol):
    def __call__(self, X, y, **kwargs):
        ...

@dataclass
class Filter:
    """
    Conserves the order in which attributes are selected by a filter preprocessor.
    """
    metric: type(Metric) = mutual_info_classif
    attribute_ratio: float = 1

    def fit(self, X, y):
        X_scaled = Standardiser()(X)(X)
        # calculate metric
        metric_values = self.metric(X_scaled, y)
        # sort it in descending order and store it
        self.ordered_features_ = np.argsort(metric_values)[::-1][:]
        return self

    def transform(self, X):
        return X[:, self.ordered_features_[:int(np.rint(X.shape[1]*self.attribute_ratio))]]


@dataclass
class FilterRuleGenerator(RuleGenerator):
    attribute_ratio: float = 1
    filter_type: type(Filter) = Filter
    metric: type(Metric) = mutual_info_classif

    def fit(self, X: np.ndarray, y: np.ndarray, types: np.ndarray = None):
        X, y = check_X_y(X, y)
        filter_it = self.filter_type(metric=self.metric, attribute_ratio=self.attribute_ratio)
        self.preprocessor_ = filter_it.fit(X, y)
        X = self.preprocessor_.transform(X)

        RuleGenerator.fit(self, X, y, types)

    def predict_proba(self, X: np.ndarray, normalized: bool = True) -> np.ndarray:
        check_is_fitted(self)
        X = self.preprocessor_.transform(X)
        return RuleGenerator.predict_proba(self, X, normalized)


    def __str__(self) -> str:
        if hasattr(self, "approximation_"):
            return f"@frfs-frri\n@approximation: {self.approximation_}\n@scaler: {self.scaler}"
        else:
            return f"@frfs-frri-NOT-YET-FITTED"
