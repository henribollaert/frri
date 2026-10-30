"""
Prediction for objects for which no rule fires. FRRI gives such an object membership 0 for every class, so argmax picks
the first class. The core inference of RuleGenerator is not changed: the fallbacks are in a subclass.
"""
from dataclasses import dataclass

import numpy as np
from sklearn.utils.validation import check_array, check_is_fitted

from fuzzyroughrules.operators import triangular_relation
from fuzzyroughrules.rule_induction_base import RuleGenerator

FALLBACKS = ("first", "relation", "default")


@dataclass
class FallbackRuleGenerator(RuleGenerator):
    """
    Attributes:
        fallback (str): What to predict for an object for which no rule fires.
            "first": the first class, as RuleGenerator does.
            "relation": the class of the rule with the highest relation R_r(x_r, u) to the object, i.e. ignoring the
            confidence. This needs no other information than the rules.
            "default": default_label, e.g. the majority class. This is a baseline: in the federated setting it needs
            the class counts of the clients, which the protocol does not send. (default = "relation")
        default_label: The label predicted by the "default" fallback. (default = None)
    """
    fallback: str = "relation"
    default_label: object = None

    def rule_relations(self, X: np.ndarray) -> np.ndarray:
        """
        :param X: objects, not scaled
        :return: for each object and class, the highest relation to a rule of that class, without the confidence
        """
        check_is_fitted(self)
        X_scaled = self.scaler_.transform(check_array(X))
        relations = np.zeros((len(X_scaled), self.n_classes_))
        for rule in self.rules_:
            relation = triangular_relation(X_scaled, rule.antecedent, rule.slopes, rule.reducts).reshape(len(X_scaled))
            relations[:, rule.decision] = np.maximum(relations[:, rule.decision], relation)
        return relations

    def no_rule_fires(self, X: np.ndarray) -> np.ndarray:
        """
        :return: boolean mask of the objects with membership 0 for every class
        """
        return np.all(self.predict_proba(X, normalized=False) == 0, axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.fallback not in FALLBACKS:
            raise ValueError(f"Unknown fallback: {self.fallback}. Use one of {FALLBACKS}.")
        memberships = self.predict_proba(X, normalized=False)
        predictions = self.classes_[np.argmax(memberships, 1)]
        silent = np.all(memberships == 0, axis=1)
        if not np.any(silent) or self.fallback == "first":
            return predictions
        if self.fallback == "default":
            if self.default_label is None:
                raise ValueError("The default fallback needs a default_label.")
            predictions[silent] = self.default_label
            return predictions
        # an object without any relation to a rule still gets the first class
        predictions[silent] = self.classes_[np.argmax(self.rule_relations(np.asarray(X)[silent]), 1)]
        return predictions


def with_fallback(model: RuleGenerator, fallback: str = "relation", default_label=None) -> FallbackRuleGenerator:
    """
    :param model: a fitted RuleGenerator
    :param fallback: see FallbackRuleGenerator
    :param default_label: see FallbackRuleGenerator
    :return: a FallbackRuleGenerator with the same rules, scaler and classes as the model
    """
    check_is_fitted(model)
    result = FallbackRuleGenerator(fallback=fallback, default_label=default_label)
    for key, value in vars(model).items():
        if key.endswith("_") or key == "scaler":
            setattr(result, key, value)
    return result
