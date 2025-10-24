from sklearn.preprocessing import normalize
import numpy as np
from sklearn.utils.validation import check_is_fitted, check_array

from fuzzyroughrules.rule_induction_base import RuleGenerator
import fuzzy_operators as fo
from fuzzyroughrules.operators import triangular_relation


# TODO IMPLEMENT

class CCRuleGenerator(RuleGenerator):
    def predict_proba(self, X: np.ndarray, normalized: bool = True) -> np.ndarray:
        check_is_fitted(self)
        # rescale X
        X = check_array(X)
        X_test = self.scaler_.transform(X)
        # calculate credibility of each rule
        credibility_predictions = []
        for i in range(self.n_classes_):
            credibility_predictions.append([])
        for rule in self.rules_:
            credibility_predictions[rule.decision].append(
                fo.lukasiewicz_t_norm(
                    triangular_relation(
                        X_test,
                        rule.antecedent,
                        rule.slopes,
                        rule.reducts
                    ),
                    rule.credibility
                ).reshape((len(X_test)))
            )

        # sum credibility for each class
        cumulative_credibility = np.zeros((X.shape[0], self.n_classes_))
        for i in range(self.n_classes_):
            if len(credibility_predictions[i]) != 0:  # easy fix for 0 size array problem!
                cumulative_credibility[:, i] = np.max(np.array(credibility_predictions[i]), 0)
        if normalized:
            cumulative_credibility = normalize(cumulative_credibility, norm='l1')

        return cumulative_credibility

