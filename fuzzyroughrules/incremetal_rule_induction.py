from dataclasses import dataclass

from sklearn.utils.validation import check_is_fitted
import fuzzyroughrules.fuzzy_operators as fo
from fuzzyroughrules.rule_induction_base import RuleGenerator
from fuzzyroughrules.operators import triangular_relation
import numpy as np


@dataclass
class IncrementalRuleInduction(RuleGenerator):
    """
    covering_threshold: minimum covering degree for the confidence of the rule to be updated
    """
    covering_threshold:float = 0.0

    def increment(self, x, y):
        check_is_fitted(self)
        x = self.scaler_.transform(x.reshape(1, -1))
        # y = None  # todo why is this even here?
        # if y in self.classes_:
        #     y = np.where(self.classes_ == y)[0][0]
        # calculate credibility of each rule
        for rule in self.rules_:
            if rule.decision == y: continue # alpha increases
            matching_degree = triangular_relation(
                    x,
                    rule.antecedent,
                    rule.slopes,
                    rule.reducts
                )
            covering_degree = fo.lukasiewicz_t_norm(matching_degree, rule.credibility)
            if covering_degree > self.covering_threshold:
                # update credibility
                # credibility is: self.positive_region_[obj] with:
                rule.credibility = min(rule.credibility, fo.lukasiewicz_t_norm(matching_degree, 0))

        return self
