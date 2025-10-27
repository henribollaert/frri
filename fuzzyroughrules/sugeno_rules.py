from dataclasses import dataclass
from typing import override

from sklearn.preprocessing import normalize
import numpy as np

from fuzzyroughrules.rule_induction_base import RuleGenerator


@dataclass
class SugenoRuleGenerator(RuleGenerator):
    learn_q:bool = False
    q:float = 1.0

    @override
    def predict_proba(self, X: np.ndarray, normalized: bool = True) -> np.ndarray:
        X, credibility_predictions = self.pre_predict(X)

        # aggregate credibility for each class
        aggregated_credibility = np.zeros((X.shape[0], self.n_classes_))
        for i in range(self.n_classes_):
            if len(credibility_predictions[i]) != 0:  # easy fix for 0 size array problem!
                # array of column vectors for each test obj containing the covering degree by all rules of this class
                cred_preds = np.array(credibility_predictions[i])
                cred_preds = np.sort(cred_preds, axis=0)
                power_measure = np.power(np.arange(len(cred_preds))/len(cred_preds), self.q)
                sug = np.minimum(cred_preds, power_measure[:, np.newaxis])

                aggregated_credibility[:, i] = np.max(sug, 0)
        if normalized:
            aggregated_credibility = normalize(aggregated_credibility, norm='l1')

        return aggregated_credibility

