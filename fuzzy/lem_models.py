"""
This is a quick and dirty class that converts Joannes Lemmens' code to a class
that follows the Model protocol from quickrules.data_handling
"""
import numpy as np
from fuzzy import classes as fc
from fuzzy.fuzzy_rough_lem2 import FuzzyRoughLEM2
from typing import Optional


class FRLEMModel:
    def __init__(
            self,
            no_sum=True,
            ming=0.9,
            alpha=1,
            t_norm=fc.FuzzySet.t_norm_min,
            implicator=fc.FuzzySet.implicator_kd
    ):
        self.no_sum = no_sum
        self.ming = ming
        self.alpha = alpha
        self.t_norm = t_norm
        self.implicator = implicator

        self.decision_system: Optional[fc.DecisionSystem] = None
        self.certain_rules = None
        self.possible_rules = None
        self.g_cons_C = {}  # Dictionary of the g-consistency of C
        self.g_cons_cr = {}  # Dictionary of the g-consistency of C' for the certain rules
        self.g_cons_pr = {}  # Dictionary of the g-consistency of C' for the possible rules

    def _initialisation(self, data, labels):
        ds = fc.DecisionSystem()
        ds.t_norm = self.t_norm
        ds.implicator = self.implicator
        # Set the cases of the decision system to be the training dataset
        ds.cases = [x.tolist() + [y] for (x, y) in zip(data, labels)]
        # The decision value is the last index of each case
        ds.decision = data.shape[1]
        # This function automatically sets up all other variables
        ds.setup_variables()
        self.decision_system = ds

    def fit(self, data, labels):
        self._initialisation(data, labels)

        # Induce rules from the training data
        self.certain_rules = {}  # Dictionary of certain rules for each concept
        self.possible_rules = {}  # Dictionary of possible rules for each concept
        self.g_cons_C = {}  # Dictionary of the g-consistency of C
        self.g_cons_cr = {}  # Dictionary of the g-consistency of C' for the certain rules
        self.g_cons_pr = {}  # Dictionary of the g-consistency of C' for the possible rules
        for key, concept in self.decision_system.concepts.items():  # For each concept, induce a set of rules
            concept = set(concept)
            self.certain_rules[key], self.possible_rules[key], self.g_cons_cr[key], self.g_cons_pr[key] = \
                FuzzyRoughLEM2(self.decision_system, concept, key, a=self.alpha)
            self.g_cons_C[key] = self.decision_system.g_cons(concept, alpha=self.alpha)

    def get_info(self) -> str:
        return "FR-LEM-model"

    def _predict_single(self, sample: np.ndarray):
        test_case = sample.tolist() + ['UNKNOWN']
        max_concept = None
        max_concept_match = 0.0
        for concept in self.decision_system.concepts:
            max_rule_match = 0
            for rule in self.certain_rules[concept]:
                match = 1
                for pair in rule:
                    match = self.t_norm(
                        match,
                        self.decision_system.R_av_case(pair[0], pair[1], test_case, alpha=self.alpha)
                    )
                max_rule_match = max(max_rule_match, match)
            if max_rule_match > max_concept_match:
                max_concept = concept
                max_concept_match = max_rule_match
        return max_concept

    def predict(self, datax: np.ndarray) -> np.ndarray:
        return np.array([self._predict_single(sample) for sample in datax])

    def get_rules_as_string(self) -> list[str]:
        certain_rules = ["### Certain rules"]
        for key in self.decision_system.concepts:
            certain_rules.append(
                f"# Concept: {key},"
                f" g(C) = {self.g_cons_C[key]},"
                f" g(C')) = {self.g_cons_cr[key]}"
            )
            for rule in self.certain_rules[key]:
                certain_rules.append(f"{str(rule)}")

        possible_rules = ["### Possible rules"]
        for key in self.decision_system.concepts:
            possible_rules.append(
                f"# Concept: {key},"
                f" g(C) = {self.g_cons_C[key]},"
                f" g(C') = {self.g_cons_pr[key]}"
            )
            for rule in self.possible_rules[key]:
                possible_rules.append(f"{str(rule)}")
        return certain_rules + possible_rules
