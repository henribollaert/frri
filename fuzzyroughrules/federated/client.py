"""
The steps that run on a client. A client only ever gets its train_local data.
"""
import numpy as np

from fuzzyroughrules.federated.partition import FixedRangeScaler
from fuzzyroughrules.federated.payload import RulePayload, rule_to_payload, payload_to_rule
from fuzzyroughrules.rule_induction_base import RuleGenerator, Rule, evaluate_rules, rule_coverage


class FederatedClient:
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, **frri_params):
        self.X_train = X_train
        self.y_train = y_train
        self.frri_params = frri_params

    def min_max(self) -> tuple[np.ndarray, np.ndarray]:
        return np.min(self.X_train, axis=0), np.max(self.X_train, axis=0)

    def labels(self) -> np.ndarray:
        return np.unique(self.y_train)

    def label_counts(self, classes: np.ndarray) -> np.ndarray:
        """
        Not part of the protocol: only the default-rule baseline needs it.
        :param classes: the global classes
        :return: the number of local objects of each class
        """
        return np.array([np.sum(self.y_train == label) for label in classes])

    def fit_local(self, scaler: FixedRangeScaler, classes: np.ndarray) -> list[RulePayload]:
        """
        Runs FRRI on the local data.
        :param scaler: the scaler with the global range
        :param classes: the global classes
        :return: the rules of the local model, with the confidence calculated by the local model
        """
        self.scaler = scaler
        self.classes = classes
        self.model = RuleGenerator(**self.frri_params)
        self.model.set_params(scaler=scaler)
        self.model.fit(self.X_train, self.y_train)
        self.X_scaled = scaler.transform(self.X_train)
        self.y_encoded = np.searchsorted(classes, self.y_train)
        return [rule_to_payload(rule, label) for rule, label in zip(self.model.rules_, self.model.rule_labels())]

    def _to_rules(self, payloads: list[RulePayload]) -> list[Rule]:
        return [payload_to_rule(payload, self.classes, self.X_scaled.shape[1]) for payload in payloads]

    def local_minima(self, payloads: list[RulePayload]) -> np.ndarray:
        """
        :param payloads: rules broadcast by the server
        :return: one number per rule: its lower approximation minimum over the local data
        """
        return evaluate_rules(self._to_rules(payloads), self.X_scaled, self.y_encoded)[0]

    def coverage_columns(self, payloads: list[RulePayload], seed: int) -> np.ndarray:
        """
        :param payloads: rules broadcast by the server, with the confidence that should be used for coverage
        :param seed: seed for shuffling the objects
        :return: binary coverage matrix of shape (n_rules, n_local_objects), with the objects in random order
        """
        coverage = rule_coverage(self._to_rules(payloads), self.X_scaled, self.model.covering_threshold)
        return np.random.default_rng(seed).permutation(coverage, axis=1)

    def coverage_counts(self, payloads: list[RulePayload]) -> np.ndarray:
        """
        :param payloads: rules of this client, with their local confidence
        :return: the number of local objects covered by each rule. Only needed to rank rules by coverage.
        """
        return np.sum(rule_coverage(self._to_rules(payloads), self.X_scaled, self.model.covering_threshold), axis=1)
