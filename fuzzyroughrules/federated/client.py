"""
The steps that run on a client. A client only ever gets its train_local data.
"""
import numpy as np

from fuzzyroughrules.federated.partition import FixedRangeScaler
from fuzzyroughrules.federated.payload import RulePayload, rule_to_payload, payload_to_rule
from fuzzyroughrules.operators import RelationTypes, triangular_relation
from fuzzyroughrules.rule_induction_base import RuleGenerator, Rule, rule_coverage, rule_minima


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
        self.local_rules = [
            rule_to_payload(rule, label) for rule, label in zip(self.model.rules_, self.model.rule_labels())
        ]
        self.start_coverage_rounds(self.local_rules)
        return self.local_rules

    def start_coverage_rounds(self, candidates: list[RulePayload]):
        """
        Coverage rounds do not send a rule that left this client before: its local rules, and its current candidates,
        e.g. repaired versions of the local rules.
        """
        self.sent = set(self.local_rules) | set(candidates)

    def _to_rules(self, payloads: list[RulePayload]) -> list[Rule]:
        return [payload_to_rule(payload, self.classes, self.X_scaled.shape[1]) for payload in payloads]

    def local_minima(self, payloads: list[RulePayload]) -> np.ndarray:
        """
        :param payloads: rules broadcast by the server
        :return: one number per rule: its lower approximation minimum over the local data
        """
        return rule_minima(self._to_rules(payloads), self.X_scaled, self.y_encoded)

    def extend_rule(self, own_index: int, payload: RulePayload) -> RulePayload | None:
        """
        Rule repair: adds one attribute that the rule does not use yet, with the value of the object that the rule was
        built from, relation INDISCERNIBLE and slope 1, where the reduction of FRRI starts. The attribute is the one
        that separates the object best from the local objects of other classes, weighted by their relation to the
        current rule. The confidence of the payload stays the local confidence.
        :param own_index: position of the rule in the rules that fit_local returned
        :param payload: the current version of that rule
        :return: the rule with one more attribute, or None if it already uses all attributes
        """
        n_features = self.X_scaled.shape[1]
        unused = np.array([attribute for attribute in range(n_features) if attribute not in payload.attributes])
        if len(unused) == 0:
            return None
        rule = self._to_rules([payload])[0]
        values = self.model.rules_[own_index].antecedent
        other = self.y_encoded != rule.decision
        weights = triangular_relation(
            self.X_scaled[other], rule.antecedent, rule.slopes, rule.reducts
        ).reshape(np.count_nonzero(other))
        separation = np.zeros(len(unused))
        if np.sum(weights) > 0:
            distances = np.minimum(np.abs(self.X_scaled[other][:, unused] - values[unused]), 1)
            separation = weights @ distances / np.sum(weights)
        added = int(unused[np.argmax(separation)])
        order = np.argsort(list(payload.attributes) + [added], kind="stable")
        return RulePayload(
            tuple(int(a) for a in np.array(list(payload.attributes) + [added])[order]),
            tuple(float(v) for v in np.array(list(payload.values) + [float(values[added])])[order]),
            tuple(str(t) for t in np.array(list(payload.relation_types) + [RelationTypes.INDISCERNIBLE.name])[order]),
            tuple(float(v) for v in np.array(list(payload.slopes) + [1.])[order]),
            payload.label,
            payload.confidence
        )

    def uncovered_rules(self, selected: list[RulePayload]) -> list[RulePayload]:
        """
        Coverage round: the FRRI rules of the local objects that the selected rules do not cover, if they were not
        sent before. These are the rules that fit built for every object before its own rule selection.
        :param selected: the rules selected by the server, with their global confidence
        :return: new candidate rules, with their local confidence
        """
        covered = np.zeros(len(self.X_scaled), dtype=bool)
        if selected:
            covered = np.any(rule_coverage(self._to_rules(selected), self.X_scaled, self.model.covering_threshold), 0)
        local_labels = np.searchsorted(self.model.classes_, self.y_train)
        new = []
        for index in np.flatnonzero(~covered):
            rule = Rule(
                self.X_scaled[index],
                self.model.reducts_[index],
                self.model.slopes_[index],
                self.model.positive_region_[index],
                local_labels[index]
            )
            payload = rule_to_payload(rule, self.y_train[index])
            if payload not in self.sent:
                self.sent.add(payload)
                new.append(payload)
        return new

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
