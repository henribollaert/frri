"""
The steps that run on the server: global confidence, selection of rules and construction of the global model.
"""
from dataclasses import replace
from math import ceil
from typing import Callable

import numpy as np

from fuzzyroughrules.approximations import LowerApproximation
from fuzzyroughrules.federated.partition import FixedRangeScaler
from fuzzyroughrules.federated.payload import RulePayload, payload_to_rule
from fuzzyroughrules.rule_induction_base import RuleGenerator, select_rules_from_coverage


def global_confidence(client_minima: list[np.ndarray]) -> np.ndarray:
    """
    :param client_minima: for each client, its local minimum for each rule
    :return: the global confidence of each rule
    """
    return np.min(client_minima, axis=0)


def _with_confidence(rules: list[RulePayload], confidence: np.ndarray) -> list[RulePayload]:
    return [replace(rule, confidence=float(value)) for rule, value in zip(rules, confidence)]


def _can_fire(rules: list[RulePayload]) -> list[RulePayload]:
    """
    A rule with confidence 0 gets membership 0 for every object, so it has no influence on the predictions.
    """
    return [rule for rule in rules if rule.confidence > 0]


def union_local(client_rules: list[list[RulePayload]]) -> list[RulePayload]:
    """
    Variant (a): all local rules, with their local confidence.
    :param client_rules: the rules of each client
    :return: the rules of the global model
    """
    return [rule for rules in client_rules for rule in rules]


def union_global(client_rules: list[list[RulePayload]], confidence: np.ndarray) -> list[RulePayload]:
    """
    Variant (b): all local rules, with their global confidence.
    :param client_rules: the rules of each client
    :param confidence: the global confidence of the rules, in the order of union_local(client_rules)
    :return: the rules of the global model
    """
    return _can_fire(_with_confidence(union_local(client_rules), confidence))


def top_k_per_client(
        client_rules: list[list[RulePayload]],
        confidence: np.ndarray,
        k: float = 0.5,
        rank_by: str = "confidence",
        coverage_counts: list[np.ndarray] = None
) -> list[RulePayload]:
    """
    Variant (c): the best fraction of the rules of each client, with their global confidence.
    :param client_rules: the rules of each client
    :param confidence: the global confidence of the rules, in the order of union_local(client_rules)
    :param k: fraction of the rules of each client that is kept, rounded up
    :param rank_by: "confidence" uses the global confidence. "coverage" uses the number of local objects covered by
    the rule, as in Beata's protocol.
    :param coverage_counts: for each client, the result of coverage_counts. Only needed if we rank by coverage.
    :return: the rules of the global model
    """
    if rank_by not in ("confidence", "coverage"):
        raise ValueError(f"Unknown ranking: {rank_by}.")
    if rank_by == "coverage" and coverage_counts is None:
        raise ValueError("Ranking by coverage needs the coverage counts of the clients.")

    selected = []
    start = 0
    for client, rules in enumerate(client_rules):
        client_confidence = confidence[start:start + len(rules)]
        start += len(rules)
        scores = client_confidence if rank_by == "confidence" else coverage_counts[client]
        best = np.argsort(-np.asarray(scores), kind="stable")[:ceil(k * len(rules))]
        selected += _with_confidence([rules[i] for i in best], client_confidence[best])
    return _can_fire(selected)


def plain_set_cover(rules: list[RulePayload]) -> np.ndarray:
    """
    Objective for the fewest rules that cover all objects.
    """
    return np.ones(len(rules))


def set_cover(
        rules: list[RulePayload],
        coverage: np.ndarray,
        objective: Callable[[list[RulePayload]], np.ndarray] = plain_set_cover,
        solver: str = "auto",
        time_limit: float = None
) -> tuple[list[RulePayload], int]:
    """
    Variant (d): solves the set cover problem over the coverage pooled from all clients. Objects that are covered by
    none of the rules are left out of the problem. FRRI adds a rule for such an object, but that is not possible here
    because the rule of the object is not one of the candidates.
    :param rules: the candidate rules, with their global confidence
    :param coverage: pooled binary coverage of shape (n_rules, n_objects), in the order of the rules
    :param objective: gives the cost of selecting each rule
    :param solver: solver of the set cover, see select_rules_from_coverage
    :param time_limit: seconds that the solver may use, see select_rules_from_coverage
    :return: the rules of the global model, the number of objects that are not covered
    """
    uncovered = np.flatnonzero(np.sum(coverage, axis=0) == 0)
    selected = select_rules_from_coverage(coverage, objective(rules), uncovered, solver, time_limit)
    return [rules[i] for i in selected], len(uncovered)


def build_global_model(rules: list[RulePayload], classes: np.ndarray, scaler: FixedRangeScaler) -> RuleGenerator:
    """
    :param rules: the rules of the global model
    :param classes: the global classes
    :param scaler: the scaler with the global range
    :return: a fitted RuleGenerator, so prediction is standard FRRI inference
    """
    model = RuleGenerator()
    model.scaler = scaler
    model.scaler_ = scaler
    model.approximation_ = LowerApproximation()
    model.classes_ = classes
    model.n_classes_ = len(classes)
    model.n_features_in_ = scaler.n_features_in_
    model.rules_ = [payload_to_rule(rule, classes, scaler.n_features_in_) for rule in rules]
    return model
