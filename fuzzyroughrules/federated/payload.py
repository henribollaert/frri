"""
What a client sends to the server about a rule. Rule.antecedent is a complete training object, so it can not be sent.
"""
from dataclasses import dataclass, asdict

import numpy as np

from fuzzyroughrules.operators import RelationTypes
from fuzzyroughrules.rule_induction_base import Rule


@dataclass(frozen=True)
class RulePayload:
    attributes: tuple[int, ...]
    values: tuple[float, ...]
    relation_types: tuple[str, ...]
    slopes: tuple[float, ...]
    label: object  # global class label
    confidence: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "RulePayload":
        return cls(
            tuple(payload["attributes"]),
            tuple(payload["values"]),
            tuple(payload["relation_types"]),
            tuple(payload["slopes"]),
            payload["label"],
            payload["confidence"]
        )


def rule_to_payload(rule: Rule, label) -> RulePayload:
    """
    :param rule: a rule of a local model
    :param label: the class label of the rule, not the index in the classes of the local model
    :return: the rule, without the values of the attributes that it does not use
    """
    used = np.flatnonzero(rule.reducts != RelationTypes.UNUSED)
    return RulePayload(
        tuple(int(attribute) for attribute in used),
        tuple(float(rule.antecedent[attribute]) for attribute in used),
        tuple(rule.reducts[attribute].name for attribute in used),
        tuple(float(rule.slopes[attribute]) for attribute in used),
        np.asarray(label).item(),
        float(rule.credibility)
    )


def payload_to_rule(payload: RulePayload, classes: np.ndarray, n_features: int) -> Rule:
    """
    Unused attributes get the value NaN: triangular_relation does not read them.
    :param payload: the received rule
    :param classes: the global classes. The decision of the rule becomes an index in this array.
    :param n_features: number of features of the data
    :return: rule that works with the standard FRRI functions
    """
    antecedent = np.full(n_features, np.nan)
    reducts = np.full(n_features, RelationTypes.UNUSED, dtype=RelationTypes)
    slopes = np.ones(n_features, dtype=float)
    for attribute, value, relation_type, slope in zip(
            payload.attributes, payload.values, payload.relation_types, payload.slopes
    ):
        antecedent[attribute] = value
        reducts[attribute] = RelationTypes[relation_type]
        slopes[attribute] = slope
    return Rule(antecedent, reducts, slopes, payload.confidence, int(np.flatnonzero(classes == payload.label)[0]))
