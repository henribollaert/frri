"""
The claim of the federated approach: for a fixed set of rules, federated confidence and coverage are the same as when
they are calculated centrally on the union of the train_local data of the clients.

The central values are calculated from the formulas with plain loops, so that they share no code with the federated
implementation. Rules are compared to the rule-relation formula. This is not what fit stores: fit uses all attributes,
so a centralised FRRI model can give a higher confidence to the same object.
"""
import numpy as np
import pytest
from sklearn.datasets import load_iris, load_wine

from fuzzyroughrules.federated.aggregate import global_confidence, set_cover, union_global, union_local
from fuzzyroughrules.federated.client import FederatedClient
from fuzzyroughrules.federated.partition import (
    FixedRangeScaler, global_classes, global_min_max, horizontal_split
)
from fuzzyroughrules.federated.payload import RulePayload
from fuzzyroughrules.federated.rounds import REPAIR_TOLERANCE, coverage_rounds, repair_rules
from fuzzyroughrules.rule_induction_base import RuleGenerator, evaluate_rules

DATASETS = {"iris": load_iris, "wine": load_wine}
COVERING_THRESHOLD = RuleGenerator().covering_threshold


def attribute_relation(value, rule_value, slope, relation_type):
    if relation_type == "DOMINATED":
        difference = rule_value - value
    elif relation_type == "DOMINANT":
        difference = value - rule_value
    else:
        difference = abs(value - rule_value)
    return max(min(1 - difference / slope, 1), 0)


def relation(rule: RulePayload, obj: np.ndarray) -> float:
    memberships = [
        attribute_relation(obj[attribute], value, slope, relation_type)
        for attribute, value, slope, relation_type in zip(
            rule.attributes, rule.values, rule.slopes, rule.relation_types
        )
    ]
    return min(memberships, default=1)


def central_confidence(rule: RulePayload, X: np.ndarray, y: np.ndarray) -> float:
    return min(min(1 - relation(rule, obj) + (rule.label == label), 1) for obj, label in zip(X, y))


def central_coverage(rules: list[RulePayload], X: np.ndarray, y: np.ndarray) -> np.ndarray:
    # the confidence stored in the rules is not used, as that is part of what is being tested
    return np.array([
        [max(relation(rule, obj) + central_confidence(rule, X, y) - 1, 0) > COVERING_THRESHOLD for obj in X]
        for rule in rules
    ], dtype=float)


def sorted_columns(matrix: np.ndarray) -> np.ndarray:
    return matrix[:, np.lexsort(matrix[::-1])]


@pytest.fixture(params=[
    ("iris", 2, {}), ("iris", 3, {}), ("wine", 2, {}), ("wine", 5, {}),
    ("iris", 2, {"optimise_slopes": True}), ("wine", 3, {"optimise_slopes": True}),
], ids=str)
def federation(request):
    name, n_clients, frri_params = request.param
    data = DATASETS[name]()
    split = horizontal_split(data.data, data.target, n_clients, seed=0)
    clients = [FederatedClient(c.X_train, c.y_train, **frri_params) for c in split.clients]
    scaler = FixedRangeScaler(*global_min_max([client.min_max() for client in clients]))
    classes = global_classes([client.labels() for client in clients])
    client_rules = [client.fit_local(scaler, classes) for client in clients]
    X_union = scaler.transform(np.vstack([c.X_train for c in split.clients]))
    y_union = np.concatenate([c.y_train for c in split.clients])
    return clients, client_rules, classes, X_union, y_union


def test_global_confidence_equals_central_confidence(federation):
    clients, client_rules, _, X_union, y_union = federation
    rules = union_local(client_rules)
    federated = global_confidence([client.local_minima(rules) for client in clients])
    central = [central_confidence(rule, X_union, y_union) for rule in rules]
    assert federated == pytest.approx(central, abs=1e-12)


def test_global_confidence_is_exactly_the_core_evaluation_on_the_union(federation):
    clients, client_rules, classes, X_union, y_union = federation
    rules = union_local(client_rules)
    federated = global_confidence([client.local_minima(rules) for client in clients])
    y_encoded = np.searchsorted(classes, y_union)
    central, _ = evaluate_rules(clients[0]._to_rules(rules), X_union, y_encoded)
    assert np.array_equal(federated, central)


def test_pooled_coverage_equals_central_coverage_up_to_column_order(federation):
    clients, client_rules, _, X_union, y_union = federation
    rules = union_local(client_rules)
    confidence = global_confidence([client.local_minima(rules) for client in clients])
    rated = union_global(client_rules, confidence)
    pooled = np.hstack([client.coverage_columns(rated, seed=i) for i, client in enumerate(clients)])
    assert pooled.shape == (len(rated), len(X_union))
    assert np.array_equal(sorted_columns(pooled), sorted_columns(central_coverage(rated, X_union, y_union)))


def test_shuffling_does_not_change_the_selection_problem(federation):
    clients, client_rules, _, _, _ = federation
    rules = union_local(client_rules)
    rated = union_global(client_rules, global_confidence([client.local_minima(rules) for client in clients]))
    first = np.hstack([client.coverage_columns(rated, seed=1) for client in clients])
    second = np.hstack([client.coverage_columns(rated, seed=2) for client in clients])
    assert not np.array_equal(first, second)
    assert np.array_equal(sorted_columns(first), sorted_columns(second))
    assert set_cover(rated, first)[1] == set_cover(rated, second)[1]


def test_global_confidence_is_at_most_the_minimum_on_the_own_data(federation):
    clients, client_rules, _, _, _ = federation
    start = 0
    rules = union_local(client_rules)
    confidence = global_confidence([client.local_minima(rules) for client in clients])
    for client, own_rules in zip(clients, client_rules):
        own = client.local_minima(own_rules)
        assert np.all(confidence[start:start + len(own_rules)] <= own)
        start += len(own_rules)


@pytest.mark.parametrize("name", DATASETS)
def test_rule_relation_minimum_on_own_data_agrees_with_stored_credibility(name):
    # the reduction keeps the rule within the class, up to the inclusion threshold. This links the formula to FRRI.
    data = DATASETS[name]()
    split = horizontal_split(data.data, data.target, 2, seed=0)
    client = FederatedClient(split.clients[0].X_train, split.clients[0].y_train)
    scaler = FixedRangeScaler(*client.min_max())
    rules = client.fit_local(scaler, global_classes([client.labels()]))
    stored = np.array([rule.confidence for rule in rules])
    assert client.local_minima(rules) == pytest.approx(stored, abs=1e-6)


def federated_confidence(clients, client_rules):
    return global_confidence([client.local_minima(union_local(client_rules)) for client in clients])


def test_repaired_rules_get_the_central_confidence(federation):
    clients, client_rules, _, X_union, y_union = federation
    repaired, confidence, log = repair_rules(clients, client_rules, federated_confidence(clients, client_rules))
    assert log, "this federation must need repair, or the test checks nothing"
    central = [central_confidence(rule, X_union, y_union) for rule in union_local(repaired)]
    assert confidence == pytest.approx(central, abs=1e-12)


def test_repair_adds_attributes_of_the_own_object_and_stops_when_recovered(federation):
    clients, client_rules, _, X_union, y_union = federation
    repaired, _, log = repair_rules(clients, client_rules, federated_confidence(clients, client_rules))
    n_features = X_union.shape[1]
    for client, original, final in zip(clients, client_rules, repaired):
        for index, (before, after) in enumerate(zip(original, final)):
            target = before.confidence - REPAIR_TOLERANCE
            added = set(after.attributes) - set(before.attributes)
            if central_confidence(before, X_union, y_union) >= target:
                assert after == before
                continue
            assert added
            kept = {a: (v, t, s) for a, v, t, s in zip(after.attributes, after.values, after.relation_types, after.slopes)}
            for a, v, t, s in zip(before.attributes, before.values, before.relation_types, before.slopes):
                assert kept[a] == (v, t, s)
            own_object = client.model.rules_[index].antecedent
            for a in added:
                assert kept[a] == (own_object[a], "INDISCERNIBLE", 1.)
            recovered = central_confidence(after, X_union, y_union) >= target
            assert recovered or len(after.attributes) == n_features
            if recovered:
                # one round less leaves the rule below its local confidence
                shorter, _, _ = repair_rules(
                    clients, client_rules, federated_confidence(clients, client_rules), max_rounds=len(added) - 1
                )
                assert central_confidence(shorter[clients.index(client)][index], X_union, y_union) < target


def central_uncovered(rules, X, y, own_X):
    """Objects of own_X that none of the rules covers, with the confidence calculated on the union X, y."""
    confidence = [central_confidence(rule, X, y) for rule in rules]
    return [
        i for i, obj in enumerate(own_X)
        if not any(max(relation(rule, obj) + c - 1, 0) > COVERING_THRESHOLD for rule, c in zip(rules, confidence))
    ]


def test_new_candidates_are_the_rules_of_the_centrally_uncovered_objects(federation):
    clients, client_rules, classes, X_union, y_union = federation
    rated = union_global(client_rules, federated_confidence(clients, client_rules))
    selected, _ = set_cover(rated, np.hstack([client.coverage_columns(rated, seed=i) for i, client in enumerate(clients)]))
    for client, own in zip(clients, client_rules):
        expected = set()
        for i in central_uncovered(selected, X_union, y_union, client.X_scaled):
            used = [a for a in range(X_union.shape[1]) if client.model.reducts_[i][a].name != "UNUSED"]
            expected.add((tuple(used), tuple(client.X_scaled[i][used]), client.y_train[i]))
        expected -= {(rule.attributes, rule.values, rule.label) for rule in own}
        new = client.uncovered_rules(selected)
        assert {(rule.attributes, rule.values, rule.label) for rule in new} == expected
        # a rule is only sent once
        assert client.uncovered_rules(selected) == []


def test_coverage_rounds_cover_what_the_central_candidates_cover(federation):
    clients, client_rules, _, X_union, y_union = federation
    selected, n_uncovered, candidates, log = coverage_rounds(
        clients, client_rules, federated_confidence(clients, client_rules), seed=0
    )
    assert [rule.confidence for rule in candidates] == pytest.approx(
        [central_confidence(rule, X_union, y_union) for rule in candidates], abs=1e-12
    )
    coverable = np.any(central_coverage(candidates, X_union, y_union), axis=0)
    assert n_uncovered == np.sum(~coverable)
    assert np.array_equal(np.any(central_coverage(selected, X_union, y_union), axis=0), coverable)
    assert log[0]["n_uncovered"] >= log[-1]["n_uncovered"]
