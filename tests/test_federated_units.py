import json

import numpy as np
import pytest
from sklearn.datasets import load_iris, load_wine
from sklearn.preprocessing import MinMaxScaler

from fuzzyroughrules.federated.aggregate import (
    build_global_model, global_confidence, plain_set_cover, set_cover, top_k_per_client, union_global, union_local
)
from fuzzyroughrules.federated.client import FederatedClient
from fuzzyroughrules.federated.partition import (
    FixedRangeScaler, global_classes, global_min_max, horizontal_split
)
from fuzzyroughrules.federated.payload import RulePayload, payload_to_rule, rule_to_payload
from fuzzyroughrules.operators import RelationTypes, triangular_relation
from fuzzyroughrules.rule_induction_base import RuleGenerator


def object_ids(split):
    parts = [split.X_test[:, 0]]
    for client in split.clients:
        parts += [client.X_train[:, 0], client.X_test[:, 0]]
    return parts


@pytest.mark.parametrize("stratify", [True, False])
@pytest.mark.parametrize("n_clients", [2, 3, 4, 5])
def test_split_is_a_disjoint_partition_with_protocol_sizes(n_clients, stratify):
    n = 1000
    X = np.arange(n, dtype=float).reshape(-1, 1)
    y = np.arange(n) % 4
    split = horizontal_split(X, y, n_clients, seed=1, stratify=stratify)

    parts = object_ids(split)
    assert sorted(np.concatenate(parts)) == list(range(n))
    assert len(split.X_test) == pytest.approx(0.2 * n, abs=2)
    sizes = [len(c.X_train) + len(c.X_test) for c in split.clients]
    assert max(sizes) - min(sizes) <= 1
    for client in split.clients:
        size = len(client.X_train) + len(client.X_test)
        assert len(client.X_test) == pytest.approx(0.2 * size, abs=2)


def test_split_depends_only_on_seed():
    X = np.arange(200, dtype=float).reshape(-1, 1)
    y = np.arange(200) % 3
    same = [np.concatenate(object_ids(horizontal_split(X, y, 3, seed=4))) for _ in range(2)]
    assert np.array_equal(*same)
    assert not np.array_equal(same[0], np.concatenate(object_ids(horizontal_split(X, y, 3, seed=5))))


def test_stratified_split_handles_class_with_fewer_objects_than_clients():
    X = np.arange(300, dtype=float).reshape(-1, 1)
    y = np.zeros(300, dtype=int)
    y[:3] = 1
    split = horizontal_split(X, y, 5, seed=0)
    assert sorted(np.concatenate(object_ids(split))) == list(range(300))
    per_client = [np.sum(c.y_train == 1) + np.sum(c.y_test == 1) for c in split.clients]
    assert max(per_client) <= 1


def test_global_range_only_uses_train_local():
    X = load_wine().data.copy()
    y = load_wine().target
    split = horizontal_split(X, y, 3, seed=0)
    ranges = [FederatedClient(c.X_train, c.y_train).min_max() for c in split.clients]
    mins, maxs = global_min_max(ranges)
    pooled = np.vstack([c.X_train for c in split.clients])
    assert np.array_equal(mins, pooled.min(axis=0))
    assert np.array_equal(maxs, pooled.max(axis=0))
    # the test objects are not part of the range, which the data of this split must be able to show
    everything = np.vstack([pooled, split.X_test, *[c.X_test for c in split.clients]])
    assert np.any(everything.min(axis=0) < mins) or np.any(everything.max(axis=0) > maxs)


def test_fixed_range_scaler_is_not_refitted_and_matches_sklearn():
    X = load_iris().data
    scaler = FixedRangeScaler(X.min(axis=0), X.max(axis=0))
    assert np.array_equal(scaler.fit_transform(X[:10]), MinMaxScaler().fit(X).transform(X[:10]))

    model = RuleGenerator()
    model.set_params(scaler=scaler)
    model.fit(X[:60], load_iris().target[:60])
    assert model.scaler_ is scaler
    assert np.array_equal(model.scaler_.transform(X), MinMaxScaler().fit(X).transform(X))


def fitted_model():
    X, y = load_iris().data, load_iris().target
    return RuleGenerator().fit(X, y), X


def test_payload_contains_only_used_attributes():
    model, _ = fitted_model()
    assert any(np.any(rule.reducts == RelationTypes.UNUSED) for rule in model.rules_)
    for rule in model.rules_:
        unused = rule.reducts == RelationTypes.UNUSED
        secret = rule.antecedent.copy()
        secret[unused] = 0.123456789
        marked = type(rule)(secret, rule.reducts, rule.slopes, rule.credibility, rule.decision)
        payload = rule_to_payload(marked, "class")
        assert "0.123456789" not in json.dumps(payload.to_dict())
        assert len(payload.attributes) == np.count_nonzero(~unused)


def test_payload_survives_json_and_gives_the_same_relation():
    model, X = fitted_model()
    classes = np.array(["a", "b", "c"])
    for rule in model.rules_:
        payload = rule_to_payload(rule, classes[rule.decision])
        received = RulePayload.from_dict(json.loads(json.dumps(payload.to_dict())))
        assert received == payload
        restored = payload_to_rule(received, classes, X.shape[1])
        assert np.all(np.isnan(restored.antecedent[rule.reducts == RelationTypes.UNUSED]))
        assert restored.decision == rule.decision
        scaled = model.scaler_.transform(X)
        assert np.array_equal(
            triangular_relation(scaled, restored.antecedent, restored.slopes, restored.reducts),
            triangular_relation(scaled, rule.antecedent, rule.slopes, rule.reducts)
        )


def test_global_model_with_unchanged_rules_predicts_like_the_local_model():
    X, y = load_wine().data, load_wine().target
    client = FederatedClient(X, y)
    scaler = FixedRangeScaler(*client.min_max())
    payloads = client.fit_local(scaler, global_classes([client.labels()]))
    model = build_global_model(payloads, client.classes, scaler)
    assert np.array_equal(model.predict(X), client.model.predict(X))
    assert np.array_equal(model.predict_proba(X), client.model.predict_proba(X))
    assert model.average_rule_length() == client.model.average_rule_length()


def federation(data, n_clients, seed=0, **frri_params):
    split = horizontal_split(data.data, data.target, n_clients, seed=seed)
    clients = [FederatedClient(c.X_train, c.y_train, **frri_params) for c in split.clients]
    scaler = FixedRangeScaler(*global_min_max([client.min_max() for client in clients]))
    classes = global_classes([client.labels() for client in clients])
    client_rules = [client.fit_local(scaler, classes) for client in clients]
    return split, clients, scaler, classes, client_rules


def test_client_missing_a_class_keeps_global_labels():
    data = load_iris()
    keep = data.target != 2
    missing_class = FederatedClient(data.data[keep][::2], data.target[keep][::2])
    other = FederatedClient(data.data[1::2], data.target[1::2])
    clients = [missing_class, other]
    scaler = FixedRangeScaler(*global_min_max([client.min_max() for client in clients]))
    classes = global_classes([client.labels() for client in clients])
    assert classes.tolist() == [0, 1, 2]
    client_rules = [client.fit_local(scaler, classes) for client in clients]

    assert {rule.label for rule in client_rules[1]} == {0, 1, 2}
    rules = union_local(client_rules)
    minima = [client.local_minima(rules) for client in clients]
    model = build_global_model(union_global(client_rules, global_confidence(minima)), classes, scaler)
    assert set(model.predict(data.data)) <= {0, 1, 2}
    assert set(model.rule_labels()) <= {0, 1, 2}


def test_top_k_rounds_up_and_ranks_within_each_client():
    def rule(index):
        return RulePayload((0,), (float(index),), ("INDISCERNIBLE",), (1.,), 0, 0.5)

    client_rules = [[rule(0), rule(1), rule(2)], [rule(3), rule(4)]]
    confidence = np.array([.1, .3, .2, .4, .6])
    selected = top_k_per_client(client_rules, confidence, k=0.5)
    assert [r.values[0] for r in selected] == [1., 2., 4.]
    assert [r.confidence for r in selected] == [.3, .2, .6]

    counts = [np.array([9, 1, 5]), np.array([7, 2])]
    selected = top_k_per_client(client_rules, confidence, k=0.5, rank_by="coverage", coverage_counts=counts)
    assert [r.values[0] for r in selected] == [0., 2., 3.]
    assert [r.confidence for r in selected] == [.1, .2, .4]

    with pytest.raises(ValueError):
        top_k_per_client(client_rules, confidence, rank_by="coverage")
    with pytest.raises(ValueError):
        top_k_per_client(client_rules, confidence, rank_by="length")


def test_rules_that_can_not_fire_are_dropped():
    def rule(index):
        return RulePayload((0,), (float(index),), ("INDISCERNIBLE",), (1.,), 0, 0.5)

    client_rules = [[rule(0), rule(1)]]
    assert len(union_global(client_rules, np.array([0., .2]))) == 1
    assert len(top_k_per_client(client_rules, np.array([0., .2]), k=1)) == 1


def test_set_cover_is_minimal_and_reports_uncovered_objects():
    rules = [RulePayload((0,), (float(i),), ("INDISCERNIBLE",), (1.,), 0, 0.5) for i in range(3)]
    coverage = np.array([[1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0]], dtype=float)
    selected, uncovered = set_cover(rules, coverage)
    assert [r.values[0] for r in selected] == [0., 2.]
    assert uncovered == 1


def test_set_cover_objective_is_pluggable():
    rules = [RulePayload((0,), (float(i),), ("INDISCERNIBLE",), (1.,), 0, 0.5) for i in range(3)]
    coverage = np.array([[1, 1, 0], [1, 0, 0], [0, 1, 1]], dtype=float)
    assert len(set_cover(rules, coverage)[0]) == 2

    def expensive_first_rule(candidates):
        weights = plain_set_cover(candidates)
        weights[0] = 10
        return weights

    selected, _ = set_cover(rules, coverage, expensive_first_rule)
    assert [r.values[0] for r in selected] == [1., 2.]
