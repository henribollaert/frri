"""
Smoke test of federated FRRI in the horizontal setting: one run of Beata's protocol, not an experiment.

Run from the root of the repository:
    python -m experiments.horizontal_fl_frri --data-dir <keel-data>/yeast-10-fold

A new data format only needs a new function in LOADERS, that returns the features and the labels.
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from fuzzyroughrules.federated.aggregate import (
    build_global_model, global_confidence, set_cover, top_k_per_client, union_global, union_local
)
from fuzzyroughrules.federated.client import FederatedClient
from fuzzyroughrules.federated.partition import (
    FixedRangeScaler, global_classes, global_min_max, horizontal_split, load_keel_dataset
)
from fuzzyroughrules.rule_induction_base import RuleGenerator

LOADERS = {"keel": load_keel_dataset}
TOP_K = (0.5, 0.7)


def no_rule_fires(model: RuleGenerator, X: np.ndarray) -> float:
    """
    Fraction of the objects with membership 0 for all classes: these get the first class by default.
    """
    return float(np.mean(np.all(model.predict_proba(X, normalized=False) == 0, axis=1)))


def describe(model: RuleGenerator, split, name: str) -> dict:
    row = {
        "model": name,
        "n_rules": len(model.rules_),
        "avg_rule_length": model.average_rule_length(),
        "global_test": balanced_accuracy_score(split.y_test, model.predict(split.X_test)),
        "no_rule_fires_global_test": no_rule_fires(model, split.X_test),
    }
    for index, client in enumerate(split.clients):
        row[f"test_local_{index}"] = balanced_accuracy_score(client.y_test, model.predict(client.X_test))
    return row


def run(X: np.ndarray, y: np.ndarray, n_clients: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    split = horizontal_split(X, y, n_clients, seed)
    clients = [FederatedClient(client.X_train, client.y_train) for client in split.clients]

    # round 1: shared scale and classes, local induction
    scaler = FixedRangeScaler(*global_min_max([client.min_max() for client in clients]))
    classes = global_classes([client.labels() for client in clients])
    client_rules = [client.fit_local(scaler, classes) for client in clients]

    # round 2: global confidence
    candidates = union_local(client_rules)
    confidence = global_confidence([client.local_minima(candidates) for client in clients])

    # round 3: pooled coverage for the set cover, with the global confidence
    rated = union_global(client_rules, confidence)
    coverage = np.hstack([client.coverage_columns(rated, seed + index) for index, client in enumerate(clients)])
    covered_rules, n_uncovered = set_cover(rated, coverage)

    counts = [client.coverage_counts(rules) for client, rules in zip(clients, client_rules)]
    variants = {"a union local conf": union_local(client_rules), "b union global conf": rated}
    for k in TOP_K:
        variants[f"c top {k:.0%} confidence"] = top_k_per_client(client_rules, confidence, k)
        variants[f"c top {k:.0%} coverage"] = top_k_per_client(
            client_rules, confidence, k, rank_by="coverage", coverage_counts=counts
        )
    variants["d set cover"] = covered_rules

    central = RuleGenerator()
    central.set_params(scaler=scaler)
    central.fit(np.vstack([client.X_train for client in clients]), np.concatenate([c.y_train for c in clients]))

    rows = [describe(build_global_model(rules, classes, scaler), split, name) for name, rules in variants.items()]
    rows.append(describe(central, split, "centralised FRRI"))

    local_rows = [{
        "client": index,
        "n_train_local": len(client.X_train),
        "n_rules": len(client.model.rules_),
        "avg_rule_length": client.model.average_rule_length(),
        "own_test_local": balanced_accuracy_score(
            split.clients[index].y_test, client.model.predict(split.clients[index].X_test)
        ),
    } for index, client in enumerate(clients)]

    info = {
        "n_objects": len(X),
        "n_global_test": len(split.X_test),
        "n_classes": len(classes),
        "n_candidates": len(candidates),
        "n_candidates_zero_global_confidence": int(np.sum(confidence == 0)),
        "n_candidates_confidence_lowered": int(
            np.sum(confidence < np.array([rule.confidence for rule in candidates]) - 1e-9)
        ),
        "n_pooled_objects": coverage.shape[1],
        "n_uncovered_in_set_cover": n_uncovered,
    }
    return pd.DataFrame(rows), pd.DataFrame(local_rows), info


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--loader", choices=LOADERS, default="keel")
    parser.add_argument("--clients", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # a test_local part can miss a class, in which case balanced accuracy only uses the classes that are present
    warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

    X, y = LOADERS[args.loader](args.data_dir)
    models, local_models, info = run(X, y, args.clients, args.seed)

    pd.set_option("display.width", 200, "display.max_columns", None)
    print(f"clients: {args.clients}, seed: {args.seed}")
    for key, value in info.items():
        print(f"{key}: {value}")
    print("\nGlobal models and centralised FRRI")
    print(models.round(4).to_string(index=False))
    print("\nLocal models")
    print(local_models.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
