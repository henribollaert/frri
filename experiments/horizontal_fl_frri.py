"""
Smoke test of federated FRRI in the horizontal setting: one run of Beata's protocol, not an experiment.

Run from the root of the repository:
    python -m experiments.horizontal_fl_frri --data-dir <keel-data>/yeast-10-fold --log-file runs.csv

A new data format only needs a new function in LOADERS, that returns the features and the labels.

The clients run one after the other here, so each client is timed separately. The training time of a model is reported
in two ways. The sequential time is the sum over all clients. The parallel time is what the same measurements give
when clients work at the same time: the slowest client of each round. Both include the set cover where it is used, but
neither includes communication, which is not simulated.
"""
import argparse
import warnings
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from experiments.datasets import DATASETS
from fuzzyroughrules.federated.aggregate import (
    build_global_model, global_confidence, set_cover, top_k_per_client, union_global, union_local
)
from fuzzyroughrules.federated.client import FederatedClient
from fuzzyroughrules.federated.inference import with_fallback
from fuzzyroughrules.federated.partition import (
    FixedRangeScaler, global_classes, global_min_max, horizontal_split, load_keel_dataset
)
from fuzzyroughrules.rule_induction_base import RuleGenerator

LOADERS = {
    "keel": load_keel_dataset,
    # --data-dir is the name of a dataset in experiments/datasets.py, e.g. --loader builtin --data-dir wine
    "builtin": lambda path: DATASETS[Path(path).name](),
}
TOP_K = (0.5, 0.7)


def no_rule_fires(model: RuleGenerator, X: np.ndarray) -> float:
    """
    Fraction of the objects with membership 0 for all classes: these get the first class by default.
    """
    return float(np.mean(np.all(model.predict_proba(X, normalized=False) == 0, axis=1)))


def timed_per_client(items: Iterable, step: Callable) -> tuple[list, list[float]]:
    """
    Runs a step for each client and measures each of them separately.
    :param items: one item for each client
    :param step: what to run for an item
    :return: the results, the duration in seconds for each client
    """
    results, durations = [], []
    for item in items:
        start = perf_counter()
        results.append(step(item))
        durations.append(perf_counter() - start)
    return results, durations


def training_times(rounds: list[list[float]], server_time: float = 0.) -> dict:
    """
    :param rounds: for each round that the model needs, the duration for each client
    :param server_time: time spent on the server
    :return: the parallel and sequential training time
    """
    return {
        "train_s_parallel": sum(max(durations) for durations in rounds) + server_time,
        "train_s_sequential": sum(sum(durations) for durations in rounds) + server_time,
    }


def describe(model: RuleGenerator, split, name: str, times: dict, default_label) -> dict:
    """
    The scores without suffix use FRRI inference, which gives the first class when no rule fires. The suffix
    _relation_fallback means that such objects get the class of the rule with the highest relation, and _default_rule
    that they get default_label.
    """
    relation = with_fallback(model, "relation")
    default = with_fallback(model, "default", default_label)
    row = {
        "model": name,
        "n_rules": len(model.rules_),
        "avg_rule_length": model.average_rule_length(),
        **times,
        "global_test": balanced_accuracy_score(split.y_test, model.predict(split.X_test)),
        "global_test_relation_fallback": balanced_accuracy_score(split.y_test, relation.predict(split.X_test)),
        "global_test_default_rule": balanced_accuracy_score(split.y_test, default.predict(split.X_test)),
        "no_rule_fires_global_test": no_rule_fires(model, split.X_test),
    }
    local_scores = [balanced_accuracy_score(client.y_test, model.predict(client.X_test)) for client in split.clients]
    row["mean_test_local"] = np.mean(local_scores)
    row["mean_test_local_relation_fallback"] = np.mean([
        balanced_accuracy_score(client.y_test, relation.predict(client.X_test)) for client in split.clients
    ])
    for index, score in enumerate(local_scores):
        row[f"test_local_{index}"] = score
    return row


def class_recalls(model: RuleGenerator, split, name: str, classes: np.ndarray, default_label) -> list[dict]:
    """
    Recall of each global class on the global test set, for each way of handling objects for which no rule fires.
    A class without test objects gets NaN.
    """
    rows = []
    silent = np.all(model.predict_proba(split.X_test, normalized=False) == 0, axis=1)
    for fallback in ("first", "relation", "default"):
        predictions = with_fallback(model, fallback, default_label).predict(split.X_test)
        for label in classes:
            present = split.y_test == label
            rows.append({
                "model": name,
                "fallback": fallback,
                "label": label,
                "n_global_test": int(np.sum(present)),
                "recall": float(np.mean(predictions[present] == label)) if np.any(present) else np.nan,
                "no_rule_fires": float(np.mean(silent[present])) if np.any(present) else np.nan,
            })
    return rows


def run(
        X: np.ndarray,
        y: np.ndarray,
        n_clients: int,
        seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    run_start = perf_counter()
    split = horizontal_split(X, y, n_clients, seed)
    clients = [FederatedClient(client.X_train, client.y_train) for client in split.clients]

    # round 1: shared scale and classes, local induction
    scaler = FixedRangeScaler(*global_min_max([client.min_max() for client in clients]))
    classes = global_classes([client.labels() for client in clients])
    client_rules, fit_times = timed_per_client(clients, lambda client: client.fit_local(scaler, classes))

    # round 2: global confidence
    candidates = union_local(client_rules)
    minima, confidence_times = timed_per_client(clients, lambda client: client.local_minima(candidates))
    confidence = global_confidence(minima)

    # round 3: pooled coverage for the set cover, with the global confidence
    rated = union_global(client_rules, confidence)
    coverage, coverage_times = timed_per_client(
        enumerate(clients), lambda item: item[1].coverage_columns(rated, seed + item[0])
    )
    coverage = np.hstack(coverage)
    start = perf_counter()
    covered_rules, n_uncovered = set_cover(rated, coverage)
    set_cover_time = perf_counter() - start

    # only needed to rank the rules by coverage
    counts, count_times = timed_per_client(
        zip(clients, client_rules), lambda item: item[0].coverage_counts(item[1])
    )

    with_confidence = training_times([fit_times, confidence_times])
    variants = {
        "a union local conf": (union_local(client_rules), training_times([fit_times])),
        "b union global conf": (rated, with_confidence),
    }
    for k in TOP_K:
        variants[f"c top {k:.0%} confidence"] = (top_k_per_client(client_rules, confidence, k), with_confidence)
        variants[f"c top {k:.0%} coverage"] = (
            top_k_per_client(client_rules, confidence, k, rank_by="coverage", coverage_counts=counts),
            training_times([fit_times, confidence_times, count_times])
        )
    variants["d set cover"] = (
        covered_rules,
        training_times([fit_times, confidence_times, coverage_times], set_cover_time)
    )

    central = RuleGenerator()
    central.set_params(scaler=scaler)
    start = perf_counter()
    central.fit(np.vstack([client.X_train for client in clients]), np.concatenate([c.y_train for c in clients]))
    central_time = perf_counter() - start

    # only for the default-rule baseline: the class counts are not part of the protocol
    default_label = classes[np.argmax(np.sum([client.label_counts(classes) for client in clients], axis=0))]
    models = {name: (build_global_model(rules, classes, scaler), times) for name, (rules, times) in variants.items()}
    models["centralised FRRI"] = (central, training_times([[central_time]]))
    rows = [describe(model, split, name, times, default_label) for name, (model, times) in models.items()]
    recall_rows = [
        row for name, (model, _) in models.items() for row in class_recalls(model, split, name, classes, default_label)
    ]

    local_rows = [{
        "client": index,
        "n_train_local": len(client.X_train),
        "n_rules": len(client.model.rules_),
        "avg_rule_length": client.model.average_rule_length(),
        "train_s": fit_times[index],
        "own_test_local": balanced_accuracy_score(
            split.clients[index].y_test, client.model.predict(split.clients[index].X_test)
        ),
    } for index, client in enumerate(clients)]

    round_rows = [
        {"round": name, **{f"client_{index}_s": duration for index, duration in enumerate(durations)}}
        for name, durations in {
            "local fit": fit_times,
            "confidence": confidence_times,
            "coverage": coverage_times,
            "coverage counts": count_times,
        }.items()
    ]
    round_rows.append({"round": "set cover (server)", "server_s": set_cover_time})
    round_rows.append({"round": "centralised fit", "server_s": central_time})

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
        "run_s": perf_counter() - run_start,
    }
    return pd.DataFrame(rows), pd.DataFrame(local_rows), pd.DataFrame(round_rows), info, pd.DataFrame(recall_rows)


def log_run(log_file: Path, models: pd.DataFrame, data_dir: Path, X: np.ndarray, n_clients: int, seed: int):
    """
    Appends one row per model. The scores of the separate clients are left out, so that runs with a different number
    of clients fit in the same file. The same function logs the class recalls, with one row per model and class.
    """
    row = models.drop(columns=models.filter(regex=r"^test_local_\d").columns).assign(
        finished_at=datetime.now().isoformat(timespec="seconds"),
        dataset=data_dir.name,
        n_objects=X.shape[0],
        n_features=X.shape[1],
        n_clients=n_clients,
        seed=seed,
    )
    row.to_csv(log_file, mode="a", header=not log_file.exists(), index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--loader", choices=LOADERS, default="keel")
    parser.add_argument("--clients", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-file", type=Path, help="CSV file that each run is appended to")
    parser.add_argument("--class-log-file", type=Path, help="CSV file that the recall of each class is appended to")
    args = parser.parse_args()

    # a test_local part can miss a class, in which case balanced accuracy only uses the classes that are present
    warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

    X, y = LOADERS[args.loader](args.data_dir)
    models, local_models, rounds, info, recalls = run(X, y, args.clients, args.seed)
    if args.log_file is not None:
        log_run(args.log_file, models, args.data_dir, X, args.clients, args.seed)
    if args.class_log_file is not None:
        log_run(args.class_log_file, recalls, args.data_dir, X, args.clients, args.seed)

    pd.set_option("display.width", 200, "display.max_columns", None)
    print(f"clients: {args.clients}, seed: {args.seed}")
    for key, value in info.items():
        print(f"{key}: {value}")
    print("\nGlobal models and centralised FRRI")
    print(models.round(4).to_string(index=False))
    print("\nLocal models")
    print(local_models.round(4).to_string(index=False))
    print("\nDuration of each round in seconds")
    print(rounds.round(4).to_string(index=False, na_rep=""))
    print("\nRecall per class on the global test set, by fallback for objects for which no rule fires")
    print(recalls.pivot_table(index=["model", "fallback"], columns="label", values="recall", sort=False).round(4))


if __name__ == "__main__":
    main()
