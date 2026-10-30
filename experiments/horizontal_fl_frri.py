"""
Smoke test of federated FRRI in the horizontal setting: one run of Beata's protocol, not an experiment.

Run from the root of the repository:
    python -m experiments.horizontal_fl_frri --data-dir <keel-data>/yeast-10-fold --log-file runs.csv
    python -m experiments.horizontal_fl_frri --loader builtin --data-dir wine --clients 3

A new data format only needs a new function in LOADERS, that returns the features and the labels.

The clients run one after the other here, so each client is timed separately. The training time of a model is reported
in two ways. The sequential time is the sum over all clients. The parallel time is what the same measurements give
when clients work at the same time: the slowest client of each round. Both include the server steps that the model
needs, but neither includes communication, which is not simulated. Bytes and memory are explained in
experiments/measure.py.
"""
import argparse
import warnings
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from experiments.datasets import DATASETS
from experiments.measure import RoundRecorder
from fuzzyroughrules.federated.aggregate import (
    build_global_model, global_confidence, plain_set_cover, set_cover, top_k_per_client, union_global, union_local
)
from fuzzyroughrules.federated.client import FederatedClient
from fuzzyroughrules.federated.inference import with_fallback
from fuzzyroughrules.federated.partition import (
    FixedRangeScaler, global_classes, global_min_max, horizontal_split, load_keel_dataset
)
from fuzzyroughrules.federated.rounds import confidence_weighted_set_cover, coverage_rounds, repair_rules
from fuzzyroughrules.operators import RelationTypes
from fuzzyroughrules.rule_induction_base import RuleGenerator

LOADERS = {
    "keel": load_keel_dataset,
    # --data-dir is the name of a dataset in experiments/datasets.py, e.g. --loader builtin --data-dir wine
    "builtin": lambda path: DATASETS[Path(path).name](),
}
TOP_K = (0.5, 0.7)
COVERAGE_ROUND_VARIANTS = {
    # name: (prefix of the rounds, starts from the repaired rules, objective)
    "f coverage rounds": ("f plain: ", False, plain_set_cover),
    "f coverage rounds, conf-weighted": ("f weighted: ", False, confidence_weighted_set_cover),
    "f repair + coverage rounds": ("f repair: ", True, plain_set_cover),
}


def no_rule_fires(model: RuleGenerator, X: np.ndarray) -> float:
    """
    Fraction of the objects with membership 0 for all classes: these get the first class by default.
    """
    return float(np.mean(np.all(model.predict_proba(X, normalized=False) == 0, axis=1)))


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


def local_score(model: RuleGenerator, client) -> float:
    """
    Balanced accuracy on the test_local data of a client, NaN if it has none, which a non-IID split can give.
    """
    if len(client.y_test) == 0:
        return np.nan
    return balanced_accuracy_score(client.y_test, model.predict(client.X_test))


def describe(model: RuleGenerator, split, name: str, measures: dict, default_label) -> dict:
    """
    The scores without suffix use FRRI inference, which gives the first class when no rule fires. The suffix
    _relation_fallback means that such objects get the class of the rule with the highest relation, and _default_rule
    that they get default_label. pooled_test_local is the balanced accuracy on the test_local data of all clients
    together, so that every model is scored on the same objects and classes, also when a class is missing from the
    test_local data of a client.
    """
    relation = with_fallback(model, "relation")
    default = with_fallback(model, "default", default_label)
    X_pooled = np.vstack([client.X_test for client in split.clients])
    y_pooled = np.concatenate([client.y_test for client in split.clients])
    row = {
        "model": name,
        "n_rules": len(model.rules_),
        "avg_rule_length": model.average_rule_length() if model.rules_ else 0.,
        **measures,
        "global_test": balanced_accuracy_score(split.y_test, model.predict(split.X_test)),
        "global_test_relation_fallback": balanced_accuracy_score(split.y_test, relation.predict(split.X_test)),
        "global_test_default_rule": balanced_accuracy_score(split.y_test, default.predict(split.X_test)),
        "no_rule_fires_global_test": no_rule_fires(model, split.X_test),
        "pooled_test_local": balanced_accuracy_score(y_pooled, model.predict(X_pooled)),
        "pooled_test_local_relation_fallback": balanced_accuracy_score(y_pooled, relation.predict(X_pooled)),
    }
    local_scores = [local_score(model, client) for client in split.clients]
    row["mean_test_local"] = np.nanmean(local_scores)
    row["mean_test_local_relation_fallback"] = np.nanmean([local_score(relation, client) for client in split.clients])
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
        seed: int,
        dirichlet_alpha: float = None,
        repair_max_rounds: int = None,
        coverage_max_rounds: int = 3,
        trace_memory: bool = False,
        solver: str = "auto",
        time_limit: float = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    """
    :param dirichlet_alpha: None for the protocol split, otherwise label skew, see horizontal_split
    :param repair_max_rounds: maximum number of repair rounds, None for no limit
    :param coverage_max_rounds: maximum number of coverage rounds after the first set cover
    :param trace_memory: measure the peak memory of every round, which makes the durations longer
    :param solver: solver of every set cover and of centralised FRRI
    :param time_limit: time limit of every set cover
    :return: one row per model, one row per local model, one row per round, information about the run, the recall of
    every class for every model
    """
    run_start = perf_counter()
    split = horizontal_split(X, y, n_clients, seed, dirichlet_alpha=dirichlet_alpha)
    clients = [FederatedClient(client.X_train, client.y_train, solver=solver) for client in split.clients]
    record = RoundRecorder(trace_memory)
    try:
        # round 1: shared scale and classes, local induction
        ranges = record("range", clients, lambda _, client: (client.min_max(), client.labels()))
        global_range = global_min_max([client_range for client_range, _ in ranges])
        scaler = FixedRangeScaler(*global_range)
        classes = global_classes([labels for _, labels in ranges])
        client_rules = record(
            "local fit", clients, lambda _, client: client.fit_local(scaler, classes), (global_range, classes)
        )

        # round 2: global confidence
        candidates = union_local(client_rules)
        minima = record("confidence", clients, lambda _, client: client.local_minima(candidates), candidates)
        confidence = global_confidence(minima)

        # round 3: pooled coverage for the set cover, with the global confidence
        rated = union_global(client_rules, confidence)
        coverage = np.hstack(record(
            "coverage", clients, lambda index, client: client.coverage_columns(rated, seed + index), rated
        ))
        covered_rules, n_uncovered = record.server(
            "set cover", lambda: set_cover(rated, coverage, solver=solver, time_limit=time_limit)
        )

        # only needed to rank the rules by coverage
        counts = record("coverage counts", clients, lambda index, client: client.coverage_counts(client_rules[index]))

        # rule repair, and a set cover of the repaired rules
        repaired, repaired_confidence, repair_log = repair_rules(
            clients, client_rules, confidence, repair_max_rounds, map_clients=record
        )
        repaired_rated = union_global(repaired, repaired_confidence)
        repaired_coverage = np.hstack(record(
            "repair: coverage", clients, lambda index, client: client.coverage_columns(repaired_rated, seed + index),
            repaired_rated
        ))
        repaired_cover, n_repaired_uncovered = record.server(
            "repair: set cover", lambda: set_cover(repaired_rated, repaired_coverage, solver=solver,
                                                   time_limit=time_limit)
        )

        # coverage rounds
        with_rounds = {}
        for name, (prefix, from_repaired, objective) in COVERAGE_ROUND_VARIANTS.items():
            recorder = record.prefixed(prefix)
            with_rounds[name] = coverage_rounds(
                clients,
                repaired if from_repaired else client_rules,
                repaired_confidence if from_repaired else confidence,
                seed,
                coverage_max_rounds,
                objective,
                solver,
                time_limit,
                map_clients=recorder,
                run_on_server=recorder.server
            )

        central = RuleGenerator(solver=solver, solver_time_limit=time_limit)
        central.set_params(scaler=scaler)
        record.server("centralised fit", lambda: central.fit(
            np.vstack([client.X_train for client in clients]), np.concatenate([c.y_train for c in clients])
        ))
    finally:
        record.close()

    base = ["range", "local fit"]
    with_confidence = base + ["confidence"]
    repair_rounds = with_confidence + ["repair: extend", "repair: confidence"]
    variants = {
        "a union local conf": (union_local(client_rules), base),
        "b union global conf": (rated, with_confidence),
    }
    for k in TOP_K:
        variants[f"c top {k:.0%} confidence"] = (top_k_per_client(client_rules, confidence, k), with_confidence)
        variants[f"c top {k:.0%} coverage"] = (
            top_k_per_client(client_rules, confidence, k, rank_by="coverage", coverage_counts=counts),
            with_confidence + ["coverage counts"]
        )
    variants["d set cover"] = (covered_rules, with_confidence + ["coverage", "set cover"])
    variants["e repair, union global conf"] = (repaired_rated, repair_rounds)
    variants["e repair, set cover"] = (repaired_cover, repair_rounds + ["repair: coverage", "repair: set cover"])
    for name, (prefix, from_repaired, _) in COVERAGE_ROUND_VARIANTS.items():
        variants[name] = (with_rounds[name][0], (repair_rounds if from_repaired else with_confidence) + record.names(prefix))

    # only for the default-rule baseline: the class counts are not part of the protocol
    default_label = classes[np.argmax(np.sum([client.label_counts(classes) for client in clients], axis=0))]
    models = {
        name: (build_global_model(rules, classes, scaler), record.totals(rounds))
        for name, (rules, rounds) in variants.items()
    }
    models["centralised FRRI"] = (central, record.totals(["centralised fit"]))
    rows = [describe(model, split, name, measures, default_label) for name, (model, measures) in models.items()]
    recall_rows = [
        row for name, (model, _) in models.items() for row in class_recalls(model, split, name, classes, default_label)
    ]

    fit_times = record.stats["local fit"].calls[0]
    local_rows = [{
        "client": index,
        "n_train_local": len(client.X_train),
        "n_rules": len(client.model.rules_),
        "avg_rule_length": client.model.average_rule_length(),
        "train_s": fit_times[index],
        "own_test_local": local_score(client.model, split.clients[index]),
    } for index, client in enumerate(clients)]

    needed_repair = [
        position for position, (rule, value) in enumerate(zip(candidates, confidence))
        if value < rule.confidence - 1e-6
    ]
    repaired_union = union_local(repaired)
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
        "repair_n_rules_below_local": len(needed_repair),
        "repair_n_rounds": len(repair_log),
        "repair_n_recovered": sum(entry["n_recovered"] for entry in repair_log),
        "repair_n_extra_attributes_revealed": sum(
            len(after.attributes) - len(before.attributes) for before, after in zip(candidates, repaired_union)
        ),
        "repair_n_rules_with_all_attributes": sum(
            len(repaired_union[p].attributes) == X.shape[1] for p in needed_repair
        ),
        "repair_n_uncovered_in_set_cover": n_repaired_uncovered,
    }
    for name, (_, n_left, _, log) in with_rounds.items():
        info[f"{name}: n_rounds"] = len(log) - 1
        info[f"{name}: n_extra_rules_exposed"] = sum(entry["n_new_candidates"] for entry in log)
        info[f"{name}: n_uncovered_first_last"] = (log[0]["n_uncovered"], n_left)
    info["run_s"] = perf_counter() - run_start
    return pd.DataFrame(rows), pd.DataFrame(local_rows), pd.DataFrame(record.table(n_clients)), info, pd.DataFrame(
        recall_rows
    )


def log_run(
        log_file: Path,
        models: pd.DataFrame,
        data_dir: Path,
        X: np.ndarray,
        n_clients: int,
        seed: int,
        **context
):
    """
    Appends one row per model. The scores of the separate clients are left out, so that runs with a different number
    of clients fit in the same file. The same function logs the class recalls, with one row per model and class.
    :param context: extra columns, e.g. the options of the run
    """
    row = models.drop(columns=models.filter(regex=r"^test_local_\d").columns).assign(
        finished_at=datetime.now().isoformat(timespec="seconds"),
        dataset=data_dir.name,
        n_objects=X.shape[0],
        n_features=X.shape[1],
        n_clients=n_clients,
        seed=seed,
        **context,
    )
    if log_file.exists() and pd.read_csv(log_file, nrows=0).columns.tolist() != row.columns.tolist():
        # an older file with other columns is rewritten with the union of the columns, so no column is shifted
        row = pd.concat([pd.read_csv(log_file), row], ignore_index=True)
        row.to_csv(log_file, index=False)
        return
    row.to_csv(log_file, mode="a", header=not log_file.exists(), index=False)


def add_run_options(parser: argparse.ArgumentParser):
    parser.add_argument("--log-file", type=Path, help="CSV file that each run is appended to")
    parser.add_argument("--class-log-file", type=Path, help="CSV file that the recall of each class is appended to")
    parser.add_argument("--dirichlet-alpha", type=float, help="non-IID split with label skew, see horizontal_split")
    parser.add_argument("--repair-max-rounds", type=int, help="maximum number of repair rounds, default no limit")
    parser.add_argument("--coverage-max-rounds", type=int, default=3)
    parser.add_argument("--trace-memory", action="store_true", help="measure peak memory, makes durations longer")
    parser.add_argument("--solver", choices=("auto", "gurobi", "scipy"), default="auto")
    parser.add_argument("--time-limit", type=float, help="time limit in seconds of every set cover")


def run_options(args: argparse.Namespace) -> dict:
    return {
        "dirichlet_alpha": args.dirichlet_alpha,
        "repair_max_rounds": args.repair_max_rounds,
        "coverage_max_rounds": args.coverage_max_rounds,
        "trace_memory": args.trace_memory,
        "solver": args.solver,
        "time_limit": args.time_limit,
    }


def run_context(args: argparse.Namespace) -> dict:
    """
    The options that are logged with every row. A missing alpha means the protocol split.
    """
    options = run_options(args)
    options["partition"] = "iid" if options.pop("dirichlet_alpha") is None else f"dirichlet {args.dirichlet_alpha:g}"
    return options


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--loader", choices=LOADERS, default="keel")
    parser.add_argument("--clients", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    add_run_options(parser)
    args = parser.parse_args()

    # a test_local part can miss a class, in which case balanced accuracy only uses the classes that are present
    warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

    X, y = LOADERS[args.loader](args.data_dir)
    models, local_models, rounds, info, recalls = run(X, y, args.clients, args.seed, **run_options(args))
    context = run_context(args)
    if args.log_file is not None:
        log_run(args.log_file, models, args.data_dir, X, args.clients, args.seed, **context)
    if args.class_log_file is not None:
        log_run(args.class_log_file, recalls, args.data_dir, X, args.clients, args.seed, **context)

    pd.set_option("display.width", 200, "display.max_columns", None)
    print(f"clients: {args.clients}, seed: {args.seed}")
    for key, value in info.items():
        print(f"{key}: {value}")
    print("\nGlobal models and centralised FRRI")
    print(models.round(4).to_string(index=False))
    print("\nLocal models")
    print(local_models.round(4).to_string(index=False))
    print("\nRounds: duration in seconds, bytes sent by the clients (up) and the server (down), peak memory")
    print(rounds.round(4).to_string(index=False, na_rep=""))
    print("\nRecall per class on the global test set, by fallback for objects for which no rule fires")
    print(recalls.pivot_table(index=["model", "fallback"], columns="label", values="recall", sort=False).round(4))


if __name__ == "__main__":
    main()
