"""
Scaling of federated FRRI (set cover variant) and centralised FRRI on synthetic data of growing size.

Every size runs in a fresh child process with a time limit and a memory limit (an address space limit), so that the
peak resident memory of the process (ru_maxrss) belongs to that size alone and a size that is too large cannot take the
machine down. The sizes grow until a size fails, separately for the federated and the centralised run. The set covers
use solver auto: with the size-limited Gurobi licence that comes with pip, problems with more than 2000 rules or
objects fall back on scipy, see select_rules_from_coverage. Repair and coverage rounds are not part of this run.

Run from the root of the repository:
    python -m experiments.scaling --sizes 1000 2000 4000 8000 --clients 3 --out scaling.csv
"""
import argparse
import json
import resource
import subprocess
import sys
import warnings
from time import perf_counter

import numpy as np
import pandas as pd

from experiments.datasets import synthetic
from experiments.measure import RoundRecorder
from fuzzyroughrules.federated.aggregate import global_confidence, set_cover, union_global, union_local
from fuzzyroughrules.federated.client import FederatedClient
from fuzzyroughrules.federated.partition import FixedRangeScaler, global_classes, global_min_max, horizontal_split
from fuzzyroughrules.rule_induction_base import RuleGenerator


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # kilobytes on Linux


def federated(X: np.ndarray, y: np.ndarray, n_clients: int, time_limit: float) -> dict:
    split = horizontal_split(X, y, n_clients, seed=0)
    clients = [FederatedClient(c.X_train, c.y_train) for c in split.clients]
    record = RoundRecorder()
    ranges = record("range", clients, lambda _, client: (client.min_max(), client.labels()))
    global_range = global_min_max([r for r, _ in ranges])
    scaler = FixedRangeScaler(*global_range)
    classes = global_classes([labels for _, labels in ranges])
    client_rules = record("local fit", clients, lambda _, client: client.fit_local(scaler, classes),
                          (global_range, classes))
    candidates = union_local(client_rules)
    confidence = global_confidence(record("confidence", clients, lambda _, c: c.local_minima(candidates), candidates))
    rated = union_global(client_rules, confidence)
    coverage = np.hstack(record("coverage", clients, lambda i, c: c.coverage_columns(rated, i), rated))
    selected, n_uncovered = record.server("set cover", lambda: set_cover(rated, coverage, time_limit=time_limit))
    rounds = ["range", "local fit", "confidence", "coverage", "set cover"]
    result = {"n_candidates": len(candidates), "n_pooled_objects": coverage.shape[1], "n_rules": len(selected),
              "n_uncovered": n_uncovered, **record.totals(rounds)}
    for name in rounds:
        result[f"{name}_parallel_s"] = record.totals([name])["train_s_parallel"]
    return result


def centralised(X: np.ndarray, y: np.ndarray, n_clients: int, time_limit: float) -> dict:
    split = horizontal_split(X, y, n_clients, seed=0)
    X_train = np.vstack([c.X_train for c in split.clients])
    y_train = np.concatenate([c.y_train for c in split.clients])
    model = RuleGenerator(solver_time_limit=time_limit)
    start = perf_counter()
    model.fit(X_train, y_train)
    duration = perf_counter() - start
    return {"n_pooled_objects": len(X_train), "n_rules": len(model.rules_), "train_s_parallel": duration,
            "train_s_sequential": duration}


def child(args: argparse.Namespace):
    resource.setrlimit(resource.RLIMIT_AS, (args.memory_limit_gb * 2 ** 30, resource.RLIM_INFINITY))
    X, y = synthetic(args.size, args.features, args.classes)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = (federated if args.child == "federated" else centralised)(X, y, args.clients, args.set_cover_limit)
    messages = [str(w.message) for w in caught]
    result["gurobi_fallback"] = any("Gurobi can not be used" in m for m in messages)
    result["set_cover_time_limit_hit"] = any("time limit" in m for m in messages)
    result["peak_rss_mb"] = peak_rss_mb()
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", nargs="+", type=int, default=[1000, 2000, 4000, 8000])
    parser.add_argument("--features", type=int, default=10)
    parser.add_argument("--classes", type=int, default=4)
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--time-limit", type=float, default=1200, help="seconds for one size and one model")
    parser.add_argument("--memory-limit-gb", type=int, default=10)
    parser.add_argument("--set-cover-limit", type=float, default=300, help="seconds for one set cover")
    parser.add_argument("--models", nargs="+", default=["federated", "centralised"])
    parser.add_argument("--out", help="CSV file for the results")
    parser.add_argument("--child", choices=["federated", "centralised"], help=argparse.SUPPRESS)
    parser.add_argument("--size", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        child(args)
        return

    rows = []
    for model in args.models:
        for size in args.sizes:
            command = [sys.executable, "-m", "experiments.scaling", "--child", model, "--size", str(size),
                       "--features", str(args.features), "--classes", str(args.classes), "--clients",
                       str(args.clients), "--memory-limit-gb", str(args.memory_limit_gb), "--set-cover-limit",
                       str(args.set_cover_limit)]
            row = {"model": model, "n_objects": size, "n_features": args.features, "n_clients": args.clients}
            start = perf_counter()
            try:
                completed = subprocess.run(command, capture_output=True, text=True, timeout=args.time_limit)
                lines = completed.stdout.strip().splitlines()
                if completed.returncode == 0 and lines:
                    row.update(json.loads(lines[-1]), status="ok")
                else:
                    error = completed.stderr.strip().splitlines()[-1:] or ["no output"]
                    row["status"] = f"failed: {error[0][:120]}"
            except subprocess.TimeoutExpired:
                row["status"] = f"time limit of {args.time_limit:g}s"
            row["wall_s"] = perf_counter() - start
            rows.append(row)
            print(json.dumps(row), flush=True)
            if row["status"] != "ok":
                break
    results = pd.DataFrame(rows)
    if args.out:
        results.to_csv(args.out, index=False)
    pd.set_option("display.width", 250, "display.max_columns", None)
    print(results.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
