"""
Real parallel run of the clients, to check the derived parallel training time of experiments/horizontal_fl_frri.py.

Every client lives in its own process for the whole run and keeps its state there, like a separate machine would.
The server sends a method name and its arguments through a pipe, and gets the result back. For the rounds local fit,
confidence and coverage this script reports:
- derived: the slowest client when the clients run one after the other in one process, as horizontal_fl_frri does,
- in worker: the slowest client when every client runs in its own process at the same time, timed inside the worker,
- wall: the time of the whole round seen from the server, including pickling and sending the messages.
With more clients than CPU cores, the clients have to share cores, so the parallel times grow.

Run from the root of the repository:
    python -m experiments.parallel_clients --dataset synthetic --clients 2 3 4 --seeds 0 1 2
"""
import argparse
import os
import warnings
from multiprocessing import Pipe, Process
from time import perf_counter

import numpy as np
import pandas as pd

from experiments.datasets import DATASETS
from fuzzyroughrules.federated.aggregate import global_confidence, union_global, union_local
from fuzzyroughrules.federated.client import FederatedClient
from fuzzyroughrules.federated.partition import FixedRangeScaler, global_classes, global_min_max, horizontal_split


def serve(connection, X_train: np.ndarray, y_train: np.ndarray):
    warnings.filterwarnings("ignore", message="Gurobi can not be used")
    client = FederatedClient(X_train, y_train)
    while True:
        message = connection.recv()
        if message is None:
            break
        method, args = message
        start = perf_counter()
        result = getattr(client, method)(*args)
        connection.send((result, perf_counter() - start))
    connection.close()


class ProcessClients:
    def __init__(self, split):
        self.connections, self.processes = [], []
        for client in split.clients:
            server_end, client_end = Pipe()
            process = Process(target=serve, args=(client_end, client.X_train, client.y_train), daemon=True)
            process.start()
            self.connections.append(server_end)
            self.processes.append(process)

    def call(self, method: str, args_per_client: list[tuple]) -> tuple[list, list[float], float]:
        start = perf_counter()
        for connection, args in zip(self.connections, args_per_client):
            connection.send((method, args))
        replies = [connection.recv() for connection in self.connections]
        wall = perf_counter() - start
        return [result for result, _ in replies], [duration for _, duration in replies], wall

    def close(self):
        for connection in self.connections:
            connection.send(None)
        for process in self.processes:
            process.join()


def sequential_call(clients: list[FederatedClient], method: str, args_per_client: list[tuple]):
    results, durations = [], []
    for client, args in zip(clients, args_per_client):
        start = perf_counter()
        results.append(getattr(client, method)(*args))
        durations.append(perf_counter() - start)
    return results, durations


def protocol(call, n_clients: int, ranges, labels, seed: int) -> dict:
    """
    Runs local fit, confidence and coverage through call(method, args_per_client) and returns the timings per round.
    """
    scaler = FixedRangeScaler(*global_min_max(ranges))
    classes = global_classes(labels)
    timings = {}
    client_rules, timings["local fit"] = call("fit_local", [(scaler, classes)] * n_clients)
    candidates = union_local(client_rules)
    minima, timings["confidence"] = call("local_minima", [(candidates,)] * n_clients)
    rated = union_global(client_rules, global_confidence(minima))
    _, timings["coverage"] = call("coverage_columns", [(rated, seed + index) for index in range(n_clients)])
    return timings


def compare(X: np.ndarray, y: np.ndarray, n_clients: int, seed: int) -> list[dict]:
    split = horizontal_split(X, y, n_clients, seed)
    ranges = [(np.min(c.X_train, axis=0), np.max(c.X_train, axis=0)) for c in split.clients]
    labels = [np.unique(c.y_train) for c in split.clients]

    clients = [FederatedClient(c.X_train, c.y_train) for c in split.clients]
    derived = protocol(
        lambda method, args: (lambda r: (r[0], max(r[1])))(sequential_call(clients, method, args)),
        n_clients, ranges, labels, seed
    )
    workers = ProcessClients(split)
    try:
        measured = protocol(
            lambda method, args: (lambda r: (r[0], (max(r[1]), r[2])))(workers.call(method, args)),
            n_clients, ranges, labels, seed
        )
    finally:
        workers.close()
    return [{
        "n_clients": n_clients,
        "seed": seed,
        "round": name,
        "derived_parallel_s": derived[name],
        "in_worker_parallel_s": measured[name][0],
        "wall_parallel_s": measured[name][1],
    } for name in derived]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=DATASETS, default="synthetic")
    parser.add_argument("--clients", nargs="+", type=int, default=[2, 3, 4])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--out", help="CSV file for the results")
    args = parser.parse_args()
    warnings.filterwarnings("ignore", message="Gurobi can not be used")

    X, y = DATASETS[args.dataset]()
    rows = [row for n_clients in args.clients for seed in args.seeds for row in compare(X, y, n_clients, seed)]
    results = pd.DataFrame(rows)
    if args.out:
        results.to_csv(args.out, index=False)
    pd.set_option("display.width", 200)
    print(f"dataset: {args.dataset}, CPU cores: {os.cpu_count()}")
    print(results.round(4).to_string(index=False))
    summary = results.groupby(["n_clients", "round"])[
        ["derived_parallel_s", "in_worker_parallel_s", "wall_parallel_s"]
    ].mean()
    summary["wall / derived"] = summary["wall_parallel_s"] / summary["derived_parallel_s"]
    print("\nMean over seeds")
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
