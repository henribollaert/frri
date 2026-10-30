"""
Runs the protocol of experiments/horizontal_fl_frri.py for every combination of dataset, number of clients and seed,
and appends every run to the CSV logs.

Run from the root of the repository:
    python -m experiments.protocol_driver --datasets iris wine --log-file runs.csv --class-log-file classes.csv
    python -m experiments.protocol_driver --keel-dirs keel-data/yeast-10-fold --log-file runs.csv
"""
import argparse
import warnings
from pathlib import Path
from time import perf_counter

from experiments.datasets import DATASETS
from experiments.horizontal_fl_frri import add_run_options, log_run, run, run_context, run_options
from fuzzyroughrules.federated.partition import PROTOCOL_SEEDS, load_keel_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="*", default=[], choices=DATASETS, help="datasets without files")
    parser.add_argument("--keel-dirs", nargs="*", type=Path, default=[], help="KEEL folders")
    parser.add_argument("--clients", nargs="+", type=int, default=[2, 3, 4, 5])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(PROTOCOL_SEEDS))
    add_run_options(parser)
    args = parser.parse_args()
    if args.log_file is None:
        parser.error("--log-file is required")
    warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

    sources = [(Path(name), DATASETS[name]) for name in args.datasets]
    sources += [(path, lambda path=path: load_keel_dataset(path)) for path in args.keel_dirs]
    for name, load in sources:
        X, y = load()
        for n_clients in args.clients:
            for seed in args.seeds:
                start = perf_counter()
                models, _, _, _, recalls = run(X, y, n_clients, seed, **run_options(args))
                log_run(args.log_file, models, name, X, n_clients, seed, **run_context(args))
                if args.class_log_file is not None:
                    log_run(args.class_log_file, recalls, name, X, n_clients, seed, **run_context(args))
                print(f"{name.name}, {n_clients} clients, seed {seed}: {perf_counter() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
