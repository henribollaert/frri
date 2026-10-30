"""
Diagnosis of the drop from local to global confidence in federated FRRI.

Hypothesis: a rule is shortened on the data of its own client, so the attributes that it drops were only unnecessary
there, and objects of other classes on other clients still match the shortened rule.

For each local rule this script reports its length, its confidence on the data of its own client (local) and on the
pooled train_local data (global), and how many objects of another class on other clients pull the global confidence
below the local one. As a paired control, the unshortened rule of the same object (all attributes, INDISCERNIBLE, slope
1, which is where the reduction of FRRI starts) is evaluated in the same way. If shortening causes the drop, the
unshortened rules should lose much less confidence.

This is an analysis of a simulation: it pools the data of the clients, which the protocol itself never does.

Run from the root of the repository:
    python -m experiments.diagnose_confidence --datasets iris wine --clients 2 5 --rules-file rules.csv
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from experiments.datasets import DATASETS
from fuzzyroughrules.federated.aggregate import build_global_model, global_confidence, union_global, union_local
from fuzzyroughrules.federated.client import FederatedClient
from fuzzyroughrules.federated.partition import (
    PROTOCOL_SEEDS, FixedRangeScaler, global_classes, global_min_max, horizontal_split
)
from fuzzyroughrules.operators import RelationTypes, triangular_relation
from fuzzyroughrules.rule_induction_base import Rule, RuleGenerator, evaluate_rules

TOLERANCE = 1e-9


def unshortened(rule: Rule) -> Rule:
    n_features = len(rule.antecedent)
    return Rule(
        rule.antecedent,
        np.full(n_features, RelationTypes.INDISCERNIBLE, dtype=RelationTypes),
        np.ones(n_features),
        rule.credibility,
        rule.decision
    )


def rule_records(clients: list[FederatedClient], client_rules: list, X_pooled: np.ndarray, y_pooled: np.ndarray,
                 owner_of_object: np.ndarray) -> list[dict]:
    """
    One record per local rule. Local rules are evaluated with the global classes, like the protocol does.
    """
    payloads = union_local(client_rules)
    confidence = global_confidence([client.local_minima(payloads) for client in clients])
    records = []
    index = 0
    for owner, (client, own_payloads) in enumerate(zip(clients, client_rules)):
        rules = client._to_rules(own_payloads)
        # the same objects, with all attributes
        full_rules = [unshortened(rule) for rule in client.model.rules_]
        for rule in full_rules:
            rule.decision = int(np.searchsorted(client.classes, client.model.classes_[rule.decision]))
        own_minima = evaluate_rules(rules, client.X_scaled, client.y_encoded)[0]
        full_own = evaluate_rules(full_rules, client.X_scaled, client.y_encoded)[0]
        full_global = evaluate_rules(full_rules, X_pooled, y_pooled)[0]
        for position, rule in enumerate(rules):
            relation = triangular_relation(X_pooled, rule.antecedent, rule.slopes, rule.reducts).reshape(len(X_pooled))
            # objects of another class whose implication I(R(x, u), 0) = 1 - R(x, u) is below the local confidence
            limiting = (y_pooled != rule.decision) & (1 - relation < own_minima[position] - TOLERANCE)
            records.append({
                "client": owner,
                "label": client.classes[rule.decision],
                "length": int(np.count_nonzero(rule.reducts != RelationTypes.UNUSED)),
                "stored_confidence": own_payloads[position].confidence,
                "local_confidence": own_minima[position],
                "global_confidence": confidence[index],
                "drop": own_minima[position] - confidence[index],
                "limiting_other_clients": int(np.sum(limiting & (owner_of_object != owner))),
                "limiting_own_client": int(np.sum(limiting & (owner_of_object == owner))),
                "full_local_confidence": full_own[position],
                "full_global_confidence": full_global[position],
                "full_drop": full_own[position] - full_global[position],
            })
            index += 1
    return records


def class_rows(model: RuleGenerator, X: np.ndarray, y: np.ndarray, classes: np.ndarray, name: str) -> list[dict]:
    """
    Recall of each class and the fraction of the objects of each class for which no rule fires.
    """
    predictions = model.predict(X)
    silent = np.all(model.predict_proba(X, normalized=False) == 0, axis=1)
    rows = []
    for label in classes:
        present = y == label
        rows.append({
            "model": name,
            "label": label,
            "n_test": int(np.sum(present)),
            # a class without test objects has no recall
            "recall": float(np.mean(predictions[present] == label)) if np.any(present) else np.nan,
            "no_rule_fires": float(np.mean(silent[present])) if np.any(present) else np.nan,
        })
    return rows


def diagnose(X: np.ndarray, y: np.ndarray, n_clients: int, seed: int) -> tuple[list[dict], list[dict]]:
    split = horizontal_split(X, y, n_clients, seed)
    clients = [FederatedClient(client.X_train, client.y_train) for client in split.clients]
    scaler = FixedRangeScaler(*global_min_max([client.min_max() for client in clients]))
    classes = global_classes([client.labels() for client in clients])
    client_rules = [client.fit_local(scaler, classes) for client in clients]

    X_pooled = np.vstack([client.X_scaled for client in clients])
    y_pooled = np.concatenate([client.y_encoded for client in clients])
    owner_of_object = np.concatenate([np.full(len(client.X_train), i) for i, client in enumerate(clients)])
    records = rule_records(clients, client_rules, X_pooled, y_pooled, owner_of_object)

    confidence = np.array([record["global_confidence"] for record in records])
    central = RuleGenerator()
    central.set_params(scaler=scaler)
    central.fit(np.vstack([c.X_train for c in clients]), np.concatenate([c.y_train for c in clients]))
    models = {
        "a union local conf": build_global_model(union_local(client_rules), classes, scaler),
        "b union global conf": build_global_model(union_global(client_rules, confidence), classes, scaler),
        "centralised FRRI": central,
    }
    classes_rows = []
    for name, model in models.items():
        classes_rows += class_rows(model, split.X_test, split.y_test, classes, name)
    return records, classes_rows


def length_table(rules: pd.DataFrame) -> pd.DataFrame:
    return rules.groupby(["dataset", "length"]).agg(
        n_rules=("drop", "size"),
        local=("local_confidence", "mean"),
        glob=("global_confidence", "mean"),
        drop=("drop", "mean"),
        dropped=("drop", lambda d: np.mean(d > TOLERANCE)),
        glob_zero=("global_confidence", lambda c: np.mean(c <= TOLERANCE)),
        limiting=("limiting_other_clients", "mean"),
        full_drop=("full_drop", "mean"),
    ).reset_index()


def dataset_table(rules: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, n_clients), group in rules.groupby(["dataset", "n_clients"]):
        correlation = spearmanr(group["length"], group["drop"]).correlation if group["length"].nunique() > 1 else np.nan
        rows.append({
            "dataset": dataset,
            "n_clients": n_clients,
            "n_rules": len(group),
            "n_features": group["n_features"].iloc[0],
            "mean_length": group["length"].mean(),
            "mean_drop": group["drop"].mean(),
            "mean_full_drop": group["full_drop"].mean(),
            "frac_dropped": np.mean(group["drop"] > TOLERANCE),
            "frac_full_dropped": np.mean(group["full_drop"] > TOLERANCE),
            "frac_glob_zero": np.mean(group["global_confidence"] <= TOLERANCE),
            "frac_full_glob_zero": np.mean(group["full_global_confidence"] <= TOLERANCE),
            "spearman_length_drop": correlation,
            "mean_limiting": group["limiting_other_clients"].mean(),
            "max_limiting_own": group["limiting_own_client"].max(),
            "max_stored_vs_local": (group["stored_confidence"] - group["local_confidence"]).abs().max(),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=["iris", "wine", "breast_cancer", "synthetic"],
                        choices=DATASETS)
    parser.add_argument("--clients", nargs="+", type=int, default=[2, 5])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(PROTOCOL_SEEDS))
    parser.add_argument("--rules-file", help="CSV file for one row per rule")
    parser.add_argument("--classes-file", help="CSV file for the recall of each class")
    args = parser.parse_args()
    warnings.filterwarnings("ignore", message="Gurobi can not be used")

    all_rules, all_classes = [], []
    for dataset in args.datasets:
        X, y = DATASETS[dataset]()
        for n_clients in args.clients:
            for seed in args.seeds:
                records, classes_rows = diagnose(X, y, n_clients, seed)
                context = {"dataset": dataset, "n_features": X.shape[1], "n_clients": n_clients, "seed": seed}
                all_rules += [{**context, **record} for record in records]
                all_classes += [{**context, **row} for row in classes_rows]
                print(f"done: {dataset}, {n_clients} clients, seed {seed}", flush=True)
    rules = pd.DataFrame(all_rules)
    classes = pd.DataFrame(all_classes)
    if args.rules_file:
        rules.to_csv(args.rules_file, index=False)
    if args.classes_file:
        classes.to_csv(args.classes_file, index=False)

    pd.set_option("display.width", 220, "display.max_columns", None, "display.max_rows", None)
    print("\nPer dataset and number of clients, over all seeds")
    print(dataset_table(rules).round(3).to_string(index=False))
    print("\nBy rule length, over all clients and seeds")
    print(length_table(rules).round(3).to_string(index=False))
    print("\nRecall per class on the global test set, mean over clients and seeds")
    recall = classes.groupby(["dataset", "model", "label"]).agg(
        recall=("recall", "mean"), no_rule_fires=("no_rule_fires", "mean")
    ).reset_index()
    print(recall.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
