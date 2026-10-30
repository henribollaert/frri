"""
Summarises CSV logs of the protocol driver: means over clients and seeds, and paired comparisons of variants.

    python -m experiments.checks.summarise_runs runs_iid.csv runs_noniid.csv
"""
import sys
import numpy as np
import pandas as pd
pd.set_option("display.width", 250, "display.max_rows", None, "display.max_columns", None)
runs = pd.concat([pd.read_csv(f) for f in sys.argv[1:]], ignore_index=True)
runs["kb_total"] = (runs["bytes_up"] + runs["bytes_down"]) / 1000
cols = ["global_test", "global_test_relation_fallback", "pooled_test_local", "no_rule_fires_global_test", "n_rules",
        "avg_rule_length", "n_client_rounds", "kb_total"]
key = ["partition", "dataset", "model"]
print("runs per (partition, dataset):", runs.groupby(["partition", "dataset"]).apply(lambda d: d.groupby(["n_clients", "seed"]).ngroups).to_dict())
print("\nMean over 2-5 clients and seeds 0-4")
print(runs.groupby(key, sort=False)[cols].mean().round(3).to_string())

print("\nglobal_test by number of clients (iid), selected models")
sel = runs[runs.model.isin(["a union local conf", "b union global conf", "d set cover", "e repair, union global conf", "e repair, set cover", "f coverage rounds", "f coverage rounds, conf-weighted", "f repair + coverage rounds", "centralised FRRI"])]
print(sel.pivot_table(index=["partition", "dataset", "model"], columns="n_clients", values="global_test", sort=False).round(3).to_string())

def paired(a, b, metric="global_test"):
    x = runs[runs.model == a].set_index(["partition", "dataset", "n_clients", "seed"])[metric]
    y = runs[runs.model == b].set_index(["partition", "dataset", "n_clients", "seed"])[metric]
    d = (x - y).dropna()
    return d.groupby(level=["partition", "dataset"]).agg(mean_diff="mean", wins=lambda v: int((v > 1e-12).sum()), losses=lambda v: int((v < -1e-12).sum()), n="size")

for a, b, m in [("e repair, union global conf", "b union global conf", "global_test"),
                ("e repair, set cover", "d set cover", "global_test"),
                ("f coverage rounds", "d set cover", "global_test"),
                ("f coverage rounds, conf-weighted", "f coverage rounds", "global_test"),
                ("f repair + coverage rounds", "d set cover", "global_test"),
                ("e repair, set cover", "d set cover", "global_test_relation_fallback"),
                ("f coverage rounds", "d set cover", "global_test_relation_fallback"),
                ("d set cover", "a union local conf", "global_test"),
                ("centralised FRRI", "a union local conf", "global_test")]:
    print(f"\n{a}  minus  {b}  on {m}")
    print(paired(a, b, m).round(3).to_string())
