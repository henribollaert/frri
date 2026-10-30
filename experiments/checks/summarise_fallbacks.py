"""
Summarises the fallbacks for objects for which no rule fires from a run log and a class log of the protocol driver.

    python -m experiments.checks.summarise_fallbacks runs.csv classes.csv
"""
import sys
import pandas as pd
runs, classes = pd.read_csv(sys.argv[1]), pd.read_csv(sys.argv[2])
pd.set_option("display.width", 250, "display.max_rows", None)
cols = ["global_test", "global_test_relation_fallback", "global_test_default_rule", "no_rule_fires_global_test"]
t = runs.groupby(["dataset", "model"], sort=False)[cols].mean()
t["relation_minus_first"] = t["global_test_relation_fallback"] - t["global_test"]
t["default_minus_first"] = t["global_test_default_rule"] - t["global_test"]
print(f"mean over {runs.groupby(['dataset','n_clients','seed']).ngroups} runs (clients x seeds per dataset)")
print(t.round(3).to_string())
# paired: in how many runs does relation beat / lose against first and default
r = runs.assign(rel_vs_first=lambda d: (d.global_test_relation_fallback - d.global_test),
                rel_vs_default=lambda d: (d.global_test_relation_fallback - d.global_test_default_rule))
w = r[r.no_rule_fires_global_test > 0].groupby("dataset").agg(
    n=("rel_vs_first", "size"),
    rel_better_first=("rel_vs_first", lambda x: (x > 1e-12).sum()), rel_worse_first=("rel_vs_first", lambda x: (x < -1e-12).sum()),
    rel_better_default=("rel_vs_default", lambda x: (x > 1e-12).sum()), rel_worse_default=("rel_vs_default", lambda x: (x < -1e-12).sum()))
print("\nmodel rows with at least one silent global test object")
print(w.to_string())
print("\nrecall per class, global test, mean over runs; selected models")
sel = classes[classes.model.isin(["b union global conf", "d set cover", "centralised FRRI"])]
p = sel.pivot_table(index=["dataset", "model", "label"], columns="fallback", values="recall", sort=False)
print(p.round(3).to_string())
