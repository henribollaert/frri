"""
Compares Gurobi and scipy rule selection on the configurations of the regression test.

    python -m experiments.checks.compare_solvers
"""
import sys
import numpy as np
from tests.test_regression_rulegenerator import CONFIGS, DATASETS
from sklearn.model_selection import train_test_split

for dataset, load in DATASETS.items():
    data = load()
    x_train, x_test, y_train, _ = train_test_split(data.data, data.target, test_size=0.3, random_state=0, stratify=data.target)
    for config, (generator, params) in CONFIGS.items():
        models = {s: generator(**params, solver=s).fit(x_train, y_train) for s in ("gurobi", "scipy")}
        g, c = models["gurobi"], models["scipy"]
        same_rules = [r.antecedent.tolist() for r in g.rules_] == [r.antecedent.tolist() for r in c.rules_]
        agree = np.mean(g.predict(x_test) == c.predict(x_test))
        print(f"{dataset}/{config}: n_rules gurobi={len(g.rules_)} scipy={len(c.rules_)} same_rules={same_rules} prediction_agreement={agree:.3f}")
