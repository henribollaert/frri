"""
Compares the Gurobi set cover of the code, built from the non-zero terms, with the old build with all terms, on the set
covers of centralised FRRI and of federated FRRI (5 seeds, 2 and 5 clients, plain and confidence-weighted costs).

    python -m experiments.checks.set_cover_build
"""
import time, sys
import numpy as np
import gurobipy as gb
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split
import fuzzyroughrules.rule_induction_base as base
from fuzzyroughrules.rule_induction_base import RuleGenerator, Rule, rule_coverage
from fuzzyroughrules.federated.client import FederatedClient
from fuzzyroughrules.federated.partition import FixedRangeScaler, global_min_max, global_classes, horizontal_split
from fuzzyroughrules.federated.aggregate import union_local, union_global, global_confidence


def all_terms(coverage, weights, uncovered):
    """The build before the change: every rule enters every constraint, with coefficient 0 or 1."""
    n_rules, n_objects = coverage.shape
    m = gb.Model("discrete_rule_induction")
    rules = [m.addVar(vtype=gb.GRB.BINARY, obj=weights[i]) for i in range(n_rules)]
    m.modelSense = gb.GRB.MINIMIZE
    skip = set(int(i) for i in uncovered)
    columns = coverage.T
    for i in range(n_objects):
        if i in skip:
            continue
        m.addConstr(gb.quicksum([coverage[j][i] * rules[j] for j in range(n_rules)]) >= 1)
    m.setParam("OutputFlag", 0)
    m.optimize()
    return np.array([j for j in range(n_rules) if rules[j].x > 0.99], dtype=int)


problems = []
# coverage matrices of centralised FRRI (as fit builds them) and of federated set covers
for name, (X, y) in {"iris": load_iris(return_X_y=True), "wine": load_wine(return_X_y=True),
                     "breast_cancer": load_breast_cancer(return_X_y=True),
                     "synthetic": make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=0, n_classes=4, random_state=0)}.items():
    for seed in range(5):
        Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
        captured = []
        original = base.select_rules_from_coverage
        base.select_rules_from_coverage = lambda c, weights=None, uncovered=(), solver="auto", time_limit=None: captured.append((c, uncovered)) or original(c, weights, uncovered, solver, time_limit)
        RuleGenerator().fit(Xtr, ytr)
        base.select_rules_from_coverage = original
        problems += [(f"{name} central seed {seed}", c, np.ones(c.shape[0]), u) for c, u in captured]
        for n_clients in (2, 5):
            split = horizontal_split(X, y, n_clients, seed)
            clients = [FederatedClient(c.X_train, c.y_train) for c in split.clients]
            scaler = FixedRangeScaler(*global_min_max([c.min_max() for c in clients]))
            classes = global_classes([c.labels() for c in clients])
            cr = [c.fit_local(scaler, classes) for c in clients]
            rated = union_global(cr, global_confidence([c.local_minima(union_local(cr)) for c in clients]))
            cov = np.hstack([c.coverage_columns(rated, seed + i) for i, c in enumerate(clients)])
            unc = np.flatnonzero(cov.sum(0) == 0)
            problems.append((f"{name} federated C={n_clients} seed {seed}", cov, np.ones(len(rated)), unc))
            w = 2 - np.array([r.confidence for r in rated])
            problems.append((f"{name} federated C={n_clients} seed {seed} weighted", cov, w, unc))

n_same, t_code, t_all = 0, 0., 0.
for label, cov, w, unc in problems:
    t = time.perf_counter(); a = base._select_with_gurobi(cov, w, unc); t_code += time.perf_counter() - t
    t = time.perf_counter(); b = all_terms(cov, w, unc); t_all += time.perf_counter() - t
    same = np.array_equal(a, b)
    n_same += same
    if not same:
        print("DIFFERENT:", label, cov.shape, len(a), len(b), w[a].sum(), w[b].sum())
print(f"{n_same} of {len(problems)} problems give the same selection; build and solve time: code (non-zero terms) {t_code:.1f}s, all terms {t_all:.1f}s")
