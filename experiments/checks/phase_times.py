"""
Times the phases of RuleGenerator.fit and the given solvers on synthetic data of a given size.

    python -m experiments.checks.phase_times 1000 gurobi scipy
"""
import sys, time, warnings
import numpy as np
from sklearn.datasets import make_classification
from fuzzyroughrules.rule_induction_base import RuleGenerator, Rule, rule_coverage, select_rules_from_coverage
n = int(sys.argv[1])
X, y = make_classification(n_samples=n, n_features=8, n_informative=5, n_classes=3, random_state=0)
m = RuleGenerator()
t = time.perf_counter()
# reproduce fit up to the coverage, as fit does it
from sklearn.preprocessing import MinMaxScaler
from fuzzyroughrules.approximations import LowerApproximation
from fuzzyroughrules.operators import ImplicatorInclusion
from fuzzyroughrules.feature_preprocessors import QuickReduct
import fuzzyroughrules.fuzzy_operators as fo
m.classes_, yy = np.unique(y, return_inverse=True); m.n_classes_ = len(m.classes_); m.priors_ = np.ones(3)
m.types_ = None; m.n_samples_, m.n_features_in_ = X.shape
m.inclusion_measure_ = ImplicatorInclusion(); m.attribute_ordering_ = QuickReduct(); m.slope_options_ = [1]
Xs = MinMaxScaler().fit_transform(X)
m.positive_region_ = LowerApproximation().get_approximation(Xs, yy)
print(f"positive region {time.perf_counter()-t:.1f}s", flush=True); t = time.perf_counter()
m.rel_matrix_y_ = fo.discernibility_matrix(yy, yy)
m.reducts_, m.slopes_ = m._get_reducts(Xs, yy)
print(f"reducts {time.perf_counter()-t:.1f}s", flush=True); t = time.perf_counter()
cov = rule_coverage([Rule(Xs[i], m.reducts_[i], m.slopes_[i], m.positive_region_[i], yy[i]) for i in range(n)], Xs)
unc = np.flatnonzero(cov.sum(0) == 0)
print(f"coverage {time.perf_counter()-t:.1f}s density {cov.mean():.4f} uncovered {len(unc)}", flush=True)
for solver in sys.argv[2:]:
    t = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel = select_rules_from_coverage(cov, uncovered=unc, solver=solver)
    print(f"{solver}: {len(sel)} rules {time.perf_counter()-t:.1f}s", flush=True)
