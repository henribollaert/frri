"""
Peak traced memory (tracemalloc) of the lower approximation and of a relation of all objects to all objects.

    python -m experiments.checks.memory_phases 1500 10
"""
import sys, tracemalloc, time
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from fuzzyroughrules.approximations import LowerApproximation
from fuzzyroughrules.operators import triangular_relation
n, d = int(sys.argv[1]), int(sys.argv[2])
X, y = make_classification(n_samples=n, n_features=d, n_informative=d // 2, n_classes=3, random_state=0)
Xs = MinMaxScaler().fit_transform(X)
tracemalloc.start()
for name, f in [("lower approximation", lambda: LowerApproximation().get_approximation(Xs, y)),
                ("relation X vs X (operators.triangular_relation)", lambda: triangular_relation(Xs, Xs))]:
    tracemalloc.reset_peak(); start = tracemalloc.get_traced_memory()[0]; t = time.perf_counter()
    result = f()
    print(f"{name}: peak {(tracemalloc.get_traced_memory()[1] - start) / 2**20:.1f} MB, {time.perf_counter() - t:.2f}s, checksum {float(np.sum(result)):.10f}")
    del result
