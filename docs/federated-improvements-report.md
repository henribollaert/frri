# Federated FRRI improvements: report

Branch `federated-improvements`, starting from `federated` at `9c1a6c0`. Items A to H of the task, in order.

**All results here are provisional.** They come from one cloud machine (4 CPU cores, 15 GB memory) and from datasets
without files (scikit-learn's iris, wine, breast cancer, and `make_classification` data), not from the KEEL datasets of
the project. OpenML could not be reached from this machine (the proxy refused the connection with 403), so yeast was not
run. Every number below was produced in this session by the command next to it. The scripts named
`experiments/checks/...` are committed. Everything should be repeated on the owner's machine (see the end).

## Environment and what was found before any change

- Python 3.10.20 (installed with `uv`), with `pip install -r requirements.txt`, `pip install -e .` and pytest.
- **The licence bundled with gurobipy 10.0.3 (what `gurobipy~=10.0.1` installs) expired on 2024-10-28.** With it,
  every Gurobi call fails. On the unmodified commit `9c1a6c0`, `python -m pytest -q tests` gave 45 failed, 30 passed
  and 32 errors, all `GurobiError: License expired 2024-10-28`. I did not try to get a licence.
- To have a working Gurobi, I made a second environment with the same requirements except gurobipy 12.0.3. Its
  restricted licence is size-limited (2,000 variables and 2,000 constraints) and **expires on 2026-11-23**, so a
  session after that date will not have it either. All experiments below ran in this environment.
- **Regression baseline.** On the unmodified `9c1a6c0` with gurobipy 12.0.3, `tests/test_regression_rulegenerator.py`
  fails for `iris-default` and `iris-granular` and passes the other 8 configurations. In both, one rule of a tied set
  cover is a different one (9 rules in both), and the predictions and probabilities are identical. This is the Gurobi
  version, not a code change. As asked, I recorded a local baseline from the unmodified `9c1a6c0`
  (`python tests/test_regression_rulegenerator.py --record`, then restored the committed file with `git checkout`),
  kept it outside the repository and did not commit it. All regression checks below compare against that local
  baseline. To make this possible without touching the committed file, the test now reads another baseline path from
  `FRRI_REGRESSION_BASELINE` (default unchanged).

Things that contradict the code or documentation, recorded and not changed:

- `fuzzyroughrules/approximate_rule_induction.py` imports `typing.override`, which only exists from Python 3.12, so
  that module can not be imported on Python 3.10.
- `LowerApproximation(implicator=...)` stores the implicator but never uses it.
- The README says the regression baseline should only be re-recorded "on an unmodified `main`", but the regression test
  only exists on the `federated` branch.

## Final test status

Last run, after all changes (`python -m pytest -q tests`, Python 3.10.20):

| environment | baseline | result |
|---|---|---|
| gurobipy 12.0.3 (restricted licence) | local re-recording from unmodified `9c1a6c0` (`FRRI_REGRESSION_BASELINE`) | 162 passed |
| gurobipy 12.0.3 | committed baseline (gurobipy 10.0.1) | regression test: 2 failed (`iris-default`, `iris-granular`), 8 passed: the same 2 that fail on the unmodified commit, see above |
| gurobipy 10.0.3 (requirements, licence expired) | committed baseline | 8 failed, 145 passed, 9 skipped |

In the last environment, all 8 failures are in `test_regression_rulegenerator.py`: "auto" falls back on scipy, which
resolves tied set covers differently from the Gurobi that recorded the baseline. They are exactly the 8 configurations
in which `compare_solvers` (A) found other rules with the same number of rules. The 9 skipped tests are the ones in
`test_rule_selection_solvers.py` that need a working Gurobi. On the unmodified commit this environment gave 45 failures
and 32 errors. No test was skipped or changed to hide a failure.

## A. Solver fallback: done

What: `select_rules_from_coverage(coverage, weights, uncovered, solver="auto", time_limit=None)` in
`fuzzyroughrules/rule_induction_base.py`. `"gurobi"` is the old code path, `"scipy"` uses `scipy.optimize.milp`
(HiGHS), `"auto"` uses Gurobi and falls back on scipy with a warning when gurobipy is not installed or raises a
`GurobiError` (expired licence, size limit). `RuleGenerator` has the fields `solver` and `solver_time_limit`, and the
federated `set_cover` passes both on. The `gurobipy` imports in `rule_induction_base.py` and `fuzzy_operators.py` are
now in `try/except ImportError` (nothing else changed there).

Decisions:
- A time limit was added (commit `5020cc8`) because HiGHS is much slower than Gurobi on FRRI set covers: on 1000
  synthetic objects both found 124 rules, Gurobi in 10.8 s and scipy in 62.7 s (`python -m experiments.checks.phase_times
  1000 gurobi scipy`, at that time with the old Gurobi build). With a limit, both solvers return the best selection so
  far with a warning. Without one, nothing changes.
- With Gurobi working, `"auto"` runs exactly the old Gurobi code; a test checks that scipy is not called then.

Measured:
- Regression and estimator tests with gurobipy 12.0.3 against the local baseline: all pass, after every change to
  `fuzzyroughrules/` outside `federated/` (last run: see "Final test status").
- `python -m experiments.checks.compare_solvers` (both solvers on the 10 regression configurations): the number of
  rules is the same in all 10 (optimal value equal), the rules themselves differ in 8 of 10 (ties), and prediction
  agreement on the test split is between 0.978 and 1.000.
- Consequently, in the environment with the expired gurobipy 10.0.3, where `"auto"` falls back on scipy, 8 of the 10
  regression tests fail: the baseline was recorded with Gurobi, and scipy resolves ties differently. I did not change
  the tests for this.
- `tests/test_rule_selection_solvers.py`: unit tests for both solvers, the fallback and its warning, the time limit,
  and equal optima on random weighted problems. The tests that need Gurobi are skipped when it is not usable.

Failed: a first check of the size-limit fallback, centralised FRRI on 2,600 synthetic objects with `"auto"`, did not
finish within 20 minutes (at that time with the old, slow Gurobi build followed by HiGHS). The scaling experiment (F)
shows the fallback working on larger problems with the time limit.

## B. Diagnosis of the confidence drop: done

What: `experiments/diagnose_confidence.py`. For every local rule over seeds 0 to 4: its length, its confidence on its own
client (local), its global confidence, how many objects of another class on other clients pull it below the local
confidence, and as a paired control the same for the **unshortened rule of the same object** (all attributes,
`INDISCERNIBLE`, slope 1, where FRRI's reduction starts). Also the recall and no-rule-fires per class.

Command: `python -m experiments.diagnose_confidence --datasets iris wine breast_cancer synthetic --clients 2 3 5
--rules-file B_rules.csv --classes-file B_classes.csv` (synthetic: 1000 objects, 10 features, 4 classes).

| dataset | clients | rules | mean length | mean drop | mean drop unshortened | rules that drop | unshortened rules that drop | Spearman(length, drop) | mean limiting objects |
|---|---|---|---|---|---|---|---|---|---|
| breast_cancer | 2 | 78 | 4.00 | 0.057 | 0.020 | 0.872 | 0.538 | 0.237 | 3.8 |
| breast_cancer | 3 | 87 | 3.69 | 0.080 | 0.033 | 0.943 | 0.782 | 0.085 | 6.2 |
| breast_cancer | 5 | 118 | 3.21 | 0.110 | 0.047 | 1.000 | 0.856 | 0.010 | 11.1 |
| iris | 2 | 43 | 1.77 | 0.032 | 0.018 | 0.558 | 0.419 | -0.205 | 0.9 |
| iris | 3 | 58 | 1.71 | 0.048 | 0.019 | 0.672 | 0.414 | -0.133 | 1.9 |
| iris | 5 | 83 | 1.59 | 0.082 | 0.041 | 0.892 | 0.639 | -0.187 | 4.2 |
| synthetic | 2 | 983 | 6.36 | 0.052 | 0.013 | 0.951 | 0.505 | 0.026 | 4.3 |
| synthetic | 3 | 1066 | 6.08 | 0.074 | 0.024 | 0.989 | 0.686 | 0.005 | 8.1 |
| synthetic | 5 | 1157 | 5.74 | 0.099 | 0.037 | 0.999 | 0.810 | -0.016 | 15.5 |
| wine | 2 | 45 | 3.73 | 0.082 | 0.026 | 0.778 | 0.422 | 0.190 | 2.4 |
| wine | 3 | 63 | 3.62 | 0.120 | 0.039 | 0.968 | 0.603 | 0.045 | 5.3 |
| wine | 5 | 93 | 3.45 | 0.164 | 0.068 | 0.978 | 0.785 | -0.022 | 9.2 |

From the same per-rule file (paired, same rules): the drop of the shortened rule is 1.8 to 3.9 times that of the
unshortened rule; the unshortened rule loses strictly less in 21% to 94% of the rules and never more (as it must: more
attributes can only lower the relation). The unshortened rule recovers the full local confidence for only 14% to 59% of
the rules. The relative drop is 13% to 46% of the local confidence. Sanity checks: the local confidence computed with
`evaluate_rules` on the own client equals the stored confidence (largest difference 0.0 at 3 decimals), and no object of
the own client limits a rule.

Conclusion: **the hypothesis partly holds.**
- Supported: shortening is the main cause. The shortened rule loses about 2 to 4 times as much as the same object's
  unshortened rule.
- Not supported in the form "shorter rules lose more": the correlation between length and drop is near zero (-0.21 to
  0.24), also within a dataset.
- A real part of the drop is not caused by shortening: even unshortened rules lose confidence on most rules once more
  data is seen, and the drop grows with the number of clients.

The diagnosis also shows the mechanism behind "no rule fires": global confidences are low (for synthetic, about 0.13 on
average), and a rule only fires where `R > 1 - confidence`. With global confidence (variant b), no rule fires for 25% to
35% of the global test objects of each synthetic class.

Decision: B supports D enough to try it, because most of the drop comes from shortening. But B also predicts that
repair can not bring most rules back to their local confidence, because even the unshortened rules usually stay below
it. D was implemented with that expectation, and the results confirm it.

## C. No-rule-fires fallback: done

What: `fuzzyroughrules/federated/inference.py`: `FallbackRuleGenerator`, a subclass of `RuleGenerator` that only
overrides `predict` (core inference unchanged), and `with_fallback(model, fallback, default_label)`. Fallback
`"relation"` gives an object for which no rule fires the class of the rule with the highest relation `R_r(x_r, u)`,
ignoring confidence; this needs no new information. `"default"` is the baseline: a default rule with the majority class
of the pooled `train_local` data, which needs the class counts of the clients. These are not part of the protocol
(`FederatedClient.label_counts` exists only for this baseline). `"first"` is FRRI's behaviour. The experiment reports
`global_test_relation_fallback`, `global_test_default_rule`, and per-class recall with `--class-log-file`.

Command: `python -m experiments.protocol_driver --datasets iris wine breast_cancer synthetic --clients 2 3 5 --log-file
C_runs.csv --class-log-file C_classes.csv` (60 runs per dataset), summarised with `python -m
experiments.checks.summarise_fallbacks C_runs.csv C_classes.csv`.

Balanced accuracy on the global test set, mean over 60 runs, for the variants where it matters most:

| dataset | model | first (FRRI) | relation | default rule | no rule fires |
|---|---|---|---|---|---|
| breast_cancer | b union global conf | 0.912 | 0.953 | 0.948 | 0.078 |
| breast_cancer | d set cover | 0.912 | 0.948 | 0.946 | 0.083 |
| breast_cancer | centralised FRRI | 0.947 | 0.950 | 0.949 | 0.012 |
| synthetic | b union global conf | 0.486 | 0.514 | 0.494 | 0.308 |
| synthetic | d set cover | 0.473 | 0.510 | 0.487 | 0.345 |
| synthetic | centralised FRRI | 0.530 | 0.539 | 0.536 | 0.070 |
| wine | d set cover | 0.923 | 0.932 | 0.916 | 0.033 |
| wine | centralised FRRI | 0.937 | 0.959 | 0.953 | 0.037 |
| iris | d set cover | 0.947 | 0.949 | 0.947 | 0.020 |

Over all model rows that have at least one global test object for which no rule fires, the relation fallback beats
"first" in 277 of 343 rows and loses in 25; against the default rule it wins in 209 and loses in 56.

Per-class recall shows why: "first" gives the silent objects to class 0, which inflates its recall (synthetic, set cover:
recall of class 0 is 0.712 with "first", 0.500 with "relation"; classes 1 to 3 go from 0.493/0.311/0.376 to
0.599/0.444/0.499). The default rule moves the same objects to the majority class instead. The relation fallback gives a
more even recall over the classes. The lowest class recall per run, averaged over the G runs (below), is higher with
"relation" than with "first" for union with global confidence, set cover and repair on every dataset and partition;
for coverage rounds it is lower on breast cancer (IID 0.914 → 0.907, non-IID 0.924 → 0.914) and higher elsewhere.

The relation fallback does not close the gap: on synthetic data the union of local rules with local confidence (0.543)
stays better than set cover with the relation fallback (0.510).

## D. Rule repair: done (with negative results)

What: `repair_rules` in `fuzzyroughrules/federated/rounds.py` and `FederatedClient.extend_rule`. While a rule's global
confidence is below its local confidence (tolerance 1e-6, the inclusion threshold of FRRI), its owner adds one
attribute of the object it still holds, and every client returns its minimum for the longer rule. A rule stops when it is
back at its local confidence or uses all attributes.

Decisions:
- The added attribute is the unused one that best separates the object from the owner's **local** objects of other
  classes, weighted by their relation to the current rule. It gets `INDISCERNIBLE` and slope 1. The owner knows nothing
  about the other clients' data, so this is a local heuristic. Proposing all attributes at once and letting the server
  choose would reveal the whole object in one round, so I did not do that.
- The payload keeps the local confidence; the global confidence is recomputed.
- A rule that does not recover keeps its last (longest) version. It has a higher confidence but a smaller relation.
- `--repair-max-rounds` limits the rounds; the default is no limit.

Tests: `tests/test_federated_equivalence.py` checks that the repaired rules' global confidence equals the confidence
computed centrally with plain loops, that repair only adds attributes with the owner's values, that it stops when
recovered (one round fewer leaves the rule below), and that unrepaired rules stay unchanged. Breaking the code on
purpose made these tests fail: global confidence with max instead of min (12 failures), never stopping (5), values of
another object (6).

Command (with E and G): `python -m experiments.protocol_driver --datasets iris wine breast_cancer synthetic --log-file
G_iid_runs.csv --class-log-file G_iid_classes.csv`, and the same with `--dirichlet-alpha 0.5` (2 to 5 clients, seeds 0
to 4, 20 runs per dataset and partition), summarised with `python -m experiments.checks.summarise_runs
G_iid_runs.csv G_dir_runs.csv`.

Results (IID, mean over 20 runs):

| dataset | d set cover | e repair, set cover | e repair, union global | b union global | avg rule length d → e | client rounds d → e | kB d → e |
|---|---|---|---|---|---|---|---|
| iris | 0.950 | 0.918 | 0.922 | 0.957 | 1.7 → 2.8 | 4 → 10.9 | 17.9 → 45.2 |
| wine | 0.916 | 0.859 | 0.875 | 0.931 | 3.5 → 10.2 | 4 → 26.5 | 29.9 → 277.0 |
| breast_cancer | 0.912 | 0.873 | 0.877 | 0.912 | 3.6 → 24.0 | 4 → 62.5 | 41.7 → 1714.2 |
| synthetic | 0.466 | 0.429 | 0.443 | 0.479 | 6.1 → 9.4 | 4 → 19.5 | 582.0 → 2156.3 |

Repair is worse than the same selection without repair in most paired runs, for both partitions: "e repair, set cover"
minus "d set cover" on the global test set is -0.022 to -0.061 per dataset (losses in 9 to 18 of 20 runs, wins in 1 to
5). It also has more objects for which no rule fires (e.g. breast cancer IID: 0.084 → 0.161), because longer rules fire
on a smaller region. With the relation fallback the loss is smaller (-0.044 to +0.005) but remains on most datasets.

**Cost, as the task asked to note:** every repair round reveals one more attribute value of each rule still being
repaired, and adds a confidence round. In the run `python -m experiments.horizontal_fl_frri --loader builtin --data-dir
wine --clients 3 --seed 0`, 12 of 13 rules needed repair, 3 recovered, 9 ended with all 13 attributes (complete training
objects), there were 11 repair rounds and the revealed attribute values went from 48 to 138. Over the G runs, the
average rule length on breast cancer went to 24 of 30 attributes.

Conclusion: rule repair as designed is not worth it on these datasets. It confirms B: most rules can not recover,
because the drop is partly caused by more data and not by shortening.

## E. Coverage rounds: done

What: `coverage_rounds` in `fuzzyroughrules/federated/rounds.py` and `FederatedClient.uncovered_rules`. After a set
cover, every client computes which of its objects the selected rules do not cover and sends the FRRI rules that `fit`
built for these objects (before its own rule selection), if it did not send them before. They get their global
confidence, the coverage is computed again and the set cover is solved again. At most 3 rounds (`--coverage-max-rounds`).
`confidence_weighted_set_cover` gives a rule the cost `2 - confidence` (between 1 and 2), so the number of rules still
dominates and confidence breaks ties. The experiment runs three variants: from the local rules with the plain and the
confidence-weighted objective, and after repair.

Decisions: a client remembers which rules it sent (its local rules and the current candidates) and never sends one again;
the columns are shuffled again in every round with a new seed; the experiment logs `n_extra_rules_exposed`.

Tests: the new candidates equal the FRRI rules of the objects that are uncovered by a central, plain-loop computation;
a rule is not sent twice; after the rounds, the candidates' confidence equals the central confidence, and the selection
covers exactly the objects that any candidate covers centrally. Breaking the code made them fail: all objects treated as
uncovered (6 failures), confidences merged in the wrong order (3), rules sent twice (4).

Results (same G runs; global test, mean over 20 runs):

| partition | dataset | d set cover | f coverage rounds | f conf-weighted | no rule fires d → f | rules d → f | client rounds d → f | kB d → f |
|---|---|---|---|---|---|---|---|---|
| iid | iris | 0.950 | 0.955 | 0.953 | 0.020 → 0.017 | 5.0 → 6.1 | 4 → 7.1 | 17.9 → 31.6 |
| iid | wine | 0.916 | 0.918 | 0.921 | 0.043 → 0.021 | 8.1 → 10.4 | 4 → 8.0 | 29.9 → 67.9 |
| iid | breast_cancer | 0.912 | 0.932 | 0.930 | 0.084 → 0.034 | 14.5 → 26.1 | 4 → 8.0 | 41.7 → 170.5 |
| iid | synthetic | 0.466 | 0.505 | 0.505 | 0.370 → 0.202 | 172 → 300 | 4 → 8.0 | 582 → 1952 |
| dirichlet 0.5 | iris | 0.885 | 0.917 | 0.917 | 0.088 → 0.055 | 5.4 → 7.2 | 4 → 7.6 | 15.6 → 36.6 |
| dirichlet 0.5 | wine | 0.920 | 0.941 | 0.943 | 0.072 → 0.031 | 8.0 → 10.5 | 4 → 8.0 | 23.7 → 63.7 |
| dirichlet 0.5 | breast_cancer | 0.909 | 0.943 | 0.939 | 0.106 → 0.038 | 11.6 → 23.6 | 4 → 8.0 | 32.6 → 152.2 |
| dirichlet 0.5 | synthetic | 0.454 | 0.501 | 0.504 | 0.418 → 0.184 | 129 → 288 | 4 → 8.0 | 400 → 1764 |

Paired over 20 runs, coverage rounds beat plain set cover by +0.002 to +0.047 per dataset and partition (wins 3 to 19,
losses 0 to 6). With the relation fallback applied to both, the gain is smaller (-0.006 to +0.024). The
confidence-weighted objective makes no clear difference (paired mean difference -0.004 to +0.003). The cost is more rules
(up to about 2 times as many), twice the client rounds, 1.8 to 4.7 times the bytes (the coverage matrix is sent again in
every round), and every extra candidate is another training object that is revealed.

## F. Scalability evidence: done

### Measurements in the experiment

`experiments/measure.py`: a `RoundRecorder` runs every round of the experiment, times every client, and counts the bytes
sent up (by the clients) and down (by the server) with a stated encoding (compact JSON for rules, 1 bit per coverage
entry, 8 bytes per number). With `--trace-memory` it also records the peak memory per round with `tracemalloc`
(standard library), which slows Python down, so durations measured with it are not comparable. Per model, the CSV log
gets `n_client_rounds`, `bytes_up`, `bytes_down` and `peak_extra_mb`; the round table is printed. The rounds of the extra
variants (repair, coverage rounds) are recorded the same way. `FederatedClient.local_minima` now calls the new
`rule_minima` and no longer computes a coverage matrix that was thrown away (`evaluate_rules` returns the same as before).

### Real multiprocessing

`python -m experiments.parallel_clients --dataset synthetic --clients 2 3 4 5 --seeds 0 1 2` keeps every client in its
own process for the whole run and compares the real parallel round time with the time derived from the sequential
simulation (the slowest client). Mean over 3 seeds, machine with 4 cores:

| clients | round | derived (s) | in worker (s) | wall, incl. messages (s) | wall / derived |
|---|---|---|---|---|---|
| 2 | local fit | 1.459 | 1.644 | 1.646 | 1.13 |
| 3 | local fit | 0.936 | 0.889 | 0.904 | 0.97 |
| 4 | local fit | 0.647 | 0.681 | 0.684 | 1.06 |
| 5 | local fit | 0.499 | 0.617 | 0.623 | 1.25 |
| 2 | confidence | 0.050 | 0.042 | 0.044 | 0.88 |
| 5 | confidence | 0.050 | 0.068 | 0.075 | 1.50 |
| 5 | coverage | 0.035 | 0.063 | 0.070 | 2.00 |

For local fit, the round that dominates, the derived parallel time is within 13% of the real one with 2 to 4 clients
(one core each). The short rounds (about 0.05 s) take up to 1.3 times the derived time with 4 clients. With 5 clients
on 4 cores the real time is 1.25 (local fit) to 2.0 (coverage) times the derived time, because clients share cores and
the short rounds are dominated by sending the messages.

### Relations in blocks of rows

`triangular_dominance` (`fuzzyroughrules/operators.py`) and `LowerApproximation` compute the relation in blocks of rows so
the intermediate (rows, rows, attributes) array stays below 2^22 elements. `tests/test_row_blocks.py` checks with tiny
blocks that the result is bitwise the same; the regression test is unchanged. With `python -m
experiments.checks.memory_phases 1500 10` (1500 synthetic objects, 10 features), the traced peak of the lower
approximation went from 703.8 MB to 130.9 MB and that of the all-objects relation from 686.9 MB to 140.8 MB, with
identical checksums. (The durations of that run are not reported because other runs were active.)

### Set cover from the non-zero terms

The Gurobi set cover now puts only the rules that cover an object into its constraint. I first checked it on 100 FRRI
set covers (centralised and federated, 5 seeds, 2 and 5 clients, plain and weighted costs, iris, wine, breast cancer and
synthetic): the selection was identical in all 100 (`python -m experiments.checks.set_cover_build`). Since it did not
change any selected rule, I changed the core path instead of adding a federated-only path; the regression test is
unchanged. Build and solve time over the 100 problems: 4.1 s with the new build against 56.3 s with the old one. On
1000 synthetic objects (centralised) the Gurobi selection now takes 0.7 s instead of 10.8 s (`python -m
experiments.checks.phase_times 1000 gurobi`). Risk: this was checked with gurobipy 12.0.3 only; the committed
regression baseline with gurobipy 10.0.1 has to confirm it (see the end). The commit (`59537b0`) is separate and can be
reverted on its own.

### Scaling experiment

`python -m experiments.scaling --sizes 1000 2000 4000 8000 16000 --clients 3 --time-limit 1200 --memory-limit-gb 10
--set-cover-limit 300` (synthetic data, 10 features, 4 classes, 3 clients; every size in a fresh process; 20 minutes and
10 GB per size and model; 300 s per set cover). The set covers use "auto", so above 2,000 rules or constraints the
restricted Gurobi licence fails and scipy takes over.

| objects | model | pooled `train_local` | rules | parallel (s) | sequential (s) | local fit, slowest client (s) | global set cover (s) | peak RSS (MB) | kB up / down | scipy fallback | time limit hit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1000 | federated | 642 | 180 | 1.1 | 3.1 | 1.0 | 0.02 | 150 | 92 / 414 | no | no |
| 2000 | federated | 1278 | 301 | 2.7 | 7.8 | 2.5 | 0.03 | 200 | 198 / 773 | no | no |
| 4000 | federated | 2559 | 482 | 7.8 | 22.0 | 7.3 | 0.10 | 297 | 415 / 1283 | no | no |
| 8000 | federated | 5119 | 873 | 23.0 | 66.9 | 21.3 | 0.23 | 388 | 1127 / 2479 | yes | no |
| 16000 | federated | 10237 | 1938 | 384.2 | 1143.1 | 379.1 | 0.49 | 1185 | 3825 / 5379 | yes | yes |
| 1000 | centralised | 642 | 174 | 3.8 | 3.8 | | | 255 | | no | no |
| 2000 | centralised | 1278 | 299 | 11.7 | 11.7 | | | 257 | | no | no |
| 4000 | centralised | 2559 | 503 | 341.1 | 341.1 | | | 725 | | yes | yes |
| 8000 | centralised | 5119 | 899 | 470.1 | 470.1 | | | 1331 | | yes | yes |
| 16000 | centralised | 10237 | 2037 | 991.6 | 991.6 | | | 2641 | | yes | yes |

(The federated set cover at 4000 objects stays under the Gurobi limit because the objects that no candidate covers are
left out: 2559 - 869 = 1690 constraints.)

Observations, all under the restricted licence:
- Up to 8000 objects the federated run is dominated by the local fits; confidence, coverage and the global set cover
  take below 1 s each. The parallel time is about a third of the sequential time with 3 clients.
- Centralised FRRI jumps from 11.7 s to 341 s at 4000 objects, because its set cover (2559 x 2559) falls back on scipy
  and runs into the 300 s limit. From there on its times mostly measure that limit, and its rules are not proven
  optimal.
- At 16000 objects every client has about 3400 objects, so the **local** set cover of each client also falls back on
  scipy and hits the 300 s limit; that is most of the 379 s of local fit. Peak memory stays far below the 10 GB limit
  (1.2 GB federated, 2.6 GB centralised, in one process).
- No size failed in the end, so the limit of this machine was not reached at 16000 objects.

What failed: in the first run, the federated run at 16000 objects hit the 20-minute limit, while centralised FRRI
finished. The cause was a bug in my scaling script: the time limit was only passed to the global set cover, not to the
set covers inside the clients' local FRRI, so those ran in scipy without a limit. I fixed that (also in
`experiments/horizontal_fl_frri.py`, where `--time-limit` now also reaches the local fits) and reran 8000 and 16000
objects (`--models federated --sizes 8000 16000`, same options); the table shows the rerun. The 8000 result was the same
as in the first run (873 rules).

## G. Protocol driver and evaluation: done

What: `experiments/protocol_driver.py` runs every combination of datasets (built-in names or KEEL folders), numbers of
clients (default 2 to 5) and `PROTOCOL_SEEDS`, appending every run to the CSV logs with the run options. Models are
compared on the same objects and classes with the new `pooled_test_local` (balanced accuracy on the `test_local` data of
all clients together), since a rare class can be missing from a single client's `test_local` data. (On a single test set
all models already share the classes, because balanced accuracy only averages over the classes in `y_true`; I checked
that with scikit-learn 1.2.2.) Non-IID: `horizontal_split(..., dirichlet_alpha=...)` and `--dirichlet-alpha` divide the
objects of each class over the clients with Dirichlet proportions (label skew), redrawing until each client has at
least 5 objects; global test and `train_local`/`test_local` splits stay stratified. A client can then have no
`test_local` data; its score is NaN and the mean skips it. An older CSV log with other columns is rewritten with the
union of the columns instead of getting shifted columns.

Command: see D. 160 runs, 13 models each. Global test balanced accuracy, mean over 20 runs per cell:

| partition | dataset | a union local | b union global | d set cover | f coverage rounds | centralised FRRI |
|---|---|---|---|---|---|---|
| iid | iris | 0.958 | 0.957 | 0.950 | 0.955 | 0.962 |
| iid | wine | 0.943 | 0.931 | 0.916 | 0.918 | 0.942 |
| iid | breast_cancer | 0.946 | 0.912 | 0.912 | 0.932 | 0.946 |
| iid | synthetic | 0.544 | 0.479 | 0.466 | 0.505 | 0.528 |
| dirichlet 0.5 | iris | 0.793 | 0.893 | 0.885 | 0.917 | 0.957 |
| dirichlet 0.5 | wine | 0.800 | 0.928 | 0.920 | 0.941 | 0.944 |
| dirichlet 0.5 | breast_cancer | 0.863 | 0.911 | 0.909 | 0.943 | 0.946 |
| dirichlet 0.5 | synthetic | 0.462 | 0.456 | 0.454 | 0.501 | 0.525 |

Observations:
- IID: the union of local rules with local confidence (a) is as good as centralised FRRI and better than every variant
  with global confidence; on synthetic it is even better than centralised (paired: centralised minus a = -0.016, 5 wins,
  15 losses).
- Non-IID: (a) falls far behind (paired, 0.14 to 0.16 below centralised on iris and wine), and the global confidence
  pays off: set cover beats (a) on average on iris, wine and breast cancer. Coverage rounds (with either objective) are the best federated
  variant there, still 0.003 to 0.04 below centralised.
- The lowest class recall per run (mean) shows the same: e.g. iris non-IID 0.545 for (a), 0.735 for set cover, 0.810 for
  coverage rounds, 0.900 for centralised; with the relation fallback 0.825 and 0.850 for the last two federated ones.

The training times in the G logs are not clean measurements: the IID and non-IID runs ran at the same time on the 4
cores, and the IID run started before the row blocks and the non-zero set cover build were committed (commit
`12ae3b7`; the non-IID run started at `79aa937`). Both changes give identical results (tests above), so the accuracies are
not affected.

## H. Privacy write-up: done

`docs/federated-privacy.md`: message by message what is revealed, the cost of coverage patterns and of
`rank_by="coverage"`, the new rounds, and options (rounding, intervals, public range, no local confidence, secure
aggregation of minima, counts instead of patterns). Differential privacy is out of scope. Two findings there that I
checked numerically on wine, breast cancer and synthetic data with 2 and 5 clients:
- the local confidence that a rule carries equals `min(1, D)`, with `D` the L-infinity distance from the rule's object
  to the nearest object of another class on the same client (largest difference 5.6e-17), so it reveals a distance to a
  second object;
- with global confidence, a rule never covers an object of another class, so every covered column of the coverage matrix
  reveals the class of its object (0 exceptions over all covered objects checked).

## What to repeat on the owner's machine (yeast, full Gurobi licence)

1. `python -m pytest -q tests` with gurobipy 10.0.1 and the full licence, against the **committed** baseline. This is
   the real check that A ("auto" with working Gurobi) and the non-zero set cover build (F, commit `59537b0`) leave the
   non-federated results unchanged. If only the regression test fails, compare with `python -m
   experiments.checks.set_cover_build`; if the build is the cause, revert `59537b0` or move it to a federated-only path.
2. `python -m experiments.checks.compare_solvers` to see how often scipy and Gurobi differ there.
3. `python -m experiments.diagnose_confidence` with yeast and the other KEEL datasets (add a KEEL loader to
   `experiments/datasets.py`, or put the data in a loader), to see whether B's conclusion holds on the data of the paper.
4. `python -m experiments.protocol_driver --keel-dirs <keel-data>/yeast-10-fold ... --log-file runs.csv --class-log-file
   classes.csv`, with and without `--dirichlet-alpha 0.5`, then `python -m experiments.checks.summarise_runs`. Yeast has
   rare classes, so `pooled_test_local` and the per-class recall matter there.
5. `python -m experiments.scaling` with the full licence (no fallback to scipy, no 2,000 limit) and larger sizes, and
   `python -m experiments.parallel_clients` on a machine with more cores.
6. Decide whether to keep rule repair at all; on this data it costs a lot of rounds and reveals whole objects without
   gaining accuracy.
