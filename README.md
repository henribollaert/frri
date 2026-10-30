# Fuzzy-Rough Rule Induction

Rule induction for classification with fuzzy-rough sets, with a scikit-learn style API (`fit`, `predict`,
`predict_proba`).

## 1. What this is

The repository contains these rule induction algorithms:

- **FRRI**: fuzzy-rough rule induction. Each rule is built from a training object, shortened by a feature reduction
  step, and the final rules are chosen by solving a set cover problem. Implemented by `RuleGenerator` in
  `fuzzyroughrules/rule_induction_base.py`, with variants in the other modules of `fuzzyroughrules/`.
- **QuickRules** and **OWA-QuickRules**, in the package `quickrules/`, also with pruning.
- **Federated FRRI** (on the `federated` branch, in `fuzzyroughrules/federated/`): FRRI for the horizontal setting,
  where several clients hold different objects with the same features and only exchange rules and a few numbers,
  never their training objects. See [section 5](#5-federated-frri).

Reference for the rule induction approach: Richard Jensen, Chris Cornelis and Qiang Shen,
[Hybrid fuzzy-rough rule induction and feature selection](https://cwi.ugent.be/Chris/fuzzieee2009b.pdf),
Fuzz-IEEE 2009, pp. 1151-1156.

The Jupyter notebooks in the repository root are research notebooks. They read datasets from folders such as
`keel-data/`, which are not part of the repository.

## 2. Install

You need Python 3.10 or later (the code uses `X | Y` type annotations).

```bash
git clone https://github.com/henribollaert/frri.git
cd frri
git checkout federated

python3.10 -m venv .venv            # or any Python >= 3.10
source .venv/bin/activate           # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` installs the packages `fuzzyroughrules`, `fuzzyroughrules.federated`, `quickrules` and `hhelper`.
The dependencies are in `requirements.txt` and are not installed by `setup.py`, so run both `pip` commands.

**Gurobi.** The rule selection step of FRRI is an optimisation problem that is solved with Gurobi, or with
`scipy.optimize.milp`. `gurobipy` is an optional import: `rule_induction_base.py` still imports without it. The version
installed by `pip` comes with a limited licence of 2,000 variables and 2,000 constraints. The selection has one
variable for each candidate rule and one constraint for each training object, so larger training sets need a full
licence. Gurobi offers a free academic licence: see https://support.gurobi.com/hc/en-us/articles/360040541251.
The licence that is bundled with `gurobipy` 10.0.x from `pip` has also expired (on 2024-10-28). With the pinned
version in `requirements.txt` and no full licence of your own, the scipy fallback below is therefore what runs.

The solver is chosen with the `solver` argument of `RuleGenerator` (and of `select_rules_from_coverage` and the
federated `set_cover`):

| `solver` | Behaviour |
|---|---|
| `"auto"` (default) | Uses Gurobi. If `gurobipy` is not installed or raises an error (for example an expired or size-limited licence), it falls back on `scipy.optimize.milp` and gives a warning. |
| `"gurobi"` | Gurobi only. Raises an error if it can not be used. |
| `"scipy"` | Always `scipy.optimize.milp`. |

Both solvers find an optimal selection, but a tied optimum can be resolved differently, so the selected rules, and
therefore the results, can differ slightly between Gurobi and scipy. `solver_time_limit` of `RuleGenerator` (`time_limit`
of `select_rules_from_coverage` and `set_cover`) is a number of seconds. When it is reached, the best solution found so
far is returned, with a warning. The default is no limit.

**Mosek.** `fuzzyroughrules/fuzzy_operators.py` imports `mosek`, so the package is imported with the rest of the code
and is installed by `requirements.txt`. Mosek is commercial software with its own licensing.

**fr-learn.** `fuzzyroughrules/filter_frri.py` and `fuzzyroughrules/frfs_frri.py` import `frlearn`, which is not in
`requirements.txt`. Install `fr-learn` yourself if you use those two modules. Nothing else in this README needs it.

## 3. Running the tests

From the root of the repository, with the environment active:

```bash
pip install pytest
python -m pytest -q tests
```

Use `python -m pytest` rather than plain `pytest`: it puts the repository root on the path, which
`tests/test_horizontal_fl_experiment.py` needs to import `experiments`. The tests cover the scikit-learn estimator
checks, the federated code (`test_federated_units.py`, `test_federated_equivalence.py`,
`test_horizontal_fl_experiment.py`), the rule selection solvers (`test_rule_selection_solvers.py`), the computation of
relations in blocks of rows (`test_row_blocks.py`) and a regression test of `RuleGenerator`. Some tests in
`tests/test_rule_selection_solvers.py` compare scipy with Gurobi. They are skipped when Gurobi is not usable, i.e. when
`gurobipy` is not installed or has no working licence.

**Regression baseline.** `tests/test_regression_rulegenerator.py` compares the rules and predictions of the
non-federated generators on iris and wine with `tests/data/rulegenerator_baseline.json`. The baseline depends on the
Gurobi version, because tied set covers can be resolved differently. If that test fails after a change of Gurobi
version, record the baseline again, but only on an unmodified `main` so that it keeps describing the non-federated code:

```bash
git checkout main
python tests/test_regression_rulegenerator.py --record
```

To compare against a baseline that was recorded elsewhere, for example on another machine or with another Gurobi
version, set the environment variable `FRRI_REGRESSION_BASELINE` to the path of that file:

```bash
FRRI_REGRESSION_BASELINE=/path/to/rulegenerator_baseline.json python -m pytest -q tests/test_regression_rulegenerator.py
```

## 4. Basic use of `RuleGenerator`

```python
from sklearn.datasets import load_wine
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

from fuzzyroughrules.rule_induction_base import RuleGenerator

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

model = RuleGenerator()
model.fit(X_train, y_train)

predictions = model.predict(X_test)        # class labels
memberships = model.predict_proba(X_test)  # one row per object, one column per class
print("balanced accuracy:", balanced_accuracy_score(y_test, predictions))
print("number of rules:", len(model.rules_))
print("average rule length:", model.average_rule_length())
```

Features must be numeric and without missing values. By default `fit` scales the features to [0, 1] with a
`MinMaxScaler`. The rules are in `model.rules_`, and `model.get_rules_as_string()` gives them as text. The
constructor arguments are documented in the docstring of `RuleGenerator`, for example `discard_uncertain_objects`,
`optimise_slopes`, `approximation`, `inclusion_measure` and `scaler`.

## 5. Federated FRRI

### Setting

Horizontal federated learning: every client has the same features but different objects. A server coordinates, and
combines the rules of the clients into one global model. Federated FRRI is evaluated here by simulation in one process.
There is no networking: the clients are ordinary `FederatedClient` objects that are run one after the other.

The data protocol (`horizontal_split` in `fuzzyroughrules/federated/partition.py`):

1. The objects are split 80/20 into a global training set and a global test set.
2. The training objects are dealt out over the clients, disjointly and as equally as possible.
3. Each client splits its objects 80/20 into `train_local` and `test_local`.

The splits are stratified by class by default, and depend only on the seed.

Non-IID option: `horizontal_split(..., dirichlet_alpha=alpha)` does not deal the training objects out equally. The
objects of each class are divided over the clients in proportions drawn from a symmetric Dirichlet distribution with
parameter `alpha`, so a small `alpha` gives clients with few classes. Draws are repeated until every client has at
least 5 objects. The global test split and the `train_local`/`test_local` splits are as before. A client can end up
with an empty `test_local` set, and its local scores are then NaN. This is not the protocol above.

### Steps

1. **Shared range and classes.** Each client sends the per-feature minimum and maximum, and the set of labels, of its
   `train_local` data. The server combines them into one global min-max range (`global_min_max`) and one sorted list of
   classes (`global_classes`). The range uses `train_local` data only, never test data. All clients scale with the
   same fixed range (`FixedRangeScaler`, which does not refit), so rules from different clients are comparable.
2. **Local FRRI.** Each client runs FRRI on its own `train_local` data (`FederatedClient.fit_local`) and sends its
   rules to the server, with the confidence that its local model computed.
3. **Global confidence.** The server broadcasts the union of all local rules. Each client computes, for every rule, its
   lower approximation minimum over its own data, and sends this one number per rule back. The global confidence of a
   rule is the minimum over the clients (`global_confidence`). The tests check that this equals what a central
   calculation on the pooled `train_local` data gives.
4. **Coverage.** The server broadcasts the rules with their global confidence. Each client computes which of its objects
   each rule covers and returns a binary matrix of shape (rules x local objects), with the objects in random order
   (`FederatedClient.coverage_columns`). The server stacks these side by side. The tests check that the result equals
   the central coverage up to the order of the columns.
5. **Selection.** The server picks the rules of the global model with one of the variants below.
6. **Inference.** `build_global_model` turns the selected rules into a fitted `RuleGenerator` with the shared scaler and
   classes. Prediction is then ordinary FRRI inference: `predict`, `predict_proba`.

### Extra rounds: rule repair and coverage rounds

Steps 1 to 6 are the base protocol. Two optional rounds, in `fuzzyroughrules/federated/rounds.py`, come on top. Both
cost more communication and reveal more than the base protocol, see [What leaves a client](#what-leaves-a-client) and
[docs/federated-privacy.md](docs/federated-privacy.md).

**Rule repair** (`repair_rules`). A rule is shortened on the data of its own client, so other clients can hold objects
of another class that still match it, and its global confidence can be lower than its local confidence. While the
global confidence of a rule is below its local confidence, the client that owns the rule adds one attribute
(`FederatedClient.extend_rule`) and sends the longer rule. Every client then returns its lower approximation minimum
for the extended rules, and the server computes the new global confidence. A rule stops when it is back at its local
confidence (up to a tolerance of 1e-6) or when it uses all attributes. The new attribute is the one that best separates
the object of the rule from the local objects of other classes, weighted by their relation to the rule. It gets the
value of that object, the relation type `INDISCERNIBLE` and slope 1. `max_rounds` limits the number of rounds. Every
round reveals one more attribute value of the object of each rule that is still being repaired.

**Coverage rounds** (`coverage_rounds`). After a set cover, every client sends the FRRI rules of its own objects that
the selected rules do not cover (`FederatedClient.uncovered_rules`). These are rules that `fit_local` built for every
object before its own rule selection, and that were not sent before. The new candidates get their global confidence
(the rule minima, as in step 3), the coverage is computed again and the set cover is solved again. This repeats for at
most `max_rounds` rounds (default 3), and stops earlier when no client has a new rule. The objective of `set_cover` can be
replaced: `confidence_weighted_set_cover` gives a rule the cost 2 minus its confidence, so that confident
rules are preferred. Coverage rounds can start from the local rules or from the repaired rules.

### Objects for which no rule fires

FRRI gives an object for which no rule fires membership 0 for every class, and `argmax` then picks the first class.
`FallbackRuleGenerator` in `fuzzyroughrules/federated/inference.py` is a subclass of `RuleGenerator` that handles these
objects differently. `with_fallback(model, fallback, default_label)` copies a fitted model into one. The core inference
of `RuleGenerator` is not changed.

| `fallback` | What an object for which no rule fires gets |
|---|---|
| `"first"` | The first class, as `RuleGenerator` does. |
| `"relation"` (default) | The class of the rule with the highest relation to the object, ignoring the confidence. This needs nothing but the rules. |
| `"default"` | `default_label`, for example the majority class. This is a baseline: it needs the class counts of the clients (`FederatedClient.label_counts`), which the protocol does not send. |

### Selection variants

| Variant | Function | What it does |
|---|---|---|
| (a) | `union_local` | All local rules, with their local confidence. |
| (b) | `union_global` | All local rules, with their global confidence. Rules with global confidence 0 are dropped, because they cannot influence a prediction. |
| (c) | `top_k_per_client` | The best fraction `k` of the rules of each client (rounded up), with their global confidence. Rules with confidence 0 are dropped. |
| (d) | `set_cover` | Solves the set cover problem on the pooled coverage matrix, with Gurobi or scipy (see `solver` in [section 2](#2-install)). |
| (e) | `repair_rules` | Rule repair, then either the union of the repaired rules with their global confidence (`e repair, union global conf`) or a set cover of them (`e repair, set cover`). |
| (f) | `coverage_rounds` | Set cover with coverage rounds, from the local rules (`f coverage rounds`, and `f coverage rounds, conf-weighted` with `confidence_weighted_set_cover`) or from the repaired rules (`f repair + coverage rounds`). |

Details:

- `top_k_per_client(..., rank_by=...)`: with `rank_by="confidence"` (the default) the rules of a client are ranked by
  global confidence. With `rank_by="coverage"` they are ranked by the number of local objects that the rule covers, and
  you must pass `coverage_counts` (one array per client, from `FederatedClient.coverage_counts`). The experiment script
  runs `k` = 0.5 and 0.7 with both rankings.
- `set_cover(rules, coverage, objective=plain_set_cover)`: the cost of selecting each rule is given by `objective`; the
  default gives every rule cost 1, i.e. the fewest rules. Objects that no candidate rule covers are left out of the
  problem, because FRRI would add the rule of that object, which is not available here. Their number is returned.
  `set_cover(..., solver="auto", time_limit=None)` passes the solver choice and the time limit on to
  `select_rules_from_coverage`. Coverage rounds exist to cover such objects.

### What leaves a client

| Step | Client sends | Client receives |
|---|---|---|
| 1 | per-feature minimum and maximum of `train_local`, the set of labels | nothing |
| 2 | its rules (`RulePayload`): the indices of the used attributes, the rule's values on them, the relation type and slope of each, the class label, the local confidence | the global range and classes |
| 3 | one number per candidate rule | all local rules (the union) |
| 4 | a binary coverage matrix (rules x local objects), objects in random order | the rules, with global confidence |
| 5, only for variant (c) ranked by coverage | for each of its own rules, the number of local objects that it covers | nothing |
| Rule repair (variant (e), (f) after repair), repeated until the rules recover or use all attributes | the owner of a rule: the rule with one more attribute (the attribute value of the object of the rule). Every client: one number per extended rule | the extended rules (from every owner, to every client) |
| Coverage rounds (variant (f)), repeated | every client: the FRRI rules of its objects that the selected rules do not cover (new candidate rules). Then, as in steps 3 and 4: one number per new rule, and the coverage matrix of all rules | the selected rules, then the new candidate rules, then all candidates with their global confidence |

Training objects are never sent. The values in a rule are the values of the training object that the rule was built
from, restricted to the attributes that the rule uses, so this is not a formal privacy guarantee. Rule repair and
coverage rounds reveal more than the base protocol: every repair round reveals one more attribute value of the object
of each rule that is still being repaired, and a rule that ends with all attributes is the complete training object.
Every new candidate of a coverage round is another training object (the attributes it uses and its label) that is
revealed. [docs/federated-privacy.md](docs/federated-privacy.md) lists, message by message, what each one reveals,
and which options could reveal less (none of them are implemented).

The default-rule baseline (`fallback="default"`, the column suffix `_default_rule`) needs the class counts of the
clients. These are not part of the protocol, and the experiment script reads them directly from the clients only for
this baseline.

### Module overview

| File | Contents |
|---|---|
| `fuzzyroughrules/federated/partition.py` | `load_keel_dataset`, `horizontal_split` (with the non-IID option `dirichlet_alpha`), `global_min_max`, `global_classes`, `FixedRangeScaler` |
| `fuzzyroughrules/federated/payload.py` | `RulePayload` (what is sent for a rule, with `to_dict` and `from_dict`), `rule_to_payload`, `payload_to_rule` |
| `fuzzyroughrules/federated/client.py` | `FederatedClient`: `min_max`, `labels`, `label_counts` (not part of the protocol), `fit_local`, `local_minima`, `coverage_columns`, `coverage_counts`, `extend_rule` (repair), `uncovered_rules` and `start_coverage_rounds` (coverage rounds) |
| `fuzzyroughrules/federated/aggregate.py` | `global_confidence`, `union_local`, `union_global`, `top_k_per_client`, `set_cover` (with `solver` and `time_limit`), `plain_set_cover`, `build_global_model` |
| `fuzzyroughrules/federated/rounds.py` | `repair_rules`, `coverage_rounds`, `confidence_weighted_set_cover`, `sequential` and `PerClient` (how a round is run on the clients, which an experiment can replace to time or measure it) |
| `fuzzyroughrules/federated/inference.py` | `FallbackRuleGenerator` and `with_fallback`: what to predict for an object for which no rule fires |
| `experiments/horizontal_fl_frri.py` | Script that runs the whole protocol once and compares all variants with centralised FRRI |
| `experiments/measure.py` | `RoundRecorder`: time per client and round, bytes sent per round and peak memory |
| `experiments/datasets.py` | Datasets without files (`DATASETS`): scikit-learn datasets, synthetic data and a public OpenML dataset |
| `experiments/protocol_driver.py` | Runs the protocol for many datasets, numbers of clients and seeds |
| `experiments/diagnose_confidence.py` | Diagnosis of the drop from local to global confidence |
| `experiments/parallel_clients.py` | Runs every client in its own process, to check the derived parallel time |
| `experiments/scaling.py` | Federated and centralised FRRI on synthetic data of growing size |

`FederatedClient(X_train, y_train, **frri_params)` passes extra keyword arguments on to `RuleGenerator`.

## 6. Adding your own dataset

`experiments/horizontal_fl_frri.py` reads data through the dictionary `LOADERS`, which maps a loader name to a function.
A loader takes the path given with `--data-dir` and returns `(X, y)`: numeric features and labels, as NumPy arrays,
**not normalised**. Normalisation happens inside the protocol, with the range computed over the clients. The existing
example is `load_keel_dataset` in `fuzzyroughrules/federated/partition.py`, which reads a KEEL folder.

To use a CSV file with a header row, numeric feature columns and the label in the last column, add this function to
`experiments/horizontal_fl_frri.py` (`Path`, `np` and `pd` are already imported there) and replace the line that
defines `LOADERS`:

```python
def load_csv_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """One header row, numeric features in all columns but the last, labels in the last column."""
    data = pd.read_csv(path)
    return data.iloc[:, :-1].to_numpy(dtype=float), data.iloc[:, -1].to_numpy()


LOADERS = {"keel": load_keel_dataset, "csv": load_csv_dataset}
```

Then run it from the root of the repository:

```bash
python -m experiments.horizontal_fl_frri --loader csv --data-dir path/to/data.csv --clients 2 --seed 0 --log-file runs.csv
```

Use the name of your own loader for `--loader`. The options:

| Option | Meaning |
|---|---|
| `--data-dir` | Required. Passed unchanged to the loader. It can be a file, if your loader reads a file. For the `builtin` loader it is the name of a dataset. |
| `--loader` | Key in `LOADERS`: `keel` (default) or `builtin`, plus your own. |
| `--clients` | Number of clients. Default 2. |
| `--seed` | Seed of the splits. Default 0. |
| `--log-file` | Optional CSV file that each run is appended to. |
| `--class-log-file` | Optional CSV file that the recall of each class is appended to, one row per model, fallback and class. |
| `--dirichlet-alpha` | Non-IID split with label skew (see `horizontal_split`). Smaller values give clients with fewer classes. Default: the equal, stratified split of the protocol. |
| `--repair-max-rounds` | Maximum number of rule repair rounds. Default: no limit. |
| `--coverage-max-rounds` | Maximum number of coverage rounds after the first set cover. Default 3. |
| `--trace-memory` | Measure the peak memory of every round with `tracemalloc`. This makes the durations longer, so durations with and without it are not comparable. |
| `--solver` | Solver of every set cover and of centralised FRRI: `auto` (default), `gurobi` or `scipy`. See [section 2](#2-install). |
| `--time-limit` | Time limit in seconds of every set cover. Default: no limit. |

**The `builtin` loader.** `--loader builtin` reads no files. `--data-dir` is the name of a dataset in `DATASETS` in
`experiments/datasets.py`: `iris`, `wine`, `breast_cancer`, `digits`, `synthetic` (synthetic data with the default
arguments of `synthetic`) or `yeast` (downloaded from OpenML into the scikit-learn cache, so it needs network access the
first time, and has not been tested). For example:

```bash
python -m experiments.horizontal_fl_frri --loader builtin --data-dir wine --clients 3 --seed 0
```

Categorical features and missing values are not handled: encode or remove them in your loader.

## 7. Reading the output

The script prints a few counts, then four tables.

**Counts.** `n_candidates` is the number of local rules over all clients. `n_candidates_zero_global_confidence` is how
many of them have global confidence 0, and `n_candidates_confidence_lowered` how many have a global confidence below
their local confidence. `n_pooled_objects` is the number of objects in the pooled coverage matrix and
`n_uncovered_in_set_cover` the number of those that no candidate rule covers. `n_global_test` and `n_classes` are the
size of the global test set and the number of classes. `run_s` is the duration of the whole run.

Lines that start with `repair_` describe rule repair:

| Line | Meaning |
|---|---|
| `repair_n_rules_below_local` | Number of candidate rules whose global confidence was below their local confidence, i.e. the rules that needed repair. |
| `repair_n_rounds` | Number of repair rounds that were run. |
| `repair_n_recovered` | Number of rule extensions after which the rule was back at its local confidence, summed over the rounds. |
| `repair_n_extra_attributes_revealed` | Number of attribute values that repair added to the rules, summed over all rules. |
| `repair_n_rules_with_all_attributes` | Of the rules that needed repair, how many end with all attributes, i.e. the complete training object. |
| `repair_n_uncovered_in_set_cover` | The same as `n_uncovered_in_set_cover`, for the set cover of the repaired rules. |

There are three lines for each variant with coverage rounds (`f coverage rounds`, `f coverage rounds, conf-weighted`
and `f repair + coverage rounds`), that start with the name of the variant:

| Line | Meaning |
|---|---|
| `<variant>: n_rounds` | Number of coverage rounds after the first set cover. |
| `<variant>: n_extra_rules_exposed` | Number of new candidate rules that the clients sent in these rounds. Each is the rule of another training object. |
| `<variant>: n_uncovered_first_last` | A pair: the number of objects that no candidate covers in the first set cover and in the last one. |

**Global models and centralised FRRI.** One row per selection variant (`a union local conf`, `b union global conf`,
`c top 50% confidence`, `c top 50% coverage`, the same for 70%, `d set cover`, `e repair, union global conf`,
`e repair, set cover`, `f coverage rounds`, `f coverage rounds, conf-weighted` and `f repair + coverage rounds`) and one
row for `centralised FRRI`, which is FRRI on the pooled `train_local` data of all clients, with the same global range.

The scores without a suffix use ordinary FRRI inference. An object for which no rule fires gets the first class. With
the suffix `_relation_fallback` such an object gets the class of the rule with the highest relation to it, and with
`_default_rule` the majority class of the training data of all clients (see [Objects for which no rule
fires](#objects-for-which-no-rule-fires)).

| Column | Meaning |
|---|---|
| `n_rules`, `avg_rule_length` | Number of rules, and mean number of attributes used per rule. |
| `train_s_parallel`, `train_s_sequential` | Training time in seconds, see below. |
| `n_client_rounds` | Number of rounds with communication that the model needs. A round that is repeated (a repair round, a coverage round) counts each time. |
| `bytes_up`, `bytes_down` | Bytes sent by the clients to the server, and by the server to the clients, in the rounds that the model needs. Counted for an encoding of the messages, not measured on a network: see `experiments/measure.py`. |
| `peak_extra_mb` | Largest peak memory over the rounds that the model needs, in MB, above what was in use when the step started. It is 0 unless `--trace-memory` is given. |
| `global_test` | Balanced accuracy of the model on the global test set. |
| `global_test_relation_fallback`, `global_test_default_rule` | The same, with the fallback for objects for which no rule fires. |
| `no_rule_fires_global_test` | Fraction of the global test objects with membership 0 for every class. No rule applies to these objects. |
| `pooled_test_local` | Balanced accuracy on the `test_local` data of all clients together. Every model is scored on the same objects and classes, also when a class is missing from the `test_local` data of a client. |
| `pooled_test_local_relation_fallback` | The same, with the relation fallback. |
| `mean_test_local` | Mean over the clients of the balanced accuracy of the model on the `test_local` data of each client. A client without `test_local` data, which a non-IID split can give, is skipped in the mean. |
| `mean_test_local_relation_fallback` | The same, with the relation fallback. |
| `test_local_0`, `test_local_1`, ... | The same as `mean_test_local`, for each client separately. NaN for a client without `test_local` data. |

**Local models.** One row per client: `n_train_local`, `n_rules`, `avg_rule_length`, `train_s` (time of its local fit)
and `own_test_local` (balanced accuracy of its own local model on its own `test_local` data, NaN if it has none).

**Rounds.** One row per round: its name, `n_calls` (how often it ran), the total time of each client in it
(`client_0_s`, ...), `server_s` (time on the server), and `bytes_up`, `bytes_down` and `peak_extra_mb` as above. The
rounds are `range`, `local fit`, `confidence`, `coverage`, `set cover`, `coverage counts`, `repair: extend`,
`repair: confidence`, `repair: coverage`, `repair: set cover`, the rounds of the coverage round variants (with the
prefix `f plain: `, `f weighted: ` or `f repair: `: `coverage`, `set cover`, `coverage: new candidates` and
`coverage: confidence`) and `centralised fit`.

**Recall per class.** The recall of every global class on the global test set, for each model and each fallback
(`first`, `relation`, `default`). The same table is written to `--class-log-file`, with the columns `model`,
`fallback`, `label`, `n_global_test` (test objects of that class), `recall` (NaN if the class has no test objects) and
`no_rule_fires` (the fraction of the objects of that class for which no rule fires).

**`train_s_parallel` and `train_s_sequential`.** The clients run one after the other in this simulation, so each
client is timed separately in each round. A variant needs some of the rounds: (a) needs `range` and `local fit`; (b)
and (c) ranked by confidence also need `confidence`; (c) ranked by coverage also needs `coverage counts`; (d) needs
`confidence` and `coverage`, plus the set cover on the server; the variants (e) and (f) also need the repair and
coverage round rows that belong to them. `train_s_sequential` is the sum of the times of all clients over the rounds
that the variant needs, plus the server time. `train_s_parallel` is derived from the same measurements, for clients
that work at the same time: in each round only the slowest client counts, plus the server time. Neither includes
communication, which is not simulated. For `centralised FRRI` both are the time of its single fit.
`experiments/parallel_clients.py` below checks the derived parallel time.

**CSV log.** With `--log-file`, the rows of the models table are appended to that CSV file, which is created with a
header if it does not exist. Each row also gets the columns `finished_at`, `dataset` (the last part of `--data-dir`),
`n_objects`, `n_features`, `n_clients` and `seed`, and the options of the run: `repair_max_rounds`,
`coverage_max_rounds`, `trace_memory`, `solver` and `time_limit`, and `partition`, which is `iid` for the protocol split
and `dirichlet <alpha>` for `--dirichlet-alpha`. The per-client columns `test_local_0`, ... are left out, so runs with a
different number of clients fit in the same file. If the file exists with other columns, for example a log written by
an older version, the whole file is rewritten with the union of the columns, so that no column shifts. Rows that lack a
column get an empty value. The local-model and round tables are not logged. `--class-log-file` gets the same extra
columns.

### More scripts

All of these run from the root of the repository. Each has a module docstring with details.

| Script | What it does | Example |
|---|---|---|
| `experiments/protocol_driver.py` | Runs the protocol for every combination of dataset, number of clients (default 2 to 5) and seed, and appends every run to the logs. `--log-file` is required. It takes the options for the run above, and KEEL folders with `--keel-dirs`. | `python -m experiments.protocol_driver --datasets iris wine --log-file runs.csv --class-log-file classes.csv` |
| `experiments/diagnose_confidence.py` | Diagnoses the drop from local to global confidence: for each rule, its confidence on its own client and on the pooled data, and the same for the unshortened rule of the same object. It pools the data of the clients, which the protocol never does. `--rules-file` and `--classes-file` write CSV files. | `python -m experiments.diagnose_confidence --datasets iris wine --clients 2 5 --rules-file rules.csv` |
| `experiments/parallel_clients.py` | Runs every client in its own process, and compares the derived parallel time of the rounds local fit, confidence and coverage with the time in the workers and the wall time. With more clients than CPU cores the clients share cores. | `python -m experiments.parallel_clients --dataset synthetic --clients 2 3 4 --seeds 0 1 2` |
| `experiments/scaling.py` | Runs federated FRRI (set cover variant) and centralised FRRI on synthetic data of growing size, each size in a child process with a time and a memory limit. Repair and coverage rounds are not part of it. `--out` writes a CSV file. | `python -m experiments.scaling --sizes 1000 2000 4000 8000 --clients 3 --out scaling.csv` |

**Results.** The results of this work are in [docs/federated-improvements-report.md](docs/federated-improvements-report.md).
They are provisional.

## 8. Licence

GPL-3.0. See the `LICENSE` file.
