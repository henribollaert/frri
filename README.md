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

**Gurobi.** The rule selection step of FRRI is an optimisation problem that is solved with Gurobi, and
`rule_induction_base.py` imports `gurobipy`, so it must be installed. The version installed by `pip` comes with a
limited licence of 2,000 variables and 2,000 constraints. The selection has one variable for each candidate rule and
one constraint for each training object, so larger training sets need a full licence. Gurobi offers a free academic
licence: see https://support.gurobi.com/hc/en-us/articles/360040541251.

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
`test_horizontal_fl_experiment.py`) and a regression test of `RuleGenerator`.

**Regression baseline.** `tests/test_regression_rulegenerator.py` compares the rules and predictions of the
non-federated generators on iris and wine with `tests/data/rulegenerator_baseline.json`. The baseline depends on the
Gurobi version, because tied set covers can be resolved differently. If that test fails after a change of Gurobi
version, record the baseline again, but only on an unmodified `main` so that it keeps describing the non-federated code:

```bash
git checkout main
python tests/test_regression_rulegenerator.py --record
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

### Selection variants

| Variant | Function | What it does |
|---|---|---|
| (a) | `union_local` | All local rules, with their local confidence. |
| (b) | `union_global` | All local rules, with their global confidence. Rules with global confidence 0 are dropped, because they cannot influence a prediction. |
| (c) | `top_k_per_client` | The best fraction `k` of the rules of each client (rounded up), with their global confidence. Rules with confidence 0 are dropped. |
| (d) | `set_cover` | Solves the set cover problem on the pooled coverage matrix, with Gurobi. |

Details:

- `top_k_per_client(..., rank_by=...)`: with `rank_by="confidence"` (the default) the rules of a client are ranked by
  global confidence. With `rank_by="coverage"` they are ranked by the number of local objects that the rule covers, and
  you must pass `coverage_counts` (one array per client, from `FederatedClient.coverage_counts`). The experiment script
  runs `k` = 0.5 and 0.7 with both rankings.
- `set_cover(rules, coverage, objective=plain_set_cover)`: the cost of selecting each rule is given by `objective`; the
  default gives every rule cost 1, i.e. the fewest rules. Objects that no candidate rule covers are left out of the
  problem, because FRRI would add the rule of that object, which is not available here. Their number is returned.

### What leaves a client

| Step | Client sends | Client receives |
|---|---|---|
| 1 | per-feature minimum and maximum of `train_local`, the set of labels | nothing |
| 2 | its rules (`RulePayload`): the indices of the used attributes, the rule's values on them, the relation type and slope of each, the class label, the local confidence | the global range and classes |
| 3 | one number per candidate rule | all local rules (the union) |
| 4 | a binary coverage matrix (rules x local objects), objects in random order | the rules, with global confidence |
| 5, only for variant (c) ranked by coverage | for each of its own rules, the number of local objects that it covers | nothing |

Training objects are never sent. The values in a rule are the values of the training object that the rule was built
from, restricted to the attributes that the rule uses, so this is not a formal privacy guarantee.

### Module overview

| File | Contents |
|---|---|
| `fuzzyroughrules/federated/partition.py` | `load_keel_dataset`, `horizontal_split`, `global_min_max`, `global_classes`, `FixedRangeScaler` |
| `fuzzyroughrules/federated/payload.py` | `RulePayload` (what is sent for a rule, with `to_dict` and `from_dict`), `rule_to_payload`, `payload_to_rule` |
| `fuzzyroughrules/federated/client.py` | `FederatedClient`: `min_max`, `labels`, `fit_local`, `local_minima`, `coverage_columns`, `coverage_counts` |
| `fuzzyroughrules/federated/aggregate.py` | `global_confidence`, `union_local`, `union_global`, `top_k_per_client`, `set_cover`, `plain_set_cover`, `build_global_model` |
| `experiments/horizontal_fl_frri.py` | Script that runs the whole protocol once and compares all variants with centralised FRRI |

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
| `--data-dir` | Required. Passed unchanged to the loader. It can be a file, if your loader reads a file. |
| `--loader` | Key in `LOADERS`. Default `keel`. |
| `--clients` | Number of clients. Default 2. |
| `--seed` | Seed of the splits. Default 0. |
| `--log-file` | Optional CSV file that each run is appended to. |

Categorical features and missing values are not handled: encode or remove them in your loader.

## 7. Reading the output

The script prints a few counts, then three tables.

**Counts.** `n_candidates` is the number of local rules over all clients. `n_candidates_zero_global_confidence` is how
many of them have global confidence 0, and `n_candidates_confidence_lowered` how many have a global confidence below
their local confidence. `n_pooled_objects` is the number of objects in the pooled coverage matrix and
`n_uncovered_in_set_cover` the number of those that no candidate rule covers. `run_s` is the duration of the whole run.

**Global models and centralised FRRI.** One row per selection variant (`a union local conf`, `b union global conf`,
`c top 50% confidence`, `c top 50% coverage`, the same for 70%, and `d set cover`) and one row for `centralised FRRI`,
which is FRRI on the pooled `train_local` data of all clients, with the same global range.

| Column | Meaning |
|---|---|
| `n_rules`, `avg_rule_length` | Number of rules, and mean number of attributes used per rule. |
| `train_s_parallel`, `train_s_sequential` | Training time in seconds, see below. |
| `global_test` | Balanced accuracy of the model on the global test set. |
| `no_rule_fires_global_test` | Fraction of the global test objects with membership 0 for every class. No rule applies to these objects, and they get the first class by default. |
| `mean_test_local` | Mean over the clients of the balanced accuracy of the model on the `test_local` data of each client. |
| `test_local_0`, `test_local_1`, ... | The same, for each client separately. |

**Local models.** One row per client: `n_train_local`, `n_rules`, `avg_rule_length`, `train_s` (time of its local fit)
and `own_test_local` (balanced accuracy of its own local model on its own `test_local` data).

**Duration of each round.** The time per client in each round (`local fit`, `confidence`, `coverage`,
`coverage counts`), the `set cover (server)` time and the `centralised fit` time.

**`train_s_parallel` and `train_s_sequential`.** The clients run one after the other in this simulation, so each
client is timed separately in each round. A variant needs some of the rounds: (a) needs `local fit`; (b) and (c) ranked
by confidence also need `confidence`; (c) ranked by coverage also needs `coverage counts`; (d) needs `local fit`,
`confidence` and `coverage`, plus the set cover on the server. `train_s_sequential` is the sum of the times of all
clients over the rounds that the variant needs, plus the server time. `train_s_parallel` is derived from the same
measurements, for clients that work at the same time: in each round only the slowest client counts, plus the server
time. Neither includes communication, which is not simulated. For `centralised FRRI` both are the time of its single fit.

**CSV log.** With `--log-file`, the rows of the first table are appended to that CSV file, which is created with a header
if it does not exist. Each row also gets the columns `finished_at`, `dataset` (the last part of `--data-dir`),
`n_objects`, `n_features`, `n_clients` and `seed`. The per-client columns `test_local_0`, ... are left out, so runs with
a different number of clients fit in the same file. The local-model and round tables are not logged.

## 8. Licence

GPL-3.0. See the `LICENSE` file.
