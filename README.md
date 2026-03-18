# Fuzzy-Rough Rule Induction

This repository contains implementations of fuzzy-rough set–based rule induction algorithms for classification, with an API inspired by scikit-learn. The code is organized as a Python package `fuzzyroughrules` and companion modules used in experiments and notebooks.

The following algorithms/variants are included:
- QuickRules
- OWA-QuickRules
- (OWA-)QuickRules with pruning
- FRRI: a fuzzy-rough rule induction algorithm

Reference: Richard Jensen, Chris Cornelis, and Qiang Shen. [Hybrid fuzzy-rough rule induction and feature selection](https://cwi.ugent.be/Chris/fuzzieee2009b.pdf). Fuzz-IEEE 2009, pp. 1151–1156.

## Overview

`fuzzyroughrules` provides:
- Rule induction with fuzzy similarity relations (triangular relations, implicators, OWA-based inclusion, etc.).
- A `RuleGenerator` estimator compatible with scikit-learn's fit/predict interface.
- Utilities for approximations, feature preprocessing (e.g., QuickReduct), and fuzzy operators.
- A collection of Jupyter notebooks demonstrating experiments on KEEL and other datasets.

## Stack and Requirements

- Language: Python
- Packaging: `setuptools` (`setup.py`)
- Package: `fuzzyroughrules`
- Dependency management: `requirements.txt`
- Notebooks: Jupyter (`.ipynb` files in the project root)
- Tests: `pytest` style (see `tests/`)
- License: MIT (as declared in `setup.py`) — see License section and TODO below.

Core dependencies (see `requirements.txt` for exact versions):
- numpy, scipy, pandas, scikit-learn
- matplotlib (for plotting in notebooks)
- cvxopt
- gurobipy (optional/proprietary; needed if using Gurobi-based routines)
- Mosek (optional/proprietary; needed if using Mosek-based routines)
- setuptools

Notes:
- Some solvers (Gurobi, MOSEK) are proprietary and require separate installation and licensing. If you don't need them, you may comment/remove them from `requirements.txt` before installation.
- There are compiled artifacts for cvxopt present under `fuzzyroughrules/cvxopt/` which are platform- and Python-version specific. Prefer installing `cvxopt` via pip/conda for your platform rather than relying on bundled binaries.

## Installation

Option A — editable install (recommended for development):

```bash
git clone <this-repo-url>
cd 2022-fuzzylem
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .  # installs the fuzzyroughrules package in editable mode
```

Option B — minimal install (without heavy/proprietary solvers):

```bash
# Edit requirements.txt and comment out gurobipy and Mosek lines if not needed
pip install -r requirements.txt
pip install -e .
```

Python version:
- Developed with Python 3.10 (inferred from build artifacts in the repo). TODO: Confirm supported Python versions and pin in classifiers/CI.

## Quick Start

Basic usage with the scikit-learn-like API:

```python
import numpy as np
from fuzzyroughrules.rule_induction_base import RuleGenerator

# Toy data
grng = np.random.default_rng(0)
X = grng.random((100, 4))
y = (X[:, 0] + X[:, 1] > 1.0).astype(int)

# Create and fit the rule generator
clf = RuleGenerator()
clf.fit(X, y)

# Predict class labels and probabilities
y_pred = clf.predict(X)
proba = clf.predict_proba(X)

# Inspect learned rules
print(clf.get_rules_as_string())
print("Average rule length:", clf.average_rule_length())
```

For more advanced relations, inclusion measures, or feature ordering, see:
- `fuzzyroughrules/operators.py`
- `fuzzyroughrules/approximations.py`
- `fuzzyroughrules/feature_preprocessors.py`
- `fuzzyroughrules/*_rules.py`

## Entry Points and Scripts

- Python package entry: `fuzzyroughrules` (import and use from Python as shown above).
- Command-line scripts: none defined (no `console_scripts` in `setup.py`).
- Notebooks: various `*.ipynb` files in the repository root demonstrate experiments and analyses.

TODO:
- If a CLI is desired, add `console_scripts` in `setup.py` and document here.

## Environment Variables

- None are strictly required for the core `fuzzyroughrules` functionality.
- If you intend to use proprietary solvers, you likely need to configure their license environment variables. Examples (to be confirmed for your installation):
  - Gurobi: `GRB_LICENSE_FILE` or `GUROBI_HOME`, and modify `PATH`/`LD_LIBRARY_PATH` accordingly. TODO: confirm exact variables used by codepaths here.
  - MOSEK: `MOSEKLM_LICENSE_FILE` or activation key. TODO: confirm exact variables and how they integrate with any code.

## Running Tests

Tests are written with `pytest`. A scikit-learn estimator compatibility check is included.

```bash
# from the project root, with the virtual environment activated
pip install -r requirements.txt
pip install -e .
pip install pytest  # if not already present
pytest -q
```

- Example test file: `tests/test_checkestimator.py` uses `sklearn.utils.estimator_checks.parametrize_with_checks` on `RuleGenerator`.

## Project Structure

Top-level highlights:

```
2022-fuzzylem/
├─ fuzzyroughrules/           # main package with rule induction and operators
│  ├─ rule_induction_base.py  # RuleGenerator and core logic
│  ├─ operators.py            # fuzzy operators, inclusion, triangular relations
│  ├─ approximations.py       # lower approximations, etc.
│  ├─ feature_preprocessors.py
│  ├─ frfs_frri.py, sugeno_rules.py, ...
│  └─ cvxopt/                 # cvxopt related modules/binaries (platform-specific)
├─ quickrules/                # weights and QuickRules-related helpers
├─ hhelper/                   # helper utilities (e.g., data loading, scoring)
├─ tests/                     # pytest tests
├─ requirements.txt           # Python dependencies (pinning used here)
├─ setup.py                   # setuptools packaging config
├─ full-keel-data/, keel-data/  # datasets used in notebooks/experiments
├─ figures/, results/, tables/  # outputs for analyses
├─ *.ipynb                    # numerous notebooks with experiments/analysis
└─ README.md
```

Note: Several large `.ipynb` notebooks at the repository root reproduce experiments for the thesis/papers.

## How to Run Experiments

- Open any of the notebooks (e.g., `analysis.ipynb`, `testing_quickrules.ipynb`) in Jupyter and run cells.
- Many notebooks expect datasets under `keel-data/` or `full-keel-data/`.
- Some notebooks may require optional solvers (see Requirements) depending on the chosen experiment.

TODO:
- Document each main notebook with a short description and prerequisites.

## Citing

If you use this code in academic work, please cite the reference mentioned above and consider citing the repository or related publications by the authors of this codebase. TODO: Add a BibTeX entry for the implementation if available.

## License

- Declared license in `setup.py`: MIT.
- TODO: Add a top-level `LICENSE` file with the full MIT text to make the licensing explicit.

## Changelog

- TODO: Add a `CHANGELOG.md` summarizing significant changes, versions, and notes.
