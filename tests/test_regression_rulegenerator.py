"""
Regression test: non-federated rule generators must keep giving the rules and predictions that were recorded on
`main` before the federated work started.

The baseline depends on the Gurobi version, as tied set covers can resolve differently. To record it again, run this
file directly with the --record flag. Only do this on an unmodified `main`.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split

from fuzzyroughrules.full_inclusion_frri import GranularRuleGenerator
from fuzzyroughrules.rule_induction_base import RuleGenerator

BASELINE = Path(__file__).parent / "data" / "rulegenerator_baseline.json"

DATASETS = {"iris": load_iris, "wine": load_wine}
CONFIGS = {
    "default": (RuleGenerator, {}),
    "discard-uncertain": (RuleGenerator, {"discard_uncertain_objects": True, "certainty_threshold": 0.4}),
    "discard-uncertain-no-add": (RuleGenerator, {
        "discard_uncertain_objects": True,
        "certainty_threshold": 0.4,
        "add_uncovered_objects": False
    }),
    "optimise-slopes": (RuleGenerator, {"optimise_slopes": True}),
    "granular": (GranularRuleGenerator, {}),
}


def snapshot(dataset: str, config: str) -> dict:
    data = DATASETS[dataset]()
    x_train, x_test, y_train, _ = train_test_split(
        data.data, data.target, test_size=0.3, random_state=0, stratify=data.target
    )
    generator, params = CONFIGS[config]
    model = generator(**params).fit(x_train, y_train)
    return {
        "rules": [{
            "antecedent": rule.antecedent.tolist(),
            "reducts": [relation.name for relation in rule.reducts],
            "slopes": rule.slopes.tolist(),
            "credibility": float(rule.credibility),
            "decision": int(rule.decision),
        } for rule in model.rules_],
        "predictions": model.predict(x_test).tolist(),
        "probabilities": model.predict_proba(x_test).tolist(),
    }


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("dataset", DATASETS)
def test_matches_main(dataset, config):
    baseline = json.loads(BASELINE.read_text())[f"{dataset}/{config}"]
    # the round trip through json makes the comparison use the same types as the baseline
    assert json.loads(json.dumps(snapshot(dataset, config))) == baseline


if __name__ == "__main__":
    if "--record" not in sys.argv:
        sys.exit("Pass --record to overwrite the baseline.")
    BASELINE.parent.mkdir(exist_ok=True)
    BASELINE.write_text(json.dumps(
        {f"{dataset}/{config}": snapshot(dataset, config) for dataset in DATASETS for config in CONFIGS}
    ))
