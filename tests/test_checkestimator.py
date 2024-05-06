from sklearn.utils.estimator_checks import parametrize_with_checks
from fuzzyroughrules.rule_induction_base import RuleGenerator


@parametrize_with_checks([RuleGenerator()])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)
