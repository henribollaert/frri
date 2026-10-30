"""
The rule selection can be solved with Gurobi or with scipy.optimize.milp. Both must find an optimal set cover, but tied
optima can be resolved differently, so the selected rules themselves are only compared on problems with one optimum.
"""
import warnings

import numpy as np
import pytest
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split

import fuzzyroughrules.rule_induction_base as base
from fuzzyroughrules.rule_induction_base import RuleGenerator, select_rules_from_coverage


def gurobi_usable() -> bool:
    if base.gb is None:
        return False
    try:
        base._select_with_gurobi(np.ones((1, 1)), np.ones(1), ())
    except base.gb.GurobiError:
        return False
    return True


needs_gurobi = pytest.mark.skipif(not gurobi_usable(), reason="gurobipy is not installed or has no usable licence")


def test_scipy_finds_the_unique_minimum_and_skips_uncovered_objects():
    coverage = np.array([[1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=float)
    assert select_rules_from_coverage(coverage, uncovered=[3], solver="scipy").tolist() == [0, 2]


def test_scipy_uses_the_weights():
    coverage = np.array([[1, 1, 0], [1, 0, 0], [0, 1, 1]], dtype=float)
    assert select_rules_from_coverage(coverage, np.array([10., 1., 1.]), solver="scipy").tolist() == [1, 2]


def test_scipy_handles_a_problem_without_constraints():
    coverage = np.zeros((2, 3))
    assert select_rules_from_coverage(coverage, uncovered=[0, 1, 2], solver="scipy").tolist() == []


def test_unknown_solver_is_refused():
    with pytest.raises(ValueError):
        select_rules_from_coverage(np.ones((1, 1)), solver="cplex")


def test_auto_falls_back_on_scipy_when_gurobi_is_missing(monkeypatch):
    monkeypatch.setattr(base, "gb", None)
    coverage = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    assert select_rules_from_coverage(coverage).tolist() == [0, 2]
    with pytest.raises(ImportError):
        select_rules_from_coverage(coverage, solver="gurobi")


@pytest.mark.skipif(base.gb is None, reason="gurobipy is not installed")
def test_auto_falls_back_on_scipy_with_a_warning_when_gurobi_fails(monkeypatch):
    def failing(*args):
        raise base.gb.GurobiError(10010, "Model too large for size-limited license")

    monkeypatch.setattr(base, "_select_with_gurobi", failing)
    coverage = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    with pytest.warns(UserWarning, match="scipy"):
        assert select_rules_from_coverage(coverage).tolist() == [0, 2]


@needs_gurobi
def test_auto_uses_gurobi_when_it_works(monkeypatch):
    def not_called(*args):
        raise AssertionError("scipy must not be used when Gurobi works")

    monkeypatch.setattr(base, "_select_with_scipy", not_called)
    coverage = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert select_rules_from_coverage(coverage).tolist() == [0, 2]


@needs_gurobi
@pytest.mark.parametrize("params", [{}, {"optimise_slopes": True}, {"discard_uncertain_objects": True}], ids=str)
@pytest.mark.parametrize("load", [load_iris, load_wine], ids=["iris", "wine"])
def test_both_solvers_select_an_optimal_number_of_rules(load, params):
    data = load()
    x_train, _, y_train, _ = train_test_split(data.data, data.target, test_size=0.3, random_state=0, stratify=data.target)
    sizes = [len(RuleGenerator(**params, solver=solver).fit(x_train, y_train).rules_) for solver in ("gurobi", "scipy")]
    assert sizes[0] == sizes[1]


@needs_gurobi
def test_random_weighted_problems_have_the_same_optimum():
    rng = np.random.default_rng(0)
    for _ in range(20):
        coverage = (rng.random((30, 40)) < 0.15).astype(float)
        uncovered = np.flatnonzero(coverage.sum(axis=0) == 0)
        weights = rng.random(30) + 0.1
        values = [weights[select_rules_from_coverage(coverage, weights, uncovered, s)].sum() for s in ("gurobi", "scipy")]
        assert values[0] == pytest.approx(values[1], rel=1e-6)


@pytest.mark.parametrize("solver", ["scipy", pytest.param("gurobi", marks=needs_gurobi)])
def test_time_limit_gives_a_cover_and_a_warning(solver):
    # random set cover problems are hard, so this one is not solved to optimality in 0.2 seconds
    rng = np.random.default_rng(0)
    coverage = (rng.random((400, 400)) < 0.02).astype(float)
    uncovered = np.flatnonzero(coverage.sum(axis=0) == 0)
    with pytest.warns(UserWarning, match="time limit"):
        selected = select_rules_from_coverage(coverage, uncovered=uncovered, solver=solver, time_limit=0.2)
    required = np.setdiff1d(np.arange(400), uncovered)
    assert np.all(coverage[selected][:, required].sum(axis=0) >= 1)
