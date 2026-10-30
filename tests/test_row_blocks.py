"""
Relations are computed in blocks of rows to bound memory. The result must be exactly the same as without blocks.
"""
import numpy as np
import pytest

import fuzzyroughrules.operators as operators
from fuzzyroughrules.approximations import LowerApproximation
from fuzzyroughrules.operators import RelationTypes, triangular_relation


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    return rng.random((101, 7)), rng.integers(0, 3, 101)


@pytest.mark.parametrize("block_elements", [1, 7 * 13, 7 * 101 * 5])
def test_blocks_give_exactly_the_same_lower_approximation(data, block_elements, monkeypatch):
    X, y = data
    whole = LowerApproximation().get_approximation(X, y)
    monkeypatch.setattr(operators, "BLOCK_ELEMENTS", block_elements)
    assert np.array_equal(LowerApproximation().get_approximation(X, y), whole)


@pytest.mark.parametrize("block_elements", [1, 7 * 13, 7 * 101 * 5])
def test_blocks_give_exactly_the_same_relation(data, block_elements, monkeypatch):
    X, _ = data
    types = np.array([RelationTypes.DOMINATED, RelationTypes.DOMINANT, RelationTypes.INDISCERNIBLE,
                      RelationTypes.UNUSED, RelationTypes.INDISCERNIBLE, RelationTypes.DOMINATED, RelationTypes.UNUSED])
    slopes = np.array([1., .5, .1, 1., 1., .01, 1.])
    whole_matrix = triangular_relation(X, X[:50], slopes, types)
    whole_rule = triangular_relation(X, X[3], slopes, types)
    monkeypatch.setattr(operators, "BLOCK_ELEMENTS", block_elements)
    assert np.array_equal(triangular_relation(X, X[:50], slopes, types), whole_matrix)
    assert np.array_equal(triangular_relation(X, X[3], slopes, types), whole_rule)
    no_attributes = np.full(7, RelationTypes.UNUSED, dtype=RelationTypes)
    assert np.array_equal(triangular_relation(X, X[:5], slopes, no_attributes), np.ones((101, 5)))
