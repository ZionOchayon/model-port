import math

import numpy as np

from modelport_platform.drift.psi import PSI_EPSILON, categorical_shares, psi, quantile_edges, shares


def test_identical_distributions_near_zero():
    values = np.random.default_rng(0).normal(size=5000)
    edges = quantile_edges(values)
    assert psi(shares(values, edges), shares(values, edges)) < 1e-9


def test_hand_computed_three_bins():
    expected, actual = np.array([0.5, 0.3, 0.2]), np.array([0.4, 0.4, 0.2])

    def smooth(p):
        p = p + PSI_EPSILON
        return p / p.sum()

    e, a = smooth(expected), smooth(actual)
    want = sum((ai - ei) * math.log(ai / ei) for ai, ei in zip(a, e, strict=True))
    assert abs(psi(expected, actual) - want) < 1e-9
    assert abs(want - (0.1 * math.log(0.4 / 0.3) - 0.1 * math.log(0.4 / 0.5))) < 1e-3


def test_non_negative_property():
    rng = np.random.default_rng(42)
    for _ in range(1000):
        bins = int(rng.integers(2, 12))
        e, a = rng.dirichlet(np.ones(bins)), rng.dirichlet(np.ones(bins))
        assert psi(e, a) >= 0.0


def test_all_new_category_is_significant():
    categories = ["a", "b", "c"]
    expected = categorical_shares(["a", "b", "c"] * 100, categories)
    actual = categorical_shares(["z"] * 300, categories)
    assert expected[-1] == 0.0
    assert actual[-1] == 1.0
    assert psi(expected, actual) > 0.25


def test_out_of_range_value_counted_in_outer_bin():
    edges = quantile_edges(np.linspace(0.0, 1.0, 1000))
    assert edges[0] == -np.inf
    assert edges[-1] == np.inf
    counted = shares(np.array([-5.0, 50.0, 0.5]), edges)
    assert counted.sum() == 1.0
    assert counted[0] == counted[-1] == 1 / 3


def test_one_sigma_shift_significant_resample_stable():
    rng = np.random.default_rng(7)
    base = rng.normal(size=5000)
    edges = quantile_edges(base)
    expected = shares(base, edges)
    assert psi(expected, shares(rng.normal(loc=1.0, size=5000), edges)) > 0.25
    assert psi(expected, shares(rng.normal(size=5000), edges)) < 0.1
