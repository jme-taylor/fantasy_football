"""The count distributions behind the defcon threshold."""

import math

import numpy as np
import pytest
from scipy.stats import poisson

from fantasy_football.modelling.distributions import (
    CountDistribution,
    PoissonCounts,
)


def test_poisson_satisfies_the_protocol() -> None:
    """The Poisson implementation is a CountDistribution."""
    assert isinstance(PoissonCounts(), CountDistribution)


def test_fit_returns_self_so_it_chains() -> None:
    """Fitting hands the distribution back for use in one expression."""
    distribution = PoissonCounts()
    assert distribution.fit([1, 2, 3], [90, 90, 90]) is distribution


def test_p_at_least_matches_the_closed_form() -> None:
    """The survival function agrees with the analytic Poisson CDF.

    A swappable interface with no direct test is not really swappable,
    which is why the distribution is checked here rather than only
    through the component that uses it.
    """
    means = [0.5, 3.0, 7.5, 12.0]
    got = PoissonCounts().p_at_least(means, 10)
    expected = [
        1.0 - sum(math.exp(-m) * m**k / math.factorial(k) for k in range(10))
        for m in means
    ]
    assert got == pytest.approx(expected, abs=1e-12)


def test_p_at_least_agrees_with_scipy_survival() -> None:
    """P(>= k) is the survival function one below the threshold."""
    means = np.array([1.0, 6.0, 20.0])
    assert PoissonCounts().p_at_least(means, 10) == pytest.approx(
        poisson.sf(9, means)
    )


def test_probability_rises_with_the_expected_count() -> None:
    """A defender expected to do more clears the threshold more often."""
    probabilities = PoissonCounts().p_at_least([2.0, 6.0, 10.0, 20.0], 10)
    assert list(probabilities) == sorted(probabilities)


def test_zero_expected_count_never_clears_the_threshold() -> None:
    """A defender expected to do nothing scores no defcon points."""
    assert PoissonCounts().p_at_least([0.0], 10) == pytest.approx([0.0])


def test_negative_means_are_clipped_rather_than_propagated() -> None:
    """A regressor undershooting below zero cannot produce a nan.

    The rate model is a regressor with no non-negativity constraint, so
    a small negative prediction is reachable. Left alone it would give a
    nan probability and a nan points contribution, which sums silently
    into the composed total.
    """
    assert PoissonCounts().p_at_least([-0.5], 10) == pytest.approx([0.0])


def test_fit_logs_the_dispersion_it_is_ignoring(caplog) -> None:
    """Overdispersion is measured on every fit, not assumed away."""
    counts = [0, 0, 1, 2, 20, 25, 30]
    with caplog.at_level("INFO"):
        PoissonCounts().fit(counts, [90] * len(counts))

    assert "variance-to-mean" in caplog.text


def test_fit_survives_an_all_zero_sample() -> None:
    """A sample with no events cannot divide by a zero mean."""
    PoissonCounts().fit([0, 0, 0], [90, 90, 90])
