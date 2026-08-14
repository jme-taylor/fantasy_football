"""Count distributions behind the defensive-contribution threshold.

A defcon point is not a smooth quantity: it is paid when a defender's
CBIT count clears 10 in a match. What the rate model predicts is a rate
per 90, so turning that into points means asking how likely a count is
to clear the threshold given the minutes on the pitch -- which needs a
distribution over counts, not a point estimate. Running a predicted
count through a step function would be badly biased for exactly the
defenders sitting near the cliff edge.

The distribution sits behind a Protocol so swapping Poisson for
Negative Binomial is one implementation, not an edit at every call site.
Dispersion is fitted state on the implementation rather than a
constructor argument, so it travels with the model artefact.
"""

import logging
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import poisson

logger = logging.getLogger(__name__)


@runtime_checkable
class CountDistribution(Protocol):
    """A distribution over match-level event counts."""

    def fit(
        self, counts: ArrayLike, exposure: ArrayLike
    ) -> "CountDistribution":
        """Learn whatever shape parameters the distribution carries.

        Parameters
        ----------
        counts : ArrayLike
            Observed counts, one per match.
        exposure : ArrayLike
            Minutes behind each count, on a 90-minute scale.

        Returns
        -------
        CountDistribution
            Self, fitted.
        """
        ...

    def p_at_least(
        self, mean: ArrayLike, threshold: int
    ) -> NDArray[np.float64]:
        """Return P(count >= threshold) for each expected count.

        Parameters
        ----------
        mean : ArrayLike
            Expected count for each match.
        threshold : int
            The count that must be reached.

        Returns
        -------
        NDArray[np.float64]
            One probability per element of ``mean``.
        """
        ...


class PoissonCounts:
    """Counts as Poisson, where the variance is forced to equal the mean.

    TODO (JT): CBIT is overdispersed. Measured on 2025-26 defenders
    playing 60+ minutes, the variance-to-mean ratio is 1.88 against the
    1.0 Poisson assumes, so this understates P(>= 10) for the high-rate
    defenders whose right tail is fattest -- the ones worth owning.
    Replacing this with a Negative Binomial is a new implementation of
    :class:`CountDistribution` and nothing else; the fitted dispersion
    has somewhere to live because :meth:`fit` already takes the counts.
    """

    def fit(self, counts: ArrayLike, exposure: ArrayLike) -> "PoissonCounts":
        """Record the observed dispersion without acting on it.

        Poisson has no free parameter beyond the mean, so this fits
        nothing. It logs the ratio it would need if it did, which is what
        turns the TODO above from an assertion into a measurement that
        every training run re-checks.
        """
        observed = np.asarray(counts, dtype=float)
        if observed.size > 1:
            mean = float(observed.mean())
            if mean > 0:
                ratio = float(observed.var(ddof=1)) / mean
                logger.info(
                    "CBIT variance-to-mean ratio is %.2f; Poisson assumes "
                    "1.0, so P(>= threshold) is understated above it.",
                    ratio,
                )
        return self

    def p_at_least(
        self, mean: ArrayLike, threshold: int
    ) -> NDArray[np.float64]:
        """Return the Poisson survival function at ``threshold``."""
        rate = np.clip(np.asarray(mean, dtype=float), 0.0, None)
        return np.asarray(poisson.sf(threshold - 1, rate), dtype=float)
