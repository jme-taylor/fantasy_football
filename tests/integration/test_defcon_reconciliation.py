"""The CBIT reconciliation gate, run against the real stored data.

Marked ``integration``: this reads the local DuckDB rather than seeded
fixtures, and is deselected by default. Run with
``uv run pytest -m integration``.

2025-26 is the one season carrying both CBIT sources -- FPL's own
``defensive_contribution`` and FCI's four counters -- so it is where the
reconstruction used for 2024-25 can be checked against the count the
scoring rule actually ran on. Written as a test rather than a one-off
notebook check so every future season is covered by it too.
"""

from collections.abc import Iterator

import duckdb
import polars as pl
import pytest

from fantasy_football.constants import DATABASE_PATH
from fantasy_football.features.match_form import DEFCON_COMPONENT_STATS

RECONCILIATION_SEASON = "2025-26"

# Exact agreement is not achievable: the two providers disagree by one on
# a few per cent of defender gameweeks, on rows where minutes and fixture
# legs match exactly, so the gap is their event definitions rather than a
# join or competition-filter fault. What is required instead is that the
# reconstruction be *unbiased* -- symmetric noise inflates a rate
# regression's variance without moving its centre -- and that no row be
# off by enough to flip a defender across the threshold on its own.
MIN_EXACT_SHARE = 0.90
MAX_MEAN_ABSOLUTE_BIAS = 0.05
MAX_ABSOLUTE_DIFFERENCE = 4
MIN_WITHIN_TWO_SHARE = 0.99

# Reads the stored database rather than the network, but shares the
# integration marker: it needs data that has been loaded, so it cannot
# run on a clean checkout.
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not DATABASE_PATH.exists(), reason="no local database to reconcile"
    ),
]


@pytest.fixture(scope="module")
def reconciliation() -> Iterator[pl.DataFrame]:
    """Return per-defender agreement between the two CBIT sources."""
    totalled = " + ".join(
        f"coalesce(o.{stat}, 0)" for stat in DEFCON_COMPONENT_STATS
    )
    connection = duckdb.connect(str(DATABASE_PATH), read_only=True)
    try:
        yield connection.sql(f"""
            WITH opta AS (
                SELECT season, gw, element, sum({totalled}) AS reconstructed
                FROM player_match_opta AS o
                WHERE season = '{RECONCILIATION_SEASON}'
                  AND competition = 'prem'
                GROUP BY 1, 2, 3
            ),
            fpl AS (
                SELECT season, gw, element,
                       sum(defensive_contribution) AS awarded
                FROM player_match_fpl
                WHERE season = '{RECONCILIATION_SEASON}'
                GROUP BY 1, 2, 3
            )
            SELECT o.reconstructed - f.awarded AS difference
            FROM opta AS o
            JOIN fpl AS f USING (season, gw, element)
            JOIN player_season AS s
                ON s.season = o.season AND s.element = o.element
            WHERE s.position = 'DEF' AND f.awarded IS NOT NULL
        """).pl()
    finally:
        connection.close()


def test_there_is_something_to_reconcile(reconciliation) -> None:
    """Both sources cover the season, so the gate is not vacuous."""
    assert reconciliation.height > 1000


def test_most_defender_gameweeks_agree_exactly(reconciliation) -> None:
    """The reconstruction matches the awarded count on the great majority.

    A sharp fall here means a real fault -- a competition filter that has
    stopped filtering, a counter renamed at source, a join fanning out --
    rather than the provider noise the looser bounds below allow for.
    """
    exact = (reconciliation["difference"] == 0).mean()
    assert exact >= MIN_EXACT_SHARE


def test_the_reconstruction_is_unbiased(reconciliation) -> None:
    """Disagreement is symmetric, so it does not shift the target.

    This is the property the 2024-25 training rows actually rely on: a
    reconstruction that ran consistently high or low would move every
    rate that season, which recency weighting would then carry into the
    live predictions.
    """
    assert abs(reconciliation["difference"].mean()) <= MAX_MEAN_ABSOLUTE_BIAS


def test_no_row_disagrees_wildly(reconciliation) -> None:
    """No single row is off by enough to be a different quantity.

    A defender miscounted by ten would be one whose Opta row covers a
    different match, which is the failure the competition filter exists
    to prevent.
    """
    assert reconciliation["difference"].abs().max() <= MAX_ABSOLUTE_DIFFERENCE


def test_disagreement_is_almost_always_within_two(reconciliation) -> None:
    """The tail is thin enough not to move P(>= 10) materially."""
    within = (reconciliation["difference"].abs() <= 2).mean()
    assert within >= MIN_WITHIN_TWO_SHARE
