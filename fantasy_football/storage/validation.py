"""Check the two match-level sources against each other.

2024-25 and 2025-26 are published by both Vaastav and FCI, which makes
each a check on the other for the handful of stats that genuinely mean
the same thing in both. A rising disagreement rate is the earliest
signal that one source has changed shape or gone stale.

The tables cannot be joined on their own keys -- one is keyed on the FPL
``fixture`` id, the other on FCI's ``match_id`` slug -- so both sides are
summed to ``(season, gw, element)`` first. The FCI side is filtered to
the Premier League, since Vaastav has no cup or European rows to compare
against.
"""

import logging
from typing import TYPE_CHECKING

import polars as pl

from fantasy_football.storage.tables import (
    PLAYER_MATCH_FPL,
    PLAYER_MATCH_OPTA,
)

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

logger = logging.getLogger(__name__)

# (player_match_fpl column, player_match_opta column) pairs that mean the
# same thing in both sources. Deliberately short: most FCI columns have
# no Vaastav counterpart, and most Vaastav columns are FPL scoring
# artefacts FCI never publishes.
OVERLAP_COLUMNS: tuple[tuple[str, str], ...] = (
    ("minutes", "minutes_played"),
    ("goals_scored", "goals"),
    ("assists", "assists"),
    ("saves", "saves"),
    ("penalties_missed", "penalties_missed"),
)

_KEY: list[str] = ["season", "gw", "element"]


def compare_overlap(fpl: pl.DataFrame, opta: pl.DataFrame) -> pl.DataFrame:
    """Compare the two sources over the seasons both cover.

    Parameters
    ----------
    fpl : pl.DataFrame
        ``player_match_fpl`` rows, per fixture.
    opta : pl.DataFrame
        ``player_match_opta`` rows, per fixture, all competitions.

    Returns
    -------
    pl.DataFrame
        One row per compared column, with ``column``, ``compared``,
        ``disagreements`` and ``disagreement_rate``.
    """
    fpl_columns = [pair[0] for pair in OVERLAP_COLUMNS]
    opta_columns = [pair[1] for pair in OVERLAP_COLUMNS]

    left = fpl.group_by(_KEY).agg(
        [pl.col(column).sum() for column in fpl_columns]
    )
    right = (
        opta.filter(pl.col("competition") == "prem")
        .group_by(_KEY)
        .agg([pl.col(column).sum() for column in opta_columns])
    )
    joined = left.join(right, on=_KEY, how="inner", suffix="_opta")

    rows = []
    for fpl_column, opta_column in OVERLAP_COLUMNS:
        # A same-named column (e.g. ``assists``) is suffixed by the join;
        # a differently-named one keeps its own name.
        right_name = (
            f"{opta_column}_opta" if opta_column == fpl_column else opta_column
        )
        if joined.is_empty():
            rows.append(
                {
                    "column": fpl_column,
                    "compared": 0,
                    "disagreements": 0,
                    "disagreement_rate": 0.0,
                }
            )
            continue
        disagreements = int(
            joined.filter(pl.col(fpl_column) != pl.col(right_name)).height
        )
        rows.append(
            {
                "column": fpl_column,
                "compared": joined.height,
                "disagreements": disagreements,
                "disagreement_rate": disagreements / joined.height,
            }
        )
    return pl.DataFrame(rows)


def validate_overlap(connection: "DuckDBPyConnection") -> pl.DataFrame:
    """Run the overlap comparison against the stored tables.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        An open connection.

    Returns
    -------
    pl.DataFrame
        The comparison report from :func:`compare_overlap`.
    """
    report = compare_overlap(
        PLAYER_MATCH_FPL.load(connection), PLAYER_MATCH_OPTA.load(connection)
    )
    for row in report.iter_rows(named=True):
        logger.info(
            "Overlap %s: %d compared, %d disagree (%.2f%%)",
            row["column"],
            row["compared"],
            row["disagreements"],
            row["disagreement_rate"] * 100,
        )
    return report
