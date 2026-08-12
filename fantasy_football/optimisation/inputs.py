"""What the optimiser needs to know, assembled from the database.

Predicted points come from the per-position models' forward predictions in
``points_prediction``; everything else about a player -- name, club,
position, price -- comes from the current roster. Keeping both halves in one
loader means the optimiser never has to reconcile two sources for the same
player.
"""

import logging
from typing import TYPE_CHECKING

import polars as pl

from fantasy_football.features.roster import current_roster
from fantasy_football.storage.tables import FORWARD_KIND, POINTS_PREDICTION

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

logger = logging.getLogger(__name__)

INPUT_COLUMNS: list[str] = [
    "element",
    "gw",
    "name",
    "position",
    "team",
    "value",
    "predicted_points",
]


class MissingPositionPredictionsError(RuntimeError):
    """Raised when a position has no forward predictions in a gameweek."""


def _forward_points(
    season: str,
    weeks: list[int],
    connection: "DuckDBPyConnection | None",
) -> pl.DataFrame:
    """Return forward predictions summed from match to gameweek grain.

    Parameters
    ----------
    season : str
        The season to read.
    weeks : list[int]
        The gameweeks being optimised.
    connection : duckdb.DuckDBPyConnection | None
        An open connection, or None to open one.

    Returns
    -------
    pl.DataFrame
        One row per (element, gw) with summed ``predicted_points``.
    """
    return (
        POINTS_PREDICTION.load(connection)
        .filter(
            (pl.col("season") == season)
            & (pl.col("prediction_kind") == FORWARD_KIND)
            & pl.col("gw").is_in(weeks)
        )
        .group_by("element", "gw")
        .agg(pl.col("predicted_points").sum())
    )


def _check_coverage(frame: pl.DataFrame, weeks: list[int]) -> None:
    """Fail on an absent position, warn on absent individuals.

    A position with no predictions at all in a gameweek means its model did
    not run. The optimiser would still solve -- it would just field that
    position on zeroes -- so this has to raise rather than warn. A single
    player missing a row is ordinary: a blank gameweek, a new signing, or a
    player the model could not build features for.

    Parameters
    ----------
    frame : pl.DataFrame
        The joined universe, carrying ``has_prediction``.
    weeks : list[int]
        The gameweeks being optimised.

    Raises
    ------
    MissingPositionPredictionsError
        When a (position, gameweek) pair has no predictions at all.
    """
    covered = (
        frame.group_by("position", "gw")
        .agg(pl.col("has_prediction").sum().alias("covered"))
        .filter(pl.col("covered") == 0)
        .sort("position", "gw")
    )
    if not covered.is_empty():
        pairs = covered.select("position", "gw").rows()
        raise MissingPositionPredictionsError(
            f"No forward predictions for {pairs} as (position, gameweek). "
            f"A whole position is unscored, so the optimiser would field it "
            f"on zeroes and return a plan that looks valid but is not. Run "
            f"predict_forward for that position, and check it has a model "
            f"promoted to its production alias."
        )

    by_position = (
        frame.group_by("position")
        .agg(
            pl.col("has_prediction").sum().alias("covered"),
            pl.len().alias("total"),
        )
        .sort("position")
    )
    for row in by_position.iter_rows(named=True):
        if row["covered"] == row["total"]:
            continue
        logger.warning(
            "%s: %d of %d player-gameweeks have no forward prediction "
            "across gws %s; they are scored 0.0.",
            row["position"],
            row["total"] - row["covered"],
            row["total"],
            weeks,
        )


def load_optimiser_inputs(
    season: str,
    weeks: list[int],
    connection: "DuckDBPyConnection | None" = None,
) -> pl.DataFrame:
    """Return one row per roster player per gameweek, ready to optimise.

    The universe is anchored on the roster rather than on the predictions, so
    every player who can be bought is representable even when nothing has
    been predicted for them. Their predicted points are then 0.0, which the
    optimiser reads as "will not score" -- the same thing a blank gameweek
    means.

    Parameters
    ----------
    season : str
        The season to optimise, e.g. ``"2026-27"``.
    weeks : list[int]
        The gameweeks in the horizon.
    connection : duckdb.DuckDBPyConnection | None, optional
        An open connection. When None, one is opened per table read.

    Returns
    -------
    pl.DataFrame
        Columns ``element``, ``gw``, ``name``, ``position``, ``team``,
        ``value`` and ``predicted_points``.

    Raises
    ------
    ValueError
        When the season has no roster, so no player has a price.
    MissingPositionPredictionsError
        When a position has no forward predictions in a gameweek.
    """
    roster = current_roster(season, connection)
    if roster.is_empty():
        raise ValueError(
            f"Season {season} has no roster, so no player has a price or a "
            f"club. Load the player snapshot and player identity for the "
            f"season before optimising."
        )

    universe = roster.select(
        "element", "name", "position", "team", "value"
    ).join(pl.DataFrame({"gw": weeks}, schema={"gw": pl.Int64}), how="cross")

    joined = universe.join(
        _forward_points(season, weeks, connection).with_columns(
            pl.lit(1).alias("has_prediction")
        ),
        on=["element", "gw"],
        how="left",
    ).with_columns(
        pl.col("predicted_points").fill_null(0.0),
        pl.col("has_prediction").fill_null(0),
    )

    _check_coverage(joined, weeks)
    return joined.select(INPUT_COLUMNS).sort("gw", "element")


def forward_gameweeks(
    season: str, connection: "DuckDBPyConnection | None" = None
) -> list[int]:
    """Return the gameweeks that have forward predictions, in order.

    Parameters
    ----------
    season : str
        The season to read.
    connection : duckdb.DuckDBPyConnection | None, optional
        An open connection. When None, one is opened per table read.

    Returns
    -------
    list[int]
        Sorted gameweek numbers, empty when nothing has been predicted.
    """
    stored = POINTS_PREDICTION.load(connection).filter(
        (pl.col("season") == season)
        & (pl.col("prediction_kind") == FORWARD_KIND)
    )
    return sorted(stored["gw"].unique().to_list())
