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
from fantasy_football.fpl_types import PlayerBreakdown
from fantasy_football.modelling.forward import last_played_gw
from fantasy_football.storage.tables import (
    FORWARD_KIND,
    PLAYER_WEEK,
    POINTS_COMPONENT,
    POINTS_PREDICTION,
)

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
    "is_departed",
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

    Departed players are in the universe too, carrying ``is_departed`` and
    no predictions. A squad can still be holding one, and the optimiser
    has to be able to sell what it cannot buy.

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
        ``value``, ``predicted_points`` and ``is_departed``.

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
        "element", "name", "position", "team", "value", "is_departed"
    ).join(pl.DataFrame({"gw": weeks}, schema={"gw": pl.Int64}), how="cross")

    joined = universe.join(
        _forward_points(season, weeks, connection).with_columns(
            pl.lit(1).alias("has_prediction")
        ),
        on=["element", "gw"],
        how="left",
    ).with_columns(
        # A departed player keeps whatever was predicted for him before he
        # left, so his points are zeroed rather than read from the join.
        pl.when(pl.col("is_departed"))
        .then(0.0)
        .otherwise(pl.col("predicted_points").fill_null(0.0))
        .alias("predicted_points"),
        pl.col("has_prediction").fill_null(0),
    )

    # Departed players are deliberately unscored, so counting them as
    # missing coverage would warn about 90-odd players every run.
    _check_coverage(joined.filter(~pl.col("is_departed")), weeks)
    return joined.select(INPUT_COLUMNS).sort("gw", "element")


def load_component_breakdown(
    season: str,
    weeks: list[int],
    connection: "DuckDBPyConnection | None" = None,
) -> dict[tuple[int, int], PlayerBreakdown]:
    """Return each player-gameweek's predicted points split by component.

    Components are stored per fixture leg and summed to gameweek grain, to
    match the ``predicted_points`` the optimiser scored the player on. The
    leg count comes back alongside so a double gameweek can be marked as
    one -- otherwise its numbers read as a single match's.

    Parameters
    ----------
    season : str
        The season to read.
    weeks : list[int]
        The gameweeks in the horizon.
    connection : duckdb.DuckDBPyConnection | None, optional
        An open connection. When None, one is opened per table read.

    Returns
    -------
    dict[tuple[int, int], PlayerBreakdown]
        Keyed by (element, gameweek). A player with no component rows is
        absent rather than present with zeroes.
    """
    stored = POINTS_COMPONENT.load(connection).filter(
        (pl.col("season") == season)
        & (pl.col("prediction_kind") == FORWARD_KIND)
        & pl.col("gw").is_in(weeks)
    )
    summed = stored.group_by("element", "gw", "component").agg(
        pl.col("points").sum()
    )
    fixtures = stored.group_by("element", "gw").agg(
        pl.col("opponent").n_unique().alias("fixtures")
    )

    counts = {
        (row["element"], row["gw"]): row["fixtures"]
        for row in fixtures.iter_rows(named=True)
    }
    components: dict[tuple[int, int], dict[str, float]] = {}
    for row in summed.iter_rows(named=True):
        key = (row["element"], row["gw"])
        components.setdefault(key, {})[row["component"]] = row["points"]

    return {
        key: PlayerBreakdown(components=points, fixtures=counts[key])
        for key, points in components.items()
    }


def forward_gameweeks(
    season: str, connection: "DuckDBPyConnection | None" = None
) -> list[int]:
    """Return the unplayed gameweeks that have forward predictions.

    Played gameweeks are excluded even though their forward rows are
    still stored. ``replace_partition`` rewrites forward predictions only
    from the first unplayed gameweek up, which deliberately freezes each
    week's pre-deadline forecast in place. Those frozen rows are a record
    of what was predicted, not something left to plan.

    Parameters
    ----------
    season : str
        The season to read.
    connection : duckdb.DuckDBPyConnection | None, optional
        An open connection. When None, one is opened per table read.

    Returns
    -------
    list[int]
        Sorted gameweek numbers, empty when nothing is left to predict.
    """
    played = last_played_gw(PLAYER_WEEK.load(connection), season)
    stored = POINTS_PREDICTION.load(connection).filter(
        (pl.col("season") == season)
        & (pl.col("prediction_kind") == FORWARD_KIND)
        & (pl.col("gw") > played)
    )
    return sorted(stored["gw"].unique().to_list())
