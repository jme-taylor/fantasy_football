"""The stored tables, declared as specs.

One ``Table`` per stored table. Adding a table means adding a spec here
and an entry in ``TABLES``; nothing else in the storage layer changes.
"""

import duckdb
import polars as pl

from fantasy_football.storage.table import Table

# Attributes that are properties of the person, not of the season. One
# observation anywhere determines them everywhere, so they are filled
# across every season sharing a player_code on read. Everything else is
# season-varying and must never be propagated -- team_join_date in
# particular is the signal that a player is a recent signing.
PLAYER_SEASON_STATIC_COLUMNS: list[str] = [
    "first_name",
    "second_name",
    "birth_date",
    "region",
]


def gkp_to_gk(frame: pl.DataFrame) -> pl.DataFrame:
    """Collapse the legacy ``GKP`` position label to ``GK``.

    Parameters
    ----------
    frame : pl.DataFrame
        A frame carrying a ``position`` column.

    Returns
    -------
    pl.DataFrame
        The frame with ``GKP`` rewritten to ``GK``.
    """
    return frame.with_columns(
        pl.when(pl.col("position") == "GKP")
        .then(pl.lit("GK"))
        .otherwise(pl.col("position"))
        .alias("position")
    )


def propagate_static_columns(frame: pl.DataFrame) -> pl.DataFrame:
    """Fill person-level attributes across every season for each player.

    Within each ``player_code`` the first non-null value of each static
    column is used. Rows with a null ``player_code`` are left untouched,
    since there is no identity to propagate along. This matters because
    ``birth_date`` is only observable from 2022-23 onwards while the data
    reaches back to 2016-17.

    Parameters
    ----------
    frame : pl.DataFrame
        Player-season rows.

    Returns
    -------
    pl.DataFrame
        The frame with static columns filled.
    """
    return frame.with_columns(
        [
            pl.when(pl.col("player_code").is_null())
            .then(pl.col(column))
            .otherwise(pl.col(column).drop_nulls().first().over("player_code"))
            .alias(column)
            for column in PLAYER_SEASON_STATIC_COLUMNS
        ]
    )


PLAYER_WEEK = Table(
    name="player_week",
    schema={
        "season": pl.Utf8,
        "gw": pl.Int64,
        "element": pl.Int64,
        "name": pl.Utf8,
        "position": pl.Utf8,
        "team": pl.Utf8,
        "bonus": pl.Int64,
        "minutes": pl.Int64,
        "round": pl.Int64,
        "total_points": pl.Int64,
        "value": pl.Int64,
    },
    primary_key=("season", "gw", "element"),
    order_by=("season", "gw", "element"),
    normalise=gkp_to_gk,
)

TEAM_FIXTURE = Table(
    name="team_fixture",
    schema={
        "season": pl.Utf8,
        "gw": pl.Int64,
        "team": pl.Utf8,
        "is_home": pl.Boolean,
        "opposition": pl.Utf8,
        "kickoff_time": pl.Datetime("us"),
    },
    primary_key=("season", "gw", "team", "opposition"),
    order_by=("season", "gw", "team"),
)

PLAYER_MATCH = Table(
    name="player_match",
    schema={
        "season": pl.Utf8,
        "gw": pl.Int64,
        "element": pl.Int64,
        "opponent": pl.Int64,
        "is_home": pl.Boolean,
        "minutes": pl.Int64,
        "total_points": pl.Int64,
        "kickoff_time": pl.Datetime("us"),
    },
    primary_key=("season", "gw", "element", "opponent"),
    order_by=("season", "gw", "element", "opponent"),
)

PLAYER_AVAILABILITY = Table(
    name="player_availability",
    schema={
        "season": pl.Utf8,
        "gw": pl.Int64,
        "element": pl.Int64,
        "chance_of_playing_this_round": pl.Int64,
    },
    primary_key=("season", "gw", "element"),
    order_by=("season", "gw", "element"),
)

MINUTES_PREDICTION = Table(
    name="minutes_prediction",
    schema={
        "season": pl.Utf8,
        "gw": pl.Int64,
        "element": pl.Int64,
        "opponent": pl.Int64,
        "p_zero": pl.Float64,
        "p_partial": pl.Float64,
        "p_sixty_plus": pl.Float64,
        "expected_minutes": pl.Float64,
        "model_version": pl.Utf8,
    },
    primary_key=("season", "gw", "element", "opponent"),
    order_by=("season", "gw", "element", "opponent"),
)

PLAYER_SNAPSHOT = Table(
    name="player_snapshot",
    schema={
        "season": pl.Utf8,
        "captured_at": pl.Datetime("us"),
        "element": pl.Int64,
        "value": pl.Int64,
        "team": pl.Utf8,
        "position": pl.Utf8,
        "chance_of_playing_this_round": pl.Int64,
    },
    primary_key=("season", "captured_at", "element"),
    order_by=("season", "captured_at", "element"),
    normalise=gkp_to_gk,
)

PLAYER_SEASON = Table(
    name="player_season",
    schema={
        "season": pl.Utf8,
        "element": pl.Int64,
        "player_code": pl.Int64,
        "web_name": pl.Utf8,
        "first_name": pl.Utf8,
        "second_name": pl.Utf8,
        "position": pl.Utf8,
        "team_code": pl.Int64,
        "birth_date": pl.Date,
        "region": pl.Int64,
        "team_join_date": pl.Date,
    },
    primary_key=("season", "element"),
    order_by=("season", "element"),
    enrich=propagate_static_columns,
)

# Every stored table. ``get_connection`` and ``reset_database`` loop over
# this, so a new table cannot be forgotten by either.
TABLES: tuple[Table, ...] = (
    PLAYER_WEEK,
    TEAM_FIXTURE,
    PLAYER_MATCH,
    PLAYER_AVAILABILITY,
    MINUTES_PREDICTION,
    PLAYER_SEASON,
    PLAYER_SNAPSHOT,
)


def minutes_prediction_versions(
    connection: duckdb.DuckDBPyConnection,
    seasons: list[str] | None = None,
) -> set[str]:
    """Return the distinct ``model_version`` values stored.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        An open connection.
    seasons : list[str] | None, optional
        When given, restrict to these seasons (used to gate the historic
        backfill on the versions already stored for historic seasons).

    Returns
    -------
    set[str]
        Distinct non-null model versions.
    """
    if seasons is None:
        rows = connection.execute(
            "SELECT DISTINCT model_version FROM minutes_prediction"
        ).fetchall()
    else:
        placeholders = ", ".join("?" for _ in seasons)
        rows = connection.execute(
            "SELECT DISTINCT model_version FROM minutes_prediction "
            f"WHERE season IN ({placeholders})",
            seasons,
        ).fetchall()
    return {row[0] for row in rows if row[0] is not None}
