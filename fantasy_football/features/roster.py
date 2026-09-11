"""Who is in the league right now.

``player_week`` is empty before a season kicks off, so it cannot say who is
playing or what they cost. The FPL bootstrap snapshot can, and
``player_season`` turns its element ids into the ``"First Second"`` names the
rest of the pipeline keys on -- the same string ``player_week.name`` carries,
so a roster-derived name joins cleanly against player-week-derived ones.
"""

from typing import TYPE_CHECKING

import polars as pl

from fantasy_football.storage.tables import PLAYER_SEASON, PLAYER_SNAPSHOT

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

# FPL's status code for a player who is no longer in the league. It is
# kept in bootstrap-static for weeks after a transfer out, priced and
# clubbed as if nothing happened, so it is the only signal that they have
# gone -- chance_of_playing_this_round stays null for them.
DEPARTED_STATUS = "u"

ROSTER_SCHEMA: dict[str, pl.DataType] = {
    "name": pl.Utf8,
    "position": pl.Utf8,
    "team": pl.Utf8,
    "element": pl.Int64,
    "player_code": pl.Int64,
    "value": pl.Int64,
    "is_departed": pl.Boolean,
}


def is_departed() -> pl.Expr:
    """Return the expression marking players who have left the league.

    A null ``status`` -- every capture taken before the column existed --
    counts as present.

    Returns
    -------
    pl.Expr
        Boolean expression over a frame carrying ``status``.
    """
    return (pl.col("status") == DEPARTED_STATUS).fill_null(False)


def latest_snapshot(
    snapshot: pl.DataFrame, season: str, keep_departed: bool = False
) -> pl.DataFrame:
    """Return the most recent capture for a season.

    Departed players are dropped by default: they cannot be bought and
    will not play, so scoring them wastes a forward prediction and
    invites the optimiser to plan around a player who does not exist.
    ``keep_departed`` is for the roster, which must still know a squad
    player who has left so the optimiser can sell them.

    Parameters
    ----------
    snapshot : pl.DataFrame
        Rows from "PLAYER_SNAPSHOT.load()".
    season : str
        The season to filter to.
    keep_departed : bool, optional
        Keep players FPL has marked as departed. Defaults to False.

    Returns
    -------
    pl.DataFrame
        The rows sharing the maximum "captured_at"
    """
    seasonal = snapshot.filter(pl.col("season") == season)
    newest = seasonal["captured_at"].max()
    latest = seasonal.filter(pl.col("captured_at") == newest)
    return latest if keep_departed else latest.filter(~is_departed())


def current_roster(
    season: str, connection: "DuckDBPyConnection | None" = None
) -> pl.DataFrame:
    """Return this season's players with their club, position and price.

    Parameters
    ----------
    season : str
        The season to build a roster for.
    connection : duckdb.DuckDBPyConnection | None, optional
        An open connection. When None, one is opened per table read.

    Returns
    -------
    pl.DataFrame
        Columns ``name``, ``position``, ``team``, ``element``,
        ``player_code``, ``value`` (price in tenths of a million) and
        ``is_departed`` -- true for a player FPL has marked as gone, who
        can still be owned and sold but never bought. Empty
        -- but carrying that schema -- when the season has no snapshot
        capture or no identity rows, which callers read as the signal to fall
        back to player-week-derived data.
    """
    empty = pl.DataFrame(schema=ROSTER_SCHEMA)
    snapshot = latest_snapshot(
        PLAYER_SNAPSHOT.load(connection), season, keep_departed=True
    )
    if snapshot.is_empty():
        return empty
    identity = (
        PLAYER_SEASON.load(connection)
        .filter(pl.col("season") == season)
        .select(
            "season",
            "element",
            "player_code",
            (pl.col("first_name") + pl.lit(" ") + pl.col("second_name")).alias(
                "name"
            ),
        )
    )
    if identity.is_empty():
        return empty
    return (
        snapshot.select(
            "season",
            "element",
            "team",
            "position",
            "value",
            is_departed().alias("is_departed"),
        )
        .join(identity, on=["season", "element"], how="inner")
        .select(list(ROSTER_SCHEMA))
    )
