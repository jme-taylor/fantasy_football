"""Who is in the league right now.

``player_week`` is empty before a season kicks off, so it cannot say who is
playing or what they cost. The FPL bootstrap snapshot can, and
``player_season`` turns its element ids into the ``"First Second"`` names the
rest of the pipeline keys on -- the same string ``player_week.name`` carries,
so a roster-derived name joins cleanly against player-week-derived ones.
"""

import polars as pl

from fantasy_football.storage.tables import PLAYER_SEASON, PLAYER_SNAPSHOT

ROSTER_SCHEMA: dict[str, pl.DataType] = {
    "name": pl.Utf8,
    "position": pl.Utf8,
    "team": pl.Utf8,
    "element": pl.Int64,
    "player_code": pl.Int64,
    "value": pl.Int64,
}


def latest_snapshot(snapshot: pl.DataFrame, season: str) -> pl.DataFrame:
    """Return the most recent capture for a season.

    Parameters
    ----------
    snapshot : pl.DataFrame
        Rows from ``PLAYER_SNAPSHOT.load()``.
    season : str
        The season to filter to.

    Returns
    -------
    pl.DataFrame
        The rows sharing the maximum ``captured_at``, or an empty frame.
    """
    seasonal = snapshot.filter(pl.col("season") == season)
    if seasonal.is_empty():
        return seasonal
    newest = seasonal["captured_at"].max()
    return seasonal.filter(pl.col("captured_at") == newest)


def current_roster(season: str) -> pl.DataFrame:
    """Return this season's players with their club, position and price.

    Parameters
    ----------
    season : str
        The season to build a roster for.

    Returns
    -------
    pl.DataFrame
        Columns ``name``, ``position``, ``team``, ``element``,
        ``player_code`` and ``value`` (price in tenths of a million). Empty
        -- but carrying that schema -- when the season has no snapshot
        capture or no identity rows, which callers read as the signal to fall
        back to player-week-derived data.
    """
    empty = pl.DataFrame(schema=ROSTER_SCHEMA)
    snapshot = latest_snapshot(PLAYER_SNAPSHOT.load(), season)
    if snapshot.is_empty():
        return empty
    identity = (
        PLAYER_SEASON.load()
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
        snapshot.select("season", "element", "team", "position", "value")
        .join(identity, on=["season", "element"], how="inner")
        .select(list(ROSTER_SCHEMA))
    )
