"""Build the row universe for scoring fixtures that have not been played.

The minutes model's training frame starts from ``player_match`` -- the
actuals table -- so a fixture only enters it once its result is known.
This module builds the mirror-image frame: a snapshot of every player's
current state crossed with the fixtures still to come, shaped exactly
like the played rows so the same feature code serves both.
"""

import polars as pl

# Columns a forward row needs to stand in for a player_week row.
_WEEK_COLUMNS = [
    "season",
    "gw",
    "element",
    "position",
    "team",
    "value",
    "minutes",
    "chance_of_playing_this_round",
]


def last_played_gw(player_week: pl.DataFrame, season: str) -> int:
    """Return the latest gameweek with stored player-week rows.

    Parameters
    ----------
    player_week : pl.DataFrame
        Player-week rows with ``season`` and ``gw``.
    season : str
        The season to inspect.

    Returns
    -------
    int
        The highest stored gameweek, or 0 when the season has no rows --
        the pre-season case, where the aggregate is null.
    """
    stored = player_week.filter(pl.col("season") == season)
    if stored.is_empty():
        return 0
    return int(stored["gw"].max() or 0)


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


def build_forward_fixtures(
    snapshot: pl.DataFrame,
    team_fixture: pl.DataFrame,
    season: str,
    from_gw: int,
    team_name_to_id: dict[str, int],
) -> pl.DataFrame:
    """Cross a player snapshot with the fixtures still to be played.

    Joins on club name, so every player inherits their club's remaining
    fixtures. A double gameweek naturally yields two rows and a blank
    yields none -- which is the only way fixture scheduling reaches the
    model, since no feature depends on the opponent.

    Parameters
    ----------
    snapshot : pl.DataFrame
        One capture's rows, from :func:`latest_snapshot`.
    team_fixture : pl.DataFrame
        Fixture rows with ``season``, ``gw``, ``team``, ``opposition``
        and ``kickoff_time``.
    season : str
        The season being scored.
    from_gw : int
        Lowest gameweek to include; gameweeks below it have been played.
    team_name_to_id : dict[str, int]
        Maps a club name to its FPL team id, matching the ``opponent``
        column stored on ``player_match``.

    Returns
    -------
    pl.DataFrame
        Match-grain rows with a null ``minutes``, one per player-fixture.

    Raises
    ------
    ValueError
        If an ``opposition`` name has no entry in ``team_name_to_id``.
        ``opponent`` is part of ``minutes_prediction``'s primary key and
        therefore NOT NULL, so an unmapped name would otherwise surface
        much later as an opaque DuckDB constraint violation.
    """
    upcoming = team_fixture.filter(
        (pl.col("season") == season) & (pl.col("gw") >= from_gw)
    ).select(["season", "gw", "team", "opposition", "kickoff_time"])
    joined = snapshot.join(upcoming, on=["season", "team"], how="inner")
    unmapped = sorted(
        (
            name
            for name in joined["opposition"].unique().to_list()
            if name not in team_name_to_id
        ),
        key=str,
    )
    if unmapped:
        raise ValueError(
            f"No FPL team id for opposition {unmapped} in {season}; "
            f"known names are {sorted(team_name_to_id)}. Fix the club-name "
            f"mapping before scoring -- opponent cannot be null."
        )
    return joined.select(
        "season",
        "gw",
        "element",
        pl.col("opposition")
        .replace(team_name_to_id, default=None)
        .cast(pl.Int64)
        .alias("opponent"),
        "kickoff_time",
        pl.lit(None, dtype=pl.Int64).alias("minutes"),
        "position",
        "team",
        "value",
        "chance_of_playing_this_round",
    )


def forward_player_weeks(forward_fixtures: pl.DataFrame) -> pl.DataFrame:
    """Reduce match-grain forward rows to the player-week grain.

    ``player_week`` collapses double gameweeks, so the synthetic rows
    must too, or the per-gameweek feature counts would be wrong.

    Parameters
    ----------
    forward_fixtures : pl.DataFrame
        Output of :func:`build_forward_fixtures`.

    Returns
    -------
    pl.DataFrame
        One row per ``(season, gw, element)`` in ``player_week`` shape.
    """
    return (
        forward_fixtures.sort("kickoff_time")
        .unique(subset=["season", "gw", "element"], keep="first")
        .select(_WEEK_COLUMNS)
    )
