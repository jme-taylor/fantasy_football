import polars as pl

from fantasy_football.constants import ROLLING_WINDOW


def add_rolling_minutes(
    data: pl.DataFrame, rolling_window: int = ROLLING_WINDOW
) -> pl.DataFrame:
    """Add each player's average minutes over their prior gameweeks.

    For every ``(season, element)`` group the ``minutes`` column is averaged
    over the previous ``rolling_window`` gameweeks. The current gameweek is
    excluded (the window is shifted by one) so the feature only uses games
    played before the gameweek being scored — this avoids leaking the current
    week's minutes into a minutes/availability model.

    The window partitions by ``season`` as well as ``element`` because FPL
    reassigns ``element`` ids each season; partitioning by element alone would
    pool different players and bleed minutes across the summer break. Rows
    earlier than ``rolling_window`` use whatever prior games exist
    (``min_periods=1``); a player's first game of a season has no prior data
    and is therefore null.

    Parameters
    ----------
    data : pl.DataFrame
        Player-week data containing ``season``, ``gw``, ``element`` and
        ``minutes``.
    rolling_window : int
        Number of prior gameweeks to average over (defaults to
        ``ROLLING_WINDOW``).

    Returns
    -------
    pl.DataFrame
        The original dataframe with an ``avg_minutes_rolling_{rolling_window}``
        column added.

    """
    output_column = f"avg_minutes_rolling_{rolling_window}"
    return data.sort(["season", "gw"]).with_columns(
        pl.col("minutes")
        .shift(1)
        .rolling_mean(window_size=rolling_window, min_periods=1)
        .over(["season", "element"])
        .alias(output_column)
    )


def add_chance_of_playing(
    data: pl.DataFrame, availability: pl.DataFrame
) -> pl.DataFrame:
    """Join FPL's point-in-time ``chance_of_playing_this_round`` onto the data.

    Left-joins the per-``(season, gw, element)`` availability frame onto
    ``data``. Rows with no availability record (a gameweek fplcache did not
    cover) default to ``100`` — the same convention as a ``null`` chance, which
    means the player carried no injury doubt.

    Parameters
    ----------
    data : pl.DataFrame
        Player data containing ``season``, ``gw`` and ``element``.
    availability : pl.DataFrame
        Availability rows with ``season``, ``gw``, ``element`` and
        ``chance_of_playing_this_round`` (e.g. from ``load_player_availability``).

    Returns
    -------
    pl.DataFrame
        ``data`` with a ``chance_of_playing_this_round`` column added.
    """
    joined = data.join(
        availability.select(
            ["season", "gw", "element", "chance_of_playing_this_round"]
        ),
        on=["season", "gw", "element"],
        how="left",
    )
    return joined.with_columns(
        pl.col("chance_of_playing_this_round").fill_null(100)
    )
