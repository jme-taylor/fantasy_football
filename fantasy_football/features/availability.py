import polars as pl

from fantasy_football.constants import FIT_THRESHOLD, ROLLING_WINDOW


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
        coalesce=True,
    )
    return joined.with_columns(
        pl.col("chance_of_playing_this_round").fill_null(100)
    )


def add_positional_availability(
    data: pl.DataFrame, fit_threshold: int = FIT_THRESHOLD
) -> pl.DataFrame:
    """Count fit same-club, same-position rivals for each player.

    Adds two columns, both partitioned by ``(season, gw, team, position)`` and
    both derived from ``chance_of_playing_this_round`` — FPL's pre-deadline
    snapshot, so neither leaks the current week's outcome:

    - ``fit_rivals_same_pos`` — number of *other* players in the same club and
      position who are fit (``chance_of_playing_this_round >= fit_threshold``).
    - ``fit_rivals_ahead`` — number of those fit players whose ``value`` is
      *strictly* higher than this player's. These are the rivals who directly
      block a start; equal-value players do not block each other.

    Run this AFTER :func:`add_chance_of_playing`, which fills a null chance with
    ``100`` — so a player with no injury record counts as fit.

    Parameters
    ----------
    data : pl.DataFrame
        Player data with ``season``, ``gw``, ``team``, ``position``, ``value``
        and ``chance_of_playing_this_round``.
    fit_threshold : int
        Minimum ``chance_of_playing_this_round`` for a rival to count as fit
        (defaults to ``FIT_THRESHOLD``).

    Returns
    -------
    pl.DataFrame
        ``data`` with ``fit_rivals_same_pos`` and ``fit_rivals_ahead`` added.
    """
    partition = ["season", "gw", "team", "position"]

    data = data.with_columns(
        (
            pl.col("chance_of_playing_this_round").fill_null(100)
            >= fit_threshold
        )
        .cast(pl.Int32)
        .alias("_fit")
    )

    # fit_rivals_same_pos: every fit player in the group, then drop self if fit.
    data = data.with_columns(
        (pl.col("_fit").sum().over(partition) - pl.col("_fit")).alias(
            "fit_rivals_same_pos"
        )
    )

    # fit_rivals_ahead: fit players with a strictly higher value. Collapse the
    # group to one row per distinct value (summing fitness), walk those values
    # high-to-low accumulating fitness, then subtract the current value's own
    # fitness so only *strictly* higher-valued fit players remain. Joining back
    # on (partition, value) fans the per-value count back onto every player.
    per_value = (
        data.group_by(partition + ["value"])
        .agg(pl.col("_fit").sum().alias("_fit_at_value"))
        .sort(partition + ["value"], descending=True)
        .with_columns(
            pl.col("_fit_at_value").cum_sum().over(partition).alias("_cum_fit")
        )
        .with_columns(
            (pl.col("_cum_fit") - pl.col("_fit_at_value")).alias(
                "fit_rivals_ahead"
            )
        )
        .select(partition + ["value", "fit_rivals_ahead"])
    )

    data = data.join(
        per_value, on=partition + ["value"], how="left", coalesce=True
    )
    return data.drop("_fit")
