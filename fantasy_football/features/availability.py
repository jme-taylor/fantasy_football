import polars as pl

from fantasy_football.constants import FIT_THRESHOLD, ROLLING_WINDOW


def add_rolling_minutes(
    data: pl.DataFrame,
    match_stream: pl.DataFrame,
    player_season: pl.DataFrame,
    rolling_window: int = ROLLING_WINDOW,
) -> pl.DataFrame:
    """Add each player's average minutes over their last matches played.

    The window is continuous across seasons and ordered by real kickoff
    time, so a player's first match of a new season sees their final
    matches of the previous one instead of a null. It is keyed on
    ``player_code`` rather than ``element`` because FPL reassigns element
    ids each season -- 841 ids map to more than one player -- so an
    element-keyed window would silently merge different people.

    Unplayed fixtures may appear in ``match_stream`` with a null
    ``minutes``. They do not participate in the window at all: each one
    carries the mean of the player's last ``rolling_window`` *played*
    matches, frozen at the last played match and identical however far
    into the future the fixture is. A naive shifted window would instead
    feed one null in per unplayed fixture, so the value would shrink and
    then vanish -- exactly where a pre-season run needs it most, since
    every gameweek of the season is forward.

    Played rows keep the leak-free shifted window: their value is the
    mean of the matches strictly *before* them, so the row never sees its
    own minutes.

    The window is computed at match grain, then reduced to one value per
    ``(season, gw, element)`` by taking the earliest kickoff in that
    gameweek -- the state entering it. Both legs of a double gameweek
    therefore share a value.

    Parameters
    ----------
    data : pl.DataFrame
        Week-grain rows with ``season``, ``gw`` and ``element``.
    match_stream : pl.DataFrame
        Match-grain rows with ``season``, ``gw``, ``element``,
        ``kickoff_time`` and ``minutes``. Unplayed fixtures carry a null
        ``minutes``.
    player_season : pl.DataFrame
        Identity rows with ``season``, ``element`` and ``player_code``.
    rolling_window : int
        Number of prior matches to average over. Defaults to
        ``ROLLING_WINDOW``.

    Returns
    -------
    pl.DataFrame
        ``data`` with an ``avg_minutes_rolling_{rolling_window}`` column
        added. Rows whose player has no ``player_code``, or no prior
        match, keep a null.
    """
    output_column = f"avg_minutes_rolling_{rolling_window}"
    stream = (
        match_stream.select(
            ["season", "gw", "element", "kickoff_time", "minutes"]
        )
        .join(
            player_season.select(["season", "element", "player_code"]),
            on=["season", "element"],
            how="left",
            coalesce=True,
        )
        .filter(pl.col("player_code").is_not_null())
        .sort("kickoff_time")
        .with_row_index("_row")
    )
    # The frozen value an unplayed fixture inherits: the mean over the last
    # ``rolling_window`` played matches, *including* the most recent one.
    # Computed on the played rows alone so intervening unplayed fixtures
    # cannot push played matches out of the window, then forward-filled
    # along each player's timeline.
    played_carry = stream.filter(pl.col("minutes").is_not_null()).select(
        "_row",
        pl.col("minutes")
        .rolling_mean(window_size=rolling_window, min_periods=1)
        .over("player_code")
        .alias("_carry"),
    )
    stream = stream.join(played_carry, on="_row", how="left", coalesce=True)
    stream = stream.with_columns(
        pl.col("_carry").forward_fill().over("player_code")
    )
    stream = stream.with_columns(
        pl.when(pl.col("minutes").is_not_null())
        .then(
            pl.col("minutes")
            .shift(1)
            .rolling_mean(window_size=rolling_window, min_periods=1)
            .over("player_code")
        )
        .otherwise(pl.col("_carry"))
        .alias(output_column)
    )
    entering = stream.group_by(["season", "gw", "element"]).agg(
        pl.col(output_column).sort_by("kickoff_time").first()
    )
    return data.join(
        entering, on=["season", "gw", "element"], how="left", coalesce=True
    )


def add_chance_of_playing(
    data: pl.DataFrame, availability: pl.DataFrame
) -> pl.DataFrame:
    """Join FPL's point-in-time ``chance_of_playing_this_round`` onto the data.

    ``data`` may already carry a ``chance_of_playing_this_round`` column --
    the player snapshot's copy, injected onto synthetic forward rows since
    ``player_availability`` (fplcache) has no rows for a season still being
    played. When present and non-null, that value wins. Otherwise the
    per-``(season, gw, element)`` availability frame is joined in. Rows with
    neither (a gameweek fplcache did not cover, and no snapshot value)
    default to ``100`` — the same convention as a ``null`` chance, which
    means the player carried no injury doubt.

    Parameters
    ----------
    data : pl.DataFrame
        Player data containing ``season``, ``gw`` and ``element``, and
        optionally an existing ``chance_of_playing_this_round`` column.
    availability : pl.DataFrame
        Availability rows with ``season``, ``gw``, ``element`` and
        ``chance_of_playing_this_round`` (e.g. from
        ``PLAYER_AVAILABILITY.load()``).

    Returns
    -------
    pl.DataFrame
        ``data`` with a ``chance_of_playing_this_round`` column added
        (or overwritten, coalesced as described above).
    """
    has_existing = "chance_of_playing_this_round" in data.columns
    if has_existing:
        data = data.rename(
            {"chance_of_playing_this_round": "_snapshot_chance"}
        )
    joined = data.join(
        availability.select(
            ["season", "gw", "element", "chance_of_playing_this_round"]
        ),
        on=["season", "gw", "element"],
        how="left",
        coalesce=True,
    )
    if has_existing:
        joined = joined.with_columns(
            pl.col("_snapshot_chance")
            .fill_null(pl.col("chance_of_playing_this_round"))
            .alias("chance_of_playing_this_round")
        ).drop("_snapshot_chance")
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


def add_games_played_this_season(data: pl.DataFrame) -> pl.DataFrame:
    """Count each player's prior played gameweeks within the current season.

    Only rows with non-null ``minutes`` are counted as played; future fixtures
    with null ``minutes`` do not increment the count. Shifted by one so the
    current gameweek is excluded, matching :func:`add_rolling_minutes`. This is
    the column that lets a model learn when to stop leaning on last season's
    history: it is 0 at GW1 and grows from there.

    Parameters
    ----------
    data : pl.DataFrame
        Player data containing ``season``, ``gw``, ``element`` and ``minutes``.

    Returns
    -------
    pl.DataFrame
        ``data`` with an integer ``games_played_this_season`` column added.
    """
    return data.sort(["season", "gw"]).with_columns(
        pl.col("minutes")
        .is_not_null()
        .cast(pl.Int64)
        .cum_sum()
        .shift(1)
        .fill_null(0)
        .over(["season", "element"])
        .cast(pl.Int64)
        .alias("games_played_this_season")
    )
