import polars as pl


def add_team_value(data: pl.DataFrame) -> pl.DataFrame:
    """Add a player's team value and their share of it for each gameweek.

    For every ``(season, gw, team)`` group the total ``value`` of all players
    is summed and broadcast back onto each player row as ``team_value``. Each
    player's ``value_share_of_team`` is then their individual ``value`` divided
    by that total.

    Partitioning includes ``season`` because ``gw`` repeats across seasons;
    omitting it would pool different seasons' gameweeks together.

    Parameters
    ----------
    data : pl.DataFrame
        Player-week data containing ``season``, ``gw``, ``team`` and ``value``.

    Returns
    -------
    pl.DataFrame
        The original dataframe with ``team_value`` and ``value_share_of_team``
        columns added.

    """
    data = data.with_columns(
        pl.col("value")
        .sum()
        .over(["season", "gw", "team"])
        .alias("team_value")
    )
    return data.with_columns(
        (pl.col("value") / pl.col("team_value")).alias("value_share_of_team")
    )
