import logging

import polars as pl

from fantasy_football.constants import DATA_FOLDER, ROLLING_WINDOW
from fantasy_football.storage.database import (
    load_player_season,
    load_player_week,
)

logger = logging.getLogger(__name__)

TRANSFORMED_DATA_FOLDER = DATA_FOLDER.joinpath("transformed")

KNOWN_POSITIONS: tuple[str, ...] = ("GK", "DEF", "MID", "FWD")


def rolling_column_name(rolling_column: str, rolling_window: int) -> str:
    """Return the output column name produced by a rolling-average step."""
    return f"{rolling_column}_rolling_{rolling_window}"


def load_gw_data() -> pl.DataFrame:
    """Load all player-week data across every season, with ``player_code``.

    Reads the entire ``player_week`` table and left-joins the ``player_season``
    identity dimension, so downstream windows can partition on a key that is
    stable across seasons rather than on the display name. Positions are already
    normalised (``GKP`` collapsed to ``GK``) at write time.

    Returns
    -------
    pl.DataFrame
        One row per (player, gameweek) for all seasons, with ``season``, ``gw``
        and ``player_code`` columns.
    """
    return load_player_week().join(
        load_player_season().select(["season", "element", "player_code"]),
        on=["season", "element"],
        how="left",
        coalesce=True,
    )


def create_rolling_average_column(
    data: pl.DataFrame,
    grouping_columns: list[str],
    rolling_column: str,
    rolling_window: int,
) -> pl.DataFrame:
    """Create a rolling average column over a given window size.

    The frame is sorted by season and gameweek, then ``rolling_column`` is
    averaged over ``rolling_window`` rows within each ``grouping_columns``
    partition.

    Parameters
    ----------
    data : pl.DataFrame
        The data to calculate the rolling average on.
    grouping_columns : list[str]
        The columns to partition the window by. Include ``season`` to stop a
        window spanning the summer break.
    rolling_column : str
        The column to calculate the rolling average on.
    rolling_window : int
        The window size to calculate the rolling average over.

    Returns
    -------
    pl.DataFrame
        The original dataframe with the rolling average column added.

    """
    rolling_average_column_name = rolling_column_name(
        rolling_column, rolling_window
    )
    data = data.sort(["season", "gw"]).with_columns(
        pl.col(rolling_column)
        .rolling_mean(window_size=rolling_window, min_periods=1)
        .over(grouping_columns)
        .alias(rolling_average_column_name)
    )
    return data


def fill_missing_values_by_position(
    data: pl.DataFrame, column_to_fill: str
) -> pl.DataFrame:
    """Fill missing values in a column by the average of the position.

    Parameters
    ----------
    data : pl.DataFrame
        The data to fill missing values on.
    column_to_fill : str
        The column to fill missing values for.

    Returns
    -------
    pl.DataFrame
        The original dataframe with the missing values filled.

    """
    positions_in_data = set(data.get_column("position").unique().to_list())
    unknown_positions = positions_in_data - set(KNOWN_POSITIONS)
    if unknown_positions:
        logger.warning(
            "fill_missing_values_by_position encountered unknown "
            "position(s) %s; rows with these positions will not have "
            "nulls in '%s' filled.",
            sorted(unknown_positions),
            column_to_fill,
        )
    for position in KNOWN_POSITIONS:
        position_data = data.filter(pl.col("position") == position)
        position_average = (
            position_data.select(pl.col(column_to_fill)).mean().item(0, 0)
        )
        data = data.with_columns(
            pl.when(
                (pl.col("position") == position)
                & (pl.col(column_to_fill).is_null())
            )
            .then(pl.lit(position_average))
            .otherwise(pl.col(column_to_fill))
            .alias(column_to_fill)
        )
    return data


def create_rolling_points_data(
    current_season: str, rolling_window: int = ROLLING_WINDOW
) -> None:
    """Create a rolling average column for player points over a given window.

    This function first creates a whole history of game week data by loading
    all previous seasons and the current season. It then calculates the rolling
    average of total points over a window size of `rolling_window`. Finally,
    it fills any missing values by the average of the position. Doesn't return
    anything, but writes the data to a CSV file in the transformed data folder
    named "rolling_points.csv".

    Parameters
    ----------
    current_season : str
        The current season we are working with. Should be in a YYYY-YY format,
        e.g. "2020-21"
    rolling_window : int, optional
        The window size to calculate the rolling average over. Defaults to 5.

    """
    gw_data = load_gw_data()
    rolling_column = rolling_column_name("total_points", rolling_window)
    gw_data = create_rolling_average_column(
        gw_data, ["player_code", "season"], "total_points", rolling_window
    )
    gw_data = fill_missing_values_by_position(gw_data, rolling_column)
    TRANSFORMED_DATA_FOLDER.mkdir(exist_ok=True, parents=True)
    gw_data.write_csv(TRANSFORMED_DATA_FOLDER.joinpath("rolling_points.csv"))
