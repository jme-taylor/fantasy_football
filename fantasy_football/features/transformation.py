import logging

import polars as pl

from fantasy_football.constants import (
    DATA_FOLDER,
    ROLLING_WINDOW,
    VASTAAV_BRIDGE_SEASONS,
)

logger = logging.getLogger(__name__)

RAW_DATA_FOLDER = DATA_FOLDER.joinpath("raw")
TRANSFORMED_DATA_FOLDER = DATA_FOLDER.joinpath("transformed")

KNOWN_POSITIONS: tuple[str, ...] = ("GK", "DEF", "MID", "FWD")


def rolling_column_name(rolling_column: str, rolling_window: int) -> str:
    """Return the output column name produced by a rolling-average step."""
    return f"{rolling_column}_rolling_{rolling_window}"


def _load_season_merged_gw(season: str) -> pl.DataFrame:
    """Read a single season's ``merged_gw.csv`` and tag it with the season.

    Parameters
    ----------
    season : str
        The season whose per-season file to read, in YYYY-YY format,
        e.g. "2024-25".

    Returns
    -------
    pl.DataFrame
        The season's gameweek rows with a ``gw`` column and a literal
        ``season`` column.
    """
    season_columns = [
        "name",
        "position",
        "team",
        "bonus",
        "element",
        "minutes",
        "round",
        "total_points",
        "GW",
    ]
    return (
        pl.read_csv(
            RAW_DATA_FOLDER.joinpath(season, "gws", "merged_gw.csv"),
            columns=season_columns,
        )
        .rename({"GW": "gw"})
        .with_columns(pl.lit(season).alias("season"))
    )


def load_gw_data(current_season: str) -> pl.DataFrame:
    """Take all season's gameweek data and join to current season data.

    This function reads the "cleaned_merged_seasons.csv" file, which is the
    merged data from older complete seasons, then appends any Vaastav "bridge"
    seasons not yet folded into that aggregate (see
    ``VASTAAV_BRIDGE_SEASONS``), and finally appends the game week data for the
    current season, giving a complete dataset of all game weeks for all
    seasons. Bridge seasons whose file is absent are skipped with a warning;
    the current season's file is required.

    Parameters
    ----------
    current_season : str
        The current season we are working with. Should be in a YYYY-YY format,
        e.g. "2020-21"

    Returns
    -------
        pl.DataFrame: A DataFrame containing all game week data for all seasons

    """
    previous_seasons_columns = [
        "season_x",
        "name",
        "position",
        "team_x",
        "bonus",
        "element",
        "minutes",
        "round",
        "total_points",
        "GW",
    ]
    previous_seasons = pl.read_csv(
        RAW_DATA_FOLDER.joinpath("cleaned_merged_seasons.csv"),
        columns=previous_seasons_columns,
    ).rename({"season_x": "season", "GW": "gw", "team_x": "team"})

    frames = [previous_seasons]
    for season in VASTAAV_BRIDGE_SEASONS:
        if season == current_season:
            continue
        bridge_path = RAW_DATA_FOLDER.joinpath(season, "gws", "merged_gw.csv")
        if bridge_path.exists():
            frames.append(_load_season_merged_gw(season))
        else:
            logger.warning(
                "Bridge season %s data not found at %s; skipping.",
                season,
                bridge_path,
            )
    frames.append(_load_season_merged_gw(current_season))

    gw_data = pl.concat(frames, how="diagonal")
    gw_data = gw_data.with_columns(
        pl.when(pl.col("position") == "GKP")
        .then(pl.lit("GK"))
        .otherwise(pl.col("position"))
        .alias("position")
    )
    return gw_data


def create_rolling_average_column(
    data: pl.DataFrame,
    grouping_column: str,
    rolling_column: str,
    rolling_window: int,
) -> pl.DataFrame:
    """Create a rolling average column over a given window size.

    This function takes the `grouping_column` and calculates the rolling
    average of the `rolling_column` over a window size of `rolling_window`.
    It does this on a dataframe that is sorted by season and gameweek.

    Parameters
    ----------
    data : pl.DataFrame
        The data to calculate the rolling average on.
    grouping_column : str
        The column to group by.
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
        .rolling_mean(window_size=rolling_window)
        .over(pl.col(grouping_column))
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
    gw_data = load_gw_data(current_season)
    rolling_column = rolling_column_name("total_points", rolling_window)
    gw_data = create_rolling_average_column(
        gw_data, "name", "total_points", rolling_window
    )
    gw_data = fill_missing_values_by_position(gw_data, rolling_column)
    TRANSFORMED_DATA_FOLDER.mkdir(exist_ok=True, parents=True)
    gw_data.write_csv(TRANSFORMED_DATA_FOLDER.joinpath("rolling_points.csv"))
