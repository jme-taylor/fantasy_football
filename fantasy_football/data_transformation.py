from pathlib import Path

import polars as pl

from fantasy_football.constants import DATA_FOLDER, CURRENT_SEASON

RAW_DATA_FOLDER = DATA_FOLDER.joinpath("raw")
TRANSFORMED_DATA_FOLDER = DATA_FOLDER.joinpath("transformed")
TRANSFORMED_DATA_FOLDER.mkdir(exist_ok=True)

def load_gw_data(current_season: str) -> pl.DataFrame:
    previous_seasons_columns = [
        "season_x",
        "name",
        "position",
        "bonus",
        "element",
        "minutes",
        "round",
        "total_points",
        "GW"
    ]
    previous_seasons = pl.read_csv(RAW_DATA_FOLDER.joinpath("cleaned_merged_seasons.csv"), columns=previous_seasons_columns).rename({"season_x": "season", "GW": "gw"})
    current_season_columns = [
        "name",
        "position",
        "bonus",
        "element",
        "minutes",
        "round",
        "total_points",
        "GW"
    ]
    current_season_data = pl.read_csv(RAW_DATA_FOLDER.joinpath(current_season, "gws", "merged_gw.csv"), columns=current_season_columns).rename({"GW": "gw"}).with_columns(pl.lit(current_season).alias("season"))
    gw_data = pl.concat([previous_seasons, current_season_data], how="diagonal")
    gw_data = gw_data.with_columns(pl.when(pl.col("position") == "GKP").then(pl.lit("GK")).otherwise(pl.col("position")).alias("position"))
    return gw_data

def create_rolling_average_column(data: pl.DataFrame, grouping_column: str, rolling_column: str, rolling_window: int) -> pl.DataFrame:
    """
    
    This assumes we always want to sort the data by season, gw and then round

    Parameters
    ----------
    data : pl.DataFrame
        _description_
    grouping_column : str
        _description_
    rolling_column : str
        _description_
    rolling_window : int
        _description_

    Returns
    -------
    pl.DataFrame
        _description_
    """
    rolling_average_column_name = f"{rolling_column}_rolling_{rolling_window}"
    data = data.sort(["season", "gw"]).with_columns(pl.col(rolling_column).rolling_mean(window_size=rolling_window).over(pl.col(grouping_column)).alias(rolling_average_column_name))
    return data

def fill_missing_values_by_position(data: pl.DataFrame, column_to_fill: str) -> pl.DataFrame:
    for position in ["GK", "DEF", "MID", "FWD"]:
        position_data = data.filter(pl.col("position") == position)
        position_average = position_data.select(pl.col(column_to_fill)).mean().item(0, 0)
        data = data.with_columns(pl.when((pl.col("position") == position) & (pl.col(column_to_fill).is_null())).then(pl.lit(position_average)).otherwise(pl.col(column_to_fill)).alias(column_to_fill))
    return data

def create_rolling_points_data(current_season: str, rolling_window: int = 5) -> None:
    gw_data = load_gw_data(current_season)
    gw_data = create_rolling_average_column(gw_data, "name", "total_points", rolling_window)
    gw_data = fill_missing_values_by_position(gw_data, "total_points_rolling_5")
    gw_data.write_csv(TRANSFORMED_DATA_FOLDER.joinpath("rolling_points.csv"))

