import polars as pl

def get_rolling_player_conceded(data: pl.DataFrame, window: int = 5, conceded_column_name: str = "goals_conceded") -> pl.DataFrame:
    data = data.sort("kickoff_time")
    player_played_data = data.filter(pl.col("minutes") > 0).select(["name", "kickoff_time", "minutes", conceded_column_name])
    player_conceded = player_played_data.with_columns(
        pl.col(conceded_column_name).rolling_sum(window).shift(1).over("name").alias(f"player_conceded_rolling_{window}"),
        pl.col("minutes").rolling_sum(window).shift(1).over("name").alias(f"player_minutes_rolling_{window}")
    ).with_columns(
        pl.col(f"player_conceded_rolling_{window}") * 90 / pl.col(f"player_minutes_rolling_{window}").cast(pl.Float64)
    ).drop(["minutes"])
    return player_conceded

def merge_player_conceded(data: pl.DataFrame) -> pl.DataFrame:
    player_conceded = get_rolling_player_conceded(data)
    return data.join_asof(player_conceded, left_on="kickoff_time", right_on="kickoff_time", by="name", strategy="backward")

def get_player_rolling_minutes(data: pl.DataFrame, window: int = 5) -> pl.DataFrame:
    """Calculate rolling minutes played for each player over a specified window.
    
    Args:
        data: DataFrame containing player data with 'name', 'kickoff_time', and 'minutes' columns
        window: Rolling window size for calculating rolling minutes (default: 5)
        
    Returns:
        DataFrame with original data plus a new column for rolling minutes played
        
    Example:
        >>> df_with_rolling = get_player_rolling_minutes(player_data, window=3)
    """
    data = data.sort("kickoff_time")

    result = data.with_columns(
        pl.col("minutes").rolling_mean(window).shift(1).over("name").alias(f"player_minutes_rolling_{window}")
    )
    
    return result