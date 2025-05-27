import polars as pl


def get_gk_rolling_saves(data: pl.DataFrame, window: int = 5, saves_column_name: str = "saves") -> pl.DataFrame:
    """Calculate rolling saves for goalkeepers over a specified window, normalized per 90 minutes.
    
    Args:
        data: DataFrame containing goalkeeper data with 'name', 'kickoff_time', 'minutes', and saves columns
        window: Rolling window size for calculating rolling saves (default: 5)
        saves_column_name: Name of the column containing saves data (default: "saves")
        
    Returns:
        DataFrame with goalkeeper saves data including rolling saves per 90 minutes
        
    Example:
        >>> df_with_saves = get_gk_rolling_saves(gk_data, window=3)
    """
    data = data.sort("kickoff_time")
    gk_played_data = data.filter(pl.col("minutes") > 0).select(["name", "kickoff_time", "minutes", saves_column_name])
    
    gk_saves = gk_played_data.with_columns(
        pl.col(saves_column_name).rolling_sum(window).shift(1).over("name").alias(f"gk_saves_rolling_{window}"),
        pl.col("minutes").rolling_sum(window).shift(1).over("name").alias(f"gk_minutes_rolling_{window}")
    ).with_columns(
        (pl.col(f"gk_saves_rolling_{window}") * 90 / pl.col(f"gk_minutes_rolling_{window}").cast(pl.Float64)).alias(f"gk_saves_per_90_rolling_{window}")
    ).drop(["minutes"])
    
    return gk_saves


def merge_gk_rolling_saves(data: pl.DataFrame, window: int = 5, saves_column_name: str = "saves") -> pl.DataFrame:
    """Merge rolling goalkeeper saves data back to the original dataset.
    
    Args:
        data: DataFrame containing goalkeeper data with 'name', 'kickoff_time', 'minutes', and saves columns
        window: Rolling window size for calculating rolling saves (default: 5)
        saves_column_name: Name of the column containing saves data (default: "saves")
        
    Returns:
        DataFrame with original data plus rolling saves columns joined using asof join
        
    Example:
        >>> df_merged = merge_gk_rolling_saves(gk_data, window=3)
    """
    gk_saves = get_gk_rolling_saves(data, window=window, saves_column_name=saves_column_name)
    return data.join_asof(gk_saves, left_on="kickoff_time", right_on="kickoff_time", by="name", strategy="backward").drop("saves_right")
