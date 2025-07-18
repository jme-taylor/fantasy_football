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

def get_player_rolling_goals_scored(data: pl.DataFrame, window: int = 5, goals_scored_column_name: str = "goals_scored") -> pl.DataFrame:
    data = data.sort("kickoff_time")
    player_played_data = data.filter(pl.col("minutes") > 0).select(["name", "kickoff_time", "minutes", goals_scored_column_name])
    player_goals_scored = player_played_data.with_columns(
        pl.col(goals_scored_column_name).rolling_sum(window).shift(1).over("name").alias(f"player_goals_scored_rolling_{window}"),
        pl.col("minutes").rolling_sum(window).shift(1).over("name").alias(f"player_minutes_rolling_{window}")
    ).with_columns(
        pl.col(f"player_goals_scored_rolling_{window}") * 90 / pl.col(f"player_minutes_rolling_{window}").cast(pl.Float64)  
    ).drop(["minutes" f"player_minutes_rolling_{window}"])
    return player_goals_scored

def merge_player_goals_scored(data: pl.DataFrame) -> pl.DataFrame:
    player_goals_scored = get_player_rolling_goals_scored(data)
    return data.join_asof(player_goals_scored, left_on="kickoff_time", right_on="kickoff_time", by="name", strategy="backward")

def get_player_rolling_assists(data: pl.DataFrame, window: int = 5, assists_column_name: str = "assists") -> pl.DataFrame:
    data = data.sort("kickoff_time")
    player_played_data = data.filter(pl.col("minutes") > 0).select(["name", "kickoff_time", "minutes", assists_column_name])
    player_assists = player_played_data.with_columns(
        pl.col(assists_column_name).rolling_sum(window).shift(1).over("name").alias(f"player_assists_rolling_{window}"),
        pl.col("minutes").rolling_sum(window).shift(1).over("name").alias(f"player_minutes_rolling_{window}")
    ).with_columns(
        pl.col(f"player_assists_rolling_{window}") * 90 / pl.col(f"player_minutes_rolling_{window}").cast(pl.Float64)
    ).drop(["minutes", f"player_minutes_rolling_{window}"])
    return player_assists

def merge_player_assists(data: pl.DataFrame) -> pl.DataFrame:
    player_assists = get_player_rolling_assists(data)
    return data.join_asof(player_assists, left_on="kickoff_time", right_on="kickoff_time", by="name", strategy="backward") 

def get_player_rolling_yellow_cards(data: pl.DataFrame, window: int = 5, yellow_cards_column_name: str = "yellow_cards") -> pl.DataFrame:
    data = data.sort("kickoff_time")
    player_played_data = data.filter(pl.col("minutes") > 0).select(["name", "kickoff_time", "minutes", yellow_cards_column_name])
    player_yellow_cards = player_played_data.with_columns(
        pl.col(yellow_cards_column_name).rolling_sum(window).shift(1).over("name").alias(f"player_yellow_cards_rolling_{window}"),
        pl.col("minutes").rolling_sum(window).shift(1).over("name").alias(f"player_minutes_rolling_{window}")
    ).with_columns(
        pl.col(f"player_yellow_cards_rolling_{window}") * 90 / pl.col(f"player_minutes_rolling_{window}").cast(pl.Float64)
    ).drop(["minutes", f"player_minutes_rolling_{window}"])
    return player_yellow_cards

def merge_player_yellow_cards(data: pl.DataFrame) -> pl.DataFrame:
    player_yellow_cards = get_player_rolling_yellow_cards(data)
    return data.join_asof(player_yellow_cards, left_on="kickoff_time", right_on="kickoff_time", by="name", strategy="backward")

def get_player_rolling_red_cards(data: pl.DataFrame, window: int = 5, red_cards_column_name: str = "red_cards") -> pl.DataFrame:
    data = data.sort("kickoff_time")
    player_played_data = data.filter(pl.col("minutes") > 0).select(["name", "kickoff_time", "minutes", red_cards_column_name])
    player_red_cards = player_played_data.with_columns(
        pl.col(red_cards_column_name).rolling_sum(window).shift(1).over("name").alias(f"player_red_cards_rolling_{window}"),
        pl.col("minutes").rolling_sum(window).shift(1).over("name").alias(f"player_minutes_rolling_{window}")
    ).with_columns(
        pl.col(f"player_red_cards_rolling_{window}") * 90 / pl.col(f"player_minutes_rolling_{window}").cast(pl.Float64)
    ).drop(["minutes", f"player_minutes_rolling_{window}"])
    return player_red_cards

def merge_player_red_cards(data: pl.DataFrame) -> pl.DataFrame:
    player_red_cards = get_player_rolling_red_cards(data)
    return data.join_asof(player_red_cards, left_on="kickoff_time", right_on="kickoff_time", by="name", strategy="backward")