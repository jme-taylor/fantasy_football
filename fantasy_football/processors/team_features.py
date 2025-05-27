import polars as pl

def get_team_conceded(data: pl.DataFrame, window: int = 5) -> pl.DataFrame:
    team_conceded = data.filter(
        (pl.col("minutes") == 90) & (pl.col("team_x").is_not_null())
    ).group_by(["team_x", "kickoff_time"], maintain_order=True).agg(
        pl.col("goals_conceded").max().alias("team_goals_conceded")
    ).with_columns(
        pl.col("team_goals_conceded").rolling_mean(window).shift(1).over("team_x").alias(f"team_conceded_rolling_{window}")
    )
    return team_conceded

def merge_team_conceded(data: pl.DataFrame, window: int = 5) -> pl.DataFrame:
    team_conceded = get_team_conceded(data, window)
    data = data.join_asof(team_conceded, on="kickoff_time", by="team_x", strategy="backward")
    return data

def get_team_saves(data: pl.DataFrame, window: int = 5) -> pl.DataFrame:
    """Calculate rolling mean of team saves over a specified window.
    
    Args:
        data: DataFrame containing player/team data with saves column
        window: Number of games to use for rolling mean calculation
        
    Returns:
        DataFrame with team saves rolling mean
    """
    team_saves = data.filter(
        (pl.col("minutes") == 90) & (pl.col("team_x").is_not_null())
    ).group_by(["team_x", "kickoff_time"], maintain_order=True).agg(
        pl.col("saves").max().alias("team_saves")
    ).with_columns(
        pl.col("team_saves").rolling_mean(window).shift(1).over("team_x").alias(f"team_saves_rolling_{window}")
    )
    return team_saves

def merge_team_saves(data: pl.DataFrame, window: int = 5) -> pl.DataFrame:
    """Merge team saves rolling mean data with the main dataset.
    
    Args:
        data: Main DataFrame to merge saves data into
        window: Number of games to use for rolling mean calculation
        
    Returns:
        DataFrame with team saves rolling mean merged in
    """
    team_saves = get_team_saves(data, window)
    data = data.join_asof(team_saves, on="kickoff_time", by="team_x", strategy="backward")
    return data

def get_team_scored(data: pl.DataFrame, window: int = 5) -> pl.DataFrame:
    """Calculate rolling mean of team goals scored over a specified window.
    
    Args:
        data: DataFrame containing player/team data with goals_conceded column
        window: Number of games to use for rolling mean calculation
        
    Returns:
        DataFrame with team goals scored rolling mean
    """
    team_scored = data.filter(
        (pl.col("minutes") == 90) & (pl.col("opp_team_name").is_not_null())
    ).group_by(["opp_team_name", "kickoff_time"], maintain_order=True).agg(
        pl.col("goals_conceded").max().alias("team_goals_scored")
    ).with_columns(
        pl.col("team_goals_scored").rolling_mean(window).shift(1).over("opp_team_name").alias(f"team_scored_rolling_{window}")
    ).select(["opp_team_name", "kickoff_time", "team_goals_scored", f"team_scored_rolling_{window}"]
    ).rename({"opp_team_name": "team"})
    return team_scored

def merge_team_scored(data: pl.DataFrame, window: int = 5, by_left: str = "team_x") -> pl.DataFrame:
    """Merge team goals scored rolling mean data with the main dataset.
    
    Args:
        data: Main DataFrame to merge scored data into
        window: Number of games to use for rolling mean calculation
        
    Returns:
        DataFrame with team goals scored rolling mean merged in
    """
    team_scored = get_team_scored(data, window)
    data = data.join_asof(team_scored, on="kickoff_time", by_left=by_left, by_right="team", strategy="backward")
    return data