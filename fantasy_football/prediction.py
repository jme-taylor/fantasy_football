import polars as pl

from fantasy_football.types import FplPlayerFixtures, FplPlayer

def predict_player_points(rolling_data: pl.DataFrame, player: FplPlayer, season: str = "2024-25") -> float:
    """Predict a players points for a given gameweek.

    Parameters
    ----------
    rolling_data : pl.DataFrame
        The rolling data to use for predictions.
    player : FplPlayer
        The player to predict points for.
    season : str, optional
        The season to filter data for, by default "2024-25"

    Returns
    -------
    float
        Predicted points based on 5-game rolling average

    """
    season_data = rolling_data.filter(pl.col("season") == season)
    player_data = season_data.filter(pl.col("element") == player.id).sort("gw", descending=True)
    
    if len(player_data) == 0:
        return 0.0
    
    # Return the most recent rolling average as the prediction
    return player_data.select("total_points_rolling_5")[0, 0]