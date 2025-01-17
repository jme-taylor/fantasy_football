import polars as pl

from fantasy_football.types import FplPlayerFixtures, FplPlayer

def predict_player_points(rolling_data: pl.DataFrame, player: FplPlayer, season: str = "2024-25") -> FplPlayerFixtures:
    """Predict a players points for a given gameweek.

    Parameters
    ----------
    rolling_data : pl.DataFrame
        The rolling data to use for predictions.
    player : FplPlayer
        The player to predict points for.
    season : str, optional
        _description_, by default "2024-25"

    Returns
    -------
    FplPlayerFixtures
        _description_

    """
    season_data = rolling_data.filter(pl.col("season") == season)
    player_expected_points = season_data.filter(pl.col("element") == player.id).sort("gw", descending=True).select("total_points_rolling_5")[0, 0]
    print(player_expected_points)