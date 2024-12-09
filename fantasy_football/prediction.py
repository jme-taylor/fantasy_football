import polars as pl

from fantasy_football.types import Player, FplPlayerFixtures, FplPlayer

def predict_player_points(rolling_data: pl.DataFrame, player: FplPlayer, season: str = "2024-25") -> FplPlayerFixtures:
    season_data = rolling_data.filter(pl.col("season") == season)