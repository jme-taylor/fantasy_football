import polars as pl

from fantasy_football.constants import CURRENT_SEASON, TRANSFORMED_DATA_FOLDER
from fantasy_football.data_extraction import DataExtractor
from fantasy_football.data_transformation import create_rolling_points_data
from fantasy_football.elo import build_team_elo
from fantasy_football.fixtures import build_fixtures_enriched
from fantasy_football.logging_config import configure_logging
from fantasy_football.optimisation import optimise_plan
from fantasy_football.prediction import predict_points


def main(download_all_data: bool = False) -> None:
    """Download FPL data, transform it, predict points, and optimise a plan."""
    configure_logging()
    extractor = DataExtractor()
    if download_all_data:
        extractor.save_all_data_files()
    else:
        extractor.update_current_season_data(CURRENT_SEASON)

    create_rolling_points_data(CURRENT_SEASON)
    build_fixtures_enriched(CURRENT_SEASON)
    build_team_elo()
    predict_points(CURRENT_SEASON)

    predictions = pl.read_csv(
        TRANSFORMED_DATA_FOLDER.joinpath("predictions.csv")
    )
    start_gw = int(predictions["gw"].min())
    optimise_plan(CURRENT_SEASON, start_gw)


if __name__ == "__main__":
    main()
