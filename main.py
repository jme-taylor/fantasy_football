from fantasy_football.constants import CURRENT_SEASON
from fantasy_football.data_extraction import DataExtractor
from fantasy_football.data_transformation import create_rolling_points_data
from fantasy_football.elo import build_team_elo
from fantasy_football.fixtures import build_fixtures_enriched
from fantasy_football.logging_config import configure_logging
from fantasy_football.prediction import predict_points


def main(download_all_data: bool = False) -> None:
    """Download FPL data, transform it, and produce baseline predictions."""
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


if __name__ == "__main__":
    main()
