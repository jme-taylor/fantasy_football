from fantasy_football.constants import CURRENT_SEASON
from fantasy_football.data_extraction import DataExtractor
from fantasy_football.data_transformation import create_rolling_points_data


def main(download_all_data: bool = False) -> None:
    """Download Fantasy Premier League data based on configuration.

    This function will download all data from the Fantasy Premier League repository
    if `download_all_data` is True, otherwise it will update the current season data.
    It will then create a rolling points dataset for the current season.

    Parameters
    ----------
    download_all_data: bool, optional
        If True, download all data from the Fantasy Premier League repository.
        If False, update the current season data.
    """
    extractor = DataExtractor()
    if download_all_data:
        extractor.save_all_data_files()
    else:
        extractor.update_current_season_data(CURRENT_SEASON)

    create_rolling_points_data(CURRENT_SEASON)


if __name__ == "__main__":
    main()
