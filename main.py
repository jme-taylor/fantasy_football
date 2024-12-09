import argparse

from fantasy_football.constants import CURRENT_SEASON
from fantasy_football.data_transformation import create_rolling_points_data
from fantasy_football.data_extraction import save_all_data_files, update_current_season_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--update-current-season",
        default=False,
        action="store_true",
        help="Update current season data only",
    )
    parser.add_argument(
        "--download-all",
        default=False,
        action="store_true",
        help="Download all data files",
    )
    args = parser.parse_args()
    if args.download_all:
        save_all_data_files()
    if args.update_current_season:
        update_current_season_data(CURRENT_SEASON)
    create_rolling_points_data(CURRENT_SEASON)

if __name__ == "__main__":
    main()
