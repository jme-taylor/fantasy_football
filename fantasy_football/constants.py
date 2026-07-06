from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_FOLDER = PROJECT_ROOT.joinpath("data")
RAW_DATA_FOLDER = DATA_FOLDER.joinpath("raw")
TRANSFORMED_DATA_FOLDER = DATA_FOLDER.joinpath("transformed")
MODELS_FOLDER = PROJECT_ROOT.joinpath("models")
MLFLOW_DB_PATH = MODELS_FOLDER.joinpath("mlflow.db")
# Single-file duckdb database; source of truth for player-week data.
# Lives under the gitignored data/ folder.
DATABASE_PATH = DATA_FOLDER.joinpath("fantasy_football.duckdb")

FPL_ID = "7515957"
CURRENT_SEASON = "2025-26"

# Data ingestion stop date
VASTAAV_LAST_SEASON = "2024-2025"
FPL_CORE_INSIGHTS_FIRST_SEASON = "2025-2026"
# Earliest season with Randdalf/fplcache bootstrap snapshots (Aug 2022),
# the source of point-in-time chance_of_playing_this_round.
FPLCACHE_FIRST_SEASON = "2022-23"

# Special season that isn't in the cleaned_merged_seasons.csv file, but is still needed for the data pipeline.
VASTAAV_BRIDGE_SEASONS: list[str] = ["2024-25"]

# Prediction model tunables
ROLLING_WINDOW: int = 5
# A player counts as "fit" for positional-availability features when their
# chance_of_playing_this_round is at or above this percentage. FPL reports
# chance on a 0/25/50/75/100 scale, so 75 means "likely to play".
FIT_THRESHOLD: int = 75
OPPONENT_FACTOR_EXPONENT: float = 1.0
HOME_FACTOR: float = 1.10
AWAY_FACTOR: float = 0.90

# ELO scrape cache
ELO_CACHE_TTL_HOURS: int = 24

# Earliest date for which we care about historic ELO ratings.
# Chosen to be slightly before 2016-17 GW1 (2016-08-13), the earliest fixture
# in our raw FPL data.
ELO_HISTORY_START: date = date(2016, 8, 1)

# Per-position points models (MLflow experiment names)
EXPERIMENT_BY_POSITION: dict[str, str] = {
    "GK": "gk-points-model",
    "DEF": "def-points-model",
    "MID": "mid-points-model",
    "FWD": "fwd-points-model",
}

# MLflow experiment for the minutes-played classifier (single experiment;
# the model is one classifier across all positions, not per-position).
MINUTES_EXPERIMENT: str = "minutes_played_classification"

# MLflow Model Registry name and alias for the minutes classifier. Every
# training run registers a new version under MINUTES_REGISTERED_MODEL; the
# backfill loads whichever version carries the MINUTES_PRODUCTION_ALIAS alias.
# Promotion (moving the alias onto a version) is manual via the MLflow UI.
MINUTES_REGISTERED_MODEL: str = "minutes_played_classifier"
MINUTES_PRODUCTION_ALIAS: str = "production"

# Number of top players per position used for precision@k, sized to the
# number of squad slots FPL gives each position.
PRECISION_K_BY_POSITION: dict[str, int] = {
    "GK": 2,
    "DEF": 5,
    "MID": 5,
    "FWD": 3,
}

# MLflow tracking store (sqlite file under the gitignored models/ folder).
MLFLOW_TRACKING_URI: str = f"sqlite:///{MLFLOW_DB_PATH}"

# URL slugs passed to ScraperFC ClubElo's scrape_team(name).
# Update this when team set changes (promotions/relegations).
CLUBELO_SCRAPE_NAMES: list[str] = [
    "Arsenal",
    "AstonVilla",
    "Bournemouth",
    "Brentford",
    "Brighton",
    "Burnley",
    "Chelsea",
    "CrystalPalace",
    "Everton",
    "Fulham",
    "Ipswich",
    "Leeds",
    "Leicester",
    "Liverpool",
    "ManCity",
    "ManUnited",
    "Newcastle",
    "Forest",
    "Southampton",
    "Sunderland",
    "Tottenham",
    "WestHam",
    "Wolves",
]

# Mapping from the `Club` column value (as returned by ClubElo) to the
# Update this when team set changes (promotions/relegations).
CLUBELO_TO_FPL: dict[str, str] = {
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston Villa",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Brighton": "Brighton",
    "Burnley": "Burnley",
    "Chelsea": "Chelsea",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Ipswich": "Ipswich",
    "Leeds": "Leeds",
    "Leicester": "Leicester",
    "Liverpool": "Liverpool",
    "Man City": "Man City",
    "Man United": "Man Utd",
    "Newcastle": "Newcastle",
    "Forest": "Nott'm Forest",
    "Southampton": "Southampton",
    "Sunderland": "Sunderland",
    "Tottenham": "Spurs",
    "West Ham": "West Ham",
    "Wolves": "Wolves",
}

# First season with player <> team mapping
PLAYER_TEAM_MAPPING_FIRST_SEASON = "2020-21"
