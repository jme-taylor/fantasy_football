from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_FOLDER = PROJECT_ROOT.joinpath("data")
MODELS_FOLDER = PROJECT_ROOT.joinpath("models")
MLFLOW_DB_PATH = MODELS_FOLDER.joinpath("mlflow.db")

FPL_ID = "7515957"
CURRENT_SEASON = "2025-26"

# Prediction model tunables
HORIZON_N: int = 5
OPPONENT_FACTOR_EXPONENT: float = 1.0
HOME_FACTOR: float = 1.10
AWAY_FACTOR: float = 0.90

# ELO scrape cache
ELO_CACHE_TTL_HOURS: int = 24

# URL slugs passed to ScraperFC ClubElo's scrape_team(name).
# Update this when team set changes (promotions/relegations).
CLUBELO_SCRAPE_NAMES: list[str] = [
    "Arsenal", "AstonVilla", "Bournemouth", "Brentford", "Brighton",
    "Burnley", "Chelsea", "CrystalPalace", "Everton", "Fulham",
    "Ipswich", "Leeds", "Leicester", "Liverpool", "ManCity",
    "ManUnited", "Newcastle", "Forest", "Southampton", "Sunderland",
    "Tottenham", "WestHam", "Wolves",
]

# Mapping from the `Club` column value (as returned by ClubElo) to the
# team name used in our FPL data (data/raw/<season>/teams.csv).
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
