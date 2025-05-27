from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_FOLDER = PROJECT_ROOT.joinpath("data")
MODELS_FOLDER = PROJECT_ROOT.joinpath("models")
MLFLOW_DB_PATH = MODELS_FOLDER.joinpath("mlflow.db")

FPL_ID = "7515957"
CURRENT_SEASON = "2024-25"
