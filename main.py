import logging

import polars as pl

from fantasy_football.constants import CURRENT_SEASON, TRANSFORMED_DATA_FOLDER
from fantasy_football.data_extraction import DataExtractor
from fantasy_football.data_transformation import create_rolling_points_data
from fantasy_football.elo import build_team_elo
from fantasy_football.evaluation import run_evaluation
from fantasy_football.fci_extraction import FciExtractor
from fantasy_football.fixtures import build_fixtures_enriched
from fantasy_football.logging_config import configure_logging
from fantasy_football.optimisation import optimise_plan
from fantasy_football.prediction import predict_points
from fantasy_football.seasons import DataSource, source_for_season
from fantasy_football.team_input import (
    load_team_file,
    resolve_ids_to_names,
    resolve_names_to_ids,
)

logger = logging.getLogger(__name__)


def update_current_season(season: str) -> None:
    """Refresh the current season's merged_gw.csv from the correct source.

    Parameters
    ----------
    season : str
        Short-form season string, e.g. ``"2025-26"``.
    """
    if source_for_season(season) == DataSource.FCI:
        FciExtractor().build_current_season_merged_gw(season)
    else:
        # Historic seasons are served by the frozen Vaastav dataset.
        DataExtractor().save_all_data_files()


def main(
    *,
    download_all_data: bool = False,
    team_file: str | None = None,
    evaluate: bool = False,
) -> None:
    """Download FPL data, transform it, predict points, and optimise a plan.

    Parameters
    ----------
    download_all_data : bool, optional
        When True, also refresh the Vaastav historic dataset. Defaults to False.
    team_file : str | None, optional
        Path to a name-authored team JSON. When given, optimisation carries
        in that squad from its gameweek instead of free-building. Defaults to
        None.
    evaluate : bool, optional
        When True, run the rolling-origin model evaluation (logging metrics
        to MLflow) before predicting and optimising. Fatal: a failed
        evaluation aborts the run. Defaults to False.
    """
    configure_logging()
    if download_all_data:
        DataExtractor().save_all_data_files()
    update_current_season(CURRENT_SEASON)

    create_rolling_points_data(CURRENT_SEASON)
    build_fixtures_enriched(CURRENT_SEASON)
    build_team_elo()
    if evaluate:
        run_evaluation()
    team = load_team_file(team_file) if team_file is not None else None
    as_of_gw = team.gameweek - 1 if team is not None else None
    predict_points(CURRENT_SEASON, as_of_gw=as_of_gw)

    predictions = pl.read_csv(
        TRANSFORMED_DATA_FOLDER.joinpath("predictions.csv")
    )
    if predictions.is_empty():
        logger.warning(
            "No upcoming gameweeks to predict for %s; skipping optimisation. "
            "The current season's data may be complete with no future "
            "fixtures to plan for.",
            CURRENT_SEASON,
        )
        return
    if team is not None:
        ids = resolve_names_to_ids(team.players, CURRENT_SEASON)
        names = resolve_ids_to_names(ids, predictions)
        optimise_plan(
            CURRENT_SEASON,
            team.gameweek,
            initial_squad=names,
            free_transfers=team.free_transfers,
            bank=team.bank,
        )
    else:
        start_gw = int(predictions["gw"].min())
        optimise_plan(CURRENT_SEASON, start_gw)


if __name__ == "__main__":
    main(download_all_data=True, team_file="data/dummy_team.json")
