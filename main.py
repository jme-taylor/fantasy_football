import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

import polars as pl

from fantasy_football.constants import CURRENT_SEASON, TRANSFORMED_DATA_FOLDER
from fantasy_football.extraction.availability import (
    load_player_availability_data,
)
from fantasy_football.extraction.extractor import DataExtractor
from fantasy_football.extraction.fci import FciExtractor
from fantasy_football.extraction.fixtures import load_fixtures
from fantasy_football.extraction.player_identity import (
    load_player_identity_data,
)
from fantasy_football.extraction.player_match import (
    load_current_season_player_match,
)
from fantasy_football.extraction.seasons import (
    DataSource,
    previous_season,
    source_for_season,
)
from fantasy_football.extraction.snapshot import load_player_snapshot
from fantasy_football.features.elo import build_team_elo
from fantasy_football.features.fixtures import build_fixtures_enriched
from fantasy_football.features.transformation import create_rolling_points_data
from fantasy_football.logging_config import configure_logging
from fantasy_football.modelling.evaluation import run_evaluation
from fantasy_football.modelling.minutes import (
    backfill_minutes,
    run_minutes_model,
    score_forward_minutes,
)
from fantasy_football.modelling.prediction import predict_points
from fantasy_football.optimisation.optimiser import optimise_plan
from fantasy_football.optimisation.team_input import (
    load_team_file,
    resolve_ids_to_names,
    resolve_names_to_ids,
)
from fantasy_football.storage.database import get_connection, reset_database
from fantasy_football.storage.tables import PLAYER_WEEK, TEAM_FIXTURE

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

logger = logging.getLogger(__name__)


def update_current_season(
    season: str, connection: "DuckDBPyConnection"
) -> None:
    """Refresh the current season's player-week rows from the correct source.

    Parameters
    ----------
    season : str
        Short-form season string, e.g. ``"2025-26"``.
    connection : duckdb.DuckDBPyConnection
        Open connection to the player-week database.
    """
    if source_for_season(season) == DataSource.FCI:
        FciExtractor().build_current_season_merged_gw(season, connection)
    else:
        raise NotImplementedError(
            f"Current-season ingestion for the Vaastav-sourced season "
            f"{season} is not supported; current seasons come from FCI."
        )


def load_player_match_data(
    season: str,
    connection: "DuckDBPyConnection",
    extractor: DataExtractor | None = None,
    current_loader: Callable[[str, "DuckDBPyConnection"], None] = (
        load_current_season_player_match
    ),
) -> None:
    """Populate the player_match table: historic from Vaastav, current from FPL.

    Parameters
    ----------
    season : str
        Short-form current season, e.g. ``"2025-26"``.
    connection : duckdb.DuckDBPyConnection
        Open connection to the database.
    extractor : DataExtractor | None, optional
        Vaastav extractor. Defaults to a new ``DataExtractor``.
    current_loader : callable, optional
        Current-season loader. Defaults to
        ``load_current_season_player_match``.
    """
    (extractor or DataExtractor()).load_immutable_player_match_seasons(
        connection, season
    )
    current_loader(season, connection)


def check_prior_season_loaded(
    connection: "DuckDBPyConnection", season: str = CURRENT_SEASON
) -> None:
    """Assert the season before ``season`` survived ingestion.

    The season immediately before the current one is the backbone of every
    ``prev_season_*`` feature, of ``is_promoted_club``, and of the
    cross-season rolling-minutes window. It has no loader of its own -- it
    arrives either in the Vaastav historic aggregate or as a
    ``VASTAAV_BRIDGE_SEASONS`` entry -- so a missing bridge entry drops it
    silently on a rebuild rather than failing. This turns that into a loud
    failure.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        Open connection to the database, after extraction has run.
    season : str, optional
        The current season. Defaults to ``CURRENT_SEASON``.

    Raises
    ------
    RuntimeError
        If the prior season is absent from ``player_week`` or
        ``team_fixture``.
    """
    prior = previous_season(season)
    missing = [
        table.name
        for table in (PLAYER_WEEK, TEAM_FIXTURE)
        if prior not in table.seasons_present(connection)
    ]
    if missing:
        raise RuntimeError(
            f"Season {prior} (the season before {season}) is missing from "
            f"{', '.join(missing)}. Nothing downstream is trustworthy "
            f"without it: prev_season_* features reroute two seasons back "
            f"and every club looks newly promoted. Add {prior!r} to "
            f"VASTAAV_BRIDGE_SEASONS (or restore its loader) and re-run."
        )


def main(
    *,
    rebuild: bool = False,
    team_file: str | None = None,
    evaluate: bool = False,
) -> None:
    """Download FPL data, transform it, predict points, and optimise a plan.

    Parameters
    ----------
    rebuild : bool, optional
        When True, drops and reloads every season from scratch (full refresh /
        recovery escape hatch). Defaults to False, which loads only missing
        immutable seasons and upserts the current season.
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
    # A rebuild drops and recreates every table, so it is the cure for
    # schema drift rather than a victim of it; skip the guard in that case
    # or an out-of-date database file could never be rebuilt.
    connection = get_connection(check_drift=not rebuild)
    try:
        if rebuild:
            reset_database(connection)
        DataExtractor().load_immutable_seasons(connection, CURRENT_SEASON)
        update_current_season(CURRENT_SEASON, connection)
        load_fixtures(connection, CURRENT_SEASON)
        load_player_match_data(CURRENT_SEASON, connection)
        load_player_availability_data(connection, CURRENT_SEASON)
        load_player_identity_data(connection, CURRENT_SEASON)
        load_player_snapshot(CURRENT_SEASON, connection)
        check_prior_season_loaded(connection, CURRENT_SEASON)
    finally:
        connection.close()

    create_rolling_points_data(CURRENT_SEASON)
    build_fixtures_enriched(CURRENT_SEASON)
    build_team_elo()
    run_minutes_model()
    backfill_minutes()
    score_forward_minutes()
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
    main(rebuild=True)
