import logging
import os
from collections.abc import Callable
from typing import TYPE_CHECKING

from dotenv import load_dotenv

from fantasy_football.constants import CURRENT_SEASON
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
from fantasy_football.extraction.player_match_fpl import VaastavMatchLoader
from fantasy_football.extraction.player_match_opta import (
    FciMatchStatsLoader,
)
from fantasy_football.extraction.seasons import (
    DataSource,
    previous_season,
    source_for_season,
)
from fantasy_football.extraction.snapshot import load_player_snapshot
from fantasy_football.logging_config import configure_logging
from fantasy_football.modelling.assists import (
    ASSISTS_SPEC,
    AssistsRatePredictor,
)
from fantasy_football.modelling.assists import (
    EXPERIMENT_NAME as ASSISTS_EXPERIMENT_NAME,
)
from fantasy_football.modelling.assists import (
    TRAINING_SEASONS as ASSISTS_TRAINING_SEASONS,
)
from fantasy_football.modelling.components import (
    compose_points,
    write_appearance_components,
)
from fantasy_football.modelling.conceding import (
    CONCEDING_SPEC,
    GOALKEEPER_CONCEDING_SPEC,
    MIDFIELDER_CONCEDING_SPEC,
    ConcedingPredictor,
    GoalkeeperConcedingPredictor,
    MidfielderConcedingPredictor,
)
from fantasy_football.modelling.conceding import (
    EXPERIMENT_NAME as CONCEDING_EXPERIMENT_NAME,
)
from fantasy_football.modelling.conceding import (
    TRAINING_SEASONS as CONCEDING_TRAINING_SEASONS,
)
from fantasy_football.modelling.defcon import (
    CBIRT_SPEC,
    DEFCON_SPEC,
    CbirtRatePredictor,
    DefconRatePredictor,
)
from fantasy_football.modelling.defcon import (
    TRAINING_SEASONS as DEFCON_TRAINING_SEASONS,
)
from fantasy_football.modelling.folds import (
    TrainTestSplitStrategy,
)
from fantasy_football.modelling.goals import (
    EXPERIMENT_NAME as GOALS_EXPERIMENT_NAME,
)
from fantasy_football.modelling.goals import (
    GOALS_SPEC,
    GoalsRatePredictor,
)
from fantasy_football.modelling.goals import (
    TRAINING_SEASONS as GOALS_TRAINING_SEASONS,
)
from fantasy_football.modelling.minutes import (
    MINUTES_SPEC,
    MinutesPredictor,
)
from fantasy_football.modelling.saves import (
    EXPERIMENT_NAME as SAVES_EXPERIMENT_NAME,
)
from fantasy_football.modelling.saves import (
    SAVES_SPEC,
    SavesRatePredictor,
)
from fantasy_football.modelling.saves import (
    TRAINING_SEASONS as SAVES_TRAINING_SEASONS,
)
from fantasy_football.modelling.yellow_cards import (
    EXPERIMENT_NAME as YELLOW_CARDS_EXPERIMENT_NAME,
)
from fantasy_football.modelling.yellow_cards import (
    TEST_SEASONS as YELLOW_CARDS_TEST_SEASONS,
)
from fantasy_football.modelling.yellow_cards import (
    YELLOW_CARDS_SPEC,
    YellowCardsRatePredictor,
)
from fantasy_football.optimisation.inputs import forward_gameweeks
from fantasy_football.optimisation.optimiser import optimise_plan
from fantasy_football.optimisation.team_input import (
    SquadUnavailableError,
    load_api_squad,
    load_team_file,
    squad_from_team_file,
)
from fantasy_football.storage.database import get_connection, reset_database
from fantasy_football.storage.tables import (
    BACKFILL_KIND,
    FORWARD_KIND,
    PLAYER_WEEK,
    TEAM_FIXTURE,
)

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

    from fantasy_football.optimisation.team_input import Squad

load_dotenv()

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


def load_match_level_stats(
    connection: "DuckDBPyConnection",
    season: str = CURRENT_SEASON,
    vaastav: VaastavMatchLoader | None = None,
    fci: FciMatchStatsLoader | None = None,
) -> None:
    """Populate the two comprehensive match-level stats tables.

    The two sources are independent, so a failure in one must not stop
    the other: Vaastav is a frozen historic archive whose repo may go
    away entirely, while FCI is the only source for the current season.
    Failures are logged and swallowed here because neither table feeds
    the prediction pipeline yet -- nothing downstream breaks if one is
    stale, and aborting the whole run would be a worse trade.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        Open connection to the database.
    season : str, optional
        The current season, upserted rather than treated as immutable.
        Defaults to ``CURRENT_SEASON``.
    vaastav : VaastavMatchLoader | None, optional
        Vaastav loader. Defaults to a new ``VaastavMatchLoader``.
    fci : FciMatchStatsLoader | None, optional
        FCI loader. Defaults to a new ``FciMatchStatsLoader``.
    """
    for name, loader in (
        ("Vaastav", vaastav or VaastavMatchLoader()),
        ("FCI", fci or FciMatchStatsLoader()),
    ):
        try:
            loader.load(connection, current_season=season)
        except Exception:
            logger.exception(
                "%s match-level stats ingestion failed; continuing. The "
                "affected table is stale, not corrupt.",
                name,
            )


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


def carried_in_squad(team_file: str | None, start_gw: int) -> "Squad | None":
    """Get the squad being carried into ``start_gw``, if there is one.

    An explicit team file wins over the API: its remaining job is to ask
    "what if I owned this instead", which only works if passing it
    overrides the live squad. With no file and no credentials configured
    there is nothing to carry in, which is a free build at GW1 and a
    skipped optimisation after it.

    Parameters
    ----------
    team_file : str | None
        Path to a team JSON, or None to read the live squad.
    start_gw : int
        The gameweek the forward predictions start at.

    Returns
    -------
    Squad | None
        The carried-in squad, or None when neither source is configured.
    """
    if team_file is not None:
        return squad_from_team_file(load_team_file(team_file), CURRENT_SEASON)

    manager_id = os.getenv("FPL_MANAGER_ID")
    cookie = os.getenv("FPL_COOKIE")
    if not manager_id or not cookie:
        return None
    return load_api_squad(
        manager_id,
        cookie,
        season=CURRENT_SEASON,
        expected_gameweek=start_gw,
    )


def main(
    *,
    rebuild: bool = False,
    team_file: str | None = None,
) -> None:
    """Download FPL data, transform it, predict points, and optimise a plan.

    Parameters
    ----------
    rebuild : bool, optional
        When True, drops and reloads every season from scratch (full refresh /
        recovery escape hatch). Defaults to False, which loads only missing
        immutable seasons and upserts the current season.
    team_file : str | None, optional
        Path to a team JSON, overriding the live squad. With this unset the
        squad is read from the authenticated FPL ``my-team`` endpoint using
        FPL_MANAGER_ID and FPL_COOKIE. One or the other is needed to
        optimise once the season is under way. Defaults to None.
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
        load_match_level_stats(connection, CURRENT_SEASON)
        load_player_availability_data(connection, CURRENT_SEASON)
        load_player_identity_data(connection, CURRENT_SEASON)
        load_player_snapshot(CURRENT_SEASON, connection)
        check_prior_season_loaded(connection, CURRENT_SEASON)

        # TODO(JT): Add a single method to predictor to do all of these in
        # one. Five near-identical train/backfill/forward blocks now, one
        # per position plus minutes.
        minutes_predictor = MinutesPredictor(
            experiment_name="minutes_played_classification",
            params={},
            model_spec=MINUTES_SPEC,
            connection=connection,
            fold_strategy=TrainTestSplitStrategy(),
        )
        minutes_predictor.train_and_register_model()
        minutes_predictor.backfill_model_predictions()
        minutes_predictor.predict_forward()

        # Every outfield position is decomposed: appearance comes
        # free from the minutes model and the rest are their own heads
        # with their own aliases. There is no residual, so bonus and red
        # cards are in none of these predictions.
        defcon_predictor = DefconRatePredictor(
            experiment_name="def-defcon-rate-model",
            params={},
            model_spec=DEFCON_SPEC,
            connection=connection,
            fold_strategy=TrainTestSplitStrategy(
                test_seasons=DEFCON_TRAINING_SEASONS
            ),
        )
        defcon_predictor.train_and_register_model()
        defcon_predictor.backfill_model_predictions()
        defcon_predictor.predict_forward()

        # One artefact fitted on DEF, MID and FWD, writing component
        # rows for all three.
        goals_predictor = GoalsRatePredictor(
            experiment_name=GOALS_EXPERIMENT_NAME,
            params={},
            model_spec=GOALS_SPEC,
            connection=connection,
            fold_strategy=TrainTestSplitStrategy(
                test_seasons=GOALS_TRAINING_SEASONS
            ),
        )
        goals_predictor.train_and_register_model()
        goals_predictor.backfill_model_predictions()
        goals_predictor.predict_forward()

        # Pooled the same way. An assist pays three points to every
        # position, so this head needs no per-position conversion --
        # only the shared process behind the final pass justifies the
        # pooling.
        assists_predictor = AssistsRatePredictor(
            experiment_name=ASSISTS_EXPERIMENT_NAME,
            params={},
            model_spec=ASSISTS_SPEC,
            connection=connection,
            fold_strategy=TrainTestSplitStrategy(
                test_seasons=ASSISTS_TRAINING_SEASONS
            ),
        )
        assists_predictor.train_and_register_model()
        assists_predictor.backfill_model_predictions()
        assists_predictor.predict_forward()

        # Team grain: one prediction per fixture, fanned out to that
        # club's players at scoring time. A forward is paid nothing
        # either way, so only DEF and MID read it.
        conceding_predictor = ConcedingPredictor(
            experiment_name=CONCEDING_EXPERIMENT_NAME,
            params={},
            model_spec=CONCEDING_SPEC,
            connection=connection,
            fold_strategy=TrainTestSplitStrategy(
                test_seasons=CONCEDING_TRAINING_SEASONS
            ),
        )
        conceding_predictor.train_and_register_model()
        conceding_predictor.backfill_model_predictions()
        conceding_predictor.predict_forward()

        # The same artefact and the same alias, fanned out to
        # midfielders instead. It does not train: a second registration
        # of one registered model would bump the version for a fit
        # identical to the one above.
        midfielder_conceding_predictor = MidfielderConcedingPredictor(
            experiment_name=CONCEDING_EXPERIMENT_NAME,
            params={},
            model_spec=MIDFIELDER_CONCEDING_SPEC,
            connection=connection,
            fold_strategy=TrainTestSplitStrategy(
                test_seasons=CONCEDING_TRAINING_SEASONS
            ),
        )
        midfielder_conceding_predictor.backfill_model_predictions()
        midfielder_conceding_predictor.predict_forward()

        # And again for keepers, who are paid the defender's four points
        # for the sheet and docked on the same schedule.
        goalkeeper_conceding_predictor = GoalkeeperConcedingPredictor(
            experiment_name=CONCEDING_EXPERIMENT_NAME,
            params={},
            model_spec=GOALKEEPER_CONCEDING_SPEC,
            connection=connection,
            fold_strategy=TrainTestSplitStrategy(
                test_seasons=CONCEDING_TRAINING_SEASONS
            ),
        )
        goalkeeper_conceding_predictor.backfill_model_predictions()
        goalkeeper_conceding_predictor.predict_forward()

        # Pooled like goals and assists, and priced flat at minus one.
        # Trains far wider than its siblings -- cards go back to 2016-17
        # -- but is tested only where the FCI features it also reads are
        # published, so the holdout describes the regime it serves in.
        yellow_cards_predictor = YellowCardsRatePredictor(
            experiment_name=YELLOW_CARDS_EXPERIMENT_NAME,
            params={},
            model_spec=YELLOW_CARDS_SPEC,
            connection=connection,
            fold_strategy=TrainTestSplitStrategy(
                test_seasons=YELLOW_CARDS_TEST_SEASONS
            ),
        )
        yellow_cards_predictor.train_and_register_model()
        yellow_cards_predictor.backfill_model_predictions()
        yellow_cards_predictor.predict_forward()

        # The midfield and forward threshold is 12 off a count that
        # includes recoveries, which makes it a different quantity from
        # the defender head's -- hence a second head rather than a wider
        # serving list on the first.
        cbirt_predictor = CbirtRatePredictor(
            experiment_name="mid-fwd-cbirt-rate-model",
            params={},
            model_spec=CBIRT_SPEC,
            connection=connection,
            fold_strategy=TrainTestSplitStrategy(
                test_seasons=DEFCON_TRAINING_SEASONS
            ),
        )
        cbirt_predictor.train_and_register_model()
        cbirt_predictor.backfill_model_predictions()
        cbirt_predictor.predict_forward()

        saves_predictor = SavesRatePredictor(
            experiment_name=SAVES_EXPERIMENT_NAME,
            params={},
            model_spec=SAVES_SPEC,
            connection=connection,
            fold_strategy=TrainTestSplitStrategy(
                test_seasons=SAVES_TRAINING_SEASONS
            ),
        )
        saves_predictor.train_and_register_model()
        saves_predictor.backfill_model_predictions()
        saves_predictor.predict_forward()

        # Last, because it reads the rows every other component wrote:
        # appearance points cover the legs a position's other components
        # already cover, not every leg the minutes model scored.
        for kind in (BACKFILL_KIND, FORWARD_KIND):
            write_appearance_components(connection, kind)

        # Every position has written its components by now, so the
        # points table is rebuilt from them in one pass. This is the only
        # writer into it.
        compose_points(connection)

    finally:
        connection.close()

    # The forward predictions decide which gameweek is being planned, so
    # they are checked before anything is fetched: a finished season has
    # no next gameweek for FPL to agree with.
    available = forward_gameweeks(CURRENT_SEASON)
    if not available:
        logger.warning(
            "No forward predictions stored for %s; skipping optimisation. "
            "The season's fixtures may all have been played, or no position "
            "has a model promoted to its production alias.",
            CURRENT_SEASON,
        )
        return

    start_gw = available[0]
    try:
        squad = carried_in_squad(team_file, start_gw)
    except SquadUnavailableError as error:
        logger.warning(
            "%s Skipping optimisation; the data, models and predictions "
            "from this run are unaffected.",
            error,
        )
        return

    if squad is None:
        if start_gw > 1:
            logger.warning(
                "Gameweek %d is mid-season and no squad was given, so "
                "there is no squad to carry in; skipping optimisation. The "
                "data, models and predictions from this run are unaffected. "
                "Pass team_file, or set FPL_MANAGER_ID and FPL_COOKIE, to "
                "plan transfers.",
                start_gw,
            )
            return
        optimise_plan(CURRENT_SEASON, start_gw, horizon=37)
        return

    optimise_plan(
        CURRENT_SEASON,
        squad.gameweek,
        horizon=37,
        initial_squad=squad.players,
        free_transfers=squad.free_transfers,
        bank=squad.bank,
    )


if __name__ == "__main__":
    main(rebuild=False)
