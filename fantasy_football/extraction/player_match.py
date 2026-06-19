"""Build the current season's per-fixture ``player_match`` rows from the FPL API.

The live ``element-summary`` endpoint exposes one row per fixture for the
current season only, so this orchestrator pulls every player's history, stamps
the season, and upserts it into ``player_match``. Historic seasons come from
Vaastav via ``DataExtractor.load_immutable_player_match_seasons`` instead.
"""

import logging
from typing import TYPE_CHECKING

import polars as pl

from fantasy_football.extraction.fpl import FplAPI
from fantasy_football.storage.database import upsert_current_player_match

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

logger = logging.getLogger(__name__)


def load_current_season_player_match(
    season: str,
    connection: "DuckDBPyConnection",
    fpl_api: FplAPI | None = None,
) -> None:
    """Fetch every current-season player's per-fixture history and upsert it.

    Parameters
    ----------
    season : str
        Short-form season string, e.g. ``"2025-26"``.
    connection : duckdb.DuckDBPyConnection
        Open connection to the database.
    fpl_api : FplAPI | None, optional
        FPL API client. Defaults to a new ``FplAPI`` instance.
    """
    api = fpl_api or FplAPI()
    players = api.get_players()
    histories = [
        api.get_player_match_history(player.id) for player in players
    ]
    non_empty = [frame for frame in histories if frame.height > 0]
    if not non_empty:
        logger.warning("No player-match history returned for %s.", season)
        return
    combined = pl.concat(non_empty, how="vertical").with_columns(
        pl.lit(season).alias("season")
    )
    upsert_current_player_match(connection, combined, season)
    logger.info(
        "Upserted %d player-match rows for current season %s",
        combined.height,
        season,
    )
