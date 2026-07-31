"""Populate the ``player_availability`` table from fplcache snapshots.

Both historic and current seasons come from the same source — the
``Randdalf/fplcache`` bootstrap-static time-series — unlike ``player_match``
which splits Vaastav (historic) from the FPL API (current). Completed seasons
are inserted once and skipped thereafter; the current season is upserted each
run so injury news refreshes.
"""

import logging
from typing import TYPE_CHECKING

from fantasy_football.constants import CURRENT_SEASON, FPLCACHE_FIRST_SEASON
from fantasy_football.extraction.fplcache import FplCacheExtractor
from fantasy_football.extraction.seasons import seasons_in_range
from fantasy_football.storage.tables import PLAYER_AVAILABILITY

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

logger = logging.getLogger(__name__)


def load_player_availability_data(
    connection: "DuckDBPyConnection",
    current_season: str = CURRENT_SEASON,
    extractor: FplCacheExtractor | None = None,
) -> None:
    """Backfill completed seasons and upsert the current season's availability.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        Open connection to the database.
    current_season : str, optional
        Short-form current season. Defaults to ``CURRENT_SEASON``.
    extractor : FplCacheExtractor | None, optional
        fplcache extractor. Defaults to a new ``FplCacheExtractor``.
    """
    extractor = extractor or FplCacheExtractor()
    present = PLAYER_AVAILABILITY.seasons_present(connection)
    for season in seasons_in_range(FPLCACHE_FIRST_SEASON, current_season):
        if season != current_season and season in present:
            logger.info(
                "Player-availability season %s already present; skipping "
                "fetch.",
                season,
            )
            continue
        frame = extractor.build_player_chance_of_playing(season)
        if season == current_season:
            PLAYER_AVAILABILITY.upsert_current(connection, frame, season)
        else:
            PLAYER_AVAILABILITY.write_immutable(connection, frame, season)
