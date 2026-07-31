"""Capture a point-in-time snapshot of every FPL player's state.

Pre-season the database holds no ``player_week`` rows for the coming
season, so the contemporaneous model features -- value, positional rank,
fit rivals -- have no input frame. Bootstrap-static does have all of it,
but only for *now*: prices move daily. This module persists that live
state stamped with the instant it was captured, so a squad picked on one
day's prices stays reconstructible afterwards.
"""

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import polars as pl

from fantasy_football.extraction.fpl import FplAPI
from fantasy_football.storage.tables import PLAYER_SNAPSHOT

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

logger = logging.getLogger(__name__)

# FPL's element_type ids. GKP is normalised to GK on write by the table's
# gkp_to_gk hook, matching the labels used throughout player_week.
ELEMENT_TYPE_TO_POSITION: dict[int, str] = {
    1: "GKP",
    2: "DEF",
    3: "MID",
    4: "FWD",
}


def build_snapshot(
    season: str, captured_at: datetime, fpl_api: FplAPI
) -> pl.DataFrame:
    """Build snapshot rows from the live bootstrap-static payload.

    Parameters
    ----------
    season : str
        Short-form season string, e.g. ``"2026-27"``.
    captured_at : datetime
        The instant this snapshot represents. Naive UTC.
    fpl_api : FplAPI
        The FPL API client.

    Returns
    -------
    pl.DataFrame
        One row per player in ``PLAYER_SNAPSHOT`` order.

    Raises
    ------
    KeyError
        If a player references a team id absent from ``get_teams``.
    """
    teams_by_id = {team.id: team.name for team in fpl_api.get_teams()}
    players = fpl_api.get_players()
    rows = [
        {
            "season": season,
            "captured_at": captured_at,
            "element": player.id,
            "value": player.now_cost,
            "team": teams_by_id[player.team_id],
            "position": ELEMENT_TYPE_TO_POSITION[player.element_type],
            "chance_of_playing_this_round": (
                player.chance_of_playing_this_round
            ),
        }
        for player in players
    ]
    if not rows:
        return pl.DataFrame(schema=PLAYER_SNAPSHOT.schema)
    return pl.DataFrame(rows).cast(PLAYER_SNAPSHOT.schema, strict=False)


def load_player_snapshot(
    season: str,
    connection: "DuckDBPyConnection",
    captured_at: datetime | None = None,
    fpl_api: FplAPI | None = None,
) -> None:
    """Capture and store the current player snapshot for a season.

    Rows are appended, never replaced: each capture is a distinct
    ``captured_at`` and the history is what makes a past squad
    reconstructible.

    Parameters
    ----------
    season : str
        Short-form season string, e.g. ``"2026-27"``.
    connection : duckdb.DuckDBPyConnection
        Open connection to the database.
    captured_at : datetime | None, optional
        Capture instant. Defaults to the current UTC time, stored naive to
        match the ``TIMESTAMP`` columns used elsewhere.
    fpl_api : FplAPI | None, optional
        FPL API client. Defaults to a new ``FplAPI``.
    """
    api = fpl_api or FplAPI()
    stamp = captured_at or datetime.now(timezone.utc).replace(tzinfo=None)
    frame = build_snapshot(season, stamp, api)
    if frame.is_empty():
        logger.warning("No players returned for %s; nothing to store.", season)
        return
    PLAYER_SNAPSHOT.append(connection, frame)
    logger.info(
        "Captured %d player-snapshot rows for %s at %s",
        frame.height,
        season,
        stamp,
    )
