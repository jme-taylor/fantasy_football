"""Tests for the point-in-time FPL player snapshot."""

from datetime import datetime

import duckdb
import pytest_mock

from fantasy_football.extraction.snapshot import (
    build_snapshot,
    load_player_snapshot,
)
from fantasy_football.fpl_types import FplPlayer, FplTeamInfo
from fantasy_football.storage.tables import PLAYER_SNAPSHOT


def _api(mocker: pytest_mock.MockerFixture) -> object:
    """Build a mock FPL API with two players across two clubs.

    Parameters
    ----------
    mocker : pytest_mock.MockerFixture
        The pytest-mock fixture.

    Returns
    -------
    object
        A mock exposing ``get_players`` and ``get_teams``.
    """
    api = mocker.Mock()
    api.get_players.return_value = [
        FplPlayer(
            id=1,
            first_name="A",
            second_name="Keeper",
            web_name="Keeper",
            selected_by_percent=1.0,
            now_cost=45,
            team_id=1,
            element_type=1,
            chance_of_playing_this_round=None,
        ),
        FplPlayer(
            id=2,
            first_name="B",
            second_name="Striker",
            web_name="Striker",
            selected_by_percent=5.0,
            now_cost=110,
            team_id=2,
            element_type=4,
            chance_of_playing_this_round=50,
        ),
    ]
    api.get_teams.return_value = [
        FplTeamInfo(id=1, code=3, name="Arsenal", short_name="ARS"),
        FplTeamInfo(id=2, code=8, name="Chelsea", short_name="CHE"),
    ]
    return api


def test_build_snapshot_maps_position_and_team(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """element_type becomes a position label and team_id a club name."""
    captured = datetime(2026, 7, 31, 12, 0)

    frame = build_snapshot("2026-27", captured, _api(mocker))

    rows = {row["element"]: row for row in frame.iter_rows(named=True)}
    # GKP is normalised to GK on write by the table's normalise hook.
    assert rows[1]["position"] == "GKP"
    assert rows[1]["team"] == "Arsenal"
    assert rows[2]["position"] == "FWD"
    assert rows[2]["value"] == 110
    assert rows[2]["chance_of_playing_this_round"] == 50


def test_load_player_snapshot_writes_normalised_rows(
    db: duckdb.DuckDBPyConnection, mocker: pytest_mock.MockerFixture
) -> None:
    """Stored rows carry the capture instant and a GK position label."""
    captured = datetime(2026, 7, 31, 12, 0)

    load_player_snapshot("2026-27", db, captured, _api(mocker))

    stored = PLAYER_SNAPSHOT.load(db)
    assert stored.height == 2
    assert set(stored["position"].to_list()) == {"GK", "FWD"}
    assert stored["captured_at"].to_list() == [captured, captured]
