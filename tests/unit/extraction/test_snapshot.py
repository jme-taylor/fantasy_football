"""Tests for the point-in-time FPL player snapshot."""

import logging
from datetime import datetime

import duckdb
import polars as pl
import pytest_mock

from fantasy_football.extraction.snapshot import (
    build_snapshot,
    load_player_snapshot,
    warn_unidentified_snapshot,
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


def test_warn_unidentified_snapshot_reports_the_coverage_gap(caplog) -> None:
    """Snapshot elements with no player_season row are counted and logged."""
    snapshot = pl.DataFrame(
        {
            "season": ["2026-27"] * 3,
            "element": [1, 2, 3],
        }
    )
    player_season = pl.DataFrame(
        {
            "season": ["2026-27", "2026-27"],
            "element": [1, 2],
            "player_code": [111, None],
        },
        schema_overrides={"player_code": pl.Int64},
    )

    with caplog.at_level(logging.WARNING):
        missing = warn_unidentified_snapshot(
            snapshot, player_season, "2026-27"
        )

    # Element 2 has a row but a null player_code, so it is unusable too.
    assert missing == 2
    assert "2 of 3" in caplog.text


def test_warn_unidentified_snapshot_silent_when_fully_covered(
    caplog,
) -> None:
    """Full identity coverage logs nothing and reports zero."""
    snapshot = pl.DataFrame({"season": ["2026-27"], "element": [1]})
    player_season = pl.DataFrame(
        {"season": ["2026-27"], "element": [1], "player_code": [111]}
    )

    with caplog.at_level(logging.WARNING):
        missing = warn_unidentified_snapshot(
            snapshot, player_season, "2026-27"
        )

    assert missing == 0
    assert caplog.text == ""
