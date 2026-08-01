"""Tests for the pipeline entry point's post-ingestion guards."""

from datetime import datetime

import duckdb
import polars as pl
import pytest

import main
from fantasy_football.constants import (
    CURRENT_SEASON,
    VASTAAV_BRIDGE_SEASONS,
)
from fantasy_football.extraction.seasons import previous_season
from fantasy_football.storage.tables import PLAYER_WEEK, TEAM_FIXTURE
from main import check_prior_season_loaded


def test_previous_season_steps_back_one_year() -> None:
    """The prior season is derived, never hard-coded."""
    assert previous_season("2026-27") == "2025-26"
    assert previous_season("2020-21") == "2019-20"


def test_prior_season_has_a_loader() -> None:
    """The season before CURRENT_SEASON must be reachable on a rebuild.

    ``reset_database`` drops everything, and reload draws only from the
    Vaastav historic aggregate (which stops at 2023-24), the bridge
    seasons, and the FCI current-season loader. Without a bridge entry the
    season before the current one has no loader at all and vanishes.
    """
    assert previous_season(CURRENT_SEASON) in VASTAAV_BRIDGE_SEASONS


def _player_week_row(season: str) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [season],
            "gw": [1],
            "element": [1],
            "name": ["P"],
            "position": ["MID"],
            "team": ["Arsenal"],
            "bonus": [0],
            "minutes": [90],
            "round": [1],
            "total_points": [6],
            "value": [70],
        }
    )


def _team_fixture_row(season: str) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [season],
            "gw": [1],
            "team": ["Arsenal"],
            "is_home": [True],
            "opposition": ["Chelsea"],
            "kickoff_time": [datetime(2025, 8, 16, 15, 0)],
        }
    )


def test_check_prior_season_loaded_passes_when_present(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """Both tables carrying the prior season is the healthy case."""
    prior = previous_season(CURRENT_SEASON)
    PLAYER_WEEK.append(db, _player_week_row(prior))
    TEAM_FIXTURE.append(db, _team_fixture_row(prior))

    check_prior_season_loaded(db, CURRENT_SEASON)


def test_check_prior_season_loaded_raises_when_player_week_missing(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """A dropped prior season fails loudly and names the season and table."""
    prior = previous_season(CURRENT_SEASON)
    TEAM_FIXTURE.append(db, _team_fixture_row(prior))

    with pytest.raises(RuntimeError) as excinfo:
        check_prior_season_loaded(db, CURRENT_SEASON)

    message = str(excinfo.value)
    assert prior in message
    assert "player_week" in message
    assert "team_fixture" not in message


def test_check_prior_season_loaded_raises_when_fixtures_missing(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """team_fixture is checked too: is_promoted_club depends on it."""
    prior = previous_season(CURRENT_SEASON)
    PLAYER_WEEK.append(db, _player_week_row(prior))

    with pytest.raises(RuntimeError) as excinfo:
        check_prior_season_loaded(db, CURRENT_SEASON)

    assert "team_fixture" in str(excinfo.value)


def test_load_match_level_stats_runs_both_loaders(mocker, db):
    """Both the Vaastav and FCI match-stat loaders are invoked."""
    vaastav = mocker.Mock()
    fci = mocker.Mock()
    main.load_match_level_stats(db, season="2026-27", vaastav=vaastav, fci=fci)
    vaastav.load.assert_called_once_with(db, current_season="2026-27")
    fci.load.assert_called_once_with(db, current_season="2026-27")


def test_load_match_level_stats_still_runs_fci_when_vaastav_fails(
    mocker, db, caplog
):
    """A dead Vaastav source must not block the current season's Opta data."""
    vaastav = mocker.Mock()
    vaastav.load.side_effect = RuntimeError("github is down")
    fci = mocker.Mock()
    with caplog.at_level("ERROR"):
        main.load_match_level_stats(
            db, season="2026-27", vaastav=vaastav, fci=fci
        )
    fci.load.assert_called_once()
    assert "github is down" in caplog.text
