"""Unit tests for fixture extraction adapters and transform."""

from datetime import datetime

import polars as pl
import pytest

from fantasy_football.extraction.fixtures import fixtures_to_team_rows


def test_transform_emits_two_rows_per_fixture() -> None:
    """Each fixture becomes a home row and an away row."""
    teams = {1: "Arsenal", 2: "Chelsea"}
    fixtures = [(1, "2023-08-11T19:00:00Z", 1, 2)]
    result = fixtures_to_team_rows(fixtures, teams, season="2023-24")
    assert result.height == 2
    home = result.filter(pl.col("team") == "Arsenal").row(0, named=True)
    away = result.filter(pl.col("team") == "Chelsea").row(0, named=True)
    assert home["opposition"] == "Chelsea"
    assert home["is_home"] is True
    assert away["opposition"] == "Arsenal"
    assert away["is_home"] is False
    for row in (home, away):
        assert row["season"] == "2023-24"
        assert row["gw"] == 1
        assert row["kickoff_time"] == datetime(2023, 8, 11, 19, 0)


def test_transform_kickoff_is_naive_utc() -> None:
    """kickoff_time is stored as a naive UTC datetime (no timezone)."""
    teams = {1: "Arsenal", 2: "Chelsea"}
    fixtures = [(1, "2023-08-11T19:00:00Z", 1, 2)]
    result = fixtures_to_team_rows(fixtures, teams, season="2023-24")
    assert result.schema["kickoff_time"] == pl.Datetime("us")
    assert result["kickoff_time"][0].tzinfo is None


def test_transform_handles_double_gameweek() -> None:
    """A team with two fixtures in one gw yields two rows."""
    teams = {1: "Arsenal", 2: "Chelsea", 3: "Spurs"}
    fixtures = [
        (29, "2026-03-14T15:00:00Z", 1, 2),
        (29, "2026-03-17T19:45:00Z", 1, 3),
    ]
    result = fixtures_to_team_rows(fixtures, teams, season="2025-26")
    arsenal = result.filter(pl.col("team") == "Arsenal").sort("kickoff_time")
    assert arsenal.height == 2
    assert arsenal["opposition"].to_list() == ["Chelsea", "Spurs"]


def test_transform_empty_returns_typed_empty_frame() -> None:
    """No fixtures yields an empty frame with the canonical schema."""
    result = fixtures_to_team_rows([], {1: "Arsenal"}, season="2023-24")
    assert result.is_empty()
    assert result.columns == [
        "season",
        "gw",
        "team",
        "is_home",
        "opposition",
        "kickoff_time",
    ]


def test_transform_unknown_team_id_raises() -> None:
    """An unknown team id raises KeyError."""
    teams = {1: "Arsenal"}  # id 2 missing
    fixtures = [(1, "2023-08-11T19:00:00Z", 1, 2)]
    with pytest.raises(KeyError):
        fixtures_to_team_rows(fixtures, teams, season="2023-24")
