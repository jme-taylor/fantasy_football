"""Tests for the current-season roster built from the FPL snapshot."""

from datetime import datetime

import polars as pl
import pytest

from fantasy_football.features.roster import current_roster
from fantasy_football.storage import database
from fantasy_football.storage.database import get_connection
from fantasy_football.storage.tables import PLAYER_SEASON, PLAYER_SNAPSHOT


def _seed(
    tmp_path,
    monkeypatch,
    season="2026-27",
    captures=1,
    departed_element: int | None = None,
) -> None:
    """Seed player_snapshot and player_season into a tmp database."""
    stamps = [datetime(2026, 7, 20, 12, 0), datetime(2026, 7, 30, 12, 0)]
    rows = []
    for stamp in stamps[:captures]:
        value = 130 if stamp == stamps[0] else 135
        rows.append(
            {
                "season": season,
                "captured_at": stamp,
                "element": 55,
                "value": value,
                "team": "Arsenal",
                "position": "MID",
                "chance_of_playing_this_round": 100,
            }
        )
    rows.append(
        {
            "season": season,
            "captured_at": stamps[captures - 1],
            "element": 56,
            "value": 45,
            "team": "Hull City",
            "position": "DEF",
            "chance_of_playing_this_round": 100,
        }
    )
    for row in rows:
        row["status"] = "u" if row["element"] == departed_element else "a"
    snapshot = pl.DataFrame(rows)
    identity = pl.DataFrame(
        {
            "season": [season] * 2,
            "element": [55, 56],
            "player_code": [999, 1000],
            "web_name": ["Saka", "Newman"],
            "first_name": ["Bukayo", "Ryan"],
            "second_name": ["Saka", "Newman"],
            "position": ["MID", "DEF"],
            "team_code": [3, 88],
            "birth_date": [None, None],
            "region": [None, None],
            "team_join_date": [None, None],
        },
        schema_overrides={
            "birth_date": pl.Date,
            "region": pl.Int64,
            "team_join_date": pl.Date,
        },
    )
    db_path = tmp_path / "roster.duckdb"
    monkeypatch.setattr(database, "DATABASE_PATH", db_path)
    connection = get_connection(db_path)
    try:
        PLAYER_SNAPSHOT.upsert_current(connection, snapshot, season)
        PLAYER_SEASON.upsert_current(connection, identity, season)
    finally:
        connection.close()


def test_current_roster_joins_snapshot_to_identity(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The roster carries snapshot club, position and price with a name."""
    _seed(tmp_path, monkeypatch)

    roster = current_roster("2026-27")

    assert sorted(roster.columns) == [
        "element",
        "is_departed",
        "name",
        "player_code",
        "position",
        "team",
        "value",
    ]
    assert sorted(roster["name"].to_list()) == ["Bukayo Saka", "Ryan Newman"]
    saka = roster.filter(pl.col("name") == "Bukayo Saka")
    assert saka["team"].item() == "Arsenal"
    assert saka["player_code"].item() == 999
    assert saka["value"].item() == 130


def test_current_roster_uses_only_the_newest_capture(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An older capture never contributes rows or stale prices."""
    _seed(tmp_path, monkeypatch, captures=2)

    roster = current_roster("2026-27")

    assert roster.filter(pl.col("name") == "Bukayo Saka").height == 1
    assert (
        roster.filter(pl.col("name") == "Bukayo Saka")["value"].item() == 135
    )


def test_current_roster_flags_departed_players(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A departed player stays on the roster, marked, so he can be sold."""
    _seed(tmp_path, monkeypatch, departed_element=56)

    roster = current_roster("2026-27")

    departed = dict(
        zip(
            roster["element"].to_list(),
            roster["is_departed"].to_list(),
            strict=True,
        )
    )
    assert departed == {55: False, 56: True}


def test_current_roster_is_empty_for_a_season_with_no_snapshot(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A season with no capture yields an empty frame, not an error."""
    _seed(tmp_path, monkeypatch)

    roster = current_roster("2024-25")

    assert roster.is_empty()
    assert "name" in roster.columns
    assert "value" in roster.columns
