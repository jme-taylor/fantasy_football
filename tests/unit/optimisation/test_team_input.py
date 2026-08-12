import json
from datetime import datetime

import polars as pl
import pytest

from fantasy_football.optimisation.team_input import (
    TeamFile,
    load_team_file,
    resolve_names_to_ids,
)
from fantasy_football.storage import database
from fantasy_football.storage.database import get_connection
from fantasy_football.storage.tables import PLAYER_SEASON, PLAYER_SNAPSHOT

SEASON = "2026-27"


def _write_team(tmp_path, payload) -> str:
    path = tmp_path / "team.json"
    path.write_text(json.dumps(payload))
    return str(path)


def test_load_team_file_reads_fields_and_defaults_bank(tmp_path) -> None:
    """load_team_file parses all fields correctly and defaults bank to 0."""
    path = _write_team(
        tmp_path,
        {
            "gameweek": 5,
            "free_transfers": 2,
            "players": [f"P{i}" for i in range(15)],
        },
    )
    team = load_team_file(path)
    assert isinstance(team, TeamFile)
    assert team.gameweek == 5
    assert team.free_transfers == 2
    assert team.bank == 0
    assert team.players == [f"P{i}" for i in range(15)]


def test_load_team_file_rejects_extra_keys(tmp_path) -> None:
    """load_team_file raises ValueError when the JSON contains unexpected keys."""
    path = _write_team(
        tmp_path,
        {
            "gameweek": 5,
            "free_transfers": 1,
            "players": [],
            "unexpected": True,
        },
    )
    with pytest.raises(ValueError):
        load_team_file(path)


def test_load_team_file_rejects_missing_field(tmp_path) -> None:
    """load_team_file raises ValueError when a required field is absent."""
    path = _write_team(tmp_path, {"gameweek": 5, "players": []})
    with pytest.raises(ValueError):
        load_team_file(path)


def _seed_roster(
    tmp_path, monkeypatch, *, names, elements, season=SEASON
) -> None:
    """Seed the snapshot and identity rows the roster is built from."""
    captured = datetime(2026, 8, 1, 12, 0)
    snapshot = pl.DataFrame(
        [
            {
                "season": season,
                "captured_at": captured,
                "element": element,
                "value": 50,
                "team": "T",
                "position": "MID",
                "chance_of_playing_this_round": 100,
            }
            for element in dict.fromkeys(elements)
        ]
    )
    identity = pl.DataFrame(
        [
            {
                "season": season,
                "element": element,
                "player_code": 10_000 + element,
                "web_name": name.split(" ", 1)[1],
                "first_name": name.split(" ")[0],
                "second_name": name.split(" ", 1)[1],
                "position": "MID",
                "team_code": element,
                "birth_date": None,
                "region": None,
                "team_join_date": None,
            }
            for name, element in zip(names, elements, strict=True)
        ],
        schema_overrides={
            "birth_date": pl.Date,
            "region": pl.Int64,
            "team_join_date": pl.Date,
        },
    )
    db_path = tmp_path / "t.duckdb"
    monkeypatch.setattr(database, "DATABASE_PATH", db_path)
    connection = get_connection(db_path)
    try:
        PLAYER_SNAPSHOT.upsert_current(connection, snapshot, season)
        PLAYER_SEASON.upsert_current(connection, identity, season)
    finally:
        connection.close()


def test_resolve_names_to_ids_maps_names(tmp_path, monkeypatch) -> None:
    """resolve_names_to_ids returns the element id for each name, in order."""
    _seed_roster(
        tmp_path,
        monkeypatch,
        names=["Mohamed Salah", "Erling Haaland"],
        elements=[328, 351],
    )
    ids = resolve_names_to_ids(["Erling Haaland", "Mohamed Salah"], SEASON)
    assert ids == [351, 328]


def test_resolve_names_to_ids_reports_unmatched(tmp_path, monkeypatch) -> None:
    """An unmatched name is named in the raised ValueError."""
    _seed_roster(
        tmp_path, monkeypatch, names=["Mohamed Salah"], elements=[328]
    )
    with pytest.raises(ValueError, match="Ghost Player"):
        resolve_names_to_ids(["Mohamed Salah", "Ghost Player"], SEASON)


def test_resolve_names_to_ids_reports_ambiguous(tmp_path, monkeypatch) -> None:
    """A name mapping to multiple elements raises an 'ambiguous' ValueError."""
    _seed_roster(
        tmp_path,
        monkeypatch,
        names=["Danny Ward", "Danny Ward"],
        elements=[11, 22],
    )
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_names_to_ids(["Danny Ward"], SEASON)


def test_resolve_names_to_ids_reports_unmatched_and_ambiguous_together(
    tmp_path, monkeypatch
) -> None:
    """Both unmatched and ambiguous offenders are named in one error."""
    _seed_roster(
        tmp_path,
        monkeypatch,
        names=["Danny Ward", "Danny Ward"],
        elements=[11, 22],
    )
    with pytest.raises(ValueError) as exc:
        resolve_names_to_ids(["Danny Ward", "Ghost Player"], SEASON)
    assert "Danny Ward" in str(exc.value)
    assert "Ghost Player" in str(exc.value)


def test_resolve_names_to_ids_finds_a_player_who_has_not_played(
    tmp_path, monkeypatch
) -> None:
    """A summer signing is on the roster, so their name resolves."""
    _seed_roster(tmp_path, monkeypatch, names=["New Signing"], elements=[500])
    assert resolve_names_to_ids(["New Signing"], SEASON) == [500]
