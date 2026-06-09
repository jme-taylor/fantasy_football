import json

import pytest

from fantasy_football.team_input import TeamFile, load_team_file


def _write_team(tmp_path, payload) -> str:
    path = tmp_path / "team.json"
    path.write_text(json.dumps(payload))
    return str(path)


def test_load_team_file_reads_fields_and_defaults_bank(tmp_path) -> None:
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
    path = _write_team(
        tmp_path,
        {
            "gameweek": 5,
            "free_transfers": 1,
            "players": [],
            "unexpected": True,
        },
    )
    with pytest.raises(Exception):
        load_team_file(path)


def test_load_team_file_rejects_missing_field(tmp_path) -> None:
    path = _write_team(tmp_path, {"gameweek": 5, "players": []})
    with pytest.raises(Exception):
        load_team_file(path)
