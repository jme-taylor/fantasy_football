import json

import polars as pl
import pytest

from fantasy_football import team_input
from fantasy_football.team_input import (
    TeamFile,
    load_team_file,
    resolve_ids_to_names,
    resolve_names_to_ids,
)


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


def _write_merged_gw(tmp_path, rows) -> None:
    raw = tmp_path / "raw" / "2025-26" / "gws"
    raw.mkdir(parents=True)
    pl.DataFrame(rows).write_csv(raw / "merged_gw.csv")


def test_resolve_names_to_ids_maps_names(tmp_path, monkeypatch) -> None:
    """resolve_names_to_ids returns the element id for each name, in order."""
    _write_merged_gw(
        tmp_path,
        {
            "name": ["Mohamed Salah", "Mohamed Salah", "Erling Haaland"],
            "element": [328, 328, 351],
            "GW": [4, 5, 5],
        },
    )
    monkeypatch.setattr(team_input, "RAW_DATA_FOLDER", tmp_path / "raw")
    ids = resolve_names_to_ids(["Erling Haaland", "Mohamed Salah"], "2025-26")
    assert ids == [351, 328]


def test_resolve_names_to_ids_reports_unmatched(tmp_path, monkeypatch) -> None:
    """An unmatched name is named in the raised ValueError."""
    _write_merged_gw(
        tmp_path,
        {"name": ["Mohamed Salah"], "element": [328], "GW": [5]},
    )
    monkeypatch.setattr(team_input, "RAW_DATA_FOLDER", tmp_path / "raw")
    with pytest.raises(ValueError, match="Ghost Player"):
        resolve_names_to_ids(["Mohamed Salah", "Ghost Player"], "2025-26")


def test_resolve_names_to_ids_reports_ambiguous(tmp_path, monkeypatch) -> None:
    """A name mapping to multiple elements raises an 'ambiguous' ValueError."""
    _write_merged_gw(
        tmp_path,
        {
            "name": ["Danny Ward", "Danny Ward"],
            "element": [11, 22],
            "GW": [5, 5],
        },
    )
    monkeypatch.setattr(team_input, "RAW_DATA_FOLDER", tmp_path / "raw")
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_names_to_ids(["Danny Ward"], "2025-26")


def test_resolve_names_to_ids_reports_unmatched_and_ambiguous_together(
    tmp_path, monkeypatch
) -> None:
    """Both unmatched and ambiguous offenders are named in one error."""
    _write_merged_gw(
        tmp_path,
        {
            "name": ["Danny Ward", "Danny Ward"],
            "element": [11, 22],
            "GW": [5, 5],
        },
    )
    monkeypatch.setattr(team_input, "RAW_DATA_FOLDER", tmp_path / "raw")
    with pytest.raises(ValueError) as exc:
        resolve_names_to_ids(["Danny Ward", "Ghost Player"], "2025-26")
    assert "Danny Ward" in str(exc.value)
    assert "Ghost Player" in str(exc.value)


def _predictions(rows) -> pl.DataFrame:
    """Build a small predictions DataFrame from a dict of columns."""
    return pl.DataFrame(rows)


def test_resolve_ids_to_names_maps_ids() -> None:
    """resolve_ids_to_names returns the name for each id, in order."""
    predictions = _predictions(
        {
            "name": ["Mohamed Salah", "Mohamed Salah", "Erling Haaland"],
            "player_id": [328, 328, 351],
            "gw": [5, 6, 5],
        }
    )
    names = resolve_ids_to_names([351, 328], predictions)
    assert names == ["Erling Haaland", "Mohamed Salah"]


def test_resolve_ids_to_names_reports_missing_id() -> None:
    """An id absent from predictions is named in the raised ValueError."""
    predictions = _predictions(
        {"name": ["Mohamed Salah"], "player_id": [328], "gw": [5]}
    )
    with pytest.raises(ValueError, match="999"):
        resolve_ids_to_names([328, 999], predictions)
