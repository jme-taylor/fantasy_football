import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from fantasy_football import elo
from fantasy_football.elo import build_team_elo, normalize_elo_frame


def _clubelo_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["Rank", "Club", "Country", "Level", "Elo", "From", "To"])


def test_normalize_elo_frame_maps_names_and_renames_columns() -> None:
    """Verify normalize_elo_frame maps team names and renames columns correctly."""
    raw = _clubelo_df([
        {"Rank": 1.0, "Club": "Arsenal", "Country": "ENG", "Level": 1,
         "Elo": 2000.0, "From": "2025-08-01", "To": "2025-08-07"},
        {"Rank": 5.0, "Club": "Tottenham", "Country": "ENG", "Level": 1,
         "Elo": 1850.0, "From": "2025-08-01", "To": "2025-08-07"},
    ])

    result = normalize_elo_frame(raw)

    assert sorted(result.columns) == ["elo", "from_date", "team", "to_date"]
    teams = sorted(result["team"].to_list())
    assert teams == ["Arsenal", "Spurs"]


def test_normalize_elo_frame_warns_and_drops_unknown_teams(caplog: pytest.LogCaptureFixture) -> None:
    """Verify normalize_elo_frame warns and drops unknown teams."""
    raw = _clubelo_df([
        {"Rank": 1.0, "Club": "Mystery FC", "Country": "ENG", "Level": 1,
         "Elo": 1500.0, "From": "2025-08-01", "To": "2025-08-07"},
    ])

    with caplog.at_level(logging.WARNING, logger="fantasy_football.elo"):
        result = normalize_elo_frame(raw)

    assert result.is_empty()
    assert any("Mystery FC" in r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)


def test_build_team_elo_uses_cache_when_fresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify build_team_elo uses cache when it is fresh."""
    cache = tmp_path / "team_elo.csv"
    pl.DataFrame(
        {
            "team": ["Arsenal"],
            "elo": [2000.0],
            "from_date": [datetime(2025, 8, 1).date()],
            "to_date": [datetime(2025, 8, 7).date()],
        }
    ).write_csv(cache)
    monkeypatch.setattr(elo, "TRANSFORMED_DATA_FOLDER", tmp_path)

    called = {"n": 0}

    class FakeClubElo:
        def scrape_team(self, team: str) -> pd.DataFrame:
            called["n"] += 1
            return _clubelo_df([])

    monkeypatch.setattr(elo, "ClubElo", FakeClubElo)
    monkeypatch.setattr(elo, "ELO_CACHE_TTL_HOURS", 24)

    result = build_team_elo(force=False)

    assert called["n"] == 0
    assert result["team"].to_list() == ["Arsenal"]


def test_build_team_elo_force_bypasses_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify build_team_elo force flag bypasses cache."""
    cache = tmp_path / "team_elo.csv"
    pl.DataFrame(
        {
            "team": ["Arsenal"],
            "elo": [1.0],
            "from_date": [datetime(2025, 8, 1).date()],
            "to_date": [datetime(2025, 8, 7).date()],
        }
    ).write_csv(cache)
    monkeypatch.setattr(elo, "TRANSFORMED_DATA_FOLDER", tmp_path)
    monkeypatch.setattr(elo, "CLUBELO_SCRAPE_NAMES", ["Arsenal"])
    monkeypatch.setattr(elo, "CLUBELO_TO_FPL", {"Arsenal": "Arsenal"})

    class FakeClubElo:
        def scrape_team(self, team: str) -> pd.DataFrame:
            return _clubelo_df([
                {"Rank": 1.0, "Club": team, "Country": "ENG", "Level": 1,
                 "Elo": 2050.0, "From": "2025-08-08", "To": "2025-08-14"},
            ])

    monkeypatch.setattr(elo, "ClubElo", FakeClubElo)

    result = build_team_elo(force=True)

    assert result["elo"].to_list() == [2050.0]


def test_build_team_elo_falls_back_to_cache_on_scrape_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify build_team_elo falls back to cache when scrape fails."""
    cache = tmp_path / "team_elo.csv"
    pl.DataFrame(
        {
            "team": ["Arsenal"],
            "elo": [1900.0],
            "from_date": [datetime(2025, 8, 1).date()],
            "to_date": [datetime(2025, 8, 7).date()],
        }
    ).write_csv(cache)
    # Make the cache file appear stale.
    stale = (datetime.now() - timedelta(hours=48)).timestamp()
    import os
    os.utime(cache, (stale, stale))

    monkeypatch.setattr(elo, "TRANSFORMED_DATA_FOLDER", tmp_path)
    monkeypatch.setattr(elo, "CLUBELO_SCRAPE_NAMES", ["Arsenal"])
    monkeypatch.setattr(elo, "CLUBELO_TO_FPL", {"Arsenal": "Arsenal"})
    monkeypatch.setattr(elo, "ELO_CACHE_TTL_HOURS", 24)

    class ExplodingClubElo:
        def scrape_team(self, team: str) -> pd.DataFrame:
            raise RuntimeError("network down")

    monkeypatch.setattr(elo, "ClubElo", ExplodingClubElo)

    with caplog.at_level(logging.WARNING, logger="fantasy_football.elo"):
        result = build_team_elo(force=False)

    assert result["elo"].to_list() == [1900.0]
    assert any("network down" in r.getMessage() or "scrape failed" in r.getMessage().lower()
               for r in caplog.records)


def test_build_team_elo_scrape_failure_no_cache_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify build_team_elo raises when scrape fails and no cache exists."""
    monkeypatch.setattr(elo, "TRANSFORMED_DATA_FOLDER", tmp_path)
    monkeypatch.setattr(elo, "CLUBELO_SCRAPE_NAMES", ["Arsenal"])
    monkeypatch.setattr(elo, "CLUBELO_TO_FPL", {"Arsenal": "Arsenal"})

    class ExplodingClubElo:
        def scrape_team(self, team: str) -> pd.DataFrame:
            raise RuntimeError("network down")

    monkeypatch.setattr(elo, "ClubElo", ExplodingClubElo)

    with pytest.raises(RuntimeError):
        build_team_elo(force=True)
