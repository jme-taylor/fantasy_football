"""Tests for the predict_points baseline heuristic."""

import logging
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from fantasy_football import prediction
from fantasy_football.prediction import predict_points


def _setup_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rolling: pl.DataFrame,
    fixtures: pl.DataFrame,
    team_elo: pl.DataFrame,
) -> Path:
    """Write CSV artifacts to tmp_path and monkeypatch the folder constant."""
    transformed = tmp_path / "transformed"
    transformed.mkdir()
    rolling.write_csv(transformed / "rolling_points.csv")
    fixtures.write_csv(transformed / "fixtures_enriched.csv")
    team_elo.write_csv(transformed / "team_elo.csv")
    monkeypatch.setattr(prediction, "TRANSFORMED_DATA_FOLDER", transformed)
    return transformed


def _baseline_rolling() -> pl.DataFrame:
    """Return a minimal single-player rolling-points DataFrame."""
    return pl.DataFrame(
        {
            "season": ["2025-26"],
            "name": ["P1"],
            "position": ["MID"],
            "team": ["Arsenal"],
            "element": [101],
            "gw": [10],
            "total_points": [6],
            "total_points_rolling_5": [4.0],
        }
    )


def _baseline_fixtures() -> pl.DataFrame:
    """Return two future fixtures for Arsenal."""
    return pl.DataFrame(
        {
            "team": ["Arsenal", "Arsenal"],
            "opponent_team": ["Chelsea", "Spurs"],
            "is_home": [True, False],
            "kickoff_date": [date(2025, 11, 1), date(2025, 11, 8)],
            "season": ["2025-26", "2025-26"],
            "gw": [11, 12],
        }
    )


def _baseline_elo() -> pl.DataFrame:
    """Return ELO ratings for Arsenal, Chelsea, and Spurs."""
    return pl.DataFrame(
        {
            "team": ["Arsenal", "Chelsea", "Spurs"],
            "elo": [2000.0, 1800.0, 1900.0],
            "from_date": [date(2025, 10, 1)] * 3,
            "to_date": [date(2025, 12, 31)] * 3,
        }
    )


def test_predict_points_formula_correct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify predicted_points matches the baseline × ELO-ratio × home/away formula."""
    _setup_artifacts(
        tmp_path,
        monkeypatch,
        _baseline_rolling(),
        _baseline_fixtures(),
        _baseline_elo(),
    )
    monkeypatch.setattr(prediction, "OPPONENT_FACTOR_EXPONENT", 2.0)
    monkeypatch.setattr(prediction, "HOME_FACTOR", 1.25)
    monkeypatch.setattr(prediction, "AWAY_FACTOR", 0.75)
    # Pin the random noise factor to its identity value so the deterministic
    # formula can be asserted exactly; production runs stay randomised.
    monkeypatch.setattr(prediction.random, "choice", lambda _seq: 1.00)

    result = predict_points("2025-26", horizon_n=2)

    home = result.filter(pl.col("gw") == 11).row(0, named=True)
    assert home["predicted_points"] == pytest.approx(
        4.0 * (2000 / 1800) ** 2.0 * 1.25
    )
    assert home["opponent_team"] == "Chelsea"
    assert home["is_home"] is True
    away = result.filter(pl.col("gw") == 12).row(0, named=True)
    assert away["predicted_points"] == pytest.approx(
        4.0 * (2000 / 1900) ** 2.0 * 0.75
    )


def test_predict_points_uses_most_recent_baseline_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify the baseline comes from the player's most recent gameweek row."""
    rolling = pl.DataFrame(
        {
            "season": ["2025-26"] * 3,
            "name": ["P1"] * 3,
            "position": ["MID"] * 3,
            "team": ["Arsenal"] * 3,
            "element": [101, 101, 101],
            "gw": [8, 9, 10],
            "total_points": [2, 4, 6],
            "total_points_rolling_5": [3.0, 3.5, 4.0],
        }
    )
    _setup_artifacts(
        tmp_path, monkeypatch, rolling, _baseline_fixtures(), _baseline_elo()
    )

    result = predict_points("2025-26", horizon_n=1)

    home = result.filter(pl.col("gw") == 11).row(0, named=True)
    assert home["baseline"] == pytest.approx(4.0)


def test_predict_points_includes_player_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """predict_points carries the FPL element id through as player_id."""
    _setup_artifacts(
        tmp_path,
        monkeypatch,
        _baseline_rolling(),
        _baseline_fixtures(),
        _baseline_elo(),
    )

    result = predict_points("2025-26", horizon_n=2)

    assert "player_id" in result.columns
    assert result["player_id"].dtype == pl.Int64
    assert result.filter(pl.col("gw") == 11)["player_id"].item() == 101
    assert result["player_id"].null_count() == 0


def test_predict_points_horizon_truncates_when_fixtures_run_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify horizon_n does not produce rows beyond available fixtures."""
    _setup_artifacts(
        tmp_path,
        monkeypatch,
        _baseline_rolling(),
        _baseline_fixtures(),
        _baseline_elo(),
    )

    result = predict_points("2025-26", horizon_n=10)

    assert set(result["gw"].to_list()) == {11, 12}


def test_predict_points_double_gameweek_produces_two_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify a double gameweek yields two prediction rows."""
    fixtures = pl.DataFrame(
        {
            "team": ["Arsenal", "Arsenal"],
            "opponent_team": ["Chelsea", "Spurs"],
            "is_home": [True, False],
            "kickoff_date": [date(2025, 11, 1), date(2025, 11, 3)],
            "season": ["2025-26", "2025-26"],
            "gw": [11, 11],
        }
    )
    _setup_artifacts(
        tmp_path, monkeypatch, _baseline_rolling(), fixtures, _baseline_elo()
    )

    result = predict_points("2025-26", horizon_n=1)

    assert result.height == 2
    assert sorted(result["opponent_team"].to_list()) == ["Chelsea", "Spurs"]


def test_predict_points_blank_gameweek_produces_no_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify a blank gameweek for a player produces no prediction row."""
    fixtures = pl.DataFrame(
        {
            "team": ["Arsenal"],
            "opponent_team": ["Spurs"],
            "is_home": [False],
            "kickoff_date": [date(2025, 11, 8)],
            "season": ["2025-26"],
            "gw": [12],
        }
    )
    _setup_artifacts(
        tmp_path, monkeypatch, _baseline_rolling(), fixtures, _baseline_elo()
    )

    result = predict_points("2025-26", horizon_n=2)

    assert result["gw"].to_list() == [12]


def test_predict_points_missing_opponent_elo_uses_median_and_warns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify missing opponent ELO falls back to median and emits a warning."""
    elo = pl.DataFrame(
        {
            "team": ["Arsenal", "Spurs", "Liverpool"],
            "elo": [2000.0, 1900.0, 1800.0],
            "from_date": [date(2025, 10, 1)] * 3,
            "to_date": [date(2025, 12, 31)] * 3,
        }
    )
    _setup_artifacts(
        tmp_path, monkeypatch, _baseline_rolling(), _baseline_fixtures(), elo
    )

    with caplog.at_level(
        logging.WARNING, logger="fantasy_football.prediction"
    ):
        result = predict_points("2025-26", horizon_n=1)

    row = result.filter(pl.col("gw") == 11).row(0, named=True)
    assert row["player_team_elo"] == pytest.approx(2000.0)
    assert row["opponent_team_elo"] == pytest.approx(1900.0)
    assert any(
        "Chelsea" in r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING
    )


def test_predict_points_as_of_elo_picks_most_recent_pre_fixture_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify as-of ELO lookup uses the most recent row with from_date <= kickoff_date."""
    fixtures = pl.DataFrame(
        {
            "team": ["Arsenal"],
            "opponent_team": ["Chelsea"],
            "is_home": [True],
            "kickoff_date": [date(2025, 11, 1)],
            "season": ["2025-26"],
            "gw": [11],
        }
    )
    elo = pl.DataFrame(
        {
            "team": ["Arsenal", "Arsenal", "Chelsea", "Chelsea"],
            "elo": [1900.0, 2100.0, 1700.0, 1850.0],
            "from_date": [
                date(2025, 10, 1),
                date(2025, 11, 15),
                date(2025, 10, 1),
                date(2025, 11, 15),
            ],
            "to_date": [
                date(2025, 11, 14),
                date(2025, 12, 31),
                date(2025, 11, 14),
                date(2025, 12, 31),
            ],
        }
    )
    _setup_artifacts(tmp_path, monkeypatch, _baseline_rolling(), fixtures, elo)

    result = predict_points("2025-26", horizon_n=1).row(0, named=True)

    assert result["player_team_elo"] == pytest.approx(1900.0)
    assert result["opponent_team_elo"] == pytest.approx(1700.0)


def test_predict_points_default_horizon_covers_all_future_gameweeks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify horizon_n=None predicts every remaining gameweek of the season."""
    fixtures = pl.DataFrame(
        {
            "team": ["Arsenal"] * 4,
            "opponent_team": ["Chelsea", "Spurs", "Liverpool", "Everton"],
            "is_home": [True, False, True, False],
            "kickoff_date": [
                date(2025, 11, 1),
                date(2025, 11, 8),
                date(2025, 11, 15),
                date(2025, 11, 22),
            ],
            "season": ["2025-26"] * 4,
            "gw": [11, 12, 13, 14],
        }
    )
    elo = pl.DataFrame(
        {
            "team": ["Arsenal", "Chelsea", "Spurs", "Liverpool", "Everton"],
            "elo": [2000.0, 1800.0, 1900.0, 1950.0, 1700.0],
            "from_date": [date(2025, 10, 1)] * 5,
            "to_date": [date(2025, 12, 31)] * 5,
        }
    )
    _setup_artifacts(tmp_path, monkeypatch, _baseline_rolling(), fixtures, elo)

    result = predict_points("2025-26")

    assert sorted(result["gw"].to_list()) == [11, 12, 13, 14]


def test_predict_points_default_horizon_with_no_future_fixtures_is_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify horizon_n=None returns empty result (no crash) when season is over."""
    # All fixtures are at or before last_completed gw (10).
    fixtures = pl.DataFrame(
        {
            "team": ["Arsenal", "Arsenal"],
            "opponent_team": ["Chelsea", "Spurs"],
            "is_home": [True, False],
            "kickoff_date": [date(2025, 8, 1), date(2025, 9, 1)],
            "season": ["2025-26", "2025-26"],
            "gw": [1, 2],
        }
    )
    _setup_artifacts(
        tmp_path, monkeypatch, _baseline_rolling(), fixtures, _baseline_elo()
    )

    result = predict_points("2025-26")

    assert result.is_empty()
