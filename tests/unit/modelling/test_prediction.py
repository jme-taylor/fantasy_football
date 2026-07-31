"""Tests for the predict_points baseline heuristic."""

import logging
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from fantasy_football.modelling import prediction
from fantasy_football.modelling.prediction import (
    _baselines,
    _latest_rolling_by_code,
    predict_points,
)
from fantasy_football.storage import database


def _setup_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rolling: pl.DataFrame,
    fixtures: pl.DataFrame,
    team_elo: pl.DataFrame,
) -> Path:
    """Write CSV artifacts to tmp_path and monkeypatch the folder constant.

    Also points ``DATABASE_PATH`` at a fresh, empty database in ``tmp_path``,
    so ``current_roster`` -- called by ``_predict`` whenever ``as_of_gw`` is
    None -- resolves hermetically to an empty roster (falling back to the
    pre-existing current-season behaviour) instead of touching the real,
    developer-local database.
    """
    transformed = tmp_path / "transformed"
    transformed.mkdir()
    rolling.write_csv(transformed / "rolling_points.csv")
    fixtures.write_csv(transformed / "fixtures_enriched.csv")
    team_elo.write_csv(transformed / "team_elo.csv")
    monkeypatch.setattr(prediction, "TRANSFORMED_DATA_FOLDER", transformed)
    monkeypatch.setattr(database, "DATABASE_PATH", tmp_path / "test.duckdb")
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
            "player_code": [999],
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


def _two_season_rolling() -> pl.DataFrame:
    """Return rolling rows for one player spanning a season boundary."""
    return pl.DataFrame(
        {
            "season": ["2025-26", "2025-26", "2026-27"],
            "name": ["P1"] * 3,
            "position": ["MID"] * 3,
            "team": ["Arsenal"] * 3,
            "element": [101, 101, 55],
            "player_code": [999, 999, 999],
            "gw": [37, 38, 1],
            "total_points": [6, 8, 2],
            "total_points_rolling_5": [4.0, 5.0, 3.0],
        }
    )


def test_latest_rolling_by_code_uses_prior_season_when_current_is_empty() -> (
    None
):
    """With no current-season rows, the baseline comes from last season."""
    rolling = _two_season_rolling().filter(pl.col("season") == "2025-26")

    result = _latest_rolling_by_code(rolling, "2026-27", None)

    assert result.height == 1
    assert result["player_code"].to_list() == [999]
    assert result["baseline"].to_list() == [5.0]  # GW38, not GW37


def test_latest_rolling_by_code_prefers_the_current_season() -> None:
    """A current-season row wins over any prior-season row."""
    result = _latest_rolling_by_code(_two_season_rolling(), "2026-27", None)

    assert result["baseline"].to_list() == [3.0]


def test_latest_rolling_by_code_respects_the_as_of_cutoff() -> None:
    """as_of_gw excludes later current-season rows but keeps prior seasons."""
    rolling = pl.concat(
        [
            _two_season_rolling(),
            pl.DataFrame(
                {
                    "season": ["2026-27"],
                    "name": ["P1"],
                    "position": ["MID"],
                    "team": ["Arsenal"],
                    "element": [55],
                    "player_code": [999],
                    "gw": [5],
                    "total_points": [9],
                    "total_points_rolling_5": [7.0],
                }
            ),
        ],
        how="vertical",
    )

    result = _latest_rolling_by_code(rolling, "2026-27", as_of_gw=1)

    assert result["baseline"].to_list() == [3.0]  # GW1, not the GW5 row


def test_latest_rolling_by_code_ignores_a_season_later_than_current() -> None:
    """A row from a season after current_season never contributes a baseline.

    Eligibility must be ordering-aware (season <= current_season), not just
    equality-aware (season != current_season). Otherwise a future-season row
    would sail straight through the as_of_gw filter unfiltered, since it
    isn't equal to current_season.
    """
    rolling = pl.DataFrame(
        {
            "season": ["2025-26", "2027-28"],
            "name": ["P1", "P1"],
            "position": ["MID", "MID"],
            "team": ["Arsenal", "Arsenal"],
            "element": [101, 9],
            "player_code": [999, 999],
            "gw": [38, 1],
            "total_points": [8, 20],
            "total_points_rolling_5": [5.0, 20.0],
        }
    )

    result = _latest_rolling_by_code(rolling, "2026-27", as_of_gw=None)

    assert result["baseline"].to_list() == [5.0]  # 2025-26, not 2027-28


def test_latest_rolling_by_code_drops_rows_with_no_player_code() -> None:
    """A null player_code has no stable identity, so it is excluded."""
    rolling = pl.DataFrame(
        {
            "season": ["2025-26"],
            "name": ["Unknown"],
            "position": ["MID"],
            "team": ["Arsenal"],
            "element": [7],
            "player_code": [None],
            "gw": [38],
            "total_points": [3],
            "total_points_rolling_5": [3.0],
        },
        schema_overrides={"player_code": pl.Int64},
    )

    result = _latest_rolling_by_code(rolling, "2026-27", None)

    assert result.is_empty()


def test_baselines_without_a_roster_is_unchanged() -> None:
    """Omitting roster preserves the pre-existing current-season behaviour."""
    result = _baselines(_baseline_rolling(), "2025-26")

    assert result["name"].to_list() == ["P1"]
    assert result["baseline"].to_list() == [4.0]


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
        logging.WARNING, logger="fantasy_football.modelling.prediction"
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


def test_baselines_as_of_gw_uses_row_at_or_before_pivot() -> None:
    """as_of_gw selects each player's latest rolling row with gw <= pivot."""
    rolling = pl.DataFrame(
        {
            "season": ["2025-26"] * 4,
            "name": ["P1"] * 4,
            "position": ["MID"] * 4,
            "team": ["Arsenal"] * 4,
            "element": [101, 101, 101, 101],
            "gw": [3, 4, 5, 6],
            "total_points": [2, 4, 6, 8],
            "total_points_rolling_5": [2.0, 3.0, 4.0, 5.0],
        }
    )
    result = _baselines(rolling, "2025-26", as_of_gw=4)
    row = result.row(0, named=True)
    # The gw4 rolling value (3.0), not the later gw5/gw6 values.
    assert row["baseline"] == pytest.approx(3.0)


def test_baselines_without_as_of_uses_latest_row() -> None:
    """With as_of_gw=None the baseline is the player's latest rolling row."""
    rolling = pl.DataFrame(
        {
            "season": ["2025-26"] * 3,
            "name": ["P1"] * 3,
            "position": ["MID"] * 3,
            "team": ["Arsenal"] * 3,
            "element": [101, 101, 101],
            "gw": [4, 5, 6],
            "total_points": [4, 6, 8],
            "total_points_rolling_5": [3.0, 4.0, 5.0],
        }
    )
    result = _baselines(rolling, "2025-26")
    assert result.row(0, named=True)["baseline"] == pytest.approx(5.0)


def test_predict_points_as_of_gw_predicts_window_with_as_of_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """as_of_gw=4 predicts GW>=5 using the as-of-GW4 baseline, not later form."""
    rolling = pl.DataFrame(
        {
            "season": ["2025-26"] * 4,
            "name": ["P1"] * 4,
            "position": ["MID"] * 4,
            "team": ["Arsenal"] * 4,
            "element": [101, 101, 101, 101],
            "gw": [3, 4, 5, 6],
            "total_points": [2, 4, 6, 8],
            "total_points_rolling_5": [2.0, 3.0, 4.0, 5.0],
        }
    )
    fixtures = pl.DataFrame(
        {
            "team": ["Arsenal", "Arsenal"],
            "opponent_team": ["Chelsea", "Spurs"],
            "is_home": [True, False],
            # Kickoffs must fall inside the _baseline_elo interval
            # (2025-10-01..2025-12-31) so the as-of ELO join matches.
            "kickoff_date": [date(2025, 11, 1), date(2025, 11, 8)],
            "season": ["2025-26", "2025-26"],
            "gw": [5, 6],
        }
    )
    _setup_artifacts(tmp_path, monkeypatch, rolling, fixtures, _baseline_elo())
    monkeypatch.setattr(prediction, "OPPONENT_FACTOR_EXPONENT", 2.0)
    monkeypatch.setattr(prediction, "HOME_FACTOR", 1.25)
    monkeypatch.setattr(prediction, "AWAY_FACTOR", 0.75)

    result = predict_points("2025-26", as_of_gw=4)

    assert sorted(result["gw"].unique().to_list()) == [5, 6]
    gw5 = result.filter(pl.col("gw") == 5).row(0, named=True)
    # Baseline is the as-of-GW4 rolling value (3.0), NOT the GW6 value (5.0).
    assert gw5["baseline"] == pytest.approx(3.0)
    assert gw5["predicted_points"] == pytest.approx(
        3.0 * (2000 / 1800) ** 2.0 * 1.25
    )


def test_predict_pure_path_writes_no_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_predict returns predictions without writing predictions.csv."""
    transformed = _setup_artifacts(
        tmp_path,
        monkeypatch,
        _baseline_rolling(),
        _baseline_fixtures(),
        _baseline_elo(),
    )
    result = prediction._predict("2025-26", horizon_n=2)
    assert not (transformed / "predictions.csv").exists()
    assert result.height == 2


def test_predict_drops_unknown_position(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Players whose position is not in KNOWN_POSITIONS are absent from output.

    Build a rolling DataFrame with two players: one standard MID (element 101)
    and one with position "AM" (element 202) which is not a known position.
    Assert that element 202 does not appear in the result while element 101
    does, confirming unknown-position rows are dropped by _apply_models.
    """
    rolling = pl.DataFrame(
        {
            "season": ["2025-26", "2025-26"],
            "name": ["P1", "P2"],
            "position": ["MID", "AM"],
            "team": ["Arsenal", "Arsenal"],
            "element": [101, 202],
            "gw": [10, 10],
            "total_points": [6, 4],
            "total_points_rolling_5": [4.0, 3.0],
        }
    )
    _setup_artifacts(
        tmp_path, monkeypatch, rolling, _baseline_fixtures(), _baseline_elo()
    )

    result = prediction._predict("2025-26", horizon_n=1)

    assert 101 in result["player_id"].to_list()
    assert 202 not in result["player_id"].to_list()


def test_predict_points_is_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two runs with identical inputs produce identical predictions."""
    _setup_artifacts(
        tmp_path,
        monkeypatch,
        _baseline_rolling(),
        _baseline_fixtures(),
        _baseline_elo(),
    )
    first = predict_points("2025-26", horizon_n=2)[
        "predicted_points"
    ].to_list()
    second = predict_points("2025-26", horizon_n=2)[
        "predicted_points"
    ].to_list()
    assert first == second


def test_predict_points_works_with_no_current_season_rolling_rows(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pre-season case: last season's form plus this season's roster.

    This is the regression test for the whole bug -- predictions.csv was
    written header-only for 2026-27, so main() skipped optimisation.
    """
    monkeypatch.setattr(
        prediction,
        "current_roster",
        lambda season: pl.DataFrame(
            {
                "name": ["Bukayo Saka"],
                "position": ["MID"],
                "team": ["Arsenal"],
                "element": [55],
                "player_code": [999],
                "value": [130],
            }
        ),
    )
    rolling = pl.DataFrame(
        {
            "season": ["2025-26"],
            "name": ["Bukayo Saka"],
            "position": ["MID"],
            "team": ["Arsenal"],
            "element": [101],
            "player_code": [999],
            "gw": [38],
            "total_points": [8],
            "total_points_rolling_5": [5.0],
        }
    )
    fixtures = pl.DataFrame(
        {
            "team": ["Arsenal"],
            "opponent_team": ["Hull City"],
            "is_home": [True],
            "kickoff_date": [date(2026, 8, 15)],
            "season": ["2026-27"],
            "gw": [1],
        }
    )
    team_elo = pl.DataFrame(
        {
            "team": ["Arsenal", "Hull City"],
            "elo": [2000.0, 1400.0],
            "from_date": [date(2026, 7, 1)] * 2,
            "to_date": [date(2026, 12, 1)] * 2,
        }
    )
    _setup_artifacts(tmp_path, monkeypatch, rolling, fixtures, team_elo)

    result = predict_points("2026-27")

    assert not result.is_empty()
    assert result["gw"].to_list() == [1]
    saka = result.filter(pl.col("name") == "Bukayo Saka")
    assert saka["baseline"].item() == 5.0
    assert saka["predicted_points"].item() is not None


def test_predict_points_ignores_the_roster_when_backtesting(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A backtest never consults the snapshot, so no future state leaks in."""
    called: list[str] = []
    monkeypatch.setattr(
        prediction,
        "current_roster",
        lambda season: called.append(season) or pl.DataFrame(),
    )
    _setup_artifacts(
        tmp_path,
        monkeypatch,
        _baseline_rolling(),
        _baseline_fixtures(),
        pl.DataFrame(
            {
                "team": ["Arsenal", "Chelsea", "Spurs"],
                "elo": [2000.0, 1900.0, 1850.0],
                "from_date": [date(2025, 10, 1)] * 3,
                "to_date": [date(2025, 12, 1)] * 3,
            }
        ),
    )

    predict_points("2025-26", as_of_gw=10, is_backtest=True)

    assert called == []


def test_predict_points_uses_the_roster_when_as_of_gw_is_set_live(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A live run with a team file still reads the roster.

    Regression test for the pre-season bug: ``main`` derives
    ``as_of_gw = team.gameweek - 1``, which is 0 for a GW1 team file. Gating
    the roster on ``as_of_gw is None`` suppressed it, the current-season
    rolling filter was empty, and predictions.csv came out header-only.
    """
    monkeypatch.setattr(
        prediction,
        "current_roster",
        lambda season: pl.DataFrame(
            {
                "name": ["Bukayo Saka"],
                "position": ["MID"],
                "team": ["Arsenal"],
                "element": [55],
                "player_code": [999],
                "value": [130],
            }
        ),
    )
    rolling = pl.DataFrame(
        {
            "season": ["2025-26"],
            "name": ["Bukayo Saka"],
            "position": ["MID"],
            "team": ["Arsenal"],
            "element": [101],
            "player_code": [999],
            "gw": [38],
            "total_points": [8],
            "total_points_rolling_5": [5.0],
        }
    )
    fixtures = pl.DataFrame(
        {
            "team": ["Arsenal"],
            "opponent_team": ["Hull City"],
            "is_home": [True],
            "kickoff_date": [date(2026, 8, 15)],
            "season": ["2026-27"],
            "gw": [1],
        }
    )
    team_elo = pl.DataFrame(
        {
            "team": ["Arsenal", "Hull City"],
            "elo": [2000.0, 1400.0],
            "from_date": [date(2026, 7, 1)] * 2,
            "to_date": [date(2026, 12, 1)] * 2,
        }
    )
    _setup_artifacts(tmp_path, monkeypatch, rolling, fixtures, team_elo)

    # as_of_gw=0 is exactly what main() passes for a GW1 team file.
    result = predict_points("2026-27", as_of_gw=0)

    assert not result.is_empty()
    assert result["gw"].to_list() == [1]
    assert result["name"].to_list() == ["Bukayo Saka"]
    assert result["player_id"].to_list() == [55]


def _mid_season_roster() -> pl.DataFrame:
    """Return a two-player roster in the shape ``current_roster`` produces."""
    return pl.DataFrame(
        {
            "name": ["P1", "P2"],
            "position": ["MID", "FWD"],
            "team": ["Arsenal", "Chelsea"],
            "element": [55, 56],
            "player_code": [999, 888],
            "value": [130, 90],
        }
    )


def test_baselines_mid_season_with_a_roster_uses_current_season_form() -> None:
    """Mid-season with a roster: form is current-season, identity is roster.

    This is the path production actually takes mid-season when no team file is
    given, so it needs cover in its own right -- the legacy no-roster branch
    is now only reached by backtests.
    """
    rolling = pl.DataFrame(
        {
            "season": ["2025-26", "2025-26", "2026-27", "2026-27"],
            "name": ["P1", "P1", "P1", "P2"],
            "position": ["MID"] * 3 + ["FWD"],
            # Stale club membership: the roster must win, not this.
            "team": ["Everton"] * 4,
            "element": [11, 11, 101, 102],
            "player_code": [999, 999, 999, 888],
            "gw": [37, 38, 9, 9],
            "total_points": [6, 8, 2, 4],
            "total_points_rolling_5": [4.0, 5.0, 7.0, 3.0],
        }
    )

    result = _baselines(
        rolling, "2026-27", as_of_gw=None, roster=_mid_season_roster()
    ).sort("name")

    # Baselines come from the latest *current-season* rows, not last season's.
    assert result["baseline"].to_list() == [7.0, 3.0]
    # Identity -- club and element -- comes from the roster.
    assert result["team"].to_list() == ["Arsenal", "Chelsea"]
    assert result["element"].to_list() == [55, 56]
    assert result["position"].to_list() == ["MID", "FWD"]


def test_baselines_roster_path_deduplicates_shared_display_names(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Two roster players with one display name cannot silently collapse.

    Downstream (``_build_problem``, ``_load_prices``) keys on ``name``, so a
    collision would become one MILP variable with last-write-wins. The
    survivor must be deterministic (lowest element) and the drop logged.
    """
    roster = pl.DataFrame(
        {
            "name": ["Danny Ward", "Danny Ward"],
            "position": ["GK", "FWD"],
            "team": ["Leicester", "Huddersfield"],
            "element": [402, 77],
            "player_code": [111, 222],
            "value": [45, 50],
        }
    )
    rolling = pl.DataFrame(
        {
            "season": ["2026-27", "2026-27"],
            "name": ["Danny Ward", "Danny Ward"],
            "position": ["GK", "FWD"],
            "team": ["Leicester", "Huddersfield"],
            "element": [402, 77],
            "player_code": [111, 222],
            "gw": [9, 9],
            "total_points": [2, 6],
            "total_points_rolling_5": [1.5, 4.5],
        }
    )

    with caplog.at_level(
        logging.WARNING, logger="fantasy_football.modelling.prediction"
    ):
        result = _baselines(rolling, "2026-27", as_of_gw=None, roster=roster)

    assert result.height == 1
    # Lowest element wins, deterministically.
    assert result["element"].to_list() == [77]
    assert result["baseline"].to_list() == [4.5]
    assert any(
        "Danny Ward" in record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
    )


def test_baselines_roster_path_logs_stale_baseline_seasons(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A baseline older than the previous season is surfaced in an INFO log."""
    roster = pl.DataFrame(
        {
            "name": ["Old Timer", "Current Star"],
            "position": ["MID", "MID"],
            "team": ["Arsenal", "Arsenal"],
            "element": [1, 2],
            "player_code": [111, 222],
            "value": [45, 130],
        }
    )
    rolling = pl.DataFrame(
        {
            "season": ["2020-21", "2025-26"],
            "name": ["Old Timer", "Current Star"],
            "position": ["MID", "MID"],
            "team": ["Arsenal", "Arsenal"],
            "element": [1, 2],
            "player_code": [111, 222],
            "gw": [38, 38],
            "total_points": [2, 8],
            "total_points_rolling_5": [1.0, 5.0],
        }
    )

    with caplog.at_level(
        logging.INFO, logger="fantasy_football.modelling.prediction"
    ):
        _baselines(rolling, "2026-27", as_of_gw=None, roster=roster)

    stale_logs = [
        record.getMessage()
        for record in caplog.records
        if "older than" in record.getMessage()
    ]
    assert len(stale_logs) == 1
    assert "Old Timer (2020-21)" in stale_logs[0]
    # The player whose baseline is from last season is not an offender.
    assert "Current Star" not in stale_logs[0]


def test_baselines_roster_path_logs_dropped_null_team_rows(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A roster row with no club is dropped with a debug log, not silently."""
    roster = pl.DataFrame(
        {
            "name": ["Clubless", "P2"],
            "position": ["MID", "FWD"],
            "team": [None, "Chelsea"],
            "element": [1, 2],
            "player_code": [111, 222],
            "value": [45, 90],
        },
        schema_overrides={"team": pl.Utf8},
    )
    rolling = pl.DataFrame(
        {
            "season": ["2026-27", "2026-27"],
            "name": ["Clubless", "P2"],
            "position": ["MID", "FWD"],
            "team": ["Arsenal", "Chelsea"],
            "element": [1, 2],
            "player_code": [111, 222],
            "gw": [9, 9],
            "total_points": [2, 6],
            "total_points_rolling_5": [3.0, 4.0],
        }
    )

    with caplog.at_level(
        logging.DEBUG, logger="fantasy_football.modelling.prediction"
    ):
        result = _baselines(rolling, "2026-27", as_of_gw=None, roster=roster)

    assert result["name"].to_list() == ["P2"]
    assert any(
        "Clubless" in record.getMessage()
        for record in caplog.records
        if record.levelno == logging.DEBUG
    )


def test_missing_elo_median_is_not_skewed_by_alias_duplication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Alias rows share one rating, so they count once towards the median.

    ``elo.normalize_elo_frame`` emits one row per FPL alias, so a multi-alias
    club appears twice with an identical rating over an identical interval.
    Counting both would drag the missing-ELO fallback median towards it.
    """
    elo = pl.DataFrame(
        {
            # Arsenal 2000, Spurs 1900, and one club under two aliases at 1000.
            "team": ["Arsenal", "Spurs", "Ipswich", "Ipswich Town"],
            "elo": [2000.0, 1900.0, 1000.0, 1000.0],
            "from_date": [date(2025, 10, 1)] * 4,
            "to_date": [date(2025, 12, 31)] * 4,
        }
    )
    _setup_artifacts(
        tmp_path, monkeypatch, _baseline_rolling(), _baseline_fixtures(), elo
    )

    result = predict_points("2025-26", horizon_n=1)

    # Deduplicated ratings are [2000, 1900, 1000] -> median 1900.
    # With the duplicate counted it would be (1900 + 1000) / 2 = 1450.
    row = result.filter(pl.col("gw") == 11).row(0, named=True)
    assert row["opponent_team_elo"] == pytest.approx(1900.0)
