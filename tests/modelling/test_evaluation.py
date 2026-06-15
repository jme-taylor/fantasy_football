"""Tests for the rolling-origin evaluation harness."""

from pathlib import Path

import polars as pl
import pytest

from fantasy_football.modelling import evaluation, prediction


def _rolling_two_seasons() -> pl.DataFrame:
    """Rolling-points rows: one player, two seasons, several gameweeks."""
    rows = []
    for season in ("2024-25", "2025-26"):
        for gw in range(1, 9):
            rows.append(
                {
                    "season": season,
                    "name": "P1",
                    "position": "MID",
                    "team": "Arsenal",
                    "element": 101,
                    "gw": gw,
                    "total_points": gw,  # actual points = gw
                    "total_points_rolling_5": float(gw),
                }
            )
    return pl.DataFrame(rows)


def test_select_pivots_respects_history_floor() -> None:
    """Pivots need ROLLING_WINDOW prior gameweeks and a next gw for actuals."""
    rolling = _rolling_two_seasons()
    pivots = evaluation._select_pivots(rolling, "2025-26", rolling_window=5)
    # gws present 1..8; need >=5 prior gws (pivot>=6) and pivot+1 present
    # (pivot<=7) -> pivots {6, 7}.
    assert pivots == [6, 7]


def _write_artifacts(transformed: Path) -> None:
    """Write the three transformed CSVs used by the harness."""
    _rolling_two_seasons().write_csv(transformed / "rolling_points.csv")
    fixtures = pl.DataFrame(
        {
            "team": ["Arsenal"] * 16,
            "opponent_team": ["Chelsea"] * 16,
            "is_home": [True] * 16,
            "kickoff_date": ["2024-09-01"] * 8 + ["2025-09-01"] * 8,
            "season": ["2024-25"] * 8 + ["2025-26"] * 8,
            "gw": list(range(1, 9)) * 2,
        }
    ).with_columns(pl.col("kickoff_date").str.to_date())
    fixtures.write_csv(transformed / "fixtures_enriched.csv")
    pl.DataFrame(
        {
            "team": ["Arsenal", "Chelsea"],
            "elo": [2000.0, 1800.0],
            "from_date": ["2024-01-01", "2024-01-01"],
            "to_date": ["2026-12-31", "2026-12-31"],
        }
    ).with_columns(
        pl.col("from_date").str.to_date(), pl.col("to_date").str.to_date()
    ).write_csv(transformed / "team_elo.csv")


def test_collect_joins_predictions_to_actuals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Collected rows carry predicted_points and the actual total_points."""
    transformed = tmp_path / "transformed"
    transformed.mkdir()
    _write_artifacts(transformed)
    monkeypatch.setattr(prediction, "TRANSFORMED_DATA_FOLDER", transformed)
    monkeypatch.setattr(evaluation, "TRANSFORMED_DATA_FOLDER", transformed)

    collected = evaluation._collect_predictions_vs_actuals(rolling_window=5)

    assert "predicted_points" in collected.columns
    assert "actual" in collected.columns
    row = collected.filter(
        (pl.col("season") == "2025-26") & (pl.col("gw") == 7)
    ).row(0, named=True)
    assert row["actual"] == 7


def test_evaluate_returns_metrics_for_present_positions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Evaluate returns a metrics dict keyed by the positions in the data."""
    transformed = tmp_path / "transformed"
    transformed.mkdir()
    _write_artifacts(transformed)
    monkeypatch.setattr(prediction, "TRANSFORMED_DATA_FOLDER", transformed)
    monkeypatch.setattr(evaluation, "TRANSFORMED_DATA_FOLDER", transformed)

    results = evaluation.evaluate(rolling_window=5)

    assert "MID" in results
    metrics = results["MID"]
    for key in (
        "mae",
        "rmse",
        "skill_score",
        "spearman",
        "precision_at_k",
        "poisson_deviance",
        "n_samples",
    ):
        assert key in metrics


def test_actuals_sums_double_gameweek_rows() -> None:
    """_actuals must SUM points across fixture rows, not take the max.

    A double gameweek produces two rolling rows for the same (season, element,
    gw). The actual target should be the sum of those rows' total_points so
    it stays symmetric with predictions, which are also summed per player-gw.
    """
    rolling = pl.DataFrame(
        {
            "season": ["2025-26", "2025-26", "2025-26"],
            "name": ["P1", "P1", "P1"],
            "position": ["MID", "MID", "MID"],
            "team": ["Arsenal", "Arsenal", "Arsenal"],
            "element": [101, 101, 101],
            "gw": [7, 7, 8],
            "total_points": [5, 8, 3],
            "total_points_rolling_5": [4.0, 4.0, 4.0],
        }
    )
    result = evaluation._actuals(rolling)
    dgw_row = result.filter(
        (pl.col("season") == "2025-26")
        & (pl.col("player_id") == 101)
        & (pl.col("gw") == 7)
    ).row(0, named=True)
    single_row = result.filter(
        (pl.col("season") == "2025-26")
        & (pl.col("player_id") == 101)
        & (pl.col("gw") == 8)
    ).row(0, named=True)
    assert dgw_row["actual"] == 13  # 5 + 8, not max(5, 8) = 8
    assert single_row["actual"] == 3


def test_metrics_for_position_groups_by_season_and_gw() -> None:
    """Rank metrics must be computed per (season, gw), not pooled by gw alone.

    When the same gw number appears in two seasons with recycled player_ids,
    pooling by gw number alone incorrectly merges them. We use MID with k=5,
    so each season-gw bucket must have >5 players for effective_k to not cover
    everyone. We use 6 players per (season, gw).

    Season A, gw=1: top-5 predicted = {1,2,3,4,5}, top-5 actual = {1,2,3,4,5}
    => precision@5 = 1.0.
    Season B, gw=1: top-5 predicted = {1,2,3,4,5}, top-5 actual = {2,3,4,5,6}
    => precision@5 = 4/5 = 0.8.

    Correct per-(season,gw) average: (1.0 + 0.8) / 2 = 0.9.

    When pooled by gw alone (12 rows, effective_k = min(5,12) = 5):
    top-5 predicted: the 5 rows with highest predicted_points. Players 1-5
    appear in both seasons so polars will pick 5 of the 12 rows; duplicate
    player_ids mean the predicted set is a mix and the pooled precision differs
    from 0.9. We assert the result equals 0.9 (the correct per-season-gw value).
    """
    season_a = [
        {
            "season": "2024-25",
            "player_id": i,
            "gw": 1,
            "position": "MID",
            # predicted rank matches actual rank (best=1)
            "predicted_points": float(10 - i),
            "baseline": 5.0,
            "actual": float(10 - i),
        }
        for i in range(1, 7)
    ]
    season_b = [
        {
            "season": "2025-26",
            "player_id": i,
            "gw": 1,
            "position": "MID",
            # predicted still ranks 1>2>3>4>5>6 but actual ranks 6>5>4>3>2>1
            # so actual top-5 = {2,3,4,5,6}, predicted top-5 = {1,2,3,4,5}
            # intersection = {2,3,4,5} -> 4/5 = 0.8
            "predicted_points": float(10 - i),
            "baseline": 5.0,
            "actual": float(i),
        }
        for i in range(1, 7)
    ]
    df = pl.DataFrame(season_a + season_b)
    result = evaluation._metrics_for_position(df, "MID")
    # Season A: precision@5 = 1.0, Season B: precision@5 = 0.8 => mean = 0.9
    assert result["precision_at_k"] == pytest.approx(0.9)


def test_evaluate_returns_empty_dict_when_no_predictions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """evaluate() returns an empty dict when _collect_predictions_vs_actuals is empty."""
    monkeypatch.setattr(
        evaluation,
        "_collect_predictions_vs_actuals",
        lambda _rw: pl.DataFrame(),
    )
    assert evaluation.evaluate() == {}


def test_log_results_to_mlflow_logs_one_run_per_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each position's metrics go to its own experiment as one run."""
    experiments: list[str] = []
    logged_metrics: list[dict] = []

    class _Run:
        def __enter__(self) -> "_Run":
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    monkeypatch.setattr(evaluation.mlflow, "set_tracking_uri", lambda _u: None)
    monkeypatch.setattr(
        evaluation.mlflow,
        "set_experiment",
        lambda name: experiments.append(name),
    )
    monkeypatch.setattr(evaluation.mlflow, "start_run", lambda: _Run())
    monkeypatch.setattr(evaluation.mlflow, "log_params", lambda _p: None)
    monkeypatch.setattr(
        evaluation.mlflow,
        "log_metrics",
        lambda m: logged_metrics.append(m),
    )

    results = {
        "MID": {"mae": 1.0, "rmse": 1.0, "n_samples": 3},
        "GK": {"mae": 2.0, "rmse": 2.0, "n_samples": 1},
    }
    evaluation.log_results_to_mlflow(results)

    assert "mid-points-model" in experiments
    assert "gk-points-model" in experiments
    assert len(logged_metrics) == 2
