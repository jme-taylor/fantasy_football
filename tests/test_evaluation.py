"""Tests for the rolling-origin evaluation harness."""

from pathlib import Path

import polars as pl
import pytest

from fantasy_football import evaluation, prediction


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
