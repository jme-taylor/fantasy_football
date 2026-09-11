"""Tests for the pure regression metric functions."""

import math

import polars as pl
import pytest

from fantasy_football.modelling.metrics import (
    PointsMetrics,
    mae,
    metric_name,
    poisson_deviance,
    precision_at_k,
    rmse,
    skill_score,
    spearman_by_gw,
    summarise,
)


def test_mae_simple() -> None:
    """MAE is the mean of absolute errors."""
    assert mae([2.0, 4.0], [3.0, 1.0]) == pytest.approx(2.0)


def test_rmse_simple() -> None:
    """RMSE is the root mean of squared errors."""
    assert rmse([2.0, 4.0], [3.0, 0.0]) == pytest.approx(
        math.sqrt((1.0 + 16.0) / 2)
    )


def test_skill_score_positive_when_model_beats_baseline() -> None:
    """Model error half the baseline error gives a skill score of 0.5."""
    actual = [4.0, 4.0]
    predicted = [3.0, 5.0]  # MAE 1.0
    baseline = [2.0, 6.0]  # MAE 2.0
    assert skill_score(predicted, actual, baseline) == pytest.approx(0.5)


def test_skill_score_zero_baseline_error_returns_nan() -> None:
    """A perfect baseline (zero error) yields nan rather than dividing by 0."""
    assert math.isnan(skill_score([1.0], [2.0], [2.0]))


def test_poisson_deviance_perfect_prediction_is_zero() -> None:
    """A perfect prediction has zero Poisson deviance."""
    assert poisson_deviance([3.0, 5.0], [3.0, 5.0]) == pytest.approx(0.0)


def test_poisson_deviance_handles_zero_prediction() -> None:
    """Zero predictions are clipped so deviance is finite, not inf."""
    value = poisson_deviance([0.0, 2.0], [1.0, 2.0])
    assert math.isfinite(value)


def test_spearman_by_gw_perfect_ranking_is_one() -> None:
    """Perfect ranking per gameweek gives a Spearman correlation of 1.0."""
    df = pl.DataFrame(
        {
            "gw": [1, 1, 1, 2, 2, 2],
            "predicted_points": [1.0, 2.0, 3.0, 9.0, 5.0, 1.0],
            "actual": [10.0, 20.0, 30.0, 90.0, 50.0, 10.0],
        }
    )
    assert spearman_by_gw(df) == pytest.approx(1.0)


def test_spearman_by_gw_skips_single_player_gameweeks() -> None:
    """A gameweek with one player is skipped (rank corr undefined there)."""
    df = pl.DataFrame(
        {
            "gw": [1, 2, 2],
            "predicted_points": [5.0, 1.0, 2.0],
            "actual": [5.0, 10.0, 20.0],
        }
    )
    # Only gw2 contributes, and it ranks perfectly.
    assert spearman_by_gw(df) == pytest.approx(1.0)


def test_precision_at_k_counts_top_k_overlap_per_gw() -> None:
    """Precision@k is the per-gameweek top-k overlap fraction."""
    df = pl.DataFrame(
        {
            "gw": [1, 1, 1, 1],
            "player_id": [1, 2, 3, 4],
            "predicted_points": [4.0, 3.0, 2.0, 1.0],
            "actual": [4.0, 1.0, 2.0, 3.0],
        }
    )
    # Predicted top-2 = {1, 2}; actual top-2 = {1, 4}; overlap 1 of 2 = 0.5.
    assert precision_at_k(df, k=2) == pytest.approx(0.5)


def test_precision_at_k_clamps_k_to_available_players() -> None:
    """K is clamped to the number of players present in the gameweek."""
    df = pl.DataFrame(
        {
            "gw": [1],
            "player_id": [1],
            "predicted_points": [4.0],
            "actual": [4.0],
        }
    )
    assert precision_at_k(df, k=5) == pytest.approx(1.0)


def test_poisson_deviance_negative_actual_returns_finite() -> None:
    """Negative actuals are floored to 0 so deviance stays finite."""
    assert math.isfinite(poisson_deviance([1.0, 2.0], [-1.0, 2.0]))


def _points_metrics(mae_value: float) -> PointsMetrics:
    """Return a metrics object differing only in its MAE."""
    return PointsMetrics(
        mae=mae_value,
        rmse=2.0,
        skill_score=0.1,
        spearman=0.5,
        precision_at_k=0.4,
    )


def test_summarise_prefixes_and_aggregates_several_folds() -> None:
    """Many folds keep the mean and standard deviation, under a prefix."""
    summary = summarise(
        [_points_metrics(1.0), _points_metrics(3.0)], "cv", aggregated=True
    )
    assert summary["cv_mae_mean"] == pytest.approx(2.0)
    assert summary["cv_mae_std"] == pytest.approx(1.0)


def test_summarise_keeps_aggregate_names_when_one_fold_survives() -> None:
    """A short season leaves one fold; the run must still be queryable.

    Naming from the fold count would drop such a run out of every query
    keyed on the aggregate names, silently.
    """
    summary = summarise([_points_metrics(1.0)], "cv", aggregated=True)
    assert summary["cv_mae_mean"] == pytest.approx(1.0)
    assert summary["cv_mae_std"] == pytest.approx(0.0)


def test_summarise_reports_a_single_fold_without_a_spread() -> None:
    """One measurement has no spread, so no zero std is reported."""
    summary = summarise([_points_metrics(1.5)], None, aggregated=False)
    assert summary == {
        "mae": 1.5,
        "rmse": 2.0,
        "skill_score": 0.1,
        "spearman": 0.5,
        "precision_at_k": 0.4,
    }


def test_metric_name_leaves_an_unprefixed_strategy_bare() -> None:
    """A strategy with no prefix logs the metric under its own name."""
    assert metric_name(None, "auc") == "auc"
    assert metric_name("cv", "auc") == "cv_auc"


def test_summarise_of_no_folds_is_empty() -> None:
    """Nothing scored means nothing to log."""
    assert summarise([], None, aggregated=False) == {}
