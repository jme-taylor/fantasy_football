"""Tests for the per-position points-model registry."""

import polars as pl
import pytest

from fantasy_football.features.transformation import KNOWN_POSITIONS
from fantasy_football.modelling.models import (
    MODELS_BY_POSITION,
    RollingFormulaModel,
    StoredPredictionModel,
)


def test_rolling_formula_model_is_deterministic_product() -> None:
    """Predict returns baseline x opponent_factor x home_away_factor."""
    features = pl.DataFrame(
        {
            "baseline": [4.0, 2.0],
            "opponent_factor": [1.2, 0.5],
            "home_away_factor": [1.1, 0.9],
        }
    )
    result = RollingFormulaModel().predict(features)
    assert result.to_list() == pytest.approx(
        [4.0 * 1.2 * 1.1, 2.0 * 0.5 * 0.9]
    )


def test_registry_covers_every_known_position() -> None:
    """Every known position has a model instance."""
    assert set(MODELS_BY_POSITION) == set(KNOWN_POSITIONS)
    for model in MODELS_BY_POSITION.values():
        assert isinstance(model, RollingFormulaModel)


def test_sums_match_grain_predictions_to_gameweek() -> None:
    """Two legs of a double gameweek sum to one gameweek total."""
    predictions = pl.DataFrame(
        {
            "season": ["2026-27", "2026-27"],
            "gw": [5, 5],
            "element": [11, 11],
            "predicted_points": [3.0, 2.5],
        }
    )
    features = pl.DataFrame(
        {"season": ["2026-27"], "gw": [5], "element": [11]}
    )

    result = StoredPredictionModel(predictions).predict(features)

    assert result.to_list() == [5.5]


def test_summing_does_not_pool_across_gameweeks_or_players() -> None:
    """Only rows sharing (season, gw, element) are summed together."""
    predictions = pl.DataFrame(
        {
            "season": ["2026-27"] * 4,
            "gw": [5, 5, 6, 5],
            "element": [11, 11, 11, 12],
            "predicted_points": [3.0, 2.5, 100.0, 40.0],
        }
    )
    features = pl.DataFrame(
        {
            "season": ["2026-27"] * 3,
            "gw": [5, 6, 5],
            "element": [11, 11, 12],
        }
    )

    result = StoredPredictionModel(predictions).predict(features)

    assert result.to_list() == [5.5, 100.0, 40.0]


def test_preserves_input_row_order() -> None:
    """Predictions come back aligned to the feature frame's row order."""
    predictions = pl.DataFrame(
        {
            "season": ["2026-27"] * 3,
            "gw": [5, 5, 6],
            "element": [1, 2, 1],
            "predicted_points": [1.0, 2.0, 3.0],
        }
    )
    features = pl.DataFrame(
        {
            "season": ["2026-27"] * 3,
            "gw": [6, 5, 5],
            "element": [1, 2, 1],
        }
    )

    result = StoredPredictionModel(predictions).predict(features)

    assert result.to_list() == [3.0, 2.0, 1.0]


def test_missing_prediction_is_null_not_a_formula_fallback() -> None:
    """An unmatched row scores null rather than falling back to a formula."""
    predictions = pl.DataFrame(
        {
            "season": ["2026-27"],
            "gw": [5],
            "element": [1],
            "predicted_points": [1.0],
        }
    )
    features = pl.DataFrame(
        {"season": ["2026-27"] * 2, "gw": [5, 5], "element": [1, 99]}
    )

    result = StoredPredictionModel(predictions).predict(features)

    assert result.to_list() == [1.0, None]


def test_ignores_extra_feature_columns() -> None:
    """Only season, gw and element are consumed; other columns are ignored."""
    predictions = pl.DataFrame(
        {
            "season": ["2026-27"],
            "gw": [5],
            "element": [1],
            "predicted_points": [4.0],
        }
    )
    features = pl.DataFrame(
        {
            "season": ["2026-27"],
            "gw": [5],
            "element": [1],
            "opponent": [7],
            "baseline": [99.0],
        }
    )

    result = StoredPredictionModel(predictions).predict(features)

    assert result.to_list() == [4.0]
    assert result.name == "predicted_points"
