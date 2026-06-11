"""Tests for the per-position points-model registry."""

import polars as pl
import pytest

from fantasy_football.data_transformation import KNOWN_POSITIONS
from fantasy_football.models import MODELS_BY_POSITION, RollingFormulaModel


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
