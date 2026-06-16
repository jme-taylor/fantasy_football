"""Per-position points models.

Today every position uses the same deterministic rolling-points formula. The
registry gives each position its own model instance so a position can later be
swapped for a different model without touching the others.
"""

from dataclasses import dataclass
from typing import Protocol

import polars as pl

from fantasy_football.features.transformation import KNOWN_POSITIONS


class PointsModel(Protocol):
    """Predicts points for a set of feature rows."""

    def predict(self, features: pl.DataFrame) -> pl.Series:
        """Return predicted points, one value per feature row.

        Parameters
        ----------
        features : pl.DataFrame
            Must contain ``baseline``, ``opponent_factor`` and
            ``home_away_factor`` columns.

        Returns
        -------
        pl.Series
            Predicted points, aligned to ``features`` row order.
        """
        ...


@dataclass
class RollingFormulaModel:
    """baseline x opponent_factor x home_away_factor. Deterministic."""

    def predict(self, features: pl.DataFrame) -> pl.Series:
        """Multiply the three feature columns into a predicted-points series.

        Parameters
        ----------
        features : pl.DataFrame
            Must contain ``baseline``, ``opponent_factor`` and
            ``home_away_factor`` columns.

        Returns
        -------
        pl.Series
            Predicted points, aligned to ``features`` row order.
        """
        return (
            features["baseline"]
            * features["opponent_factor"]
            * features["home_away_factor"]
        ).alias("predicted_points")


MODELS_BY_POSITION: dict[str, PointsModel] = {
    position: RollingFormulaModel() for position in KNOWN_POSITIONS
}
