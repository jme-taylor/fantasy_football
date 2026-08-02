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
            Fixture-grain rows. Each implementation documents which
            columns it consumes: :class:`RollingFormulaModel` needs
            ``baseline``, ``opponent_factor`` and ``home_away_factor``;
            :class:`StoredPredictionModel` needs ``season``, ``gw`` and
            ``element``.

        Returns
        -------
        pl.Series
            Predicted points, one value per input row and aligned to
            ``features`` row order.
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


@dataclass
class StoredPredictionModel:
    """Serves stored per-match predictions, summed to gameweek grain.

    The predictions frame is injected rather than read here, so
    ``predict`` stays pure column work and stays testable without a
    database. Summing over ``(season, gw, element)`` is where double
    gameweeks resolve: two legs, two match-grain predictions, one
    gameweek total.

    There is deliberately no fallback to
    :class:`RollingFormulaModel`. Mixing formula-scored and
    model-scored players in one column would put two uncalibrated
    scales side by side and make the optimiser's comparison between
    them meaningless, invisibly. A null here is a coverage bug and must
    read as one -- ``prediction._check_defender_coverage`` turns it into
    a named error before the optimiser ever sees the column.

    Attributes
    ----------
    predictions : pl.DataFrame
        Match-grain rows carrying ``season``, ``gw``, ``element`` and
        ``predicted_points``.
    """

    predictions: pl.DataFrame

    def predict(self, features: pl.DataFrame) -> pl.Series:
        """Return the stored prediction for each feature row.

        Parameters
        ----------
        features : pl.DataFrame
            Fixture-grain rows carrying ``season``, ``gw`` and
            ``element``. No other column is consumed.

        Returns
        -------
        pl.Series
            Predicted points, aligned to ``features`` row order. Null
            where no prediction is stored.
        """
        keys = ["season", "gw", "element"]
        totals = self.predictions.group_by(keys).agg(
            pl.col("predicted_points").sum()
        )
        joined = features.select(keys).join(
            totals, on=keys, how="left", maintain_order="left"
        )
        return joined["predicted_points"].alias("predicted_points")


MODELS_BY_POSITION: dict[str, PointsModel] = {
    position: RollingFormulaModel() for position in KNOWN_POSITIONS
}
