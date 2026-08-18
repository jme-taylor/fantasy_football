"""The goalkeeper saves rate model.

FPL pays a point per three saves, which is a floor rather than a rate,
so this head predicts *saves per 90* and :class:`SavesComponent` does
the flooring at composition. Regressing on ``floor(saves / 3)`` instead
would round twice -- once in the target and again in the component --
and throw away the difference between a keeper expected to make 2.9
saves and one expected to make 2.1, who are paid differently in
expectation and identically after rounding.

Unpooled, unlike the goals and assists heads. There is nobody to pool
with: an outfielder's saves column is a mass of exact zeros, and the
process producing a keeper's has no outfield counterpart.

The training window is FCI's, and the reason is the team form rather
than anything about keepers. Saves are mostly shots faced, which is a
team quantity, and ``team_match_form`` is built from FCI player rows --
so ``xg_against_rolling_5`` and its siblings do not exist before
2024-25. Vaastav's ``saves`` column goes back to 2016-17, but a longer
frame bought by median-filling the danger features would be fitting the
population mean to eight seasons of keepers.

Known omissions, both deliberate. Penalty saves pay five points and are
in no component: a keeper faces about four penalties a season and saves
one, so a head for it would model roughly five points of noise. And
there is no shot-quality feature -- ``xgot_faced`` is exactly that, and
it is the obvious first addition once there are more than two seasons
of it.
"""

import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import ClassVar, override

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from fantasy_football.modelling.components import (
    SAVES_DIVISOR,
    Component,
    SavesComponent,
    expected_floor,
)
from fantasy_football.modelling.conceding import (
    SCORING_MINUTES_FLOOR as CONCEDING_SCORING_FLOOR,
)
from fantasy_football.modelling.distributions import PoissonCounts
from fantasy_football.modelling.metrics import count_poisson_deviance
from fantasy_football.modelling.points import PositionPointsPredictor
from fantasy_football.modelling.predictor import ModelSpec
from fantasy_football.storage.coverage import FCI_SEASONS
from fantasy_football.storage.tables import (
    POINTS_COMPONENT,
    TEST_POINTS_PREDICTION,
)

logger = logging.getLogger(__name__)

POSITION = "GK"

REGISTERED_MODEL = "goalkeeper_saves_regressor"
PRODUCTION_ALIAS = "production"
EXPERIMENT_NAME = "goalkeeper-saves-model"

OPPOSITION_PREFIX = "opposition_"

#: Seasons the team-form features this head leans on exist in.
TRAINING_SEASONS = FCI_SEASONS

# Matched to the keeper's sibling components rather than chosen: saves,
# conceding and appearance have to cover the same legs, because compose
# drops a fixture leg missing any one of them rather than summing what
# it has.
SCORING_MINUTES_FLOOR = CONCEDING_SCORING_FLOOR

# Higher than the outfield heads' thirty. A keeper playing under an hour
# is almost always an injury or a sending-off replacement, not the
# rotation a substitute striker represents, and the per-90 rate off that
# denominator is arithmetic noise.
MINUTES_FLOOR = 60


def saves_count_sql(fpl: str = "sf") -> str:
    """Return FPL's save count.

    Aliased rather than fixed so a caller can point it at its own joins.
    One definition, two aliasings -- a second spelling is how a target
    and the points it is paid quietly stop agreeing.

    Parameters
    ----------
    fpl : str, optional
        Alias of the ``player_match_fpl`` relation.

    Returns
    -------
    str
        A scalar expression, unaliased.
    """
    return f"{fpl}.saves"


# The save source at match grain.
SAVES_JOINS = """
LEFT JOIN player_match_fpl AS sf
    ON  sf.season        = m.season
    AND sf.gw            = m.gw
    AND sf.element       = m.element
    AND sf.opponent_team = m.opponent"""


@dataclass(frozen=True, slots=True)
class SavesMetrics:
    """Scores for the saves head over one fold.

    Scored at match scale rather than on the per-90 rate: the rate is
    what the model emits, but the points come off the count, and the two
    disagree most for exactly the rotation cases that matter.

    ``calibration`` is mean predicted over mean actual, so above one is
    a head expecting busier keepers than the league produces.
    ``points_mae`` scores the floored value the component actually pays,
    which is the number a bad fit has to be bad at to cost anything.
    """

    poisson_deviance: float
    mae: float
    calibration: float
    points_mae: float
    base_rate_points_mae: float
    skill_score: float

    def as_dict(self) -> dict[str, float]:
        """Return the metric names and values, one level deep."""
        return asdict(self)


class SavesRatePredictor(PositionPointsPredictor):
    """Predicts saves per 90 for keepers, with no minutes features."""

    POSITION = POSITION
    TARGET = "saves_per_90"
    COMPONENT = Component.SAVES
    COMPONENT_IMPL: ClassVar[SavesComponent] = SavesComponent()
    TRAINING_SEASONS = TRAINING_SEASONS
    OPPOSITION_PREFIX = OPPOSITION_PREFIX
    WEIGHT_COLUMN = "minutes"
    EXTRA_COLUMNS = (
        f"{saves_count_sql()} AS saves",
        "m.minutes AS minutes",
    )

    FEATURES = [
        "is_home",
        "saves_per90_rolling_5",
        "xg_against_rolling_5",
        "goals_against_rolling_5",
        "opposition_xg_for_rolling_5",
        "opposition_goals_for_rolling_5",
    ]

    PLAYER_FORM_COLUMNS = ["saves_per90_rolling_5"]
    # A keeper's own club for the danger it lets through, the opposition
    # for the danger it creates. Both sides, because neither alone says
    # how many shots arrive: a good defence facing a good attack is not
    # the same fixture as a bad one facing a bad attack.
    OWN_TEAM_COLUMNS = ["xg_against_rolling_5", "goals_against_rolling_5"]
    OPPOSITION_COLUMNS = ["xg_for_rolling_5", "goals_for_rolling_5"]
    MINUTES_COLUMNS: ClassVar[list[str]] = []

    @property
    @override
    def target_sql(self) -> str:
        """Return saves per 90, as FPL published them."""
        return (
            f"{saves_count_sql()} * 90.0 "
            "/ nullif(m.minutes, 0) AS saves_per_90"
        )

    @property
    @override
    def extra_joins(self) -> str:
        """Join the FPL match table at match grain."""
        return SAVES_JOINS

    @property
    @override
    def row_filter(self) -> str:
        """Match the keeper's other components' floor, and nothing tighter."""
        return f"\n  AND m.minutes >= {SCORING_MINUTES_FLOOR}"

    @property
    @override
    def training_row_filter(self) -> str:
        """Drop cameos and legs with no published save count.

        The count is null wherever the FPL join missed, which is every
        leg of a season Vaastav has not published; such a leg has no
        target at all. Both restrictions are fit-only -- the rows are
        still scored, or the keeper loses his whole composed prediction
        rather than just his saves term.
        """
        return (
            f"\n  AND m.minutes >= {MINUTES_FLOOR}"
            f"\n  AND {saves_count_sql()} IS NOT NULL"
        )

    @override
    def make_pipeline(self) -> Pipeline:
        """Build the median-imputing Poisson-loss pipeline.

        The conceding head's reasoning on the same shape of target: a
        count fed straight into a Poisson survival function, which a
        squared-error fit respects neither the non-negativity nor the
        mean-variance link of.

        Returns
        -------
        sklearn.pipeline.Pipeline
            Unfitted pipeline ending in a Poisson regressor.
        """
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("model", HistGradientBoostingRegressor(loss="poisson")),
            ]
        )

    @override
    def fold_metrics(
        self, test_df: pl.DataFrame, predicted: Sequence[float]
    ) -> SavesMetrics:
        """Score the fold on the count and on the points it is paid.

        Actual minutes build the expected count rather than the minutes
        model's forecast, which isolates this head's error from that
        model's.
        """
        distribution = PoissonCounts()
        minutes = test_df["minutes"].cast(pl.Float64).to_numpy()
        rate = np.clip(np.asarray(predicted, dtype=float), 0.0, None)
        expected = rate * minutes / 90.0
        saves = test_df["saves"].cast(pl.Float64).to_numpy()
        paid = np.floor(saves / SAVES_DIVISOR)
        predicted_points = expected_floor(
            expected, SAVES_DIVISOR, distribution
        )
        # "Predict the average keeper's workload every time", carried
        # through the same floor, so the skill score compares two
        # numbers of points rather than a rate against a count.
        mean_actual = float(np.mean(saves))
        base_points = expected_floor(
            np.full_like(expected, mean_actual), SAVES_DIVISOR, distribution
        )
        points_mae = float(np.mean(np.abs(predicted_points - paid)))
        base_points_mae = float(np.mean(np.abs(base_points - paid)))
        return SavesMetrics(
            poisson_deviance=count_poisson_deviance(saves, expected),
            mae=float(np.mean(np.abs(expected - saves))),
            calibration=(
                float(np.mean(expected)) / mean_actual if mean_actual else 0.0
            ),
            points_mae=points_mae,
            base_rate_points_mae=base_points_mae,
            skill_score=(
                1.0 - points_mae / base_points_mae if base_points_mae else 0.0
            ),
        )


SAVES_SPEC = ModelSpec(
    registered_model_name=REGISTERED_MODEL,
    production_alias=PRODUCTION_ALIAS,
    table=POINTS_COMPONENT,
    evaluation_table=TEST_POINTS_PREDICTION,
    position=POSITION,
    component=Component.SAVES,
)
