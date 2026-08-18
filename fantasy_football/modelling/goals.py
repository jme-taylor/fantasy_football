"""The goals rate model, pooled across the outfield positions.

FPL pays six points for a defender's goal, five for a midfielder's and
four for a forward's. That is one underlying quantity -- how often this
player scores -- worth three different amounts, which is why the model
predicts *goals per 90* and the component converts. Folding the points
value into the target instead would give the same underlying rate three
different targets, and pooling would stop making sense.

Pooled is the point. A defender scores roughly one goal every twenty
appearances, far too rare to estimate from defenders alone, but the
process producing it -- get into the box, get a shot away, finish it --
is the same one that produces a forward's. So the model trains on
defenders, midfielders and forwards together with the position carried
as three indicator columns, and ``POSITION`` says only which position's
rows this instance scores and writes.

Goalkeepers are excluded outright. Their goals are so rare that their
rows are a mass of exact zeros, and all they would do is drag the fit.

Like the defcon head, this reads no minutes features. It predicts a rate
and minutes turn that rate into an expected count at composition, so the
minutes forecast is applied exactly once across every component.

The target prefers FPL's ``goals_scored`` and falls back to FCI's
``goals``, the same hybrid the defcon head's count uses and for the same
reason: Vaastav's per-fixture stats stop at 2025-26, so a single-source
target would be null for every row of the live season. ``player_match``
itself carries no goals column, which is why both sources are joined
rather than read off the match row.

FCI's raw table spans every competition, but ``opta_match`` -- the
bridge this reads, not the table -- is already filtered to the league,
so the fallback cannot carry a cup goal into a league gameweek.

"""

import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import ClassVar, override

import numpy as np
import polars as pl
from sklearn.metrics import brier_score_loss, log_loss

from fantasy_football.features.match_form import (
    NO_FORM_COLUMN,
    PENALTY_EXPOSURE_COLUMN,
)
from fantasy_football.modelling.components import (
    GOALS_POINTS_BY_POSITION,
    Component,
    RateComponent,
)
from fantasy_football.modelling.defcon import (
    MINUTES_FLOOR as DEFCON_MINUTES_FLOOR,
)
from fantasy_football.modelling.distributions import PoissonCounts
from fantasy_football.modelling.metrics import (
    count_poisson_deviance,
    top_decile_ratio,
)
from fantasy_football.modelling.points import (
    PositionPointsPredictor,
    position_dummy_names,
)
from fantasy_football.modelling.predictor import ModelSpec
from fantasy_football.storage.coverage import FCI_SEASONS
from fantasy_football.storage.tables import (
    POINTS_COMPONENT,
    TEST_POINTS_PREDICTION,
)

logger = logging.getLogger(__name__)

#: The model's primary position, which keys its registered name and
#: its evaluation. It scores and writes every position in
#: :data:`TRAINING_POSITIONS` -- see ``SERVING_POSITIONS``.
POSITION = "DEF"

#: The outfield positions the one artefact is fitted on, and
#: writes component rows for.
TRAINING_POSITIONS: tuple[str, ...] = ("DEF", "MID", "FWD")

REGISTERED_MODEL = "goals_rate_regressor"
PRODUCTION_ALIAS = "production"
EXPERIMENT_NAME = "goals-rate-model"

#: Seasons with the expected-goals and penalty counters this reads.
TRAINING_SEASONS = FCI_SEASONS

# Higher than the defcon head's fifteen, deliberately. Goals are far
# spikier: a substitute who scores in fifteen minutes enters the fit at
# six goals per 90 against a population mean nearer 0.15, and minutes
# weighting alone does not take that leverage away.
#
# It bounds what this head *learns* from, not what it scores. The
# scoring floor has to match the sibling components' or a short
# appearance loses its whole composed prediction rather than just its
# goals term.
MINUTES_FLOOR = 30
SCORING_MINUTES_FLOOR = DEFCON_MINUTES_FLOOR


def goals_count_sql(fpl: str = "gf", opta: str = "go") -> str:
    """Return the goal count, preferring FPL's own published figure.

    Aliased rather than fixed so a caller can point it at whichever
    same expression off its own joins. One definition, two aliasings --
    a second spelling of this is how the components quietly stop summing
    to the total.

    Parameters
    ----------
    fpl : str, optional
        Alias of the ``player_match_fpl`` relation.
    opta : str, optional
        Alias of the ``opta_match`` relation.

    Returns
    -------
    str
        A scalar expression, unaliased.
    """
    return f"coalesce({fpl}.goals_scored, {opta}.goals)"


# Both goal sources at match grain. The goals head reads them to build
# its target.
GOALS_JOINS = """
LEFT JOIN player_match_fpl AS gf
    ON  gf.season        = m.season
    AND gf.gw            = m.gw
    AND gf.element       = m.element
    AND gf.opponent_team = m.opponent
LEFT JOIN opta_match AS go
    ON  go.season   = m.season
    AND go.gw       = m.gw
    AND go.element  = m.element
    AND go.opponent = m.opponent"""


@dataclass(frozen=True, slots=True)
class GoalsMetrics:
    """Scores for the goals head over one fold.

    Scored at match scale rather than on the per-90 rate, for the same
    reason the defcon head is: almost every row is a zero, and a rate
    error is minimised by predicting nothing at all.

    ``top_decile_ratio`` is predicted over actual goals among the tenth
    of rows with the highest predicted rate. Below one means the head
    under-predicts its own best players, which is the compression a
    squared-error forest is expected to show -- and that decile is the
    one holding every player worth captaining.
    """

    poisson_deviance: float
    brier: float
    logloss: float
    base_rate_brier: float
    skill_score: float
    scoring_rate: float
    rate_mae: float
    top_decile_ratio: float

    def as_dict(self) -> dict[str, float]:
        """Return the metric names and values, one level deep."""
        return asdict(self)


class GoalsRatePredictor(PositionPointsPredictor):
    """Predicts goals per 90 for outfielders, with no minutes features."""

    POSITION = POSITION
    TRAINING_POSITIONS = TRAINING_POSITIONS
    SERVING_POSITIONS = TRAINING_POSITIONS
    TARGET = "goals_per_90"
    COMPONENT = Component.GOALS
    COMPONENT_IMPL = RateComponent(
        Component.GOALS, points_per_event=GOALS_POINTS_BY_POSITION
    )
    TRAINING_SEASONS = TRAINING_SEASONS
    WEIGHT_COLUMN = "minutes"
    EXTRA_COLUMNS = (
        f"{goals_count_sql()} AS goals_scored",
        "m.minutes AS minutes",
    )
    # A null flag means the form join missed entirely, which is the same
    # thing the flag exists to report. Left to the imputer it would come
    # back as the population median instead.
    FEATURE_FILLS: ClassVar[dict[str, float]] = {NO_FORM_COLUMN: 1.0}

    FEATURES = [
        "is_home",
        *position_dummy_names(TRAINING_POSITIONS),
        NO_FORM_COLUMN,
        "goals_scored_per90_rolling_5",
        "xg_per90_rolling_5",
        "xa_per90_rolling_5",
        PENALTY_EXPOSURE_COLUMN,
        "xg_for_rolling_5",
        "goals_for_rolling_5",
        "xg_against_rolling_5",
        "goals_against_rolling_5",
    ]

    PLAYER_FORM_COLUMNS = [
        NO_FORM_COLUMN,
        "goals_scored_per90_rolling_5",
        "xg_per90_rolling_5",
        "xa_per90_rolling_5",
        PENALTY_EXPOSURE_COLUMN,
    ]
    OWN_TEAM_COLUMNS = ["xg_for_rolling_5", "goals_for_rolling_5"]
    OPPOSITION_COLUMNS = ["xg_against_rolling_5", "goals_against_rolling_5"]
    MINUTES_COLUMNS: ClassVar[list[str]] = []

    @property
    @override
    def target_sql(self) -> str:
        """Return goals per 90, from whichever source published them."""
        return (
            f"{goals_count_sql()} * 90.0 "
            "/ nullif(m.minutes, 0) AS goals_per_90"
        )

    @property
    @override
    def extra_joins(self) -> str:
        """Join both goal sources at match grain."""
        return GOALS_JOINS

    @property
    @override
    def row_filter(self) -> str:
        """Match the other DEF components' floor, and nothing tighter.

        Every restriction here removes a leg from the composed
        prediction, not just from the fit: a defender with an
        ``appearance``, a ``defcon`` and an ``assists`` but no ``goals``
        is incomplete, and ``compose`` drops him entirely. So this is
        the floor the sibling components use and no more; what this head
        may not *learn* from is in :attr:`training_row_filter`.
        """
        return f"\n  AND m.minutes >= {SCORING_MINUTES_FLOOR}"

    @property
    @override
    def training_row_filter(self) -> str:
        """Drop cameos, no-form rows, and legs with no goal count.

        A cameo's per-90 rate is arithmetic noise with enormous
        leverage. A no-form row teaches the model that the imputed
        median is predictive. A leg neither provider published a count
        for has no target at all.

        All three are still *scored*. A debutant has to be, and there
        the imputer fills his rates with the pooled median -- which,
        since defenders are the largest group, reads a debut striker as
        a forward who creates nothing. The flag lets the model know it
        is guessing; it cannot help it guess better. Per-position
        imputation is the honest fix.
        """
        return (
            f"\n  AND m.minutes >= {MINUTES_FLOOR}"
            f"\n  AND mf.{NO_FORM_COLUMN} = 0"
            f"\n  AND {goals_count_sql()} IS NOT NULL"
        )

    @override
    def fold_metrics(
        self, test_df: pl.DataFrame, predicted: Sequence[float]
    ) -> GoalsMetrics:
        """Score the fold at match scale, not on the per-90 rate.

        Actual minutes build the expected count rather than the minutes
        model's forecast, which isolates this head's error from that
        model's.
        """
        minutes = test_df["minutes"].cast(pl.Float64).to_numpy()
        rate = np.clip(np.asarray(predicted, dtype=float), 0.0, None)
        expected = rate * minutes / 90.0
        goals = test_df["goals_scored"].cast(pl.Float64).to_numpy()
        scored = (goals > 0).astype(int)
        probability = PoissonCounts().p_at_least(expected, 1)
        base_rate = float(np.mean(scored))
        baseline = np.full_like(probability, base_rate)
        brier = float(brier_score_loss(scored, probability, pos_label=1))
        base_brier = float(brier_score_loss(scored, baseline, pos_label=1))
        return GoalsMetrics(
            poisson_deviance=count_poisson_deviance(goals, expected),
            brier=brier,
            # Labelled for the same reason the defcon head labels them: a
            # fold in which nobody scores is single-class, and both
            # metrics infer their labels from the data unless told.
            logloss=float(
                log_loss(
                    scored,
                    np.clip(probability, 1e-9, 1 - 1e-9),
                    labels=[0, 1],
                )
            ),
            base_rate_brier=base_brier,
            skill_score=1.0 - brier / base_brier if base_brier else 0.0,
            scoring_rate=base_rate,
            rate_mae=float(
                np.average(
                    np.abs(rate - test_df[self.TARGET].to_numpy()),
                    weights=minutes,
                )
            ),
            top_decile_ratio=top_decile_ratio(rate, goals, expected),
        )


GOALS_SPEC = ModelSpec(
    registered_model_name=REGISTERED_MODEL,
    production_alias=PRODUCTION_ALIAS,
    table=POINTS_COMPONENT,
    evaluation_table=TEST_POINTS_PREDICTION,
    position=POSITION,
    component=Component.GOALS,
)
