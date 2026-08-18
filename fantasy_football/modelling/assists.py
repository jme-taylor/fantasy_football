"""The assists rate model, pooled across the outfield positions.

An assist pays three points whatever the shirt, so unlike the goals head
this one needs no per-position conversion to justify itself. What it
shares with goals is the reason to pool: an assist is rare enough for a
defender that the position cannot estimate its own rate, while the
process behind it -- get on the ball in the final third, pick the pass,
have it finished -- is the same one a midfielder's comes from. So the
model trains on defenders, midfielders and forwards together with the
position carried as three indicator columns, and ``POSITION`` says only
which position's rows this instance scores and writes.

Goalkeepers are excluded, as they are from the goals head: their assists
are rare enough to be a mass of exact zeros, and their creation stats
describe something structurally different.

The target is FPL's ``assists`` and nothing else. This is where the head
deliberately parts company with goals, which coalesces FPL's count with
FCI's. The two providers do not measure the same event: FPL awards an
assist for the pass before an own goal, for winning a penalty that is
scored, and for a shot rebounding to a scorer, where FCI is stricter.
Over the seasons both publish, FPL logs 983 assists to FCI's 814 in
2024-25 and the disagreements run almost entirely one way. FPL settles
the points, so FPL defines the target, and a coalesce would swap that
definition on the rows it fired for.

The cost of that, accepted deliberately: ``player_match_fpl`` is
scraped from Vaastav, whose per-fixture files stop at 2025-26, while
``player_match`` and ``player_match_opta`` both fill for the live season
from sources of their own -- the FPL API and FCI. So if Vaastav does not
publish 2026-27, this head has no target for it and DEF loses every
composed prediction until it does. That is the failure the goals head's coalesce exists to prevent,
and it is not prevented here.

It fails loudly rather than quietly: ``compose`` names the missing
component and the leg count when it drops them, so the symptom arrives
as a warning naming ``assists`` rather than as defenders silently
absent from the optimiser. The fix, if Vaastav stays gone, is to carry
the count on ``player_match`` itself -- the FPL API's ``element-summary``
history already returns ``assists`` and the loader simply narrows it
away -- which would serve one FPL-settled definition live and historic
with no fallback at all.

``player_match`` carries no assists column, which is why the FPL match
table is joined rather than read off the match row.

Like the goals and defcon heads, this reads no minutes features. It
predicts a rate and minutes turn that rate into an expected count at
composition, so the minutes forecast is applied exactly once across
every component.

Known gap: there is no set-piece feature. Corners and free kicks are a
large share of a defender's assists, and the taker is exactly the sort
of thing the goals head captures with penalty exposure -- but FCI
publishes ``corners`` only from 2026-27, so there is nothing to build it
from over the training seasons. Revisit once that season accumulates.

"""

import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import ClassVar, override

import numpy as np
import polars as pl
from sklearn.metrics import brier_score_loss, log_loss

from fantasy_football.features.match_form import NO_FORM_COLUMN
from fantasy_football.modelling.components import (
    ASSIST_POINTS,
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

REGISTERED_MODEL = "assists_rate_regressor"
PRODUCTION_ALIAS = "production"
EXPERIMENT_NAME = "assists-rate-model"

#: Seasons with the expected-assist and creation counters this reads.
TRAINING_SEASONS = FCI_SEASONS

# The goals head's floor, for the goals head's reason: an assist off a
# fifteen minute cameo enters the fit at six per 90 against a population
# mean nearer 0.1, and minutes weighting alone does not take that
# leverage away. The scoring floor has to match the sibling components'
# or a short appearance loses its whole composed prediction rather than
# just its assists term.
MINUTES_FLOOR = 30
SCORING_MINUTES_FLOOR = DEFCON_MINUTES_FLOOR


def assists_count_sql(fpl: str = "af") -> str:
    """Return FPL's assist count.

    Aliased rather than fixed so a caller can point it at whichever
    same expression off its own joins. One definition, two aliasings --
    a second spelling of this is how the components quietly stop summing
    to the total.

    Parameters
    ----------
    fpl : str, optional
        Alias of the ``player_match_fpl`` relation.

    Returns
    -------
    str
        A scalar expression, unaliased.
    """
    return f"{fpl}.assists"


# The assist source at match grain. The assists head reads it to build
# its target.
ASSISTS_JOINS = """
LEFT JOIN player_match_fpl AS af
    ON  af.season        = m.season
    AND af.gw            = m.gw
    AND af.element       = m.element
    AND af.opponent_team = m.opponent"""


@dataclass(frozen=True, slots=True)
class AssistsMetrics:
    """Scores for the assists head over one fold.

    Scored at match scale rather than on the per-90 rate, for the same
    reason the goals head is: almost every row is a zero, and a rate
    error is minimised by predicting nothing at all.

    ``top_decile_ratio`` is predicted over actual assists among the tenth
    of rows with the highest predicted rate. Below one means the head
    under-predicts its own best creators.
    """

    poisson_deviance: float
    brier: float
    logloss: float
    base_rate_brier: float
    skill_score: float
    assisting_rate: float
    rate_mae: float
    top_decile_ratio: float

    def as_dict(self) -> dict[str, float]:
        """Return the metric names and values, one level deep."""
        return asdict(self)


class AssistsRatePredictor(PositionPointsPredictor):
    """Predicts assists per 90 for outfielders, with no minutes features."""

    POSITION = POSITION
    TRAINING_POSITIONS = TRAINING_POSITIONS
    SERVING_POSITIONS = TRAINING_POSITIONS
    TARGET = "assists_per_90"
    COMPONENT = Component.ASSISTS
    COMPONENT_IMPL = RateComponent(
        Component.ASSISTS, points_per_event=ASSIST_POINTS
    )
    TRAINING_SEASONS = TRAINING_SEASONS
    WEIGHT_COLUMN = "minutes"
    EXTRA_COLUMNS = (
        f"{assists_count_sql()} AS assists",
        "m.minutes AS minutes",
    )
    # A null flag means the form join missed entirely, which is the same
    # thing the flag exists to report. Left to the imputer it would come
    # back as the population median instead.
    FEATURE_FILLS: ClassVar[dict[str, float]] = {NO_FORM_COLUMN: 1.0}

    # Penalty exposure is deliberately absent, unlike the goals list:
    # FPL credits the assist to whoever *won* the penalty, not to the
    # taker, so the feature has no mechanism on this target and two
    # seasons of data are too few to spend on one.
    FEATURES = [
        "is_home",
        *position_dummy_names(TRAINING_POSITIONS),
        NO_FORM_COLUMN,
        "assists_per90_rolling_5",
        "xa_per90_rolling_5",
        "chances_created_per90_rolling_5",
        "accurate_crosses_per90_rolling_5",
        "goals_scored_per90_rolling_5",
        "xg_per90_rolling_5",
        "xg_for_rolling_5",
        "goals_for_rolling_5",
        "xg_against_rolling_5",
        "goals_against_rolling_5",
    ]

    PLAYER_FORM_COLUMNS = [
        NO_FORM_COLUMN,
        "assists_per90_rolling_5",
        "xa_per90_rolling_5",
        "chances_created_per90_rolling_5",
        "accurate_crosses_per90_rolling_5",
        "goals_scored_per90_rolling_5",
        "xg_per90_rolling_5",
    ]
    # An assist needs a teammate to finish, so a creator's own club is
    # read for what it converts and the opposition for what it concedes.
    OWN_TEAM_COLUMNS = ["xg_for_rolling_5", "goals_for_rolling_5"]
    OPPOSITION_COLUMNS = ["xg_against_rolling_5", "goals_against_rolling_5"]
    MINUTES_COLUMNS: ClassVar[list[str]] = []

    @property
    @override
    def target_sql(self) -> str:
        """Return assists per 90, as FPL published them."""
        return (
            f"{assists_count_sql()} * 90.0 "
            "/ nullif(m.minutes, 0) AS assists_per_90"
        )

    @property
    @override
    def extra_joins(self) -> str:
        """Join the FPL match table at match grain."""
        return ASSISTS_JOINS

    @property
    @override
    def row_filter(self) -> str:
        """Match the other DEF components' floor, and nothing tighter.

        Every restriction here removes a leg from the composed
        prediction, not just from the fit: a defender missing only his
        ``assists`` component is incomplete, and ``compose`` drops him
        entirely. What this head may not *learn* from is in
        :attr:`training_row_filter`.
        """
        return f"\n  AND m.minutes >= {SCORING_MINUTES_FLOOR}"

    @property
    @override
    def training_row_filter(self) -> str:
        """Drop cameos, no-form rows, and legs with no assist count.

        A cameo's per-90 rate is arithmetic noise with enormous
        leverage. A no-form row teaches the model that the imputed
        median is predictive. The count is null wherever the FPL join
        missed, which is every leg of a season Vaastav has not
        published; such a leg has no target at all.

        All three are still *scored*, and there a debutant's rates come
        back as the pooled median. The flag lets the model know it is
        guessing; it cannot help it guess better. Per-position
        imputation is the honest fix, and it belongs to both this head
        and the goals one rather than to either alone.
        """
        return (
            f"\n  AND m.minutes >= {MINUTES_FLOOR}"
            f"\n  AND mf.{NO_FORM_COLUMN} = 0"
            f"\n  AND {assists_count_sql()} IS NOT NULL"
        )

    @override
    def fold_metrics(
        self, test_df: pl.DataFrame, predicted: Sequence[float]
    ) -> AssistsMetrics:
        """Score the fold at match scale, not on the per-90 rate.

        Actual minutes build the expected count rather than the minutes
        model's forecast, which isolates this head's error from that
        model's.
        """
        minutes = test_df["minutes"].cast(pl.Float64).to_numpy()
        rate = np.clip(np.asarray(predicted, dtype=float), 0.0, None)
        expected = rate * minutes / 90.0
        assists = test_df["assists"].cast(pl.Float64).to_numpy()
        assisted = (assists > 0).astype(int)
        probability = PoissonCounts().p_at_least(expected, 1)
        base_rate = float(np.mean(assisted))
        baseline = np.full_like(probability, base_rate)
        brier = float(brier_score_loss(assisted, probability, pos_label=1))
        base_brier = float(brier_score_loss(assisted, baseline, pos_label=1))
        return AssistsMetrics(
            poisson_deviance=count_poisson_deviance(assists, expected),
            brier=brier,
            # Labelled for the same reason the goals head labels them: a
            # fold in which nobody assists is single-class, and both
            # metrics infer their labels from the data unless told.
            logloss=float(
                log_loss(
                    assisted,
                    np.clip(probability, 1e-9, 1 - 1e-9),
                    labels=[0, 1],
                )
            ),
            base_rate_brier=base_brier,
            skill_score=1.0 - brier / base_brier if base_brier else 0.0,
            assisting_rate=base_rate,
            rate_mae=float(
                np.average(
                    np.abs(rate - test_df[self.TARGET].to_numpy()),
                    weights=minutes,
                )
            ),
            top_decile_ratio=top_decile_ratio(rate, assists, expected),
        )


ASSISTS_SPEC = ModelSpec(
    registered_model_name=REGISTERED_MODEL,
    production_alias=PRODUCTION_ALIAS,
    table=POINTS_COMPONENT,
    evaluation_table=TEST_POINTS_PREDICTION,
    position=POSITION,
    component=Component.ASSISTS,
)
