"""The defensive-contribution rate models.

FPL pays two points for reaching a threshold on a count of defensive
actions. That is a threshold on a count, not a smooth quantity, and it is
the one part of a score that a points regressor has no way to represent.

The rule runs off two different counts. A defender needs 10 CBIT --
clearances, blocks, interceptions and tackles. A midfielder or forward
needs 12 CBIRT, the same four plus recoveries. Two counts and two
thresholds mean two heads: one regressor predicts one quantity, and a
single rate cannot be both.

What each model predicts is the *rate* per 90, with no minutes feature
anywhere in it. Minutes turn that rate into an expected count and the
count distribution turns the expected count into the probability of
clearing the threshold, both at composition time. Splitting it that way
is what lets minutes be applied exactly once across every component.
"""

import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import ClassVar, override

import numpy as np
import polars as pl
from sklearn.metrics import brier_score_loss, log_loss

from fantasy_football.constants import DEFCON_THRESHOLD_BY_POSITION
from fantasy_football.features.match_form import (
    DEFCON_COMPONENT_STATS,
    DEFCON_MID_FWD_COMPONENT_STATS,
    NO_FORM_COLUMN,
)
from fantasy_football.modelling.components import (
    Component,
    DefconComponent,
)
from fantasy_football.modelling.distributions import PoissonCounts
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

POSITION = "DEF"

REGISTERED_MODEL = "defender_cbit_rate_regressor"
PRODUCTION_ALIAS = "production"

#: The seasons a CBIT count can be had for at all. FPL publishes its own
#: count only from 2025-26; FCI's counters reach back one season further
#: and no earlier. The 2016-19 seasons carry the counters too, but that
#: is a different era of pressing and they are left out deliberately.
TRAINING_SEASONS = FCI_SEASONS

#: Rows below this many minutes are dropped. A per-90 rate off eight
#: minutes is arithmetic noise with enormous leverage on a regressor;
#: minutes weighting handles the rest of the range.
MINUTES_FLOOR = 15


def _opta_count(stats: tuple[str, ...], alias: str = "oc") -> str:
    """Return the SQL summing FCI's counters into one count.

    Null when FCI published none of them, rather than zero: a player
    with no data did not make no clearances.
    """
    absent = " AND ".join(f"{alias}.{stat} IS NULL" for stat in stats)
    totalled = " + ".join(f"coalesce({alias}.{stat}, 0)" for stat in stats)
    return f"CASE WHEN {absent} THEN NULL ELSE {totalled} END"


def _opta_cbit(alias: str = "oc") -> str:
    """Return the SQL summing FCI's four defender counters."""
    return _opta_count(DEFCON_COMPONENT_STATS, alias)


# FPL's own count where it is published, and FCI's reconstruction only
# where it is not. Reconciled on 2025-26, where both exist, the two agree
# exactly on 92.7% of defender gameweeks and differ by one on a further
# 6.4% -- minutes and fixture legs match on every disagreeing row, so the
# gap is the two providers counting a tackle differently rather than a
# join or competition-filter fault. Preferring FPL where it exists keeps
# the target exact for the season the rule actually ran in, and confines
# the provider noise to 2024-25.
CBIT_COUNT_SQL = f"coalesce(pmf.defensive_contribution, {_opta_cbit()})"


@dataclass(frozen=True, slots=True)
class DefconMetrics:
    """Scores for the defcon head over one fold.

    Scored at match scale rather than on the per-90 rate. The rate is not
    the quantity of interest -- P(CBIT >= threshold) is -- and scoring on
    the rate would hand the low-minutes rows the leverage the weighting
    exists to take away from them.

    ``skill_score`` is against always predicting the fold's base rate, so
    a head with no signal scores zero rather than looking good because
    the event is rare.
    """

    brier: float
    logloss: float
    base_rate_brier: float
    skill_score: float
    hit_rate: float
    rate_mae: float

    def as_dict(self) -> dict[str, float]:
        """Return the metric names and values, one level deep."""
        return asdict(self)


# Both decomposed DEF models read both CBIT sources: the rate model to
# build its target, the residual model to know what to deduct.
DEFCON_JOINS = """
LEFT JOIN opta_match AS oc
    ON  oc.season   = m.season
    AND oc.gw       = m.gw
    AND oc.element  = m.element
    AND oc.opponent = m.opponent
LEFT JOIN player_match_fpl AS pmf
    ON  pmf.season        = m.season
    AND pmf.gw            = m.gw
    AND pmf.element       = m.element
    AND pmf.opponent_team = m.opponent"""


class DefconRatePredictor(PositionPointsPredictor):
    """Predicts CBIT per 90 for defenders, with no minutes features."""

    POSITION = POSITION
    #: The scored column holding the match's actual count. Named rather
    #: than assumed, so the metrics below serve either count.
    COUNT_COLUMN: ClassVar[str] = "cbit_count"
    TARGET = "cbit_per_90"
    COMPONENT = Component.DEFCON
    COMPONENT_IMPL = DefconComponent(
        threshold=DEFCON_THRESHOLD_BY_POSITION["DEF"],
        distribution=PoissonCounts(),
    )
    TRAINING_SEASONS = TRAINING_SEASONS
    WEIGHT_COLUMN = "minutes"
    EXTRA_COLUMNS = (
        f"{CBIT_COUNT_SQL} AS cbit_count",
        "m.minutes AS minutes",
    )

    # No expected_minutes, and no bucket probabilities. A rate model that
    # read them would be predicting a quantity that already has minutes
    # in it, and composition would multiply them in a second time.
    FEATURES = [
        "is_home",
        "cbit_per90_rolling_5",
        "cbit_ten_plus_rate_rolling_5",
        "cbit_std_rolling_5",
        "tackles_per90_rolling_5",
        "interceptions_per90_rolling_5",
        "clearances_per90_rolling_5",
        "blocks_per90_rolling_5",
        "xg_against_rolling_5",
        "goals_against_rolling_5",
        "xg_for_rolling_5",
        "goals_for_rolling_5",
    ]

    PLAYER_FORM_COLUMNS = [
        "cbit_per90_rolling_5",
        "cbit_ten_plus_rate_rolling_5",
        "cbit_std_rolling_5",
        "tackles_per90_rolling_5",
        "interceptions_per90_rolling_5",
        "clearances_per90_rolling_5",
        "blocks_per90_rolling_5",
    ]
    # A defender who defends a lot plays for a club under pressure, so
    # the team columns run the opposite way to the clean-sheet model's.
    OWN_TEAM_COLUMNS = ["xg_against_rolling_5", "goals_against_rolling_5"]
    OPPOSITION_COLUMNS = ["xg_for_rolling_5", "goals_for_rolling_5"]
    MINUTES_COLUMNS: ClassVar[list[str]] = []

    @property
    @override
    def target_sql(self) -> str:
        """Return CBIT per 90 from whichever source published it."""
        return f"{CBIT_COUNT_SQL} * 90.0 / nullif(m.minutes, 0) AS cbit_per_90"

    @property
    @override
    def extra_joins(self) -> str:
        """Join both CBIT sources at match grain."""
        return DEFCON_JOINS

    @property
    @override
    def row_filter(self) -> str:
        """Drop rows with no count, or too few minutes to rate one."""
        return (
            f"\n  AND m.minutes >= {MINUTES_FLOOR}"
            f"\n  AND {CBIT_COUNT_SQL} IS NOT NULL"
        )

    @override
    def fold_metrics(
        self, test_df: pl.DataFrame, predicted: Sequence[float]
    ) -> DefconMetrics:
        """Score the fold at match scale, not on the per-90 rate.

        Actual minutes are used to build lambda rather than the minutes
        model's forecast, which isolates this head's error from the
        minutes model's.
        """
        threshold = self.COMPONENT_IMPL.threshold
        minutes = test_df["minutes"].cast(pl.Float64).to_numpy()
        rate = np.clip(np.asarray(predicted, dtype=float), 0.0, None)
        probability = self.COMPONENT_IMPL.distribution.p_at_least(
            rate * minutes / 90.0, threshold
        )
        actual = (
            (test_df[self.COUNT_COLUMN] >= threshold).cast(pl.Int64).to_list()
        )
        base_rate = float(np.mean(actual))
        baseline = np.full_like(probability, base_rate)
        brier = float(brier_score_loss(actual, probability, pos_label=1))
        base_brier = float(brier_score_loss(actual, baseline, pos_label=1))
        return DefconMetrics(
            brier=brier,
            # A fold in which no defender clears the threshold is
            # single-class, and both metrics infer their labels from the
            # data unless told. Naming them keeps a quiet gameweek a
            # score rather than a crash.
            logloss=float(
                log_loss(
                    actual,
                    np.clip(probability, 1e-9, 1 - 1e-9),
                    labels=[0, 1],
                )
            ),
            base_rate_brier=base_brier,
            skill_score=1.0 - brier / base_brier if base_brier else 0.0,
            hit_rate=base_rate,
            rate_mae=float(
                np.average(
                    np.abs(rate - test_df[self.TARGET].to_numpy()),
                    weights=minutes,
                )
            ),
        )


DEFCON_SPEC = ModelSpec(
    registered_model_name=REGISTERED_MODEL,
    production_alias=PRODUCTION_ALIAS,
    table=POINTS_COMPONENT,
    evaluation_table=TEST_POINTS_PREDICTION,
    position=POSITION,
    component=Component.DEFCON,
)


MID_FWD_POSITION = "MID"

CBIRT_REGISTERED_MODEL = "cbirt_rate_regressor"

#: The positions paid on CBIRT. Pooled into one head: the rule, the
#: count and the threshold are identical for both, so the target is the
#: same quantity and only the level differs -- which the position dummy
#: carries. Forwards clear twelve in under 1% of their appearances, far
#: too rare to fit on their own.
CBIRT_POSITIONS: tuple[str, ...] = ("MID", "FWD")

# FPL's own count first, FCI's published count second, FCI's
# reconstruction from the five counters last. Reconciled on 2025-26,
# where all three exist: FCI's published count agrees with FPL exactly on
# every row it covers, and the reconstruction agrees on 93.0% of
# midfielder and 97.0% of forward gameweeks, differing by one on most of
# the rest. Only 15 rows in 5,585 fall on opposite sides of the
# threshold, which is the only disagreement that reaches a prediction.
#
# The reconstruction is not a fallback for tidiness: FCI publishes its
# own count from 2025-26 and Vaastav's ends there, so 2024-25 has
# nothing else, and the live season has no Vaastav column at all.
CBIRT_COUNT_SQL = (
    "coalesce(pmf.defensive_contribution, oc.defensive_contributions, "
    f"{_opta_count(DEFCON_MID_FWD_COMPONENT_STATS)})"
)


class CbirtRatePredictor(DefconRatePredictor):
    """Predicts CBIRT per 90 for midfielders and forwards.

    Everything but the count, the threshold and the feature list is the
    defender head's: the same joins, the same minutes floor, the same
    match-scale metrics.
    """

    POSITION = MID_FWD_POSITION
    TRAINING_POSITIONS = CBIRT_POSITIONS
    COUNT_COLUMN = "cbirt_count"
    TARGET = "cbirt_per_90"
    COMPONENT_IMPL = DefconComponent(
        threshold=DEFCON_THRESHOLD_BY_POSITION[MID_FWD_POSITION],
        distribution=PoissonCounts(),
    )
    EXTRA_COLUMNS = (
        f"{CBIRT_COUNT_SQL} AS cbirt_count",
        "m.minutes AS minutes",
    )
    FEATURE_FILLS: ClassVar[dict[str, float]] = {NO_FORM_COLUMN: 1.0}

    # The defender head's list, on the CBIRT trio rather than the CBIT
    # one and with recoveries added -- the counter that separates the two
    # counts. The team columns run the same way for the same reason: a
    # midfielder defends more when his club is under pressure.
    FEATURES = [
        "is_home",
        *position_dummy_names(CBIRT_POSITIONS),
        "cbirt_per90_rolling_5",
        "cbirt_twelve_plus_rate_rolling_5",
        "cbirt_std_rolling_5",
        "tackles_per90_rolling_5",
        "interceptions_per90_rolling_5",
        "clearances_per90_rolling_5",
        "blocks_per90_rolling_5",
        "recoveries_per90_rolling_5",
        "xg_against_rolling_5",
        "goals_against_rolling_5",
        "xg_for_rolling_5",
        "goals_for_rolling_5",
    ]

    PLAYER_FORM_COLUMNS = [
        "cbirt_per90_rolling_5",
        "cbirt_twelve_plus_rate_rolling_5",
        "cbirt_std_rolling_5",
        "tackles_per90_rolling_5",
        "interceptions_per90_rolling_5",
        "clearances_per90_rolling_5",
        "blocks_per90_rolling_5",
        "recoveries_per90_rolling_5",
    ]
    OWN_TEAM_COLUMNS = ["xg_against_rolling_5", "goals_against_rolling_5"]
    OPPOSITION_COLUMNS = ["xg_for_rolling_5", "goals_for_rolling_5"]
    MINUTES_COLUMNS: ClassVar[list[str]] = []

    @property
    @override
    def target_sql(self) -> str:
        """Return CBIRT per 90 from whichever source published it."""
        return (
            f"{CBIRT_COUNT_SQL} * 90.0 "
            "/ nullif(m.minutes, 0) AS cbirt_per_90"
        )

    @property
    @override
    def row_filter(self) -> str:
        """Drop rows with no count, or too few minutes to rate one."""
        return (
            f"\n  AND m.minutes >= {MINUTES_FLOOR}"
            f"\n  AND {CBIRT_COUNT_SQL} IS NOT NULL"
        )


CBIRT_SPEC = ModelSpec(
    registered_model_name=CBIRT_REGISTERED_MODEL,
    production_alias=PRODUCTION_ALIAS,
    table=POINTS_COMPONENT,
    evaluation_table=TEST_POINTS_PREDICTION,
    position=MID_FWD_POSITION,
    component=Component.DEFCON,
)
