"""The defender points regressor.

Everything that reads the column lists below lives in
:mod:`fantasy_football.modelling.points`; this module states only what
makes a defender a defender.
"""

import logging
from typing import ClassVar, override

from fantasy_football.constants import (
    DEFCON_FIRST_SEASON,
    DEFCON_THRESHOLD_DEF,
)
from fantasy_football.modelling.components import (
    MINUTES_OUTPUTS,
    Component,
    RateComponent,
)
from fantasy_football.modelling.defcon import (
    CBIT_COUNT_SQL,
    DEFCON_JOINS,
    MINUTES_FLOOR,
)
from fantasy_football.modelling.points import PositionPointsPredictor
from fantasy_football.modelling.predictor import ModelSpec
from fantasy_football.storage.tables import (
    POINTS_COMPONENT,
    TEST_POINTS_PREDICTION,
)

logger = logging.getLogger(__name__)

POSITION = "DEF"

REGISTERED_MODEL = "defender_points_regressor"
PRODUCTION_ALIAS = "production"

# TODO (JT): expected_minutes is train/serve skewed. Training rows take
# backfill-kind minutes predictions, which minutes.py documents as
# in-sample -- the minutes champion scored its own training seasons, so
# they flatter. Live rows take forward-kind predictions, which are
# genuinely out-of-sample and noisier, so this model will over-trust
# expected_minutes and underperform live. Fixing it means generating
# out-of-fold minutes predictions. Cross-validation will NOT catch this,
# because CV reads the backfill values too.


class DefenderPointsPredictor(PositionPointsPredictor):
    """Points regressor for players in the DEF position."""

    POSITION = POSITION

    # Model inputs, as chosen in the notebook. Ordering matters only for
    # readability; the pipeline selects by name.
    FEATURES = [
        "is_home",
        "expected_minutes",
        "p_zero",
        "p_partial",
        "p_sixty_plus",
        "xg_per90_rolling_5",
        "xa_per90_rolling_5",
        "xg_against_rolling_5",
        "goals_against_rolling_5",
        "clean_sheet_rolling_5",
        "xg_for_rolling_5",
        "goals_for_rolling_5",
        "interceptions_per90_rolling_5",
        "tackles_per90_rolling_5",
        "clearances_per90_rolling_5",
        "blocks_per90_rolling_5",
        "yellow_cards_per90_rolling_5",
        "red_cards_per90_rolling_5",
        "yellow_cards_season_to_date",
        "red_cards_season_to_date",
        "days_since_last_appearance",
    ]

    PLAYER_FORM_COLUMNS = [
        "xg_per90_rolling_5",
        "xa_per90_rolling_5",
        "interceptions_per90_rolling_5",
        "tackles_per90_rolling_5",
        "clearances_per90_rolling_5",
        "blocks_per90_rolling_5",
        "yellow_cards_per90_rolling_5",
        "red_cards_per90_rolling_5",
        "yellow_cards_season_to_date",
        "red_cards_season_to_date",
        "days_since_last_appearance",
    ]
    # A defender is judged on what his own club concedes and on what the
    # opposition creates -- the mirror image of the forwards model.
    OWN_TEAM_COLUMNS = [
        "xg_against_rolling_5",
        "goals_against_rolling_5",
        "clean_sheet_rolling_5",
    ]
    OPPOSITION_COLUMNS = ["xg_for_rolling_5", "goals_for_rolling_5"]
    MINUTES_COLUMNS = [
        "expected_minutes",
        "p_zero",
        "p_partial",
        "p_sixty_plus",
    ]


DEFENDER_SPEC = ModelSpec(
    registered_model_name=REGISTERED_MODEL,
    production_alias=PRODUCTION_ALIAS,
    table=POINTS_COMPONENT,
    component=Component.TOTAL,
    evaluation_table=TEST_POINTS_PREDICTION,
    position=POSITION,
)


# The points left once appearance and defcon have been carved out:
# goals, assists, clean sheets, cards and bonus. Predicted per 90 and
# scaled by the minutes forecast at composition, so this model reads no
# minutes feature either.
#
# The deduction is gated on the season the rule came in, not on whether
# a provider happens to publish a column. Keying it on Vaastav's
# ``defensive_contribution`` alone would silently stop deducting in
# 2026-27 -- that source ends at 2025-26 -- so the residual would
# re-absorb points the defcon component is also predicting, and the
# composed total would double-count them in the live season. The count
# itself comes from the same hybrid the defcon model trains on, which
# falls back to FCI wherever FPL publishes nothing.
#
# Before the rule, nothing is deducted, which is correct because nothing
# was awarded. That is what makes this target the same quantity in every
# season and retires the regime break the monolithic model trains across.
#
# TODO (JT): bonus points sit in here and are not a per-90 quantity at
# all. They are awarded per match on a BPS ranking, so scaling them by
# minutes is an approximation, and they are the largest source of noise
# left in this target. Bonus is the next component to split out.
RESIDUAL_TARGET_SQL = f"""(
    m.total_points
    - CASE WHEN m.minutes >= 60 THEN 2 WHEN m.minutes > 0 THEN 1 ELSE 0 END
    - CASE
          WHEN m.season >= '{DEFCON_FIRST_SEASON}'
           AND {CBIT_COUNT_SQL} >= {DEFCON_THRESHOLD_DEF}
          THEN 2
          ELSE 0
      END
) * 90.0 / nullif(m.minutes, 0) AS residual_points_per_90"""

RESIDUAL_REGISTERED_MODEL = "defender_residual_points_regressor"


class DefenderResidualPointsPredictor(DefenderPointsPredictor):
    """Everything a defender scores bar appearance and defcon points."""

    TARGET = "residual_points_per_90"
    COMPONENT = Component.RESIDUAL
    COMPONENT_IMPL = RateComponent(Component.RESIDUAL)
    WEIGHT_COLUMN = "minutes"
    EXTRA_COLUMNS = ("m.minutes AS minutes",)

    # The monolithic list minus expected_minutes and the three bucket
    # probabilities. Nothing replaces them: a minutes proxy would put the
    # double-count back in through the side door.
    FEATURES = [
        feature
        for feature in DefenderPointsPredictor.FEATURES
        if feature not in MINUTES_OUTPUTS
    ]
    MINUTES_COLUMNS: ClassVar[list[str]] = []

    @property
    @override
    def target_sql(self) -> str:
        """Return points per 90 net of appearance and defcon."""
        return RESIDUAL_TARGET_SQL

    @property
    @override
    def extra_joins(self) -> str:
        """Join both CBIT sources, as the defcon model does."""
        return DEFCON_JOINS

    @property
    @override
    def row_filter(self) -> str:
        """Drop appearances too short to carry a meaningful per-90 rate."""
        return f"\n  AND m.minutes >= {MINUTES_FLOOR}"


DEFENDER_RESIDUAL_SPEC = ModelSpec(
    registered_model_name=RESIDUAL_REGISTERED_MODEL,
    production_alias=PRODUCTION_ALIAS,
    table=POINTS_COMPONENT,
    evaluation_table=TEST_POINTS_PREDICTION,
    position=POSITION,
    component=Component.RESIDUAL,
)
