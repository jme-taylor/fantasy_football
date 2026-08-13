"""The defender points regressor.

Everything that reads the column lists below lives in
:mod:`fantasy_football.modelling.points`; this module states only what
makes a defender a defender.
"""

import logging

from fantasy_football.modelling.points import PositionPointsPredictor
from fantasy_football.modelling.predictor import ModelSpec
from fantasy_football.storage.tables import (
    POINTS_PREDICTION,
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
    table=POINTS_PREDICTION,
    evaluation_table=TEST_POINTS_PREDICTION,
    position=POSITION,
)
