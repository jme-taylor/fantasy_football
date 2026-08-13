"""The forwards points regressor.

Everything that reads the column lists below lives in
:mod:`fantasy_football.modelling.points`; this module states only what
makes a forward a forward.
"""

import logging

from fantasy_football.modelling.points import PositionPointsPredictor
from fantasy_football.modelling.predictor import ModelSpec
from fantasy_football.storage.tables import (
    POINTS_PREDICTION,
    TEST_POINTS_PREDICTION,
)

logger = logging.getLogger(__name__)

POSITION = "FWD"

REGISTERED_MODEL = "forwards_points_regressor"
PRODUCTION_ALIAS = "production"

# TODO (JT): expected_minutes is train/serve skewed. Training rows take
# backfill-kind minutes predictions, which minutes.py documents as
# in-sample -- the minutes champion scored its own training seasons, so
# they flatter. Live rows take forward-kind predictions, which are
# genuinely out-of-sample and noisier, so this model will over-trust
# expected_minutes and underperform live. Fixing it means generating
# out-of-fold minutes predictions. Cross-validation will NOT catch this,
# because CV reads the backfill values too.

# TODO (JT): FPL introduced defensive-contribution points in 2025-26, so
# total_points before that season is a different quantity from the one
# being predicted, and defensive_contributions is null there. Training
# spans those seasons anyway -- an accepted trade for training-set size.
# Revisit once 2026-27 completes and two same-regime seasons exist.


class ForwardPointsPredictor(PositionPointsPredictor):
    """Points regressor for players in the FWD position.

    "Forward" here is the position, not the forward-in-time scoring
    direction the base class's ``predict_forward`` refers to.
    """

    POSITION = POSITION

    # Model inputs, as chosen in the notebook. Ordering matters only for
    # readability; the pipeline selects by name. Unlike the defender
    # model this deliberately omits the minutes model's bucket
    # probabilities and takes only expected_minutes.
    FEATURES = [
        "is_home",
        "expected_minutes",
        "xg_per90_rolling_5",
        "xa_per90_rolling_5",
        "xg_for_rolling_5",
        "goals_for_rolling_5",
        "xg_against_rolling_5",
        "goals_against_rolling_5",
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
    # A forward is judged on what his own club creates and on what the
    # opposition concedes -- the mirror image of the defender model.
    OWN_TEAM_COLUMNS = [
        "xg_for_rolling_5",
        "goals_for_rolling_5",
    ]
    OPPOSITION_COLUMNS = ["xg_against_rolling_5", "goals_against_rolling_5"]
    MINUTES_COLUMNS = ["expected_minutes"]


FORWARD_SPEC = ModelSpec(
    registered_model_name=REGISTERED_MODEL,
    production_alias=PRODUCTION_ALIAS,
    table=POINTS_PREDICTION,
    evaluation_table=TEST_POINTS_PREDICTION,
    position=POSITION,
)
