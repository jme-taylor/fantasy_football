"""The midfielder points regressor.

Everything that reads the column lists below lives in
:mod:`fantasy_football.modelling.points`; this module states only what
makes a midfielder a midfielder.
"""

import logging

from fantasy_football.modelling.components import Component
from fantasy_football.modelling.points import PositionPointsPredictor
from fantasy_football.modelling.predictor import ModelSpec
from fantasy_football.storage.tables import (
    POINTS_COMPONENT,
    TEST_POINTS_PREDICTION,
)

logger = logging.getLogger(__name__)

POSITION = "MID"

REGISTERED_MODEL = "midfielder_points_regressor"
PRODUCTION_ALIAS = "production"

# Both sides of the fixture are read for the same measures, so the
# opposition copies are prefixed. Without it the frame would carry two
# columns called xg_for_rolling_5 -- see OPPOSITION_PREFIX in points.py.
OPPOSITION_PREFIX = "opposition_"

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


class MidfielderPointsPredictor(PositionPointsPredictor):
    """Points regressor for players in the MID position."""

    POSITION = POSITION
    OPPOSITION_PREFIX = OPPOSITION_PREFIX

    # Model inputs. Ordering matters only for readability; the pipeline
    # selects by name.
    FEATURES = [
        "is_home",
        "expected_minutes",
        "p_zero",
        "p_partial",
        "p_sixty_plus",
        "xg_per90_rolling_5",
        "xa_per90_rolling_5",
        "xg_for_rolling_5",
        "goals_for_rolling_5",
        "xg_against_rolling_5",
        "goals_against_rolling_5",
        "clean_sheet_rolling_5",
        "opposition_xg_for_rolling_5",
        "opposition_goals_for_rolling_5",
        "opposition_xg_against_rolling_5",
        "opposition_goals_against_rolling_5",
        "opposition_clean_sheet_rolling_5",
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
    # A midfielder scores at both ends -- goals and assists, and a clean
    # sheet point -- so unlike the defender and forward models, which each
    # take one half of the team picture, this one reads every team measure
    # for both clubs and leaves the trade-off to the model.
    OWN_TEAM_COLUMNS = [
        "xg_for_rolling_5",
        "goals_for_rolling_5",
        "xg_against_rolling_5",
        "goals_against_rolling_5",
        "clean_sheet_rolling_5",
    ]
    OPPOSITION_COLUMNS = [
        "xg_for_rolling_5",
        "goals_for_rolling_5",
        "xg_against_rolling_5",
        "goals_against_rolling_5",
        "clean_sheet_rolling_5",
    ]
    # A clean sheet needs sixty minutes on the pitch, so how the minutes
    # are distributed matters here as it does for a defender.
    MINUTES_COLUMNS = [
        "expected_minutes",
        "p_zero",
        "p_partial",
        "p_sixty_plus",
    ]


MIDFIELDER_SPEC = ModelSpec(
    registered_model_name=REGISTERED_MODEL,
    production_alias=PRODUCTION_ALIAS,
    table=POINTS_COMPONENT,
    component=Component.TOTAL,
    evaluation_table=TEST_POINTS_PREDICTION,
    position=POSITION,
)
