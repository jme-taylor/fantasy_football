"""The goalkeeper points regressor.

Everything that reads the column lists below lives in
:mod:`fantasy_football.modelling.points`; this module states only what
makes a goalkeeper a goalkeeper.

Two things set this position apart from the other three.

*Its features did not exist before this model.* No keeper measure was
windowed into rolling form until now -- the shared per-90 lists are
outfield stats. The keeper stats live in their own constants in
``features/match_form.py``, and the reason they are kept apart is written
there.

*It cannot train on the whole history.* The FCI keeper columns open at
2024-25, and the training frame otherwise takes every played fixture leg,
so without ``TRAINING_SEASONS`` the model's most informative features
would be median-filled across roughly eight earlier seasons. The window
is computed from the stat lists rather than written down, so a
narrower-coverage feature narrows it automatically.
"""

import logging

from fantasy_football.features.match_form import goalkeeper_covered_seasons
from fantasy_football.modelling.points import PositionPointsPredictor
from fantasy_football.modelling.predictor import ModelSpec
from fantasy_football.storage.tables import (
    POINTS_PREDICTION,
    TEST_POINTS_PREDICTION,
)

logger = logging.getLogger(__name__)

POSITION = "GK"

REGISTERED_MODEL = "goalkeeper_points_regressor"
PRODUCTION_ALIAS = "production"

OPPOSITION_PREFIX = "opposition_"

# TODO (JT): expected_minutes is train/serve skewed.
# TODO (JT): FPL introduced defensive-contribution points in 2025-26


class GoalkeeperPointsPredictor(PositionPointsPredictor):
    """Points regressor for players in the GK position."""

    POSITION = POSITION
    OPPOSITION_PREFIX = OPPOSITION_PREFIX
    TRAINING_SEASONS = goalkeeper_covered_seasons()

    FEATURES = [
        "is_home",
        "expected_minutes",
        "p_zero",
        "p_partial",
        "p_sixty_plus",
        "saves_per90_rolling_5",
        "penalties_saved_per90_rolling_5",
        "goals_conceded_per90_rolling_5",
        "goals_prevented_per90_rolling_5",
        "xgot_faced_per90_rolling_5",
        "high_claim_per90_rolling_5",
        "sweeper_actions_per90_rolling_5",
        "yellow_cards_per90_rolling_5",
        "red_cards_per90_rolling_5",
        "yellow_cards_season_to_date",
        "red_cards_season_to_date",
        "days_since_last_appearance",
        "xg_against_rolling_5",
        "goals_against_rolling_5",
        "clean_sheet_rolling_5",
        "opposition_xg_for_rolling_5",
        "opposition_goals_for_rolling_5",
    ]

    PLAYER_FORM_COLUMNS = [
        "saves_per90_rolling_5",
        "penalties_saved_per90_rolling_5",
        "goals_conceded_per90_rolling_5",
        "goals_prevented_per90_rolling_5",
        "xgot_faced_per90_rolling_5",
        "high_claim_per90_rolling_5",
        "sweeper_actions_per90_rolling_5",
        "yellow_cards_per90_rolling_5",
        "red_cards_per90_rolling_5",
        "yellow_cards_season_to_date",
        "red_cards_season_to_date",
        "days_since_last_appearance",
    ]

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


GOALKEEPER_SPEC = ModelSpec(
    registered_model_name=REGISTERED_MODEL,
    production_alias=PRODUCTION_ALIAS,
    table=POINTS_PREDICTION,
    evaluation_table=TEST_POINTS_PREDICTION,
    position=POSITION,
)
