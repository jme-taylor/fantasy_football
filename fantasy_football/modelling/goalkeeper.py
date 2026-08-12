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
from fantasy_football.storage.tables import POINTS_PREDICTION

logger = logging.getLogger(__name__)

POSITION = "GK"

REGISTERED_MODEL = "goalkeeper_points_regressor"
PRODUCTION_ALIAS = "production"

# Set even though the own-team and opposition lists are disjoint and no
# prefix is needed to keep them apart. This position has no registered
# model to invalidate, and a reader should not have to consult two lists
# to learn which club opposition_xg_for_rolling_5 describes.
OPPOSITION_PREFIX = "opposition_"

# TODO (JT): expected_minutes is train/serve skewed. Training rows take
# backfill-kind minutes predictions, which minutes.py documents as
# in-sample -- the minutes champion scored its own training seasons, so
# they flatter. Live rows take forward-kind predictions, which are
# genuinely out-of-sample and noisier, so this model will over-trust
# expected_minutes and underperform live. Fixing it means generating
# out-of-fold minutes predictions. Cross-validation will NOT catch this,
# because CV reads the backfill values too.

# TODO (JT): nothing reads these predictions yet. prediction.py serves
# only DEF from stored model predictions; FWD, MID and now GK are
# trained, registered, backfilled and forward-scored, and the optimiser
# then scores them with the rolling formula anyway. Generalising that
# path from one model-served position to four is deliberately separate
# work -- so promoting a version here does not change team selection.

# TODO (JT): FPL introduced defensive-contribution points in 2025-26, so
# total_points before that season is a different quantity from the one
# being predicted. The training window straddles the change. Goalkeepers
# are the position least affected -- defensive contributions are an
# outfield mechanism -- but the window is not regime-pure. Revisit once
# two same-regime seasons exist.


class GoalkeeperPointsPredictor(PositionPointsPredictor):
    """Points regressor for players in the GK position."""

    POSITION = POSITION
    OPPOSITION_PREFIX = OPPOSITION_PREFIX
    TRAINING_SEASONS = goalkeeper_covered_seasons()

    # Model inputs. Ordering matters only for readability; the pipeline
    # selects by name.
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
        "xg_against_rolling_5",
        "goals_against_rolling_5",
        "clean_sheet_rolling_5",
        "opposition_xg_for_rolling_5",
        "opposition_goals_for_rolling_5",
    ]

    # No xg or xa, and none of the outfield defensive block: a keeper
    # records essentially none of them, and on a training set a fifth the
    # size of the midfielder's, six structurally-zero features are
    # overfitting surface rather than harmless noise. Shot-stopping is
    # split deliberately -- xgot_faced is the danger faced, which drives
    # save points, and goals_prevented is the skill on top of it, which
    # drives clean sheets and the conceded deduction.
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
    ]
    # A keeper's entire points profile is defence-side, so this is the
    # defender's split: what his own club concedes, and what the
    # opposition creates.
    OWN_TEAM_COLUMNS = [
        "xg_against_rolling_5",
        "goals_against_rolling_5",
        "clean_sheet_rolling_5",
    ]
    OPPOSITION_COLUMNS = ["xg_for_rolling_5", "goals_for_rolling_5"]
    # A clean sheet needs sixty minutes on the pitch, so the distribution
    # matters and not the expectation alone. p_partial sits near zero for
    # almost every keeper -- a substitution means a red card or an injury
    # -- and the forest will ignore it; it is kept until the feature
    # importances say otherwise.
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
    position=POSITION,
)
