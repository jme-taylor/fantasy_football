"""The defender points model, end to end.

Same shape as :mod:`fantasy_football.modelling.minutes`: this module owns
one model's features, cross-validation, MLflow lifecycle and scoring, and
writes its output to a table rather than returning it.

It works at *match* grain -- one row per played fixture leg, so the two
halves of a double gameweek are separate observations. Predictions are
summed back to gameweek grain only when the optimiser consumes them,
in :class:`fantasy_football.modelling.models.StoredPredictionModel`.
"""

import logging
from typing import TYPE_CHECKING

import polars as pl
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from fantasy_football.features.views import register_feature_views

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

logger = logging.getLogger(__name__)

POSITION = "DEF"
TARGET = "total_points"

# Match-grain keys. They identify a row and are never model inputs.
KEY_COLUMNS = ["season", "gw", "element", "opponent"]

# MLflow Model Registry name and alias. Every training run registers a
# new version; the scoring path loads whichever carries the alias.
# Promotion is manual, in the MLflow UI.
REGISTERED_MODEL = "defender_points_regressor"
PRODUCTION_ALIAS = "production"

# Random-forest hyperparameters, carried over from the notebook the
# champion was chosen in (notebooks/260714_initial_defender_model.ipynb).
N_ESTIMATORS = 100
RANDOM_STATE = 42

# Gameweeks the first CV fold must train on. Below roughly this, most
# rolling-form windows are still empty and the fold measures noise.
MIN_TRAIN_GWS = 10

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
]

_PLAYER_FORM_COLUMNS = [
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
]
_OWN_TEAM_COLUMNS = [
    "xg_against_rolling_5",
    "goals_against_rolling_5",
    "clean_sheet_rolling_5",
]
_OPPOSITION_COLUMNS = ["xg_for_rolling_5", "goals_for_rolling_5"]

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


def model_frame_sql() -> str:
    """Return the SELECT behind the defender model frame.

    Joins the match-grain target on :mod:`player_match` to the minutes
    forecast, the player's rolling form, and both teams' rolling form.
    ``player_week`` supplies the player's club for the season, which is
    what resolves which side of ``team_match_form`` is his own.

    Returns
    -------
    str
        A SELECT over ``player_match`` and the registered feature views.
    """
    own = ",\n    ".join(
        f"own.{column} AS {column}" for column in _OWN_TEAM_COLUMNS
    )
    opposition = ",\n    ".join(
        f"opp.{column} AS {column}" for column in _OPPOSITION_COLUMNS
    )
    player = ",\n    ".join(
        f"mf.{column} AS {column}" for column in _PLAYER_FORM_COLUMNS
    )
    return f"""
SELECT
    m.season,
    m.gw,
    m.element,
    m.opponent,
    m.{TARGET},
    m.is_home,
    mn.expected_minutes,
    mn.p_zero,
    mn.p_partial,
    mn.p_sixty_plus,
    {player},
    {own},
    {opposition}
FROM player_match AS m
INNER JOIN player_season AS s
    ON  m.element = s.element
    AND m.season  = s.season
    AND s.position = '{POSITION}'
LEFT JOIN minutes_prediction AS mn
    ON  m.season   = mn.season
    AND m.gw       = mn.gw
    AND m.element  = mn.element
    AND m.opponent = mn.opponent
LEFT JOIN player_match_form AS mf
    ON  m.season   = mf.season
    AND m.gw       = mf.gw
    AND m.element  = mf.element
    AND m.opponent = mf.opponent
LEFT JOIN player_week AS pw
    ON  pw.season  = m.season
    AND pw.gw      = m.gw
    AND pw.element = m.element
LEFT JOIN fpl_team_id AS opp_id
    ON  opp_id.season  = m.season
    AND opp_id.team_id = m.opponent
LEFT JOIN team_match_form AS own
    ON  own.season     = m.season
    AND own.gw         = m.gw
    AND own.team       = pw.team
    AND own.opposition = opp_id.team
LEFT JOIN team_match_form AS opp
    ON  opp.season     = m.season
    AND opp.gw         = m.gw
    AND opp.team       = opp_id.team
    AND opp.opposition = pw.team
WHERE m.minutes IS NOT NULL
"""


def build_model_frame(connection: "DuckDBPyConnection") -> pl.DataFrame:
    """Build the defender training frame from the store.

    Registers the feature views on ``connection`` first, so callers do
    not have to remember to.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        An open connection to the store.

    Returns
    -------
    pl.DataFrame
        One row per played defender fixture leg, with
        ``KEY_COLUMNS``, ``TARGET`` and ``FEATURES``, in that order.
    """
    register_feature_views(connection)
    frame = connection.sql(model_frame_sql()).pl()
    return frame.select(KEY_COLUMNS + [TARGET] + FEATURES)


def make_pipeline() -> Pipeline:
    """Build the median-imputing random-forest pipeline.

    Imputation lives inside the pipeline so it refits per fold on
    training data only. There is deliberately no scaler: it is a no-op
    for trees, and the notebook only carried one because it started with
    a linear model.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted pipeline ending in a ``RandomForestRegressor``.
    """
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE
                ),
            ),
        ]
    )
