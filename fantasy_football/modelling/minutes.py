import logging
from collections.abc import Sequence
from typing import override

import numpy as np
import pandas as pd
import polars as pl
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier

from fantasy_football.constants import (
    MINUTES_PRODUCTION_ALIAS,
    MINUTES_REGISTERED_MODEL,
)
from fantasy_football.features.availability import (
    add_chance_of_playing,
    add_games_played_this_season,
    add_positional_availability,
    add_rolling_minutes,
)
from fantasy_football.features.history import (
    COLD_START_FEATURES,
    HISTORY_FEATURES,
    add_cold_start_features,
    add_history_features,
)
from fantasy_football.features.transfermarkt import (
    add_transfermarkt_features,
)
from fantasy_football.features.valuation import (
    add_positional_value_rank,
    add_team_value,
)
from fantasy_football.modelling.folds import Fold, FoldResult
from fantasy_football.modelling.forward import (
    forward_player_weeks,
)
from fantasy_football.modelling.metrics import MinutesMetrics
from fantasy_football.modelling.predictor import (
    ModelSpec,
    Predictor,
    feature_json,
)
from fantasy_football.storage.coverage import FPL_STAT_SEASONS
from fantasy_football.storage.tables import (
    MINUTES_PREDICTION,
    PLAYER_AVAILABILITY,
    PLAYER_MATCH,
    PLAYER_SEASON,
    PLAYER_WEEK,
    TEAM_FIXTURE,
    TEST_MINUTES_PREDICTION,
    TM_MARKET_VALUE,
    TM_PLAYER,
    TM_PLAYER_MAP,
    TM_TRANSFER,
)

logger = logging.getLogger(__name__)

# Minutes-bucket target labels.
BUCKET_ZERO = "0_minutes"
BUCKET_PARTIAL = "1_to_59_minutes"
BUCKET_SIXTY_PLUS = "60_minutes_plus"
MINUTES_BUCKETS = [BUCKET_ZERO, BUCKET_PARTIAL, BUCKET_SIXTY_PLUS]

# Features built by features/transfermarkt.py, listed apart from the FPL
# ones so the two feature sources stay legible. Not all of them need a
# Transfermarkt match: the prev-game and to-date minutes come from
# player_match, and only the value, rank and arrival features go null for
# an unmapped player.
TM_FEATURES = [
    "value_tm",
    "players_same_tm_pos",
    "tm_pos_value_share",
    "tm_pos_value_rank_norm",
    "prev_game_minutes",
    "prev_game_minutes_2",
    "days_since_prev_game",
    "prev_pct_position_minutes",
    "minutes_to_date",
    "pct_position_minutes_to_date",
    "rivals_joined_same_pos",
    "higher_value_rivals_joined_same_pos",
    "max_rival_value_eur",
    "days_since_rival_joined",
]

NUM_FEATURES = [
    "players_same_pos",
    "chance_of_playing_this_round",
    "fit_rivals_same_pos",
    "fit_rivals_ahead",
    "avg_minutes_rolling_5",
    "games_played_this_season",
    "prev_season_minutes",
    "prev_season_start_rate",
    "pl_seasons_played",
    "seasons_since_last_pl",
    "age_years",
    *TM_FEATURES,
]
# The booleans are one-hot encoded rather than passed through: they can be
# null now that a player may have no Transfermarkt row, and "unknown" is a
# third state the tree can split on.
CAT_FEATURES = ["is_pl_newcomer", "is_promoted_club", "tm_position"]
FEATURES = NUM_FEATURES + CAT_FEATURES

# Placeholder for a null categorical, so the one-hot encoder sees a
# category rather than a NaN it cannot encode.
UNKNOWN_CATEGORY = "unknown"

TRAINING_START_SEASON = "2020-21"
TRAINING_SEASONS: tuple[str, ...] = tuple(
    season for season in FPL_STAT_SEASONS if season >= TRAINING_START_SEASON
)

# HISTORY_FEATURES and COLD_START_FEATURES are imported (rather than only used
# in features/history.py) so this assertion catches drift between this
# module's feature list and the features module the moment either changes.
# days_since_team_join and prev_season_points_per_start are deliberately
# excluded from the model, so they are the columns carved out here.
assert (set(HISTORY_FEATURES) | set(COLD_START_FEATURES)) - {
    "days_since_team_join",
    "prev_season_points_per_start",
} <= set(FEATURES)

# Representative minutes per bucket, for the expected-minutes leverage metric.
MINUTE_MIDPOINTS = {
    BUCKET_ZERO: 0.0,
    BUCKET_PARTIAL: 30.0,
    BUCKET_SIXTY_PLUS: 75.0,
}


class LabelledXGBClassifier(BaseEstimator, ClassifierMixin):
    """XGBoost classifier that keeps string class labels.

    ``XGBClassifier`` only accepts integer-encoded targets, so the encoder
    lives inside the estimator rather than beside it. Everything
    downstream -- :func:`_boundary_column`, :func:`fold_predictions`,
    :func:`score_minutes` -- matches ``classes_`` against the bucket
    label constants, so an int-classed model would silently mismatch.
    """

    def __init__(self, params: dict[str, object] | None = None) -> None:
        """Store the XGBoost keyword arguments, unpacked at fit time."""
        self.params = params

    def fit(
        self, X: pd.DataFrame, y: Sequence[str]
    ) -> "LabelledXGBClassifier":
        """Encode the labels, fit XGBoost, and expose string ``classes_``."""
        self._encoder = LabelEncoder()
        encoded = self._encoder.fit_transform(list(y))
        self._model = XGBClassifier(**(self.params or {}))
        self._model.fit(X, encoded)
        self.classes_ = list(self._encoder.classes_)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return class probabilities, columns aligned to ``classes_``."""
        return self._model.predict_proba(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted bucket labels as strings."""
        return self._encoder.inverse_transform(self._model.predict(X))


def create_minutes_bucket(
    data: pl.DataFrame, minutes_col: str = "minutes"
) -> pl.DataFrame:
    """Add a ``minutes_bucket`` target derived from a minutes column.

    Buckets are ``0_minutes`` (exactly zero), ``1_to_59_minutes`` (a partial
    appearance) and ``60_minutes_plus`` (a full-ish shift, the clean-sheet /
    appearance-point threshold).

    Parameters
    ----------
    data : pl.DataFrame
        Frame containing ``minutes_col``.
    minutes_col : str
        Name of the integer minutes column to bucket. Defaults to ``minutes``.

    Returns
    -------
    pl.DataFrame
        ``data`` with a ``minutes_bucket`` string column added.
    """
    return data.with_columns(
        pl.when(pl.col(minutes_col) == 0)
        .then(pl.lit(BUCKET_ZERO))
        .when(pl.col(minutes_col) < 60)
        .then(pl.lit(BUCKET_PARTIAL))
        .otherwise(pl.lit(BUCKET_SIXTY_PLUS))
        .alias("minutes_bucket")
    )


def build_feature_frame(
    player_week: pl.DataFrame,
    availability: pl.DataFrame,
    player_match: pl.DataFrame,
    player_season: pl.DataFrame,
    team_fixture: pl.DataFrame,
    tm_player: pl.DataFrame,
    tm_player_map: pl.DataFrame,
    tm_market_value: pl.DataFrame,
    tm_transfer: pl.DataFrame,
    forward_fixtures: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build the model feature frame at the player-week grain.

    Contemporaneous features (value share, positional rank, chance of playing,
    fit rivals) are computed per ``(season, gw, element)`` so the per-gameweek
    counts are correct. History features join through ``player_code`` to reach
    prior seasons, and cold-start features stand in for players who have none.

    Parameters
    ----------
    player_week : pl.DataFrame
        Player-week rows from :meth:`PLAYER_WEEK.load`.
    availability : pl.DataFrame
        Availability rows from :meth:`PLAYER_AVAILABILITY.load`.
    player_match : pl.DataFrame
        Per-fixture rows from :meth:`PLAYER_MATCH.load`, used for prior-season
        aggregates that must not be distorted by double gameweeks.
    player_season : pl.DataFrame
        Identity rows from :meth:`PLAYER_SEASON.load`.
    team_fixture : pl.DataFrame
        Fixture rows from :meth:`TEAM_FIXTURE.load`, used to detect promoted
        clubs.
    tm_player : pl.DataFrame
        Transfermarkt players from :meth:`TM_PLAYER.load`.
    tm_player_map : pl.DataFrame
        FPL-to-Transfermarkt bridge from :meth:`TM_PLAYER_MAP.load`.
    tm_market_value : pl.DataFrame
        Dated valuations from :meth:`TM_MARKET_VALUE.load`.
    tm_transfer : pl.DataFrame
        Transfer history from :meth:`TM_TRANSFER.load`.
    forward_fixtures : pl.DataFrame | None
        Optional match-grain rows for fixtures not yet played, with a null
        ``minutes`` (``season``, ``gw``, ``element``, ``kickoff_time``,
        ``minutes``). When given, they are appended to ``player_match`` to
        form the stream :func:`add_rolling_minutes` scans, so a future
        gameweek inherits the rolling window frozen at the player's last
        played match instead of falling back to a null. This feeds *only*
        the rolling-minutes window -- every other feature in this function
        is still computed on the week grain exactly as before.

    Returns
    -------
    pl.DataFrame
        One row per ``(season, gw, element)`` with ``position`` and every
        column in ``FEATURES``.
    """
    frame = add_team_value(player_week)
    frame = add_positional_value_rank(frame)
    frame = add_chance_of_playing(frame, availability)
    frame = add_positional_availability(frame)
    match_stream = (
        player_match
        if forward_fixtures is None
        else pl.concat(
            [
                player_match.select(
                    ["season", "gw", "element", "kickoff_time", "minutes"]
                ),
                forward_fixtures.select(
                    ["season", "gw", "element", "kickoff_time", "minutes"]
                ),
            ],
            how="vertical",
        )
    )
    frame = add_rolling_minutes(frame, match_stream, player_season)
    frame = add_games_played_this_season(frame)
    frame = add_history_features(frame, player_match, player_season)
    frame = frame.join(
        player_season.select(
            ["season", "element", "birth_date", "team_join_date"]
        ),
        on=["season", "element"],
        how="left",
        coalesce=True,
    )
    frame = add_cold_start_features(frame, team_fixture)
    frame = add_transfermarkt_features(
        frame,
        match_stream,
        player_season,
        tm_player,
        tm_player_map,
        tm_market_value,
        tm_transfer,
    )
    frame = frame.filter(pl.col("season").is_in(TRAINING_SEASONS))
    return frame.select(["season", "gw", "element", "position", *FEATURES])


# TODO (JT): Work out a cleaner way to do this (feature store?)
def build_model_frame(
    player_match: pl.DataFrame, feature_frame: pl.DataFrame
) -> pl.DataFrame:
    """Join features onto match rows and derive the bucket target.

    Match-level minutes from ``player_match`` define the target (so double
    gameweeks contribute one row per match). Features are inner-joined on
    ``(season, gw, element)`` — match rows with no feature row are dropped.

    Parameters
    ----------
    player_match : pl.DataFrame
        Match rows from :meth:`PLAYER_MATCH.load` (``season``, ``gw``,
        ``element``, ``minutes`` and more).
    feature_frame : pl.DataFrame
        Output of :func:`build_feature_frame`.

    Returns
    -------
    pl.DataFrame
        Columns ``season``, ``gw``, ``element``, ``opponent``, ``minutes``,
        ``minutes_bucket`` and every column in ``FEATURES``. ``opponent`` is
        carried through so downstream scoring keys predictions at match grain
        (one row per fixture, disambiguating double gameweeks).
    """
    joined = player_match.select(
        ["season", "gw", "element", "opponent", "minutes"]
    ).join(feature_frame, on=["season", "gw", "element"], how="inner")
    joined = create_minutes_bucket(joined)
    return joined.select(
        [
            "season",
            "gw",
            "element",
            "opponent",
            "minutes",
            "minutes_bucket",
            *FEATURES,
        ]
    )


def make_pipeline() -> Pipeline:
    """Build the gradient-boosted minutes pipeline.

    Numeric features pass through unimputed and unscaled: XGBoost learns a
    default direction per split for missing values, which carries more
    information than a median stand-in, and trees are indifferent to scale.
    Categoricals are one-hot encoded after :func:`prepare_features` has
    turned their nulls into an explicit category. All preprocessing lives
    inside the pipeline so it is refit per CV fold on train data only.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted pipeline ending in :class:`LabelledXGBClassifier`.
    """
    preprocessor = ColumnTransformer(
        [
            ("num", "passthrough", NUM_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
        ]
    )
    return Pipeline(
        [
            ("prep", preprocessor),
            ("clf", LabelledXGBClassifier()),
        ]
    )


def prepare_features(frame: pl.DataFrame) -> pd.DataFrame:
    """Shape a feature frame for the pipeline.

    Categoricals are cast to string and their nulls replaced with
    :data:`UNKNOWN_CATEGORY`, so the one-hot encoder sees a category
    rather than a NaN. Numerics keep their nulls, which reach XGBoost as
    NaN and are exactly what its missing-direction split consumes.

    Parameters
    ----------
    frame : pl.DataFrame
        Rows carrying every column in :data:`FEATURES`.

    Returns
    -------
    pd.DataFrame
        The feature columns, in :data:`FEATURES` order.
    """
    return (
        frame.with_columns(
            [
                pl.col(column).cast(pl.Utf8).fill_null(UNKNOWN_CATEGORY)
                for column in CAT_FEATURES
            ]
        )
        .select(FEATURES)
        .to_pandas()
    )


def _boundary_column(
    proba: np.ndarray, classes: list[str], label: str
) -> np.ndarray:
    """Return the probability column for ``label``.

    Raises rather than defaulting to zeros: a label the model has never
    seen and a label the model reports under a different type look
    identical here, and a zero column is a plausible-looking prediction
    that would flow all the way to the optimiser unnoticed.
    """
    classes = list(classes)
    if label not in classes:
        raise ValueError(
            f"{label!r} is not among the model's classes {classes!r}"
        )
    return proba[:, classes.index(label)]


# TODO (JT): Make a boundary metrics dataclass as output type
# TODO (JT): Go through the metrics and make sure they make sense
def boundary_metrics(
    y_true_bucket: list[str],
    proba: np.ndarray,
    classes: list[str],
    true_minutes: list[float],
) -> dict[str, float]:
    """Score the appearance and 60-minute decision boundaries.

    Parameters
    ----------
    y_true_bucket : list[str]
        True minutes-bucket labels.
    proba : np.ndarray
        ``predict_proba`` output, columns aligned to ``classes``.
    classes : list[str]
        The classifier's ``classes_`` (column order of ``proba``).
    true_minutes : list[float]
        Actual minutes, for the expected-minutes leverage metric.

    Returns
    -------
    dict[str, float]
        ``logloss_appear``, ``brier_appear``, ``logloss_60``, ``brier_60``,
        ``e_min_mae``, ``e_app_mae`` always; ``auc_appear`` / ``auc_60`` only
        when that boundary's test rows contain both classes.
    """
    y_true = np.asarray(y_true_bucket)
    minutes = np.asarray(true_minutes, dtype=float)

    p_60 = _boundary_column(proba, classes, BUCKET_SIXTY_PLUS)
    p_partial = _boundary_column(proba, classes, BUCKET_PARTIAL)
    p_appear = p_partial + p_60

    y_appear = (y_true != BUCKET_ZERO).astype(int)
    y_60 = (y_true == BUCKET_SIXTY_PLUS).astype(int)

    out: dict[str, float] = {
        "logloss_appear": float(log_loss(y_appear, p_appear, labels=[0, 1])),
        "brier_appear": float(brier_score_loss(y_appear, p_appear)),
        "logloss_60": float(log_loss(y_60, p_60, labels=[0, 1])),
        "brier_60": float(brier_score_loss(y_60, p_60)),
    }
    if len(np.unique(y_appear)) == 2:
        out["auc_appear"] = float(roc_auc_score(y_appear, p_appear))
    if len(np.unique(y_60)) == 2:
        out["auc_60"] = float(roc_auc_score(y_60, p_60))

    expected_minutes = (
        p_partial * MINUTE_MIDPOINTS[BUCKET_PARTIAL]
        + p_60 * MINUTE_MIDPOINTS[BUCKET_SIXTY_PLUS]
    )
    out["e_min_mae"] = float(np.mean(np.abs(expected_minutes - minutes)))

    return out


def _fit_predict_fold(
    train_df: pl.DataFrame, test_df: pl.DataFrame
) -> tuple[dict[str, float], np.ndarray, list[str]]:
    """Fit a fresh pipeline on the train split and score the test split.

    The probabilities and class labels come back with the metrics so the
    caller can shape stored predictions from them rather than running
    ``predict_proba`` over the same rows again.
    """
    pipe = make_pipeline()
    pipe.fit(
        prepare_features(train_df),
        train_df["minutes_bucket"].to_list(),
    )
    proba = pipe.predict_proba(prepare_features(test_df))
    classes = list(pipe.classes_)
    return (
        boundary_metrics(
            test_df["minutes_bucket"].to_list(),
            proba,
            classes,
            test_df["minutes"].to_list(),
        ),
        proba,
        classes,
    )


def fold_predictions(
    test_df: pl.DataFrame, proba: np.ndarray, classes: list[str]
) -> pl.DataFrame:
    """Shape one fold's scored rows for the evaluation table.

    ``run_id`` is left off: the fold does not know which run it belongs
    to, and the predictor stamps it on the way to storage.

    Both actuals are kept. The bucket is the target the classifier is
    scored against; the minutes are what ``expected_minutes`` is compared
    to downstream, and neither is derivable from the other.

    Parameters
    ----------
    test_df : pl.DataFrame
        The fold's held-out rows.
    proba : np.ndarray
        Class probabilities for those rows.
    classes : list[str]
        The fitted model's ``classes_``.

    Returns
    -------
    pl.DataFrame
        Keys, class probabilities, expected minutes, both actuals and the
        model's inputs.
    """
    # The scored frame holds keys and probabilities, not features, so the
    # JSON is encoded against the fold's own rows and carried across.
    # Both frames are built from ``test_df`` in its own row order, which
    # is what makes attaching the actuals positionally safe.
    features = test_df.select(feature_json(test_df, FEATURES)).to_series()
    return minutes_from_proba(test_df, proba, classes).with_columns(
        actual_bucket=test_df["minutes_bucket"],
        actual_minutes=test_df["minutes"].cast(pl.Int64),
        features=features,
    )


def score_minutes(frame: pl.DataFrame, model: Pipeline) -> pl.DataFrame:
    """Score a fitted minutes model over ``frame`` and return predictions.

    Applies ``predict_proba`` and aligns the probability columns to the model's
    ``classes_`` (a class absent from ``classes_`` yields a zero column), then
    derives ``expected_minutes`` from the bucket midpoints.

    Parameters
    ----------
    frame : pl.DataFrame
        Rows with the match keys (``season``, ``gw``, ``element``,
        ``opponent``) and every column in ``FEATURES``.
    model : sklearn.pipeline.Pipeline
        A fitted pipeline exposing ``predict_proba`` and ``classes_``.

    Returns
    -------
    pl.DataFrame
        One row per input row with ``season``, ``gw``, ``element``,
        ``opponent``, ``p_zero``, ``p_partial``, ``p_sixty_plus`` and
        ``expected_minutes``.
    """
    return minutes_from_proba(
        frame,
        model.predict_proba(prepare_features(frame)),
        list(model.classes_),
    )


def minutes_from_proba(
    frame: pl.DataFrame, proba: np.ndarray, classes: list[str]
) -> pl.DataFrame:
    """Shape already-computed class probabilities into prediction rows.

    Split out from :func:`score_minutes` so a caller holding ``proba``
    already -- fold scoring does -- can shape it without a second
    ``predict_proba`` pass over the same rows.

    Parameters
    ----------
    frame : pl.DataFrame
        The rows ``proba`` was computed over, carrying the match keys.
    proba : np.ndarray
        Class probabilities, one row per row of ``frame``.
    classes : list[str]
        The fitted model's ``classes_``, naming ``proba``'s columns.

    Returns
    -------
    pl.DataFrame
        The match keys, the three bucket probabilities and
        ``expected_minutes``, in ``frame``'s row order.
    """
    p_zero = _boundary_column(proba, classes, BUCKET_ZERO)
    p_partial = _boundary_column(proba, classes, BUCKET_PARTIAL)
    p_sixty_plus = _boundary_column(proba, classes, BUCKET_SIXTY_PLUS)
    expected_minutes = (
        p_partial * MINUTE_MIDPOINTS[BUCKET_PARTIAL]
        + p_sixty_plus * MINUTE_MIDPOINTS[BUCKET_SIXTY_PLUS]
    )
    return frame.select(["season", "gw", "element", "opponent"]).with_columns(
        p_zero=pl.Series(p_zero),
        p_partial=pl.Series(p_partial),
        p_sixty_plus=pl.Series(p_sixty_plus),
        expected_minutes=pl.Series(expected_minutes),
    )


class MinutesPredictor(Predictor):
    """Predictor for minutes."""

    @property
    @override
    def expected_features(self) -> list[str] | None:
        """Return the declared model inputs."""
        return list(FEATURES)

    def _transfermarkt_tables(self) -> tuple[pl.DataFrame, ...]:
        """Load the Transfermarkt tables, in build_feature_frame order."""
        return (
            TM_PLAYER.load(self.connection),
            TM_PLAYER_MAP.load(self.connection),
            TM_MARKET_VALUE.load(self.connection),
            TM_TRANSFER.load(self.connection),
        )

    @override
    def build_training_data(self) -> pl.DataFrame:
        player_match = PLAYER_MATCH.load(self.connection)
        feature_frame = build_feature_frame(
            PLAYER_WEEK.load(self.connection),
            PLAYER_AVAILABILITY.load(self.connection),
            player_match,
            PLAYER_SEASON.load(self.connection),
            TEAM_FIXTURE.load(self.connection),
            *self._transfermarkt_tables(),
        )
        return build_model_frame(player_match, feature_frame)

    @override
    def build_forward_data(
        self, forward_fixtures: pl.DataFrame
    ) -> pl.DataFrame:
        combined_weeks = pl.concat(
            [
                PLAYER_WEEK.load(self.connection),
                forward_player_weeks(forward_fixtures),
            ],
            how="diagonal",
        )
        feature_frame = build_feature_frame(
            combined_weeks,
            PLAYER_AVAILABILITY.load(self.connection),
            PLAYER_MATCH.load(self.connection),
            PLAYER_SEASON.load(self.connection),
            TEAM_FIXTURE.load(self.connection),
            *self._transfermarkt_tables(),
            forward_fixtures=forward_fixtures,
        )
        return forward_fixtures.select(
            ["season", "gw", "element", "opponent"]
        ).join(feature_frame, on=["season", "gw", "element"], how="inner")

    @override
    def fit_predict_fold(self, fold: Fold) -> FoldResult:
        """Fit on the fold's train split and score its test split."""
        metrics, proba, classes = _fit_predict_fold(fold.train, fold.test)
        return FoldResult(
            metrics=MinutesMetrics(**metrics),
            predictions=fold_predictions(fold.test, proba, classes),
        )

    @override
    def train_final(self, feature_frame: pl.DataFrame) -> Pipeline:
        pipe = make_pipeline()
        pipe.fit(
            prepare_features(feature_frame),
            feature_frame["minutes_bucket"].to_list(),
        )
        return pipe

    @override
    def build_prediction_rows(
        self,
        feature_frame: pl.DataFrame,
        model: Pipeline,
        version: str,
        kind: str,
    ) -> pl.DataFrame:
        return (
            score_minutes(feature_frame, model)
            .with_columns(
                model_version=pl.lit(version),
                prediction_kind=pl.lit(kind),
                snapshot_captured_at=pl.lit(None, dtype=pl.Datetime("us")),
            )
            .select(self.model_spec.table.columns)
        )


MINUTES_SPEC = ModelSpec(
    registered_model_name=MINUTES_REGISTERED_MODEL,
    production_alias=MINUTES_PRODUCTION_ALIAS,
    table=MINUTES_PREDICTION,
    evaluation_table=TEST_MINUTES_PREDICTION,
)
