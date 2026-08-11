import logging
from typing import override

import numpy as np
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
from fantasy_football.features.valuation import (
    add_positional_value_rank,
    add_team_value,
)
from fantasy_football.modelling.folds import Fold
from fantasy_football.modelling.forward import (
    forward_player_weeks,
)
from fantasy_football.modelling.metrics import MinutesMetrics
from fantasy_football.modelling.predictor import ModelSpec, Predictor
from fantasy_football.storage.tables import (
    MINUTES_PREDICTION,
    PLAYER_AVAILABILITY,
    PLAYER_MATCH,
    PLAYER_SEASON,
    PLAYER_WEEK,
    TEAM_FIXTURE,
)

logger = logging.getLogger(__name__)

# Minutes-bucket target labels.
BUCKET_ZERO = "0_minutes"
BUCKET_PARTIAL = "1_to_59_minutes"
BUCKET_SIXTY_PLUS = "60_minutes_plus"
MINUTES_BUCKETS = [BUCKET_ZERO, BUCKET_PARTIAL, BUCKET_SIXTY_PLUS]

NUM_FEATURES = [
    "value",
    "value_share_of_team",
    "pos_value_rank",
    "players_same_pos",
    "chance_of_playing_this_round",
    "fit_rivals_same_pos",
    "fit_rivals_ahead",
    "avg_minutes_rolling_5",
    "games_played_this_season",
    "prev_season_minutes",
    "prev_season_start_rate",
    "prev_season_points_per_start",
    "pl_seasons_played",
    "seasons_since_last_pl",
    "age_years",
]
CAT_FEATURES = ["position"]
BOOL_FEATURES = ["is_pl_newcomer", "is_promoted_club"]
FEATURES = NUM_FEATURES + CAT_FEATURES + BOOL_FEATURES

# HISTORY_FEATURES and COLD_START_FEATURES are imported (rather than only used
# in features/history.py) so this assertion catches drift between this
# module's feature list and the features module the moment either changes.
# days_since_team_join is deliberately excluded from the model (see the
# comment on NUM_FEATURES above), so it is the one column carved out here.
assert set(HISTORY_FEATURES) | set(COLD_START_FEATURES) - {
    "days_since_team_join"
} <= set(NUM_FEATURES) | set(BOOL_FEATURES)

# Representative minutes per bucket, for the expected-minutes leverage metric.
MINUTE_MIDPOINTS = {
    BUCKET_ZERO: 0.0,
    BUCKET_PARTIAL: 30.0,
    BUCKET_SIXTY_PLUS: 75.0,
}


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
        column in ``NUM_FEATURES`` and ``BOOL_FEATURES``.
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
    frame = frame.with_columns(
        [pl.col(column).cast(pl.Int8) for column in BOOL_FEATURES]
    )
    return frame.select(
        ["season", "gw", "element", "position", *NUM_FEATURES, *BOOL_FEATURES]
    )


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
            *NUM_FEATURES,
            *CAT_FEATURES,
            *BOOL_FEATURES,
        ]
    )


def make_pipeline() -> Pipeline:
    """Build the logistic-regression pipeline.

    Numeric features are median-imputed then standardised; the categorical
    ``position`` is one-hot encoded; the boolean ``BOOL_FEATURES`` pass straight
    through untouched -- they are cast to ``Int8`` and never null by the time
    they reach the pipeline (see :func:`build_feature_frame`), so they need
    neither imputation nor scaling. All preprocessing lives inside the pipeline
    so it is refit per CV fold on train data only.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted pipeline ending in ``LogisticRegression(max_iter=1000)``.
    """
    preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                NUM_FEATURES,
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
            ("bool", "passthrough", BOOL_FEATURES),
        ]
    )
    return Pipeline(
        [
            ("prep", preprocessor),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )


def _boundary_column(
    proba: np.ndarray, classes: list[str], label: str
) -> np.ndarray:
    """Return the probability column for ``label`` (zeros if absent)."""
    classes = list(classes)
    if label in classes:
        return proba[:, classes.index(label)]
    return np.zeros(proba.shape[0])


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
) -> dict[str, float]:
    """Fit a fresh pipeline on the train split and score the test split."""
    pipe = make_pipeline()
    pipe.fit(
        train_df.select(FEATURES).to_pandas(),
        train_df["minutes_bucket"].to_list(),
    )
    proba = pipe.predict_proba(test_df.select(FEATURES).to_pandas())
    return boundary_metrics(
        test_df["minutes_bucket"].to_list(),
        proba,
        list(pipe.classes_),
        test_df["minutes"].to_list(),
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
    proba = model.predict_proba(frame.select(FEATURES).to_pandas())
    classes = list(model.classes_)
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

    @override
    def build_training_data(self) -> pl.DataFrame:
        player_match = PLAYER_MATCH.load(self.connection)
        feature_frame = build_feature_frame(
            PLAYER_WEEK.load(self.connection),
            PLAYER_AVAILABILITY.load(self.connection),
            player_match,
            PLAYER_SEASON.load(self.connection),
            TEAM_FIXTURE.load(self.connection),
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
            forward_fixtures=forward_fixtures,
        )
        return forward_fixtures.select(
            ["season", "gw", "element", "opponent"]
        ).join(feature_frame, on=["season", "gw", "element"], how="inner")

    @override
    def fit_predict_fold(self, fold: Fold) -> MinutesMetrics:
        return MinutesMetrics(**_fit_predict_fold(fold.train, fold.test))

    @override
    def train_final(self, feature_frame: pl.DataFrame) -> Pipeline:
        pipe = make_pipeline()
        pipe.fit(
            feature_frame.select(FEATURES).to_pandas(),
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
)
