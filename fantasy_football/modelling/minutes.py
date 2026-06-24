"""End-to-end minutes-played classifier: features, CV scoring, MLflow.

A single 3-class model predicts each player's minutes bucket for a match:
benched (``0_minutes``), partial (``1_to_59_minutes``) or a full-ish shift
(``60_minutes_plus``). It is scored on the two decision boundaries downstream
points models care about — probability of any appearance and probability of a
60+ minute appearance — cross-validated with an expanding window over seasons,
then refit on all seasons and logged to MLflow.
"""

import logging

import mlflow
import mlflow.sklearn
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

from fantasy_football.constants import MINUTES_EXPERIMENT, MLFLOW_TRACKING_URI

from fantasy_football.features.availability import (
    add_chance_of_playing,
    add_positional_availability,
)
from fantasy_football.features.valuation import (
    add_positional_value_rank,
    add_team_value,
)
from fantasy_football.storage.database import (
    load_player_availability,
    load_player_match,
    load_player_week,
)

logger = logging.getLogger(__name__)

# Minutes-bucket target labels.
BUCKET_ZERO = "0_minutes"
BUCKET_PARTIAL = "1_to_59_minutes"
BUCKET_SIXTY_PLUS = "60_minutes_plus"
MINUTES_BUCKETS = [BUCKET_ZERO, BUCKET_PARTIAL, BUCKET_SIXTY_PLUS]

# Model inputs.
NUM_FEATURES = [
    "value",
    "value_share_of_team",
    "pos_value_rank",
    "players_same_pos",
    "chance_of_playing_this_round",
    "fit_rivals_same_pos",
    "fit_rivals_ahead",
]
CAT_FEATURES = ["position"]
FEATURES = NUM_FEATURES + CAT_FEATURES

# Representative minutes per bucket, for the expected-minutes leverage metric.
MINUTE_MIDPOINTS = {
    BUCKET_ZERO: 0.0,
    BUCKET_PARTIAL: 30.0,
    BUCKET_SIXTY_PLUS: 75.0,
}
# FPL appearance points: 0 for no game, 1 for <60 mins, 2 for 60+.
APPEARANCE_POINTS = {
    BUCKET_ZERO: 0.0,
    BUCKET_PARTIAL: 1.0,
    BUCKET_SIXTY_PLUS: 2.0,
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
    player_week: pl.DataFrame, availability: pl.DataFrame
) -> pl.DataFrame:
    """Build the model feature frame at the player-week grain.

    Value features (``value_share_of_team``, ``pos_value_rank``,
    ``players_same_pos``) and availability features
    (``chance_of_playing_this_round``, ``fit_rivals_same_pos``,
    ``fit_rivals_ahead``) are computed once per ``(season, gw, element)`` so the
    per-gameweek counts are correct, then narrowed to the columns the model
    consumes plus the join keys and ``position``.

    Parameters
    ----------
    player_week : pl.DataFrame
        Player-week rows from :func:`load_player_week` (``season``, ``gw``,
        ``element``, ``position``, ``team``, ``value`` and more).
    availability : pl.DataFrame
        Availability rows from :func:`load_player_availability`.

    Returns
    -------
    pl.DataFrame
        One row per ``(season, gw, element)`` with ``position`` and every
        column in ``NUM_FEATURES``.
    """
    frame = add_team_value(player_week)
    frame = add_positional_value_rank(frame)
    frame = add_chance_of_playing(frame, availability)
    frame = add_positional_availability(frame)
    return frame.select(
        ["season", "gw", "element", "position", *NUM_FEATURES]
    )


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
        Match rows from :func:`load_player_match` (``season``, ``gw``,
        ``element``, ``minutes`` and more).
    feature_frame : pl.DataFrame
        Output of :func:`build_feature_frame`.

    Returns
    -------
    pl.DataFrame
        Columns ``season``, ``gw``, ``element``, ``minutes``,
        ``minutes_bucket`` and every column in ``FEATURES``.
    """
    joined = player_match.select(
        ["season", "gw", "element", "minutes"]
    ).join(feature_frame, on=["season", "gw", "element"], how="inner")
    joined = create_minutes_bucket(joined)
    return joined.select(
        ["season", "gw", "element", "minutes", "minutes_bucket", *FEATURES]
    )


def assemble_model_frame() -> pl.DataFrame:
    """Load player data from the database and build the model frame.

    Returns
    -------
    pl.DataFrame
        The match-level model frame from :func:`build_model_frame`.
    """
    feature_frame = build_feature_frame(
        load_player_week(), load_player_availability()
    )
    return build_model_frame(load_player_match(), feature_frame)


def make_pipeline() -> Pipeline:
    """Build the logistic-regression pipeline.

    Numeric features are median-imputed then standardised; the categorical
    ``position`` is one-hot encoded. All preprocessing lives inside the pipeline
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
        ]
    )
    return Pipeline(
        [
            ("prep", preprocessor),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )


def season_folds(seasons: list[str]) -> list[tuple[list[str], str]]:
    """Build expanding-window CV folds over sorted seasons.

    Each fold trains on every prior season and tests on the next unseen one,
    mirroring deployment. Requires at least two seasons.

    Parameters
    ----------
    seasons : list[str]
        Season strings; sorted ascending internally.

    Returns
    -------
    list[tuple[list[str], str]]
        ``(train_seasons, test_season)`` pairs.
    """
    ordered = sorted(seasons)
    return [(ordered[:i], ordered[i]) for i in range(1, len(ordered))]


def _boundary_column(
    proba: np.ndarray, classes: list[str], label: str
) -> np.ndarray:
    """Return the probability column for ``label`` (zeros if absent)."""
    classes = list(classes)
    if label in classes:
        return proba[:, classes.index(label)]
    return np.zeros(proba.shape[0])


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
        "logloss_appear": float(
            log_loss(y_appear, p_appear, labels=[0, 1])
        ),
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

    expected_app = (
        p_partial * APPEARANCE_POINTS[BUCKET_PARTIAL]
        + p_60 * APPEARANCE_POINTS[BUCKET_SIXTY_PLUS]
    )
    true_app = np.where(minutes >= 60, 2.0, np.where(minutes > 0, 1.0, 0.0))
    out["e_app_mae"] = float(np.mean(np.abs(expected_app - true_app)))

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


def cross_validate(
    model_df: pl.DataFrame, folds: list[tuple[list[str], str]]
) -> tuple[list[dict[str, float]], dict[str, float]]:
    """Score the model over expanding-window folds.

    Parameters
    ----------
    model_df : pl.DataFrame
        Output of :func:`assemble_model_frame` (or an equivalent frame).
    folds : list[tuple[list[str], str]]
        ``(train_seasons, test_season)`` pairs from :func:`season_folds`.

    Returns
    -------
    tuple[list[dict], dict]
        Per-fold metric dicts, and an aggregate dict with ``{metric}_mean`` and
        ``{metric}_std`` for every metric present in all folds.
    """
    per_fold: list[dict[str, float]] = []
    for train_seasons, test_season in folds:
        train_df = model_df.filter(pl.col("season").is_in(train_seasons))
        test_df = model_df.filter(pl.col("season") == test_season)
        if train_df.is_empty() or test_df.is_empty():
            continue
        per_fold.append(_fit_predict_fold(train_df, test_df))

    agg: dict[str, float] = {}
    if per_fold:
        shared = set(per_fold[0])
        for metrics in per_fold[1:]:
            shared &= set(metrics)
        for key in sorted(shared):
            values = [metrics[key] for metrics in per_fold]
            agg[f"{key}_mean"] = float(np.mean(values))
            agg[f"{key}_std"] = float(np.std(values))
    return per_fold, agg


def train_final(model_df: pl.DataFrame) -> Pipeline:
    """Fit the pipeline on every row in ``model_df``.

    Parameters
    ----------
    model_df : pl.DataFrame
        The full model frame (all seasons).

    Returns
    -------
    sklearn.pipeline.Pipeline
        The fitted pipeline.
    """
    pipe = make_pipeline()
    pipe.fit(
        model_df.select(FEATURES).to_pandas(),
        model_df["minutes_bucket"].to_list(),
    )
    return pipe


def run_minutes_model() -> dict[str, float]:
    """Train, cross-validate and log the minutes model to MLflow.

    Assembles the model frame, scores it with expanding-window CV, fits the
    final pipeline on all seasons, and logs run params, per-fold metrics
    (stepped), aggregate mean/std metrics and the fitted model to the
    ``MINUTES_EXPERIMENT`` experiment. With fewer than two seasons there are no
    folds, so the function logs a warning and returns an empty dict.

    Returns
    -------
    dict[str, float]
        The aggregate metrics (``{metric}_mean`` / ``{metric}_std``), or an
        empty dict when there is too little data to evaluate.
    """
    model_df = assemble_model_frame()
    seasons = model_df["season"].unique().to_list()
    folds = season_folds(seasons)
    if not folds:
        logger.warning(
            "Need at least two seasons to evaluate the minutes model; "
            "found %d. Skipping.",
            len(seasons),
        )
        return {}

    per_fold, agg = cross_validate(model_df, folds)
    final_model = train_final(model_df)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MINUTES_EXPERIMENT)
    with mlflow.start_run():
        mlflow.log_params(
            {
                "model": "logistic_regression",
                "max_iter": 1000,
                "cv": "expanding_window_by_season",
                "n_folds": len(folds),
                "n_samples": model_df.height,
            }
        )
        for step, fold_metrics in enumerate(per_fold):
            for key, value in fold_metrics.items():
                mlflow.log_metric(key, value, step=step)
        mlflow.log_metrics(agg)
        mlflow.sklearn.log_model(final_model, artifact_path="model")

    logger.info("Minutes model logged to MLflow: %s", agg)
    return agg
