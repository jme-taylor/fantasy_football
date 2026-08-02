import logging
from typing import TYPE_CHECKING, Any

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

from fantasy_football.constants import (
    CURRENT_SEASON,
    MINUTES_EXPERIMENT,
    MINUTES_PRODUCTION_ALIAS,
    MINUTES_REGISTERED_MODEL,
    MLFLOW_TRACKING_URI,
)
from fantasy_football.extraction.fpl import FplAPI
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
from fantasy_football.modelling.folds import season_folds
from fantasy_football.modelling.forward import (
    build_forward_fixtures,
    forward_player_weeks,
    last_played_gw,
    latest_snapshot,
)
from fantasy_football.modelling.registry import load_production_model
from fantasy_football.storage.database import get_connection
from fantasy_football.storage.tables import (
    BACKFILL_KIND,
    FORWARD_KIND,
    MINUTES_PREDICTION,
    PLAYER_AVAILABILITY,
    PLAYER_MATCH,
    PLAYER_SEASON,
    PLAYER_SNAPSHOT,
    PLAYER_WEEK,
    TEAM_FIXTURE,
    minutes_prediction_versions,
)

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

logger = logging.getLogger(__name__)

# Minutes-bucket target labels.
BUCKET_ZERO = "0_minutes"
BUCKET_PARTIAL = "1_to_59_minutes"
BUCKET_SIXTY_PLUS = "60_minutes_plus"
MINUTES_BUCKETS = [BUCKET_ZERO, BUCKET_PARTIAL, BUCKET_SIXTY_PLUS]

# Model inputs. The first block is contemporaneous -- everything knowable at
# the deadline. The second reaches across the summer break through player_code
# (see features/history.py), which is what lets the model say anything useful
# at GW1 of a new season, before any in-season evidence exists.
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
    # days_since_team_join is deliberately excluded: FPL only began publishing
    # team_join_date around 2024 and FCI's players.csv does not carry it at
    # all, so it is 100% null in 6 of 8 training seasons and 564/564 null in
    # 2026-27 -- the season this work exists to serve. Its non-nullness
    # correlates almost perfectly with season membership, so the
    # median-imputed column risked acting as a season proxy in train_final.
    # It is still produced by add_cold_start_features and kept in
    # player_season; re-add it here once FPL's coverage reaches further back.
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


def assemble_model_frame() -> pl.DataFrame:
    """Load player data from the database and build the model frame.

    Returns
    -------
    pl.DataFrame
        The match-level model frame from :func:`build_model_frame`.
    """
    player_match = PLAYER_MATCH.load()
    feature_frame = build_feature_frame(
        PLAYER_WEEK.load(),
        PLAYER_AVAILABILITY.load(),
        player_match,
        PLAYER_SEASON.load(),
        TEAM_FIXTURE.load(),
    )
    return build_model_frame(player_match, feature_frame)


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
    ``MINUTES_EXPERIMENT`` experiment. With fewer than two seasons
    :func:`season_folds` produces no folds, so ``per_fold`` and ``agg`` are
    both empty and no CV metrics are logged, but the run still fits and
    registers the final model.

    Returns
    -------
    dict[str, float]
        The aggregate metrics (``{metric}_mean`` / ``{metric}_std``), or an
        empty dict when there is too little data to cross-validate.
    """
    model_df = assemble_model_frame()
    seasons = model_df["season"].unique().to_list()
    folds = season_folds(seasons)

    per_fold, agg = cross_validate(model_df, folds)
    final_model = train_final(model_df)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MINUTES_EXPERIMENT)
    with mlflow.start_run():
        # TODO (JT: Don't have magic strings here
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
        # TODO (JT): Add auto alias promotion if the metrics are good enough
        mlflow.sklearn.log_model(
            final_model,
            name="model",
            registered_model_name=MINUTES_REGISTERED_MODEL,
        )

    # TODO (JT: Don't logg the agg, just show the model ID
    logger.info("Minutes model logged to MLflow: %s", agg)
    return agg


def get_production_model() -> tuple[str, Any] | None:
    """Return the production-aliased version string and model, or None.

    Thin wrapper over
    :func:`fantasy_football.modelling.registry.load_production_model`,
    kept so callers and tests can keep using the minutes-specific name.

    Returns
    -------
    tuple[str, Any] | None
        The aliased model version and the loaded model, or ``None`` if no
        production alias is set.
    """
    return load_production_model(
        MINUTES_REGISTERED_MODEL, MINUTES_PRODUCTION_ALIAS
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


def _score_and_store(
    connection: "DuckDBPyConnection",
    model_frame: pl.DataFrame,
    season: str,
    model: Pipeline,
    version: str,
) -> None:
    """Score one season's rows and store them as backfill predictions.

    Rows are stamped with ``prediction_kind=BACKFILL_KIND`` and a null
    ``snapshot_captured_at``, then written via ``replace_partition``
    scoped to this season's backfill partition, so forward rows for the
    same season are left untouched.
    """
    sub = model_frame.filter(pl.col("season") == season)
    if sub.is_empty():
        return
    scored = score_minutes(sub, model).with_columns(
        model_version=pl.lit(version),
        prediction_kind=pl.lit(BACKFILL_KIND),
        snapshot_captured_at=pl.lit(None, dtype=pl.Datetime("us")),
    )
    MINUTES_PREDICTION.replace_partition(
        connection,
        scored,
        equals={"season": season, "prediction_kind": BACKFILL_KIND},
    )


def backfill_minutes() -> None:
    """Score the production minutes model over all seasons and persist rows.

    Loads the ``production``-aliased model, always re-scores the current season
    (new gameweeks arrive each run), and re-scores the historic seasons only
    when the stored predictions are missing or were produced by a different
    model version — the version-gated backfill. Every stored row is stamped
    with the production model version that produced it.

    Does nothing (logs a warning) when no ``production`` alias is set, which is
    the normal state until the first manual promotion in the MLflow UI.
    """
    production = get_production_model()
    if production is None:
        return
    prod_version, model = production
    model_frame = assemble_model_frame()
    all_seasons = set(model_frame["season"].unique().to_list())
    historic_seasons = sorted(all_seasons - {CURRENT_SEASON})

    connection = get_connection()
    # TODO(JT): Can you do this with a context manager?
    try:
        # The current season is always refreshed — new gameweeks each run.
        _score_and_store(
            connection, model_frame, CURRENT_SEASON, model, prod_version
        )

        # Gate the historic backfill on the versions/seasons already stored.
        stored_versions = minutes_prediction_versions(
            connection, seasons=historic_seasons
        )
        stored_seasons = MINUTES_PREDICTION.seasons_present(connection) & set(
            historic_seasons
        )
        historic_needs_rebuild = bool(historic_seasons) and (
            stored_versions != {prod_version}
            or stored_seasons != set(historic_seasons)
        )

        if historic_needs_rebuild:
            for season in historic_seasons:
                _score_and_store(
                    connection, model_frame, season, model, prod_version
                )
            logger.info(
                f"Backfilled historic minutes predictions for {historic_seasons} at version {prod_version}",
            )
        else:
            logger.info(
                f"Historic minutes predictions already at version {prod_version}; skipping.",
            )
    finally:
        connection.close()


def warn_unidentified_snapshot(
    snapshot: pl.DataFrame, player_season: pl.DataFrame, season: str
) -> int:
    """Log a warning for snapshot elements with no ``player_season`` row.

    The FPL bootstrap typically carries more elements than ``player_season``
    resolves identities for. An unmatched element gets a null
    ``player_code``, so it drops out of the cross-season rolling stream and
    reaches the model with null history, ``is_pl_newcomer=True`` and a null
    ``age_years`` -- yet it still scores, median-imputed, and its stored
    prediction is indistinguishable from a well-supported one. Making the
    gap visible is the point; nothing is filtered out.

    Parameters
    ----------
    snapshot : pl.DataFrame
        One capture's rows, with ``element``.
    player_season : pl.DataFrame
        Identity rows with ``season``, ``element`` and ``player_code``.
    season : str
        The season being scored.

    Returns
    -------
    int
        The number of snapshot elements with no identity row.
    """
    identified = set(
        player_season.filter(
            (pl.col("season") == season) & pl.col("player_code").is_not_null()
        )["element"].to_list()
    )
    elements = snapshot["element"].to_list()
    missing = sorted({e for e in elements if e not in identified})
    if missing:
        logger.warning(
            "%d of %d %s snapshot elements have no player_season row "
            "(e.g. %s); they score with null history, is_pl_newcomer=True "
            "and a null age_years.",
            len(missing),
            len(elements),
            season,
            missing[:10],
        )
    return len(missing)


def score_forward_minutes() -> None:
    """Score the production model over every unplayed fixture and store it.

    Builds the feature frame from the latest player snapshot crossed with
    the season's remaining fixtures, scores it, and writes the rows as
    ``FORWARD_KIND``. Only gameweeks at or after the first unplayed one
    are rewritten: once a gameweek kicks off its forecast is frozen, so
    the out-of-sample record survives to be evaluated against the result.

    Does nothing when no ``production`` alias is set, or when the season
    has no snapshot or no remaining fixtures.

    Returns
    -------
    None
        Predictions are written to ``MINUTES_PREDICTION`` as a side effect.
    """
    production = get_production_model()
    if production is None:
        return
    prod_version, model = production

    connection = get_connection()
    try:
        snapshot = latest_snapshot(
            PLAYER_SNAPSHOT.load(connection), CURRENT_SEASON
        )
        if snapshot.is_empty():
            logger.warning(
                "No player snapshot for %s; skipping forward scoring.",
                CURRENT_SEASON,
            )
            return
        captured_at = snapshot["captured_at"][0]

        player_week = PLAYER_WEEK.load(connection)
        player_season = PLAYER_SEASON.load(connection)
        warn_unidentified_snapshot(snapshot, player_season, CURRENT_SEASON)
        from_gw = last_played_gw(player_week, CURRENT_SEASON) + 1
        team_name_to_id = {team.name: team.id for team in FplAPI().get_teams()}
        forward_fixtures = build_forward_fixtures(
            snapshot,
            TEAM_FIXTURE.load(connection),
            CURRENT_SEASON,
            from_gw,
            team_name_to_id,
        )
        if forward_fixtures.is_empty():
            logger.info(
                "No unplayed %s fixtures from gw %d; nothing to score.",
                CURRENT_SEASON,
                from_gw,
            )
            return

        combined_weeks = pl.concat(
            [player_week, forward_player_weeks(forward_fixtures)],
            how="diagonal",
        )
        feature_frame = build_feature_frame(
            combined_weeks,
            PLAYER_AVAILABILITY.load(connection),
            PLAYER_MATCH.load(connection),
            player_season,
            TEAM_FIXTURE.load(connection),
            forward_fixtures=forward_fixtures,
        )
        model_frame = forward_fixtures.select(
            ["season", "gw", "element", "opponent"]
        ).join(feature_frame, on=["season", "gw", "element"], how="inner")

        scored = score_minutes(model_frame, model).with_columns(
            model_version=pl.lit(prod_version),
            prediction_kind=pl.lit(FORWARD_KIND),
            snapshot_captured_at=pl.lit(captured_at, dtype=pl.Datetime("us")),
        )
        MINUTES_PREDICTION.replace_partition(
            connection,
            scored,
            equals={
                "season": CURRENT_SEASON,
                "prediction_kind": FORWARD_KIND,
            },
            gw_from=from_gw,
        )
        logger.info(
            "Stored %d forward minutes rows for %s from gw %d at version %s",
            scored.height,
            CURRENT_SEASON,
            from_gw,
            prod_version,
        )
    finally:
        connection.close()
