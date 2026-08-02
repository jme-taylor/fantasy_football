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
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

import mlflow
import mlflow.sklearn
import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from fantasy_football.constants import (
    CURRENT_SEASON,
    EXPERIMENT_BY_POSITION,
    MLFLOW_TRACKING_URI,
    PRECISION_K_BY_POSITION,
)
from fantasy_football.extraction.fpl import FplAPI
from fantasy_football.features.match_form import (
    covered_seasons,
    rolling_identity_sql,
)
from fantasy_football.features.views import register_feature_views
from fantasy_football.modelling.folds import gameweek_folds
from fantasy_football.modelling.forward import (
    build_forward_fixtures,
    last_played_gw,
    latest_snapshot,
)
from fantasy_football.modelling.metrics import (
    mae,
    precision_at_k,
    rmse,
    skill_score,
    spearman_by_gw,
)
from fantasy_football.modelling.registry import load_production_model
from fantasy_football.storage.database import get_connection
from fantasy_football.storage.tables import (
    BACKFILL_KIND,
    FORWARD_KIND,
    MINUTES_PREDICTION,
    PLAYER_SNAPSHOT,
    PLAYER_WEEK,
    POINTS_PREDICTION,
    TEAM_FIXTURE,
    points_prediction_versions,
)

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
# The two player-form columns that reset each season. Everything else in
# _PLAYER_FORM_COLUMNS is a rolling rate that deliberately spans seasons.
_PLAYER_SEASON_TO_DATE_COLUMNS = [
    "yellow_cards_season_to_date",
    "red_cards_season_to_date",
]
_PLAYER_ROLLING_COLUMNS = [
    column
    for column in _PLAYER_FORM_COLUMNS
    if column not in _PLAYER_SEASON_TO_DATE_COLUMNS
]
_OWN_TEAM_COLUMNS = [
    "xg_against_rolling_5",
    "goals_against_rolling_5",
    "clean_sheet_rolling_5",
]
_OPPOSITION_COLUMNS = ["xg_for_rolling_5", "goals_for_rolling_5"]
_MINUTES_COLUMNS = [
    "expected_minutes",
    "p_zero",
    "p_partial",
    "p_sixty_plus",
]

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
-- prediction_kind is part of the minutes primary key, so a fixture that
-- was forward-scored before it was played and backfilled afterwards
-- carries both kinds. Joining unfiltered would fan the training row out.
-- Backfill is the right kind here: see the train/serve skew TODO above.
LEFT JOIN minutes_prediction AS mn
    ON  m.season   = mn.season
    AND m.gw       = mn.gw
    AND m.element  = mn.element
    AND m.opponent = mn.opponent
    AND mn.prediction_kind = '{BACKFILL_KIND}'
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


def fold_metrics(
    test_df: pl.DataFrame, predicted: Sequence[float]
) -> dict[str, float]:
    """Score one fold's predictions against its actuals.

    The baseline for the skill score is the mean target over the fold,
    i.e. "predict the average defender every time". Ranking metrics
    matter more than MAE here: the optimiser only needs the order of
    defenders to be right.

    Parameters
    ----------
    test_df : pl.DataFrame
        The fold's held-out rows, carrying ``KEY_COLUMNS`` and ``TARGET``.
    predicted : Sequence[float]
        Predictions aligned to ``test_df`` row order.

    Returns
    -------
    dict[str, float]
        ``mae``, ``rmse``, ``skill_score``, ``spearman`` and
        ``precision_at_k``.
    """
    actual = test_df[TARGET].to_list()
    baseline = [float(np.mean(actual))] * len(actual)
    ranked = test_df.select(
        pl.col("gw"),
        pl.col("element").alias("player_id"),
        pl.Series("predicted_points", predicted),
        pl.col(TARGET).alias("actual"),
    )
    return {
        "mae": mae(predicted, actual),
        "rmse": rmse(predicted, actual),
        "skill_score": skill_score(predicted, actual, baseline),
        "spearman": spearman_by_gw(ranked),
        "precision_at_k": precision_at_k(
            ranked, k=PRECISION_K_BY_POSITION[POSITION]
        ),
    }


def covered_folds(
    keys: "Iterable[tuple[str, int]]", min_train_gws: int = MIN_TRAIN_GWS
) -> list[tuple[list[tuple[str, int]], tuple[str, int]]]:
    """Expanding-window folds whose test gameweek has every feature.

    Eleven of the twenty features derive from FCI's Opta stats, which
    only exist from the season :func:`covered_seasons` reports. Training
    deliberately spans every season anyway (see the scoring-regime TODO
    above), but a fold *testing* an earlier gameweek is scoring a model
    the imputer has silently stripped down: with those columns entirely
    null in the training slice, ``SimpleImputer`` drops them and the
    forest fits on the nine that remain. Such folds measure a different
    model than the one being registered, and averaging them into the
    aggregate metrics buries the signal the promotion decision needs.

    Only the *test* side is restricted. A covered fold still trains on
    every earlier gameweek, uncovered seasons included, which is what
    the deployed model does.

    Parameters
    ----------
    keys : Iterable[tuple[str, int]]
        ``(season, gw)`` pairs present in the model frame.
    min_train_gws : int, optional
        Minimum gameweeks in the training side of the first fold.

    Returns
    -------
    list[tuple[list[tuple[str, int]], tuple[str, int]]]
        ``(train_keys, test_key)`` pairs. Empty when no gameweek in
        ``keys`` falls in a covered season.
    """
    covered = set(covered_seasons())
    every = gameweek_folds(keys, min_train_gws=min_train_gws)
    kept = [(train, test) for train, test in every if test[0] in covered]
    if len(kept) != len(every):
        logger.info(
            "Cross-validating on %d of %d gameweek folds; dropped %d whose "
            "test gameweek predates full feature coverage (%s).",
            len(kept),
            len(every),
            len(every) - len(kept),
            ", ".join(sorted(covered)),
        )
    return kept


def cross_validate(
    model_df: pl.DataFrame,
    folds: list[tuple[list[tuple[str, int]], tuple[str, int]]],
) -> tuple[list[dict[str, float]], dict[str, float]]:
    """Score the model over expanding-window gameweek folds.

    Parameters
    ----------
    model_df : pl.DataFrame
        Output of :func:`build_model_frame`.
    folds : list
        ``(train_keys, test_key)`` pairs from
        :func:`fantasy_football.modelling.folds.gameweek_folds`.

    Returns
    -------
    tuple[list[dict], dict]
        Per-fold metric dicts, and an aggregate dict with
        ``{metric}_mean`` and ``{metric}_std`` for every metric present
        in all folds. Both are empty when no fold has data on both sides.
    """
    per_fold: list[dict[str, float]] = []
    keys = pl.struct(["season", "gw"])
    for train_keys, test_key in folds:
        train_set = [{"season": s, "gw": g} for s, g in train_keys]
        train_df = model_df.filter(keys.is_in(train_set))
        test_df = model_df.filter(
            (pl.col("season") == test_key[0]) & (pl.col("gw") == test_key[1])
        )
        if train_df.is_empty() or test_df.is_empty():
            continue
        pipe = make_pipeline()
        pipe.fit(
            train_df.select(FEATURES).to_pandas(), train_df[TARGET].to_list()
        )
        predicted = pipe.predict(test_df.select(FEATURES).to_pandas())
        per_fold.append(fold_metrics(test_df, list(predicted)))

    agg: dict[str, float] = {}
    if per_fold:
        shared = set(per_fold[0])
        for metrics in per_fold[1:]:
            shared &= set(metrics)
        for key in sorted(shared):
            values = [metrics[key] for metrics in per_fold]
            n_nan = sum(1 for value in values if np.isnan(value))
            if n_nan:
                logger.warning(
                    "%d of %d folds had a nan %s; the aggregate is "
                    "degenerate.",
                    n_nan,
                    len(values),
                    key,
                )
            agg[f"{key}_mean"] = float(np.mean(values))
            agg[f"{key}_std"] = float(np.std(values))
    return per_fold, agg


def train_final(model_df: pl.DataFrame) -> Pipeline:
    """Fit the pipeline on every row in ``model_df``.

    Parameters
    ----------
    model_df : pl.DataFrame
        The full model frame.

    Returns
    -------
    sklearn.pipeline.Pipeline
        The fitted pipeline.
    """
    pipe = make_pipeline()
    pipe.fit(model_df.select(FEATURES).to_pandas(), model_df[TARGET].to_list())
    return pipe


def run_defender_model() -> dict[str, float]:
    """Train, cross-validate and log the defender model to MLflow.

    Assembles the model frame, scores it with expanding-window gameweek
    folds, fits the final pipeline on every row, and logs params,
    per-fold metrics (stepped), aggregate mean/std metrics and the fitted
    model. Every run registers a new version; which version is live is a
    manual alias move in the MLflow UI.

    Returns
    -------
    dict[str, float]
        The aggregate metrics, or an empty dict when there is too little
        data to form a fold.
    """
    connection = get_connection()
    try:
        model_df = build_model_frame(connection)
    finally:
        connection.close()

    folds = covered_folds(
        list(zip(model_df["season"], model_df["gw"])),
        min_train_gws=MIN_TRAIN_GWS,
    )
    per_fold, agg = cross_validate(model_df, folds)
    final_model = train_final(model_df)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_BY_POSITION[POSITION])
    with mlflow.start_run():
        mlflow.log_params(
            {
                "model": "random_forest",
                "n_estimators": N_ESTIMATORS,
                "random_state": RANDOM_STATE,
                "cv": "expanding_window_by_gameweek",
                "min_train_gws": MIN_TRAIN_GWS,
                "n_folds": len(folds),
                "n_samples": model_df.height,
                "n_features": len(FEATURES),
            }
        )
        for step, metrics in enumerate(per_fold):
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=step)
        mlflow.log_metrics(agg)
        # TODO (JT): Promote automatically once the metrics justify it.
        # A gate must re-score the incumbent on today's folds rather than
        # compare against its stored number -- the incumbent's metric was
        # computed over fewer folds, so a direct comparison would read as
        # an improvement or regression for reasons unrelated to the model.
        # Deferred until a few weeks of retrains show how much the
        # metrics wobble. Mirrors the same TODO in minutes.py.
        mlflow.sklearn.log_model(
            final_model,
            name="model",
            registered_model_name=REGISTERED_MODEL,
        )

    logger.info(
        "Defender model registered under %s over %d folds",
        REGISTERED_MODEL,
        len(folds),
    )
    return agg


def get_production_model() -> tuple[str, Any] | None:
    """Return the production-aliased version string and model, or None.

    Returns
    -------
    tuple[str, Any] | None
        The aliased model version and loaded model, or ``None`` when no
        alias is set -- the normal state until the first manual
        promotion in the MLflow UI.
    """
    return load_production_model(REGISTERED_MODEL, PRODUCTION_ALIAS)


def score_defender_points(
    frame: pl.DataFrame, model: Pipeline, version: str, kind: str
) -> pl.DataFrame:
    """Score a fitted model over ``frame`` and shape rows for storage.

    Parameters
    ----------
    frame : pl.DataFrame
        Rows carrying ``KEY_COLUMNS`` and every column in ``FEATURES``.
    model : sklearn.pipeline.Pipeline
        A fitted pipeline exposing ``predict``.
    version : str
        The registry version that produced the predictions.
    kind : str
        ``BACKFILL_KIND`` or ``FORWARD_KIND``.

    Returns
    -------
    pl.DataFrame
        One row per input row, in ``POINTS_PREDICTION`` column order.
    """
    predicted = model.predict(frame.select(FEATURES).to_pandas())
    return (
        frame.select(KEY_COLUMNS)
        .with_columns(
            position=pl.lit(POSITION),
            predicted_points=pl.Series(predicted).cast(pl.Float64),
            model_version=pl.lit(version),
            prediction_kind=pl.lit(kind),
        )
        .select(POINTS_PREDICTION.columns)
    )


def _score_and_store(
    connection: "DuckDBPyConnection",
    model_frame: pl.DataFrame,
    season: str,
    model: Pipeline,
    version: str,
) -> None:
    """Score one season's rows and store them as backfill predictions.

    Written via ``replace_partition`` scoped to this season's backfill
    partition, so forward rows for the same season are left untouched.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        An open connection.
    model_frame : pl.DataFrame
        The full model frame, filtered to ``season`` internally.
    season : str
        The season to score.
    model : sklearn.pipeline.Pipeline
        The production model.
    version : str
        The registry version that produced it.
    """
    sub = model_frame.filter(pl.col("season") == season)
    if sub.is_empty():
        return
    scored = score_defender_points(sub, model, version, BACKFILL_KIND)
    POINTS_PREDICTION.replace_partition(
        connection,
        scored,
        equals={"season": season, "prediction_kind": BACKFILL_KIND},
    )


def backfill_defender_points() -> None:
    """Score the production model over all seasons and persist the rows.

    Always re-scores the current season, since new gameweeks arrive each
    run, and re-scores historic seasons only when the stored predictions
    are missing or came from a different model version.

    These rows are not what the optimiser consumes -- that is the
    forward path. They exist so predictions can be read next to actuals
    when deciding whether to promote a newly trained version, which
    matters because promotion is manual and ``run_evaluation`` does not
    yet exercise this model.

    Does nothing (logs a warning) when no production alias is set.
    """
    production = get_production_model()
    if production is None:
        return
    prod_version, model = production

    connection = get_connection()
    try:
        model_frame = build_model_frame(connection)
        all_seasons = set(model_frame["season"].unique().to_list())
        historic = sorted(all_seasons - {CURRENT_SEASON})

        _score_and_store(
            connection, model_frame, CURRENT_SEASON, model, prod_version
        )

        # Nothing to gate when there are no historic seasons at all --
        # e.g. a fresh database holding only the current season. The
        # shared version-lookup helpers handle an empty ``seasons`` list
        # safely (see storage/tables.py); skipping here just avoids
        # logging a vacuous "backfilled the empty list" line.
        needs_rebuild = False
        if historic:
            historic_set = set(historic)
            stored_versions = points_prediction_versions(
                connection, seasons=historic
            )
            stored_seasons = (
                POINTS_PREDICTION.seasons_present(connection) & historic_set
            )
            needs_rebuild = (
                stored_versions != {prod_version}
                or stored_seasons != historic_set
            )
        if needs_rebuild:
            for season in historic:
                _score_and_store(
                    connection, model_frame, season, model, prod_version
                )
            logger.info(
                "Backfilled historic defender points for %s at version %s",
                historic,
                prod_version,
            )
        else:
            logger.info(
                "Historic defender points already at version %s; skipping.",
                prod_version,
            )
    finally:
        connection.close()


def _asof_form(
    left: pl.DataFrame,
    right: pl.DataFrame,
    by: list[str],
    columns: list[str],
) -> pl.DataFrame:
    """Attach the most recent form row at or before each fixture.

    Both sides are sorted on ``kickoff_time`` here rather than by the
    caller, and polars' own sortedness check is turned off because it
    cannot verify sortedness once ``by`` groups are given -- it would
    only emit a warning, never a guarantee. A global sort on the as-of
    key implies a sorted order within every ``by`` group, so grouping is
    safe; the guarantee comes from the ``sort`` calls two lines above the
    join, not from the check.

    Parameters
    ----------
    left : pl.DataFrame
        Forward fixtures, carrying ``kickoff_time`` and every column in
        ``by``.
    right : pl.DataFrame
        A form frame carrying ``kickoff_time``, ``by`` and ``columns``.
    by : list[str]
        Columns matched exactly before the as-of comparison. They must
        identify one subject -- a player or a club -- or a fixture would
        inherit a stranger's form.
    columns : list[str]
        Form columns to bring across.

    Returns
    -------
    pl.DataFrame
        ``left``, ordered by ``kickoff_time``, with the form columns
        attached. Exactly one row per input row: an as-of join takes at
        most one match, so a duplicated right-hand row cannot fan the
        output out -- it only changes which value arrives.
    """
    absent = [pl.lit(None, dtype=pl.Float64).alias(name) for name in columns]
    if right.is_empty():
        return left.with_columns(absent)
    slimmed = (
        right.select(by + ["kickoff_time"] + columns)
        .with_columns(pl.col("kickoff_time").cast(pl.Datetime("us")))
        # A form row with no kickoff has no position in time, so it can
        # neither be ordered nor as-of matched. team_form warns about
        # these already; dropping them here keeps the join well defined.
        .filter(pl.col("kickoff_time").is_not_null())
        .sort("kickoff_time")
    )
    if slimmed.is_empty():
        return left.with_columns(absent)
    return (
        left.with_columns(pl.col("kickoff_time").cast(pl.Datetime("us")))
        .sort("kickoff_time")
        .join_asof(
            slimmed,
            on="kickoff_time",
            by=by,
            strategy="backward",
            check_sortedness=False,
        )
    )


def _attach_rolling_identity(
    connection: "DuckDBPyConnection", frame: pl.DataFrame
) -> pl.DataFrame:
    """Add the form views' ``rolling_identity`` to forward fixture rows.

    Resolved through ``player_season`` by the same expression the view
    uses, rather than reimplemented here, so serve-time identity cannot
    drift from the identity the rolling window partitions by. A player
    with no ``player_season`` row falls back to the view's own
    ``(season, element)`` form, which the LEFT JOIN reaches because the
    fallback branch reads its season and element from the fixture side.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        An open connection holding ``player_season``.
    frame : pl.DataFrame
        Forward fixture rows carrying ``season`` and ``element``.

    Returns
    -------
    pl.DataFrame
        ``frame`` with a non-null ``rolling_identity`` string column.
        ``player_season`` is keyed on ``(season, element)``, so the join
        cannot add rows.
    """
    keys = frame.select("season", "element").unique()
    connection.register("_forward_identity_keys", keys)
    identities = connection.sql(
        "SELECT k.season, k.element, "
        f"{rolling_identity_sql('ps', 'k')} AS rolling_identity "
        "FROM _forward_identity_keys AS k "
        "LEFT JOIN player_season AS ps "
        "ON ps.season = k.season AND ps.element = k.element"
    ).pl()
    connection.unregister("_forward_identity_keys")
    return frame.join(identities, on=["season", "element"], how="left")


def build_forward_feature_frame(
    connection: "DuckDBPyConnection",
    forward_fixtures: pl.DataFrame,
) -> pl.DataFrame:
    """Attach model features to fixtures that have not been played.

    The inclusive form views are used here, not the exclusive ones the
    training frame reads. At a player's most recent appearance the
    exclusive window has already dropped that appearance, so as-of
    joining to it would make every live prediction one match stale --
    and it is the most informative match. See
    :func:`fantasy_football.features.team_form.window_frame`.

    Player form is as-of joined on ``rolling_identity`` -- the key the
    form view itself windows on -- so serve matches train: a player whose
    most recent appearance was last season carries it forward, exactly as
    his first training row of a season does. The two ``*_season_to_date``
    columns are the exception and are matched within the season, because
    they reset; see :func:`_attach_rolling_identity`.

    ``is_home`` and the opponent's *name* both come from ``team_fixture``,
    keyed on ``(season, gw, team, kickoff_time)``, which identifies one
    fixture leg. The opponent's name is what the form views are
    partitioned by, and it is deliberately not recovered from
    ``fpl_team_id``: FPL reissues team ids each season, so a club's id in
    one season belongs to a different club in another, and that view has
    no rows at all for a season before its first match is played.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        An open connection with the feature views registered.
    forward_fixtures : pl.DataFrame
        Defender rows from
        :func:`fantasy_football.modelling.forward.build_forward_fixtures`,
        carrying ``season``, ``gw``, ``element``, ``opponent``,
        ``kickoff_time`` and ``team``.

    Returns
    -------
    pl.DataFrame
        One row per input fixture, with ``KEY_COLUMNS`` and ``FEATURES``.
        Ordered by ``kickoff_time`` rather than by input order.
    """
    player_form = connection.sql(
        "SELECT * FROM player_match_form_inclusive"
    ).pl()
    team_form = connection.sql("SELECT * FROM team_match_form_inclusive").pl()
    fixtures = TEAM_FIXTURE.load(connection).select(
        "season",
        "gw",
        "team",
        "kickoff_time",
        "is_home",
        pl.col("opposition").alias("opponent_name"),
    )
    # prediction_kind is part of the minutes primary key, so both kinds
    # can sit on one fixture; joining unfiltered would fan the row out.
    minutes = MINUTES_PREDICTION.load(connection).filter(
        pl.col("prediction_kind") == FORWARD_KIND
    )

    frame = forward_fixtures.join(
        fixtures,
        on=["season", "gw", "team", "kickoff_time"],
        how="left",
    ).join(
        minutes.select(KEY_COLUMNS + _MINUTES_COLUMNS),
        on=KEY_COLUMNS,
        how="left",
    )

    # rolling_identity, not (season, element). The form view windows on
    # rolling_identity with no season term, so in training a player's
    # first row of a season carries the tail of his previous one; keying
    # the as-of join on season would match nothing until his first
    # appearance and leave every player-form feature null at exactly the
    # pre-season squad build this model exists to serve. player_code is
    # globally stable, so it still keeps a reissued element id from
    # inheriting the previous holder's form -- which is what the season
    # term was really protecting.
    frame = _attach_rolling_identity(connection, frame)
    frame = _asof_form(
        frame, player_form, ["rolling_identity"], _PLAYER_ROLLING_COLUMNS
    )
    # The season-to-date counts do reset, so they are matched inside the
    # season and default to 0 -- the same value training's coalesce gives
    # a season's first appearance.
    frame = _asof_form(
        frame,
        player_form,
        ["rolling_identity", "season"],
        _PLAYER_SEASON_TO_DATE_COLUMNS,
    ).with_columns(
        [
            pl.col(column).fill_null(0.0)
            for column in _PLAYER_SEASON_TO_DATE_COLUMNS
        ]
    )
    # Club form deliberately is not season-scoped. The form views
    # partition by club across seasons, so a club's first fixture of a
    # season inherits the tail of its previous one.
    frame = _asof_form(frame, team_form, ["team"], _OWN_TEAM_COLUMNS)
    opposition = team_form.select(
        pl.col("team").alias("opponent_name"),
        "kickoff_time",
        *_OPPOSITION_COLUMNS,
    )
    frame = _asof_form(
        frame, opposition, ["opponent_name"], _OPPOSITION_COLUMNS
    )

    frame = frame.with_columns(pl.col("is_home").cast(pl.Float64))
    missing = [name for name in FEATURES if name not in frame.columns]
    if missing:
        frame = frame.with_columns(
            [pl.lit(None, dtype=pl.Float64).alias(name) for name in missing]
        )
        logger.warning(
            "Forward defender features absent from the join: %s", missing
        )
    return frame.select(KEY_COLUMNS + FEATURES)


def score_forward_defender_points() -> None:
    """Score the production model over every unplayed fixture.

    Builds fixture rows from the latest player snapshot crossed with the
    season's remaining fixtures, filters to defenders, attaches features
    from the inclusive form views and the forward minutes forecasts, and
    stores the result as ``FORWARD_KIND``. These are the rows the
    optimiser consumes. Only gameweeks at or after the first unplayed one
    are rewritten, so a forecast for a gameweek that has kicked off stays
    frozen and can still be scored against the result.

    Does nothing when no production alias is set, or when the season has
    no snapshot or no remaining fixtures.
    """
    production = get_production_model()
    if production is None:
        return
    prod_version, model = production

    connection = get_connection()
    try:
        register_feature_views(connection)
        snapshot = latest_snapshot(
            PLAYER_SNAPSHOT.load(connection), CURRENT_SEASON
        )
        if snapshot.is_empty():
            logger.warning(
                "No player snapshot for %s; skipping defender forward "
                "scoring.",
                CURRENT_SEASON,
            )
            return

        from_gw = (
            last_played_gw(PLAYER_WEEK.load(connection), CURRENT_SEASON) + 1
        )
        team_name_to_id = {team.name: team.id for team in FplAPI().get_teams()}
        forward_fixtures = build_forward_fixtures(
            snapshot,
            TEAM_FIXTURE.load(connection),
            CURRENT_SEASON,
            from_gw,
            team_name_to_id,
        ).filter(pl.col("position") == POSITION)
        if forward_fixtures.is_empty():
            logger.info(
                "No unplayed %s defender fixtures from gw %d.",
                CURRENT_SEASON,
                from_gw,
            )
            return

        frame = build_forward_feature_frame(connection, forward_fixtures)
        if frame.height != forward_fixtures.height:
            # Every join in build_forward_feature_frame is a left join or
            # an as-of, which takes at most one match, so the height can
            # only grow: a fixture has been duplicated. That means two
            # rows sharing (season, gw, element, opponent, forward) and
            # the insert below is about to violate points_prediction's
            # primary key.
            logger.warning(
                "Forward feature join duplicated rows: %d fixtures became "
                "%d. Duplicate keys will violate the points_prediction "
                "primary key on insert.",
                forward_fixtures.height,
                frame.height,
            )

        scored = score_defender_points(
            frame, model, prod_version, FORWARD_KIND
        )
        POINTS_PREDICTION.replace_partition(
            connection,
            scored,
            equals={
                "season": CURRENT_SEASON,
                "prediction_kind": FORWARD_KIND,
            },
            gw_from=from_gw,
        )
        logger.info(
            "Stored %d forward defender rows for %s from gw %d at "
            "version %s",
            scored.height,
            CURRENT_SEASON,
            from_gw,
            prod_version,
        )
    finally:
        connection.close()
