"""Behaviour every per-position points model shares.

These tests run once per position, against the real subclass, so a
regression in :mod:`fantasy_football.modelling.points` cannot be fixed
for one position and left broken for another. Assertions that depend on
which columns a position reads live in ``test_defender.py`` and
``test_forwards.py`` instead.
"""

import json
from datetime import timedelta
from types import SimpleNamespace

import polars as pl
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

from fantasy_football.features.views import register_feature_views
from fantasy_football.modelling.components import (
    POSITION_COMPONENTS,
    Component,
    compose,
)
from fantasy_football.modelling.defender import (
    DEFENDER_SPEC,
    DefenderPointsPredictor,
)
from fantasy_football.modelling.folds import (
    ExpandingGameweekFoldStrategy,
    TrainTestSplitStrategy,
)
from fantasy_football.modelling.forwards import (
    FORWARD_SPEC,
    ForwardPointsPredictor,
)
from fantasy_football.modelling.goalkeeper import (
    GOALKEEPER_SPEC,
    GoalkeeperPointsPredictor,
)
from fantasy_football.modelling.midfielder import (
    MIDFIELDER_SPEC,
    MidfielderPointsPredictor,
)
from fantasy_football.modelling.points import (
    KEY_COLUMNS,
    TARGET,
    PositionPointsPredictor,
)
from fantasy_football.storage.tables import (
    BACKFILL_KIND,
    FORWARD_KIND,
    MINUTES_PREDICTION,
    PLAYER_MATCH,
    PLAYER_SEASON,
    PLAYER_SNAPSHOT,
    POINTS_COMPONENT,
    POINTS_PREDICTION,
    TEAM_FIXTURE,
    TEST_POINTS_PREDICTION,
)
from tests.unit.modelling.conftest import (
    ARSENAL,
    CHELSEA,
    GW1_KICKOFF,
    GW2_KICKOFF,
    GW3_KICKOFF,
    SEASON,
    SPURS,
    TEAM_IDS,
    UNITED,
    ConstantPointsModel,
    append_rows,
)

PRIOR_SEASON = "2024-25"

SPECS = {
    DefenderPointsPredictor: DEFENDER_SPEC,
    GoalkeeperPointsPredictor: GOALKEEPER_SPEC,
    ForwardPointsPredictor: FORWARD_SPEC,
    MidfielderPointsPredictor: MIDFIELDER_SPEC,
}


def _probe(predictor: PositionPointsPredictor) -> str:
    """Return a cross-season player-form column this position reads.

    The tests below need one form column to trace through the as-of
    join, and no single column serves every position -- a goalkeeper
    reads none of the attacking rates the outfield models do. Taking the
    first cross-season rolling column keeps each position's assertions on
    a column it actually carries.
    """
    return predictor.player_rolling_columns[0]


@pytest.fixture(params=list(SPECS), ids=lambda cls: cls.POSITION)
def predictor(request, connection):
    """Return each position's predictor in turn, bound to the test db."""
    cls = request.param
    return cls(
        experiment_name=f"test-{cls.POSITION}",
        params={},
        model_spec=SPECS[cls],
        connection=connection,
        fold_strategy=ExpandingGameweekFoldStrategy(),
    )


# --- Frame shape -----------------------------------------------------


def test_features_exclude_keys_and_target(predictor) -> None:
    """Neither the target nor a key column leaks into FEATURES."""
    assert TARGET not in predictor.FEATURES
    for key in KEY_COLUMNS:
        assert key not in predictor.FEATURES


def test_model_frame_sql_filters_to_position_and_played_matches(
    predictor,
) -> None:
    """The SQL restricts to this position and rows with recorded minutes."""
    sql = predictor.model_frame_sql()
    assert f"s.position = '{predictor.POSITION}'" in sql
    assert "m.minutes IS NOT NULL" in sql


def test_build_training_data_returns_keys_target_and_features(
    predictor,
) -> None:
    """The frame's columns are keys, then target, then features."""
    frame = predictor.build_training_data()
    assert frame.columns == KEY_COLUMNS + [TARGET] + predictor.FEATURES


def test_model_frame_selects_every_declared_minutes_column(
    predictor, connection, seed_model_frame
) -> None:
    """Each position's own minutes columns reach the training frame.

    The defender model reads the bucket probabilities as well as
    expected_minutes; the forwards model deliberately takes only
    expected_minutes. Generating the SELECT from MINUTES_COLUMNS is what
    keeps those apart, so this pins that the list is actually honoured.
    """
    seed_model_frame(connection, predictor.POSITION)
    append_rows(
        MINUTES_PREDICTION,
        connection,
        [
            {
                "season": SEASON,
                "gw": 2,
                "element": 1,
                "opponent": TEAM_IDS[ARSENAL],
                "p_zero": 0.1,
                "p_partial": 0.2,
                "p_sixty_plus": 0.7,
                "expected_minutes": 11.0,
                "model_version": "1",
                "prediction_kind": BACKFILL_KIND,
            }
        ],
    )

    row = predictor.build_training_data().filter(pl.col("gw") == 2)

    for column in predictor.MINUTES_COLUMNS:
        assert column in row.columns
        assert row[column].item() is not None


def test_model_frame_takes_only_backfill_minutes_predictions(
    predictor, connection, seed_model_frame
) -> None:
    """Both minutes kinds on one fixture still give one training row.

    ``prediction_kind`` is part of the minutes primary key, so a fixture
    that has been forward-scored and then backfilled carries two rows.
    Joining without the filter fans the training row out -- duplicating
    it in training and CV, and violating the points_prediction primary
    key on backfill insert.
    """
    seed_model_frame(connection, predictor.POSITION)
    for kind, expected in ((BACKFILL_KIND, 11.0), (FORWARD_KIND, 77.0)):
        append_rows(
            MINUTES_PREDICTION,
            connection,
            [
                {
                    "season": SEASON,
                    "gw": 2,
                    "element": 1,
                    "opponent": TEAM_IDS[ARSENAL],
                    "p_zero": 0.1,
                    "p_partial": 0.2,
                    "p_sixty_plus": 0.7,
                    "expected_minutes": expected,
                    "model_version": "1",
                    "prediction_kind": kind,
                }
            ],
        )

    row = predictor.build_training_data().filter(pl.col("gw") == 2)

    assert row.height == 1
    assert row["expected_minutes"].item() == pytest.approx(11.0)


# --- Training-season restriction -------------------------------------


def _seed_prior_season_match(connection, position: str) -> None:
    """Add one played fixture in the season before ``SEASON``."""
    append_rows(
        PLAYER_SEASON,
        connection,
        [{"season": PRIOR_SEASON, "element": 1, "position": position}],
    )
    append_rows(
        PLAYER_MATCH,
        connection,
        [
            {
                "season": PRIOR_SEASON,
                "gw": 38,
                "element": 1,
                "opponent": TEAM_IDS[ARSENAL],
                "is_home": True,
                "minutes": 90,
                "kickoff_time": GW1_KICKOFF - timedelta(days=90),
            }
        ],
    )


@pytest.mark.parametrize(
    "cls",
    [cls for cls in SPECS if cls.TRAINING_SEASONS is None],
    ids=lambda cls: cls.POSITION,
)
def test_training_data_spans_every_season_by_default(
    cls, connection, seed_model_frame
) -> None:
    """An unrestricted position trains on every season it has rows for.

    The three outfield models rely on this: their features go back as far
    as the data does, and narrowing them would invalidate models already
    registered against the wider frame. Parametrised over the positions
    that leave the restriction unset rather than skipping the one that
    does, so the goalkeeper never appears here as a passing case; what it
    does instead is in ``test_goalkeeper.py``.
    """
    predictor = cls(
        experiment_name=f"test-{cls.POSITION}",
        params={},
        model_spec=SPECS[cls],
        connection=connection,
        fold_strategy=ExpandingGameweekFoldStrategy(),
    )
    seed_model_frame(connection, cls.POSITION)
    _seed_prior_season_match(connection, cls.POSITION)

    seasons = set(predictor.build_training_data()["season"].to_list())

    assert seasons == {PRIOR_SEASON, SEASON}


def test_training_seasons_drops_rows_outside_the_window(
    connection, seed_model_frame
) -> None:
    """A position that sets TRAINING_SEASONS sees only those seasons.

    Without this the restriction can silently do nothing: the frame would
    look right, and the only symptom would be a model quietly fitted on
    median-imputed values for features that did not exist yet.
    """

    class Restricted(DefenderPointsPredictor):
        TRAINING_SEASONS = (SEASON,)

    predictor = Restricted(
        experiment_name="test-restricted",
        params={},
        model_spec=DEFENDER_SPEC,
        connection=connection,
        fold_strategy=ExpandingGameweekFoldStrategy(),
    )
    seed_model_frame(connection, Restricted.POSITION)
    _seed_prior_season_match(connection, Restricted.POSITION)

    seasons = set(predictor.build_training_data()["season"].to_list())

    assert seasons == {SEASON}


def test_empty_training_seasons_names_the_cause(connection) -> None:
    """A collapsed window fails with a message, not a parser error.

    The window is derived by intersecting coverage maps, so a stat whose
    seasons do not overlap the rest empties it. Emitting ``IN ()`` would
    surface as a DuckDB parser error naming neither the position nor the
    stat lists behind it.
    """

    class Collapsed(DefenderPointsPredictor):
        TRAINING_SEASONS = ()

    predictor = Collapsed(
        experiment_name="test-collapsed",
        params={},
        model_spec=DEFENDER_SPEC,
        connection=connection,
        fold_strategy=ExpandingGameweekFoldStrategy(),
    )

    with pytest.raises(ValueError, match="empty tuple"):
        predictor.model_frame_sql()


# --- Pipeline and metrics --------------------------------------------


def test_pipeline_imputes_and_has_no_scaler(predictor) -> None:
    """The pipeline median-imputes into a random forest, with no scaler."""
    steps = dict(predictor.make_pipeline().named_steps)
    assert isinstance(steps["impute"], SimpleImputer)
    assert steps["impute"].strategy == "median"
    assert isinstance(steps["model"], RandomForestRegressor)
    assert not any("scale" in name for name in steps)


def test_pipeline_fits_and_predicts_with_nulls(predictor) -> None:
    """The pipeline fits and predicts on a frame containing null features."""
    pipe = predictor.make_pipeline()
    x = pl.DataFrame(
        {name: [1.0, None, 3.0] for name in predictor.FEATURES}
    ).to_pandas()
    pipe.fit(x, [1.0, 2.0, 3.0])
    assert len(pipe.predict(x)) == 3


def test_fold_metrics_reports_every_expected_metric(
    predictor, synthetic_frame
) -> None:
    """fold_metrics returns exactly the five expected metric keys."""
    test_df = synthetic_frame(predictor, n_gws=1)
    metrics = predictor.fold_metrics(test_df, [1.0] * test_df.height)
    assert set(metrics.as_dict()) == {
        "mae",
        "rmse",
        "skill_score",
        "spearman",
        "precision_at_k",
    }


def test_fold_metrics_perfect_predictions_are_zero_error(
    predictor, synthetic_frame
) -> None:
    """Exact predictions give mae == 0.0 and rmse == 0.0, not swapped."""
    test_df = synthetic_frame(predictor, n_gws=1)
    metrics = predictor.fold_metrics(
        test_df, test_df[TARGET].to_list()
    ).as_dict()
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0


def test_fold_metrics_constant_mean_prediction_has_zero_skill(
    predictor, synthetic_frame
) -> None:
    """Predicting the fold mean everywhere scores skill_score == 0.0.

    The skill-score baseline *is* the fold mean, so a constant-mean
    prediction has identical MAE to the baseline: 1 - mae/mae == 0.0.

    This fixture's mae (1.5) and rmse (~1.77951) differ from each other,
    so pinning both catches an "mae"/"rmse" key swap in fold_metrics --
    a swap is invisible to skill_score alone, since skill_score reads
    predicted/actual directly rather than the dict's mae/rmse entries.
    Derivation: targets are element % 7 for element in 1..12, i.e.
    [1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5], mean 3.0. Absolute deviations
    from 3.0 are [2, 1, 0, 1, 2, 3, 3, 2, 1, 0, 1, 2]; their mean is
    18 / 12 = 1.5 (mae) and the root-mean-square is
    sqrt(38 / 12) = 1.7795130420052185 (rmse).
    """
    test_df = synthetic_frame(predictor, n_gws=1)
    mean_target = sum(test_df[TARGET].to_list()) / test_df.height
    metrics = predictor.fold_metrics(
        test_df, [mean_target] * test_df.height
    ).as_dict()
    assert metrics["mae"] == pytest.approx(1.5)
    assert metrics["rmse"] == pytest.approx(1.7795130420052185)
    assert metrics["skill_score"] == pytest.approx(0.0)


def test_fold_metrics_perfect_and_reversed_ranking_spearman(
    predictor, synthetic_frame
) -> None:
    """A perfectly ordered prediction scores spearman == 1.0.

    Feeding the actual target values back in as the "prediction"
    guarantees an identical ordering (so it is perfectly rank-
    correlated with itself) regardless of repeats in the target, and
    negating them reverses every pairwise order.
    """
    test_df = synthetic_frame(predictor, n_gws=1)
    actual = test_df[TARGET].to_list()
    perfect = predictor.fold_metrics(test_df, actual).as_dict()
    assert perfect["spearman"] == pytest.approx(1.0)

    reversed_metrics = predictor.fold_metrics(
        test_df, [-value for value in actual]
    ).as_dict()
    assert reversed_metrics["spearman"] == pytest.approx(-1.0)


# --- Cross validation and training -----------------------------------


def test_cross_validate_returns_per_fold_and_aggregates(
    predictor, synthetic_frame
) -> None:
    """cross_validate returns one result per fold plus aggregates."""
    model_df = synthetic_frame(predictor)
    folds = list(
        ExpandingGameweekFoldStrategy(min_train_folds=10).split(model_df)
    )
    per_fold, agg = predictor.cross_validate(folds)
    assert len(per_fold) == len(folds) == 4
    assert "cv_mae_mean" in agg
    assert "cv_mae_std" in agg


def test_cross_validate_of_a_single_fold_reports_no_spread(
    predictor, synthetic_frame
) -> None:
    """A holdout is one measurement, so it has no standard deviation."""
    predictor.fold_strategy = TrainTestSplitStrategy(test_fraction=0.2)
    model_df = synthetic_frame(predictor)
    folds = list(predictor.fold_strategy.split(model_df))
    per_fold, agg = predictor.cross_validate(folds)
    assert len(per_fold) == 1
    assert "holdout_mae" in agg
    assert not any(key.endswith(("_mean", "_std")) for key in agg)


def test_fit_predict_fold_returns_metrics_and_predictions(
    predictor, synthetic_frame
) -> None:
    """A fold result carries its scores and the rows they came from."""
    model_df = synthetic_frame(predictor)
    fold = next(
        iter(TrainTestSplitStrategy(test_fraction=0.2).split(model_df))
    )
    result = predictor.fit_predict_fold(fold)
    assert set(result.metrics.as_dict()) == {
        "mae",
        "rmse",
        "skill_score",
        "spearman",
        "precision_at_k",
    }
    assert result.predictions.height == fold.test.height
    assert result.predictions.columns == [
        column
        for column in TEST_POINTS_PREDICTION.columns
        if column != "run_id"
    ]


def test_fold_predictions_carry_the_actual_and_the_features(
    predictor, synthetic_frame
) -> None:
    """Stored rows hold the outcome and every feature the model consumed."""
    model_df = synthetic_frame(predictor)
    fold = next(
        iter(TrainTestSplitStrategy(test_fraction=0.2).split(model_df))
    )
    predictions = predictor.fit_predict_fold(fold).predictions
    assert predictions["actual_points"].to_list() == [
        float(value) for value in fold.test[TARGET].to_list()
    ]
    features = json.loads(predictions["features"][0])
    assert sorted(features) == sorted(predictor.FEATURES)


def test_store_fold_predictions_writes_one_row_per_scored_row(
    predictor, synthetic_frame
) -> None:
    """Every fold's test-side rows land in the evaluation table."""
    model_df = synthetic_frame(predictor)
    folds = list(TrainTestSplitStrategy(test_fraction=0.2).split(model_df))
    results, _ = predictor.cross_validate(folds)

    predictor.store_fold_predictions(results, "run-1")

    stored = TEST_POINTS_PREDICTION.load(predictor.connection)
    assert stored.height == sum(fold.test.height for fold in folds)
    assert stored["run_id"].unique().to_list() == ["run-1"]
    assert stored["position"].unique().to_list() == [predictor.POSITION]


def test_store_fold_predictions_of_nothing_writes_nothing(predictor) -> None:
    """A run that scored no fold stores no evaluation rows."""
    predictor.store_fold_predictions([], "run-1")
    assert TEST_POINTS_PREDICTION.load(predictor.connection).is_empty()


def test_cross_validate_handles_no_folds(predictor) -> None:
    """No folds yields empty per-fold and aggregate results."""
    per_fold, agg = predictor.cross_validate([])
    assert per_fold == []
    assert agg == {}


def test_train_final_fits_on_every_row(predictor, synthetic_frame) -> None:
    """train_final fits a pipeline that predicts every row in the frame."""
    model_df = synthetic_frame(predictor)
    pipe = predictor.train_final(model_df)
    predicted = pipe.predict(model_df.select(predictor.FEATURES).to_pandas())
    assert len(predicted) == model_df.height


def _patch_mlflow(mocker, run_id: str = "run-1"):
    """Patch the MLflow calls the base class makes, naming the run."""
    for name in ("set_experiment", "log_params"):
        mocker.patch(f"fantasy_football.modelling.predictor.mlflow.{name}")
    start_run = mocker.patch(
        "fantasy_football.modelling.predictor.mlflow.start_run"
    )
    start_run.return_value.__enter__.return_value.info.run_id = run_id
    return SimpleNamespace(
        log_metric=mocker.patch(
            "fantasy_football.modelling.predictor.mlflow.log_metric"
        ),
        log_metrics=mocker.patch(
            "fantasy_football.modelling.predictor.mlflow.log_metrics"
        ),
        log_model=mocker.patch(
            "fantasy_football.modelling.predictor.mlflow.sklearn.log_model"
        ),
    )


def test_train_and_register_model_logs_and_registers(
    predictor, synthetic_frame, mocker
) -> None:
    """Training logs the aggregate metrics and registers to MLflow.

    ``mlflow`` is patched on :mod:`fantasy_football.modelling.predictor`,
    which is where the base class does its logging.
    """
    predictor._model_dataframe = synthetic_frame(predictor)
    patched = _patch_mlflow(mocker)

    predictor.train_and_register_model()

    patched.log_metrics.assert_called_once()
    assert "cv_mae_mean" in patched.log_metrics.call_args.args[0]
    assert (
        patched.log_model.call_args.kwargs["registered_model_name"]
        == predictor.model_spec.registered_model_name
    )


def test_train_and_register_model_prefixes_per_fold_metrics(
    predictor, synthetic_frame, mocker
) -> None:
    """Per-fold series are prefixed too, or holdout and CV runs collide.

    A bare ``mae`` at step 0 would land in the same MLflow series as
    every historical cross-validated ``mae``.
    """
    predictor._model_dataframe = synthetic_frame(predictor)
    patched = _patch_mlflow(mocker)

    predictor.train_and_register_model()

    logged = {call.args[0] for call in patched.log_metric.call_args_list}
    assert logged
    assert all(name.startswith("cv_") for name in logged)


def test_train_and_register_model_stores_predictions_after_registering(
    predictor, synthetic_frame, mocker
) -> None:
    """A run that dies while registering leaves no evaluation rows."""
    predictor.fold_strategy = TrainTestSplitStrategy(test_fraction=0.2)
    predictor._model_dataframe = synthetic_frame(predictor)
    patched = _patch_mlflow(mocker)
    patched.log_model.side_effect = RuntimeError("registry down")

    with pytest.raises(RuntimeError):
        predictor.train_and_register_model()

    assert TEST_POINTS_PREDICTION.load(predictor.connection).is_empty()


def test_train_and_register_model_stores_predictions_against_its_run(
    predictor, synthetic_frame, mocker
) -> None:
    """Stored predictions carry the run whose metrics describe them."""
    predictor.fold_strategy = TrainTestSplitStrategy(test_fraction=0.2)
    predictor._model_dataframe = synthetic_frame(predictor)
    _patch_mlflow(mocker, run_id="run-7")

    predictor.train_and_register_model()

    stored = TEST_POINTS_PREDICTION.load(predictor.connection)
    assert stored["run_id"].unique().to_list() == ["run-7"]


def test_build_prediction_rows_shapes_rows_for_storage(
    predictor, synthetic_frame
) -> None:
    """build_prediction_rows shapes rows for POINTS_COMPONENT storage."""
    frame = synthetic_frame(predictor, n_gws=1, n_players=3)
    pipe = predictor.train_final(frame)
    scored = predictor.build_prediction_rows(frame, pipe, "4", BACKFILL_KIND)

    assert scored.columns == POINTS_COMPONENT.columns
    assert scored.height == frame.height
    assert scored["position"].unique().to_list() == [predictor.POSITION]
    assert scored["model_version"].unique().to_list() == ["4"]
    assert scored["prediction_kind"].unique().to_list() == [BACKFILL_KIND]
    assert scored["component"].unique().to_list() == [Component.TOTAL]


def test_undecomposed_position_composes_back_to_its_raw_prediction(
    predictor, synthetic_frame
) -> None:
    """A single-component position composes to exactly what it predicted.

    The byte-identical guarantee: decomposition must not move a number
    for a position that has not been decomposed.
    """
    frame = synthetic_frame(predictor, n_gws=1, n_players=3)
    pipe = predictor.train_final(frame)
    scored = predictor.build_prediction_rows(frame, pipe, "4", BACKFILL_KIND)

    composed = compose(scored, expected=POSITION_COMPONENTS).sort(KEY_COLUMNS)
    component_points = scored.sort(KEY_COLUMNS)["points"].to_list()

    assert composed.columns == POINTS_PREDICTION.columns
    assert composed["predicted_points"].to_list() == pytest.approx(
        component_points
    )
    assert composed["model_version"].unique().to_list() == ["4"]


# --- Backfill --------------------------------------------------------


def _wire_backfill(mocker, predictor, frame, prod_version: str) -> None:
    """Give ``predictor`` a training frame and a stubbed production model."""
    predictor._model_dataframe = frame
    mocker.patch(
        "fantasy_football.modelling.predictor.load_production_model",
        return_value=(prod_version, predictor.train_final(frame)),
    )
    mocker.patch(
        "fantasy_football.modelling.predictor.CURRENT_SEASON", "2026-27"
    )


def _two_season_frame(synthetic_frame, predictor) -> pl.DataFrame:
    """Combine a historic and a current season into one model frame."""
    return pl.concat(
        [
            synthetic_frame(predictor, n_gws=2, n_players=3, season="2025-26"),
            synthetic_frame(predictor, n_gws=2, n_players=3, season="2026-27"),
        ],
        how="vertical",
    )


def _seed_points_component(
    connection,
    predictor,
    frame: pl.DataFrame,
    season: str,
    model_version: str,
    points: float = 999.0,
) -> None:
    """Seed a sentinel backfill component per key in ``frame``."""
    sentinel = (
        frame.filter(pl.col("season") == season)
        .select(KEY_COLUMNS)
        .with_columns(
            position=pl.lit(predictor.POSITION),
            prediction_kind=pl.lit(BACKFILL_KIND),
            component=pl.lit(str(predictor.COMPONENT)),
            points=pl.lit(points),
            model_version=pl.lit(model_version),
            diagnostics=pl.lit(None, dtype=pl.Utf8),
        )
        .select(POINTS_COMPONENT.columns)
    )
    POINTS_COMPONENT.append(connection, sentinel)


def test_backfill_writes_nothing_without_a_production_alias(
    predictor, connection, synthetic_frame, mocker
) -> None:
    """No production alias: backfill returns quietly, storing nothing."""
    predictor._model_dataframe = synthetic_frame(
        predictor, n_gws=2, n_players=3
    )
    mocker.patch(
        "fantasy_football.modelling.predictor.load_production_model",
        return_value=None,
    )

    predictor.backfill_model_predictions()

    assert POINTS_COMPONENT.load(connection).is_empty()


def test_backfill_rescores_the_current_season_every_run(
    predictor, connection, synthetic_frame, mocker
) -> None:
    """The current season is always re-scored, with no historic seasons."""
    frame = synthetic_frame(predictor, n_gws=2, n_players=3)
    _wire_backfill(mocker, predictor, frame, "4")

    predictor.backfill_model_predictions()

    stored = POINTS_COMPONENT.load(connection)
    assert stored.height == frame.height
    assert stored["prediction_kind"].unique().to_list() == [BACKFILL_KIND]
    assert stored["model_version"].unique().to_list() == ["4"]


def test_backfill_rescores_historic_season_when_nothing_stored(
    predictor, connection, synthetic_frame, mocker
) -> None:
    """A historic season with no stored rows is scored on this run."""
    frame = _two_season_frame(synthetic_frame, predictor)
    _wire_backfill(mocker, predictor, frame, "4")

    predictor.backfill_model_predictions()

    historic = POINTS_COMPONENT.load(connection).filter(
        pl.col("season") == "2025-26"
    )
    assert (
        historic.height == frame.filter(pl.col("season") == "2025-26").height
    )
    assert historic["model_version"].unique().to_list() == ["4"]


def test_backfill_rescores_historic_season_on_version_change(
    predictor, connection, synthetic_frame, mocker
) -> None:
    """A historic season stored under a stale version is rewritten."""
    frame = _two_season_frame(synthetic_frame, predictor)
    _wire_backfill(mocker, predictor, frame, "4")
    _seed_points_component(connection, predictor, frame, "2025-26", "3")

    predictor.backfill_model_predictions()

    historic = POINTS_COMPONENT.load(connection).filter(
        pl.col("season") == "2025-26"
    )
    assert (
        historic.height == frame.filter(pl.col("season") == "2025-26").height
    )
    assert historic["model_version"].unique().to_list() == ["4"]
    assert 999.0 not in historic["points"].to_list()


def test_backfill_skips_historic_season_already_at_production_version(
    predictor, connection, synthetic_frame, mocker
) -> None:
    """A complete, up-to-date historic season is left untouched.

    The current season is still re-scored regardless of what the gate
    decides about historic seasons.
    """
    frame = _two_season_frame(synthetic_frame, predictor)
    _wire_backfill(mocker, predictor, frame, "4")
    _seed_points_component(connection, predictor, frame, "2025-26", "4")

    predictor.backfill_model_predictions()

    stored = POINTS_COMPONENT.load(connection)
    historic = stored.filter(pl.col("season") == "2025-26")
    # Untouched sentinel proves the historic partition was not rewritten.
    assert historic["points"].unique().to_list() == [999.0]
    assert historic["model_version"].unique().to_list() == ["4"]

    current = stored.filter(pl.col("season") == "2026-27")
    assert current.height == frame.filter(pl.col("season") == "2026-27").height
    assert current["model_version"].unique().to_list() == ["4"]


# --- Forward feature frame -------------------------------------------


def test_forward_frame_takes_the_most_recent_appearance(
    predictor, connection, stub_player_form, forward_fixture, forward_frame
) -> None:
    """The as-of join takes the latest appearance before the fixture."""
    register_feature_views(connection)
    stub_player_form(
        connection,
        {
            "season": [SEASON, SEASON],
            "element": [1, 1],
            "kickoff_time": [GW1_KICKOFF, GW2_KICKOFF],
            _probe(predictor): [0.1, 0.9],
        },
        predictor,
    )

    frame = predictor.build_forward_data(
        forward_frame([forward_fixture(position=predictor.POSITION)])
    )

    assert frame[_probe(predictor)].to_list() == [0.9]


def test_forward_frame_covers_every_rostered_player(
    predictor, connection, forward_fixture, forward_frame
) -> None:
    """Every rostered player at this position gets a row, form or no form."""
    register_feature_views(connection)
    frame = predictor.build_forward_data(
        forward_frame(
            [
                forward_fixture(element=1, position=predictor.POSITION),
                forward_fixture(element=2, position=predictor.POSITION),
            ]
        )
    )

    assert sorted(frame["element"].to_list()) == [1, 2]
    assert frame.columns == KEY_COLUMNS + predictor.FEATURES


def test_forward_frame_drops_other_positions(
    predictor, connection, forward_fixture, forward_frame
) -> None:
    """A player at another position is not scored by this model.

    Every position writes to points_prediction, whose primary key does
    not carry position. Scoring a player this model does not serve puts
    a row there that another position's model owns.
    """
    other = "GKP" if predictor.POSITION != "GKP" else "MID"
    register_feature_views(connection)

    frame = predictor.build_forward_data(
        forward_frame(
            [
                forward_fixture(element=1, position=predictor.POSITION),
                forward_fixture(element=2, position=other),
            ]
        )
    )

    assert frame["element"].to_list() == [1]


def test_forward_frame_carries_is_home_from_the_fixture(
    predictor, connection, seed_forward_history, forward_fixture, forward_frame
) -> None:
    """``is_home`` is resolved from team_fixture, not left null.

    It is a model feature, so a null here would be median-imputed on
    every live row and the model would never see who is at home.
    """
    seed_forward_history(connection, predictor.POSITION)
    register_feature_views(connection)

    home = predictor.build_forward_data(
        forward_frame([forward_fixture(position=predictor.POSITION)])
    )
    away = predictor.build_forward_data(
        forward_frame(
            [
                forward_fixture(
                    element=3,
                    team=ARSENAL,
                    opponent=TEAM_IDS[UNITED],
                    position=predictor.POSITION,
                )
            ]
        )
    )

    assert home["is_home"].item() == 1.0
    assert away["is_home"].item() == 0.0


def test_forward_frame_matches_each_player_to_his_own_last_appearance(
    predictor, connection, stub_player_form, forward_fixture, forward_frame
) -> None:
    """Interleaved appearances still resolve per player, not globally.

    Element 1's appearances straddle element 2's in time, so a join that
    lost the per-player grouping -- or sorted only globally -- would hand
    at least one of them the other's row.
    """
    register_feature_views(connection)
    stub_player_form(
        connection,
        {
            "season": [SEASON] * 4,
            "element": [1, 2, 1, 2],
            "kickoff_time": [
                GW1_KICKOFF,
                GW1_KICKOFF + timedelta(days=1),
                GW2_KICKOFF,
                GW2_KICKOFF + timedelta(days=1),
            ],
            _probe(predictor): [0.1, 5.0, 0.2, 6.0],
        },
        predictor,
    )

    frame = predictor.build_forward_data(
        forward_frame(
            [
                forward_fixture(element=1, position=predictor.POSITION),
                forward_fixture(element=2, position=predictor.POSITION),
            ]
        )
    ).sort("element")

    assert frame[_probe(predictor)].to_list() == [0.2, 6.0]


def test_forward_frame_resolves_the_opponent_by_name_not_reused_id(
    predictor,
    connection,
    seed_forward_history,
    stub_two_season_form,
    forward_fixture,
    forward_frame,
) -> None:
    """Opposition form is Arsenal's, even though id 1 was Chelsea's.

    Chelsea played more recently than Arsenal, so a join that pools both
    clubs under FPL id 1 -- which is what dropping ``season`` from the
    ``fpl_team_id`` join does -- as-of matches Chelsea's 3.0 rather than
    Arsenal's 0.5. Resolving the opponent by name through
    ``team_fixture`` cannot make that mistake.
    """
    seed_forward_history(connection, predictor.POSITION)
    register_feature_views(connection)
    stub_two_season_form(connection, predictor)

    frame = predictor.build_forward_data(
        forward_frame([forward_fixture(position=predictor.POSITION)])
    )

    for column in predictor.opposition_feature_names:
        assert frame[column].item() == pytest.approx(0.5)


def test_forward_frame_has_exactly_one_row_per_fixture(
    predictor,
    connection,
    seed_forward_history,
    stub_two_season_form,
    forward_fixture,
    forward_frame,
) -> None:
    """Three input fixtures give three output rows, keys unchanged.

    The seeded form carries a club in two seasons, so a join that
    multiplies form rows by season would show up here as extra rows.
    """
    seed_forward_history(connection, predictor.POSITION)
    register_feature_views(connection)
    stub_two_season_form(connection, predictor)
    forward = forward_frame(
        [
            forward_fixture(element=1, position=predictor.POSITION),
            forward_fixture(element=2, position=predictor.POSITION),
            forward_fixture(
                element=3,
                team=ARSENAL,
                opponent=TEAM_IDS[UNITED],
                position=predictor.POSITION,
            ),
        ]
    )

    frame = predictor.build_forward_data(forward)

    assert frame.height == forward.height
    assert sorted(frame.select(KEY_COLUMNS).rows()) == sorted(
        forward.select(KEY_COLUMNS).rows()
    )


def test_forward_frame_gives_a_double_gameweek_two_rows(
    predictor, connection, seed_forward_history, forward_fixture, forward_frame
) -> None:
    """One player with two gw3 fixtures gets two rows, not one or four."""
    seed_forward_history(connection, predictor.POSITION)
    register_feature_views(connection)
    second_kickoff = GW3_KICKOFF + timedelta(days=3)
    append_rows(
        TEAM_FIXTURE,
        connection,
        [
            {
                "season": SEASON,
                "gw": 3,
                "team": UNITED,
                "is_home": False,
                "opposition": SPURS,
                "kickoff_time": second_kickoff,
            }
        ],
    )

    frame = predictor.build_forward_data(
        forward_frame(
            [
                forward_fixture(position=predictor.POSITION),
                forward_fixture(
                    opponent=TEAM_IDS[SPURS],
                    kickoff=second_kickoff,
                    position=predictor.POSITION,
                ),
            ]
        )
    )

    assert frame.height == 2
    assert sorted(frame["opponent"].to_list()) == sorted(
        [TEAM_IDS[SPURS], TEAM_IDS[ARSENAL]]
    )
    assert sorted(frame["is_home"].to_list()) == [0.0, 1.0]


def test_forward_frame_takes_only_forward_minutes_predictions(
    predictor, connection, seed_forward_history, forward_fixture, forward_frame
) -> None:
    """Backfill minutes rows for the same key are ignored, not joined.

    ``prediction_kind`` is part of the minutes primary key, so both kinds
    can sit on one fixture. Joining without the filter would either take
    the backfill value or fan the row out.
    """
    seed_forward_history(connection, predictor.POSITION)
    register_feature_views(connection)
    for kind, expected in ((BACKFILL_KIND, 11.0), (FORWARD_KIND, 77.0)):
        append_rows(
            MINUTES_PREDICTION,
            connection,
            [
                {
                    "season": SEASON,
                    "gw": 3,
                    "element": 1,
                    "opponent": TEAM_IDS[ARSENAL],
                    "p_zero": 0.1,
                    "p_partial": 0.2,
                    "p_sixty_plus": 0.7,
                    "expected_minutes": expected,
                    "model_version": "1",
                    "prediction_kind": kind,
                }
            ],
        )

    frame = predictor.build_forward_data(
        forward_frame([forward_fixture(position=predictor.POSITION)])
    )

    assert frame.height == 1
    assert frame["expected_minutes"].item() == pytest.approx(77.0)


def test_forward_frame_never_inherits_a_previous_holder_of_the_id(
    predictor, connection, stub_player_form, forward_fixture, forward_frame
) -> None:
    """A player with no appearance this season gets null form, not another's.

    FPL reissues element ids each season, so element 1 in 2024-25 is a
    different footballer from element 1 in 2025-26. Matching on element
    alone would as-of join back into last season and hand this player a
    stranger's form, which nothing downstream could detect -- the value
    looks perfectly ordinary. A null is the honest answer; the pipeline's
    median imputer handles it.
    """
    register_feature_views(connection)
    stub_player_form(
        connection,
        {
            "season": ["2024-25"],
            "element": [1],
            "kickoff_time": [GW1_KICKOFF - timedelta(days=365)],
            _probe(predictor): [7.0],
        },
        predictor,
    )

    frame = predictor.build_forward_data(
        forward_frame([forward_fixture(position=predictor.POSITION)])
    )

    assert frame[_probe(predictor)].to_list() == [None]


def test_forward_frame_carries_form_across_the_season_boundary(
    predictor, connection, stub_player_form, forward_fixture, forward_frame
) -> None:
    """Last season's form serves a player who has not played this one.

    The training view windows on ``rolling_identity`` with no season
    term, so a player's first row of a season carries the tail of his
    previous one. Keying the forward as-of join on ``(season, element)``
    instead would match nothing before his first appearance -- so every
    player at a pre-season squad build would arrive with all their
    player-form features null and be median-imputed, which is precisely
    the moment this model exists to serve.
    """
    append_rows(
        PLAYER_SEASON,
        connection,
        [
            {"season": "2024-25", "element": 9, "player_code": 777},
            {"season": SEASON, "element": 1, "player_code": 777},
        ],
    )
    register_feature_views(connection)
    stub_player_form(
        connection,
        {
            "season": ["2024-25"],
            "element": [9],
            "rolling_identity": ["777"],
            "kickoff_time": [GW1_KICKOFF - timedelta(days=365)],
            _probe(predictor): [7.0],
        },
        predictor,
    )

    frame = predictor.build_forward_data(
        forward_frame([forward_fixture(position=predictor.POSITION)])
    )

    assert frame[_probe(predictor)].to_list() == [7.0]


def test_forward_frame_never_inherits_a_reused_id_with_player_codes(
    predictor, connection, stub_player_form, forward_fixture, forward_frame
) -> None:
    """Sharing an element id across seasons is not sharing an identity.

    Element 1 belonged to player_code 111 in 2024-25 and to 222 in
    2025-26. Carrying form across the season boundary must not carry it
    across footballers, and ``player_code`` -- not ``season`` -- is what
    keeps the two apart.
    """
    append_rows(
        PLAYER_SEASON,
        connection,
        [
            {"season": "2024-25", "element": 1, "player_code": 111},
            {"season": SEASON, "element": 1, "player_code": 222},
        ],
    )
    register_feature_views(connection)
    stub_player_form(
        connection,
        {
            "season": ["2024-25"],
            "element": [1],
            "rolling_identity": ["111"],
            "kickoff_time": [GW1_KICKOFF - timedelta(days=365)],
            _probe(predictor): [7.0],
        },
        predictor,
    )

    frame = predictor.build_forward_data(
        forward_frame([forward_fixture(position=predictor.POSITION)])
    )

    assert frame[_probe(predictor)].to_list() == [None]


def test_forward_frame_gives_a_player_with_no_history_a_null_row(
    predictor, connection, stub_player_form, forward_fixture, forward_frame
) -> None:
    """A player with no appearance anywhere still gets a row, all nulls."""
    register_feature_views(connection)
    stub_player_form(
        connection,
        {
            "season": [SEASON],
            "element": [2],
            "kickoff_time": [GW2_KICKOFF],
            _probe(predictor): [0.4],
        },
        predictor,
    )

    frame = predictor.build_forward_data(
        forward_frame(
            [forward_fixture(element=1, position=predictor.POSITION)]
        )
    )

    assert frame.height == 1
    assert frame[_probe(predictor)].to_list() == [None]


def test_forward_frame_starts_the_season_to_date_counts_at_zero(
    predictor, connection, stub_player_form, forward_fixture, forward_frame
) -> None:
    """Carrying form over a season boundary does not carry card counts.

    The rolling rates span seasons; the season-to-date totals reset, and
    training coalesces an empty season window to 0. A player whose most
    recent appearance is last season must therefore arrive with 0 cards
    this season, not last season's closing count and not a null that the
    imputer turns into a mid-season booking tally.
    """
    append_rows(
        PLAYER_SEASON,
        connection,
        [
            {"season": "2024-25", "element": 9, "player_code": 777},
            {"season": SEASON, "element": 1, "player_code": 777},
        ],
    )
    register_feature_views(connection)
    stub_player_form(
        connection,
        {
            "season": ["2024-25"],
            "element": [9],
            "rolling_identity": ["777"],
            "kickoff_time": [GW1_KICKOFF - timedelta(days=365)],
            _probe(predictor): [7.0],
            "yellow_cards_season_to_date": [8.0],
            "red_cards_season_to_date": [1.0],
        },
        predictor,
    )

    frame = predictor.build_forward_data(
        forward_frame([forward_fixture(position=predictor.POSITION)])
    )

    assert frame[_probe(predictor)].to_list() == [7.0]
    for column in predictor.season_to_date_columns:
        assert frame[column].to_list() == [0.0]


def test_forward_frame_takes_this_season_to_date_counts_when_present(
    predictor, connection, stub_player_form, forward_fixture, forward_frame
) -> None:
    """An appearance this season supplies its own running card totals."""
    append_rows(
        PLAYER_SEASON,
        connection,
        [{"season": SEASON, "element": 1, "player_code": 777}],
    )
    register_feature_views(connection)
    stub_player_form(
        connection,
        {
            "season": ["2024-25", SEASON],
            "element": [9, 1],
            "rolling_identity": ["777", "777"],
            "kickoff_time": [GW1_KICKOFF - timedelta(days=365), GW2_KICKOFF],
            "yellow_cards_season_to_date": [8.0, 3.0],
        },
        predictor,
    )

    frame = predictor.build_forward_data(
        forward_frame([forward_fixture(position=predictor.POSITION)])
    )

    assert frame["yellow_cards_season_to_date"].to_list() == [3.0]


def test_forward_frame_scores_through_a_fitted_pipeline(
    predictor,
    connection,
    seed_forward_history,
    synthetic_frame,
    forward_fixture,
    forward_frame,
) -> None:
    """The forward frame's dtypes survive a real fitted pipeline.

    Two of the form columns arrive from DuckDB as ``Decimal``, not
    ``Float64``, and reach pandas as ``object``. A frame that builds
    cleanly but cannot be predicted on is no use, so this exercises the
    real pipeline rather than a stand-in model.
    """
    seed_forward_history(connection, predictor.POSITION)
    register_feature_views(connection)
    frame = predictor.build_forward_data(
        forward_frame([forward_fixture(position=predictor.POSITION)])
    )
    pipe = predictor.train_final(
        synthetic_frame(predictor, n_gws=2, n_players=3)
    )

    scored = predictor.build_prediction_rows(frame, pipe, "1", FORWARD_KIND)

    assert scored.columns == POINTS_COMPONENT.columns
    assert scored.height == 1
    assert scored["points"].dtype == pl.Float64
    assert scored["points"].item() is not None


# --- Forward scoring -------------------------------------------------


def test_forward_scoring_skips_without_a_production_alias(
    predictor, connection, mocker
) -> None:
    """No production alias: forward scoring returns, storing nothing."""
    mocker.patch(
        "fantasy_football.modelling.predictor.load_production_model",
        return_value=None,
    )

    predictor.predict_forward()

    assert POINTS_COMPONENT.load(connection).is_empty()


def _seed_snapshot(connection, position: str) -> None:
    """Seed a two-player snapshot: element 1 at ``position``, 3 elsewhere."""
    other = "MID" if position != "MID" else "FWD"
    append_rows(
        PLAYER_SNAPSHOT,
        connection,
        [
            {
                "season": SEASON,
                "captured_at": GW2_KICKOFF + timedelta(days=1),
                "element": 1,
                "value": 50,
                "team": UNITED,
                "position": position,
                "chance_of_playing_this_round": 100,
            },
            {
                "season": SEASON,
                "captured_at": GW2_KICKOFF + timedelta(days=1),
                "element": 3,
                "value": 60,
                "team": ARSENAL,
                "position": other,
                "chance_of_playing_this_round": 100,
            },
        ],
    )


def _mock_fpl_teams(mocker):
    """Patch FplAPI so team-name-to-id resolution needs no network."""
    api = mocker.Mock()
    api.get_teams.return_value = [
        SimpleNamespace(id=team_id, name=name)
        for name, team_id in TEAM_IDS.items()
    ]
    mocker.patch(
        "fantasy_football.modelling.predictor.FplAPI", return_value=api
    )


def test_predict_forward_stores_and_freezes(
    predictor, connection, seed_forward_history, mocker
) -> None:
    """Forward rows land for gw3 while the gw2 forecast stays frozen.

    gw1 and gw2 have been played, so ``from_gw`` is 3 and the gw2
    sentinel sits below the delete floor. Losing ``gw_from`` on
    ``replace_partition`` would wipe it.
    """
    seed_forward_history(connection, predictor.POSITION)
    # predict_forward reads the inclusive form views but does not
    # register them; in the pipeline training does that first.
    register_feature_views(connection)
    _seed_snapshot(connection, predictor.POSITION)
    POINTS_COMPONENT.append(
        connection,
        POINTS_COMPONENT.coerce(
            pl.DataFrame(
                {
                    "season": [SEASON],
                    "gw": [2],
                    "element": [1],
                    "opponent": [TEAM_IDS[CHELSEA]],
                    "position": [predictor.POSITION],
                    "prediction_kind": [FORWARD_KIND],
                    "component": [str(predictor.COMPONENT)],
                    "points": [999.0],
                    "model_version": ["1"],
                    "diagnostics": [None],
                }
            )
        ),
    )
    mocker.patch(
        "fantasy_football.modelling.predictor.load_production_model",
        return_value=("9", ConstantPointsModel()),
    )
    mocker.patch("fantasy_football.modelling.predictor.CURRENT_SEASON", SEASON)
    _mock_fpl_teams(mocker)

    predictor.predict_forward()

    stored = POINTS_COMPONENT.load(connection)
    frozen = stored.filter(pl.col("gw") == 2)
    assert frozen["points"].to_list() == [999.0]
    assert frozen["model_version"].to_list() == ["1"]

    fresh = stored.filter(pl.col("gw") == 3)
    # Only this position is scored -- the other player is dropped.
    assert fresh.select("element", "opponent").rows() == [
        (1, TEAM_IDS[ARSENAL])
    ]
    assert fresh["points"].to_list() == [4.5]
    assert fresh["prediction_kind"].to_list() == [FORWARD_KIND]
    assert fresh["model_version"].to_list() == ["9"]
    assert fresh["position"].to_list() == [predictor.POSITION]


# --- Cross-position isolation ----------------------------------------


def test_positions_do_not_overwrite_each_others_predictions(
    connection, seed_forward_history, mocker
) -> None:
    """Scoring one position leaves another position's rows in place.

    ``points_prediction``'s primary key does not carry ``position``, so
    the partition rewrite must be scoped by it. Without that, the second
    model to run deletes everything the first one wrote for the same
    season, kind and gameweek range, and only the last position survives.
    """
    seed_forward_history(connection, "DEF")
    register_feature_views(connection)
    append_rows(
        PLAYER_SNAPSHOT,
        connection,
        [
            {
                "season": SEASON,
                "captured_at": GW2_KICKOFF + timedelta(days=1),
                "element": element,
                "value": 50,
                "team": UNITED,
                "position": position,
                "chance_of_playing_this_round": 100,
            }
            for element, position in [(1, "DEF"), (2, "FWD")]
        ],
    )
    mocker.patch(
        "fantasy_football.modelling.predictor.load_production_model",
        return_value=("9", ConstantPointsModel()),
    )
    mocker.patch("fantasy_football.modelling.predictor.CURRENT_SEASON", SEASON)
    _mock_fpl_teams(mocker)

    for cls, spec in SPECS.items():
        cls(
            experiment_name=f"test-{cls.POSITION}",
            params={},
            model_spec=spec,
            connection=connection,
            fold_strategy=ExpandingGameweekFoldStrategy(),
        ).predict_forward()

    stored = POINTS_COMPONENT.load(connection).filter(pl.col("gw") == 3)
    assert sorted(stored["position"].unique().to_list()) == ["DEF", "FWD"]
