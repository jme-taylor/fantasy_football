from datetime import date
from unittest import mock

import numpy as np
import polars as pl
import pytest
from sklearn.pipeline import Pipeline

from fantasy_football.constants import (
    MINUTES_REGISTERED_MODEL,
    MLFLOW_TRACKING_URI,
)
from fantasy_football.modelling.minutes import (
    BUCKET_PARTIAL,
    BUCKET_SIXTY_PLUS,
    BUCKET_ZERO,
    FEATURES,
    MINUTES_BUCKETS,
    boundary_metrics,
    build_feature_frame,
    build_model_frame,
    create_minutes_bucket,
    cross_validate,
    make_pipeline,
    run_minutes_model,
    season_folds,
    train_final,
)


def test_create_minutes_bucket_edges() -> None:
    """0 -> zero, 1..59 -> partial, 60+ -> sixty-plus (boundaries pinned)."""
    data = pl.DataFrame({"minutes": [0, 1, 59, 60, 90]})

    result = create_minutes_bucket(data)
    buckets = result["minutes_bucket"].to_list()

    assert buckets == [
        BUCKET_ZERO,
        BUCKET_PARTIAL,
        BUCKET_PARTIAL,
        BUCKET_SIXTY_PLUS,
        BUCKET_SIXTY_PLUS,
    ]


def _player_week() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": ["2022-23"] * 3,
            "gw": [1, 1, 1],
            "element": [1, 2, 3],
            "name": ["A", "B", "C"],
            "position": ["MID", "MID", "DEF"],
            "team": ["Arsenal", "Arsenal", "Arsenal"],
            "bonus": [0, 0, 0],
            "minutes": [90, 0, 45],
            "round": [1, 1, 1],
            "total_points": [6, 0, 2],
            "value": [70, 50, 40],
        }
    )


def _availability() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": ["2022-23"] * 3,
            "gw": [1, 1, 1],
            "element": [1, 2, 3],
            "chance_of_playing_this_round": [100, 75, 100],
        }
    )


def _history_player_match() -> pl.DataFrame:
    """Player-match rows for the history-feature join.

    Distinct from the match-level ``player_match`` frame used by
    ``build_model_frame``.
    """
    return pl.DataFrame(
        {
            "season": ["2022-23"] * 3,
            "gw": [1, 1, 1],
            "element": [1, 2, 3],
            "minutes": [90, 0, 45],
            "total_points": [6, 0, 2],
        }
    )


def _player_season() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": ["2022-23"] * 3,
            "element": [1, 2, 3],
            "player_code": [101, 102, 103],
            "birth_date": [date(1995, 1, 1)] * 3,
            "team_join_date": [date(2020, 1, 1)] * 3,
        },
        schema_overrides={"birth_date": pl.Date, "team_join_date": pl.Date},
    )


def _team_fixture() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": ["2022-23"],
            "gw": [1],
            "team": ["Arsenal"],
        }
    )


def test_build_feature_frame_has_one_row_per_player_week() -> None:
    """Feature frame keeps the player-week grain and the feature columns."""
    frame = build_feature_frame(
        _player_week(),
        _availability(),
        _history_player_match(),
        _player_season(),
        _team_fixture(),
    )

    assert frame.height == 3
    for column in ["season", "gw", "element", "position", *FEATURES]:
        assert column in frame.columns
    # Arsenal MID pecking order: element 1 (value 70) ranks above element 2.
    ranks = {
        row["element"]: row["pos_value_rank"]
        for row in frame.iter_rows(named=True)
    }
    assert ranks[1] == 1
    assert ranks[2] == 2


def test_build_model_frame_joins_features_onto_matches() -> None:
    """Match rows get features by (season, gw, element) and a bucket target."""
    feature_frame = build_feature_frame(
        _player_week(),
        _availability(),
        _history_player_match(),
        _player_season(),
        _team_fixture(),
    )
    player_match = pl.DataFrame(
        {
            "season": ["2022-23"] * 3,
            "gw": [1, 1, 1],
            "element": [1, 2, 3],
            "opponent": [10, 10, 10],
            "is_home": [True, True, True],
            "minutes": [90, 0, 45],
            "total_points": [6, 0, 2],
        }
    )

    model_frame = build_model_frame(player_match, feature_frame)

    assert model_frame.height == 3
    buckets = {
        row["element"]: row["minutes_bucket"]
        for row in model_frame.iter_rows(named=True)
    }
    assert buckets[1] == BUCKET_SIXTY_PLUS
    assert buckets[2] == BUCKET_ZERO
    assert buckets[3] == BUCKET_PARTIAL
    for column in ["minutes", "minutes_bucket", *FEATURES]:
        assert column in model_frame.columns


def test_season_folds_expanding_window() -> None:
    """Each fold trains on all prior seasons and tests on the next one."""
    seasons = ["2022-23", "2023-24", "2024-25", "2025-26"]

    folds = season_folds(seasons)

    assert folds == [
        (["2022-23"], "2023-24"),
        (["2022-23", "2023-24"], "2024-25"),
        (["2022-23", "2023-24", "2024-25"], "2025-26"),
    ]


def test_boundary_metrics_perfect_predictions() -> None:
    """Confident, correct probabilities give ~0 loss and AUC 1.0."""
    classes = (
        MINUTES_BUCKETS  # ["0_minutes", "1_to_59_minutes", "60_minutes_plus"]
    )
    y_true = [BUCKET_ZERO, BUCKET_SIXTY_PLUS]
    true_minutes = [0, 90]
    # rows: benched (col 0), full shift (col 2).
    proba = np.array([[0.99, 0.005, 0.005], [0.005, 0.005, 0.99]])

    m = boundary_metrics(y_true, proba, classes, true_minutes)

    assert m["logloss_appear"] < 0.05
    assert m["logloss_60"] < 0.05
    assert m["auc_appear"] == 1.0
    assert m["auc_60"] == 1.0
    assert m["e_min_mae"] < 9.0


def test_boundary_metrics_skips_auc_when_single_class() -> None:
    """AUC is omitted for a boundary whose test rows are all one class."""
    classes = MINUTES_BUCKETS
    y_true = [BUCKET_SIXTY_PLUS, BUCKET_SIXTY_PLUS]  # all appear, all 60+
    true_minutes = [90, 75]
    proba = np.array([[0.01, 0.04, 0.95], [0.02, 0.03, 0.95]])

    m = boundary_metrics(y_true, proba, classes, true_minutes)

    assert "auc_appear" not in m
    assert "auc_60" not in m
    assert "logloss_60" in m


def test_make_pipeline_fits_and_predicts_proba() -> None:
    """The pipeline trains on a tiny frame and yields 3-class probabilities."""
    rng = np.random.default_rng(0)
    n = 30
    df = pl.DataFrame(
        {
            "value": rng.integers(40, 120, n),
            "value_share_of_team": rng.random(n),
            "pos_value_rank": rng.integers(1, 6, n),
            "players_same_pos": rng.integers(1, 6, n),
            "chance_of_playing_this_round": rng.choice([0, 75, 100], n),
            "fit_rivals_same_pos": rng.integers(0, 4, n),
            "fit_rivals_ahead": rng.integers(0, 4, n),
            "avg_minutes_rolling_5": rng.random(n) * 90,
            "games_played_this_season": rng.integers(0, 10, n),
            "prev_season_minutes": rng.integers(0, 3000, n),
            "prev_season_start_rate": rng.random(n),
            "prev_season_points_per_start": rng.random(n) * 6,
            "pl_seasons_played": rng.integers(0, 5, n),
            "seasons_since_last_pl": rng.integers(0, 3, n),
            "age_years": rng.random(n) * 15 + 17,
            "days_since_team_join": rng.random(n) * 3000,
            "position": rng.choice(["GK", "DEF", "MID", "FWD"], n),
            "is_pl_newcomer": rng.integers(0, 2, n),
            "is_promoted_club": rng.integers(0, 2, n),
            "minutes_bucket": rng.choice(MINUTES_BUCKETS, n),
        }
    )

    pipe = make_pipeline()
    pipe.fit(df.select(FEATURES).to_pandas(), df["minutes_bucket"].to_list())
    proba = pipe.predict_proba(df.select(FEATURES).to_pandas())

    assert proba.shape == (n, len(MINUTES_BUCKETS))


def _synthetic_model_df(
    seasons: list[str], per_season: int = 60
) -> pl.DataFrame:
    """Return a model frame with a learnable signal across several seasons."""
    rng = np.random.default_rng(1)
    rows = []
    for season in seasons:
        for _ in range(per_season):
            rank = int(rng.integers(1, 6))
            # Higher-ranked (lower number) players play more.
            if rank <= 2:
                bucket = BUCKET_SIXTY_PLUS
                minutes = 90
            elif rank == 3:
                bucket = BUCKET_PARTIAL
                minutes = 30
            else:
                bucket = BUCKET_ZERO
                minutes = 0
            rows.append(
                {
                    "season": season,
                    "value": int(rng.integers(40, 120)),
                    "value_share_of_team": float(rng.random()),
                    "pos_value_rank": rank,
                    "players_same_pos": 5,
                    "chance_of_playing_this_round": 100,
                    "fit_rivals_same_pos": rank - 1,
                    "fit_rivals_ahead": rank - 1,
                    "avg_minutes_rolling_5": float(rng.random() * 90),
                    "games_played_this_season": int(rng.integers(0, 10)),
                    "prev_season_minutes": int(rng.integers(0, 3000)),
                    "prev_season_start_rate": float(rng.random()),
                    "prev_season_points_per_start": float(rng.random() * 6),
                    "pl_seasons_played": int(rng.integers(0, 5)),
                    "seasons_since_last_pl": int(rng.integers(0, 3)),
                    "age_years": float(rng.random() * 15 + 17),
                    "days_since_team_join": float(rng.random() * 3000),
                    "position": rng.choice(["DEF", "MID", "FWD"]),
                    "is_pl_newcomer": int(rng.integers(0, 2)),
                    "is_promoted_club": int(rng.integers(0, 2)),
                    "minutes": minutes,
                    "minutes_bucket": bucket,
                }
            )
    return pl.DataFrame(rows)


def test_cross_validate_returns_per_fold_and_aggregate() -> None:
    """One metric dict per fold, plus mean/std aggregates over shared keys."""
    seasons = ["2022-23", "2023-24", "2024-25"]
    df = _synthetic_model_df(seasons)
    folds = season_folds(seasons)

    per_fold, agg = cross_validate(df, folds)

    assert len(per_fold) == len(folds)
    assert "logloss_60_mean" in agg
    assert "logloss_60_std" in agg
    # Learnable signal -> better-than-chance appearance separation.
    assert agg["logloss_appear_mean"] < 0.69


def test_train_final_fits_on_all_rows() -> None:
    """train_final returns a fitted pipeline that predicts probabilities."""
    df = _synthetic_model_df(["2022-23", "2023-24"])

    model = train_final(df)

    assert isinstance(model, Pipeline)
    proba = model.predict_proba(df.select(FEATURES).to_pandas())
    assert proba.shape[0] == df.height


def test_run_minutes_model_logs_to_mlflow() -> None:
    """run_minutes_model logs params, metrics and the model, returns aggregates."""
    df = _synthetic_model_df(["2022-23", "2023-24", "2024-25"])

    with (
        mock.patch(
            "fantasy_football.modelling.minutes.assemble_model_frame",
            return_value=df,
        ),
        mock.patch("fantasy_football.modelling.minutes.mlflow") as mlflow_mock,
    ):
        mlflow_mock.start_run.return_value.__enter__ = mock.Mock()
        mlflow_mock.start_run.return_value.__exit__ = mock.Mock(
            return_value=False
        )

        agg = run_minutes_model()

    assert "logloss_60_mean" in agg
    mlflow_mock.set_tracking_uri.assert_called_once_with(MLFLOW_TRACKING_URI)
    mlflow_mock.set_experiment.assert_called_once()
    mlflow_mock.log_params.assert_called_once()
    assert mlflow_mock.log_metric.called  # per-fold metrics
    mlflow_mock.log_metrics.assert_called_once_with(agg)
    mlflow_mock.sklearn.log_model.assert_called_once()
    _, log_model_kwargs = mlflow_mock.sklearn.log_model.call_args
    assert (
        log_model_kwargs["registered_model_name"] == MINUTES_REGISTERED_MODEL
    )


def test_build_model_frame_double_gameweek_yields_two_rows() -> None:
    """A DGW (two opponents same gw/element) produces two model-frame rows."""
    feature_frame = build_feature_frame(
        _player_week(),
        _availability(),
        _history_player_match(),
        _player_season(),
        _team_fixture(),
    )

    # Element 1 plays twice in GW1 (opponents 10 and 20 — a double gameweek).
    player_match = pl.DataFrame(
        {
            "season": ["2022-23", "2022-23"],
            "gw": [1, 1],
            "element": [1, 1],
            "opponent": [10, 20],
            "is_home": [True, False],
            "minutes": [90, 90],
            "total_points": [6, 6],
        }
    )

    model_frame = build_model_frame(player_match, feature_frame)

    element_1_rows = model_frame.filter(pl.col("element") == 1)
    assert element_1_rows.height == 2
    assert element_1_rows["minutes_bucket"].to_list() == [
        BUCKET_SIXTY_PLUS,
        BUCKET_SIXTY_PLUS,
    ]
    # Both rows share the same player-week features.
    ranks = element_1_rows["pos_value_rank"].to_list()
    assert ranks[0] == ranks[1]


def test_build_model_frame_carries_opponent_for_match_grain() -> None:
    """Opponent survives the join so predictions key by match on a DGW."""
    feature_frame = build_feature_frame(
        _player_week(),
        _availability(),
        _history_player_match(),
        _player_season(),
        _team_fixture(),
    )
    player_match = pl.DataFrame(
        {
            "season": ["2022-23", "2022-23"],
            "gw": [1, 1],
            "element": [1, 1],
            "opponent": [10, 20],
            "is_home": [True, False],
            "minutes": [90, 45],
            "total_points": [6, 2],
        }
    )

    model_frame = build_model_frame(player_match, feature_frame)

    assert "opponent" in model_frame.columns
    element_1 = model_frame.filter(pl.col("element") == 1).sort("opponent")
    assert element_1["opponent"].to_list() == [10, 20]


from types import SimpleNamespace  # noqa: E402

from mlflow.exceptions import MlflowException  # noqa: E402

from fantasy_football.modelling.minutes import (  # noqa: E402
    get_production_model,
    score_minutes,
)


class _StubModel:
    """Minimal stand-in for a fitted pipeline: fixed classes and proba."""

    def __init__(self, classes: list[str], proba: np.ndarray) -> None:
        self.classes_ = np.asarray(classes)
        self._proba = proba

    def predict_proba(self, _x: object) -> np.ndarray:
        return self._proba


def _scoring_frame() -> pl.DataFrame:
    rows = {
        "season": ["2024-25", "2024-25"],
        "gw": [1, 1],
        "element": [5, 6],
        "opponent": [12, 12],
    }
    for feature in FEATURES:
        rows[feature] = [1.0, 2.0] if feature != "position" else ["MID", "FWD"]
    return pl.DataFrame(rows)


def test_score_minutes_full_three_classes() -> None:
    """All three classes present: probs preserved, expected minutes derived."""
    proba = np.array([[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]])
    model = _StubModel([BUCKET_ZERO, BUCKET_PARTIAL, BUCKET_SIXTY_PLUS], proba)

    out = score_minutes(_scoring_frame(), model)  # type: ignore

    assert out["p_zero"].to_list() == [0.1, 0.5]
    assert out["p_partial"].to_list() == [0.2, 0.3]
    assert out["p_sixty_plus"].to_list() == [0.7, 0.2]
    # expected_minutes = p_partial*30 + p_sixty_plus*75
    assert out["expected_minutes"].to_list() == [
        0.2 * 30 + 0.7 * 75,
        0.3 * 30 + 0.2 * 75,
    ]
    for key in ["season", "gw", "element", "opponent"]:
        assert key in out.columns


def test_score_minutes_missing_class_gives_zero_column() -> None:
    """A class absent from classes_ yields a zero probability column."""
    # classes_ omits BUCKET_PARTIAL — proba has two columns.
    proba = np.array([[0.3, 0.7]])
    model = _StubModel([BUCKET_ZERO, BUCKET_SIXTY_PLUS], proba)
    frame = _scoring_frame().head(1)

    out = score_minutes(frame, model)  # type: ignore

    assert out["p_partial"].to_list() == [0.0]
    assert out["p_zero"].to_list() == [0.3]
    assert out["p_sixty_plus"].to_list() == [0.7]
    assert out["expected_minutes"].to_list() == [0.7 * 75]


def test_get_production_model_returns_version_and_model() -> None:
    """get_production_model returns the aliased version string and model."""
    stub_model = _StubModel([BUCKET_ZERO], np.array([[1.0]]))
    with mock.patch(
        "fantasy_football.modelling.minutes.mlflow"
    ) as mlflow_mock:
        client = mlflow_mock.tracking.MlflowClient.return_value
        client.get_model_version_by_alias.return_value = SimpleNamespace(
            version="7"
        )
        mlflow_mock.sklearn.load_model.return_value = stub_model
        version, model = get_production_model()

    assert version == "7"
    assert model is stub_model


def test_get_production_model_raises_when_alias_missing() -> None:
    """A missing alias/registered model now propagates, not swallowed to None.

    Callers (e.g. ``backfill_minutes``) are expected to let this fail loudly
    rather than silently skip, so the exception must surface unchanged.
    """
    with mock.patch(
        "fantasy_football.modelling.minutes.mlflow"
    ) as mlflow_mock:
        client = mlflow_mock.tracking.MlflowClient.return_value
        client.get_model_version_by_alias.side_effect = MlflowException(
            "no such alias"
        )

        with pytest.raises(MlflowException):
            get_production_model()


from fantasy_football.constants import CURRENT_SEASON  # noqa: E402
from fantasy_football.modelling.minutes import backfill_minutes  # noqa: E402
from fantasy_football.storage.database import get_connection  # noqa: E402
from fantasy_football.storage.tables import MINUTES_PREDICTION  # noqa: E402


class _ConstantModel:
    """Returns the same 3-class probabilities for every input row."""

    def __init__(self) -> None:
        self.classes_ = np.asarray(
            [BUCKET_ZERO, BUCKET_PARTIAL, BUCKET_SIXTY_PLUS]
        )

    def predict_proba(self, x: object) -> np.ndarray:
        return np.tile([0.1, 0.2, 0.7], (len(x), 1))  # type: ignore


def _backfill_model_frame() -> pl.DataFrame:
    """Return a frame spanning one historic season and the current season."""
    seasons = ["2024-25", CURRENT_SEASON]
    rows = []
    for season in seasons:
        row = {
            "season": season,
            "gw": 1,
            "element": 5,
            "opponent": 12,
            "minutes": 90,
            "minutes_bucket": BUCKET_SIXTY_PLUS,
        }
        for feature in FEATURES:
            row[feature] = 1.0 if feature != "position" else "MID"
        rows.append(row)
    return pl.DataFrame(rows)


def _patched_backfill(tmp_path, prod_version, db_name="bf.duckdb"):
    """Context helpers: patched deps around a real temp DuckDB."""
    db_path = tmp_path / db_name
    return (
        db_path,
        mock.patch(
            "fantasy_football.modelling.minutes.get_production_model",
            return_value=(prod_version, _ConstantModel()),
        ),
        mock.patch(
            "fantasy_football.modelling.minutes.assemble_model_frame",
            return_value=_backfill_model_frame(),
        ),
        mock.patch(
            "fantasy_football.modelling.minutes.get_connection",
            side_effect=lambda: get_connection(db_path),
        ),
    )


def test_backfill_minutes_populates_all_seasons_when_empty(tmp_path) -> None:
    """First backfill scores every season and stamps the production version."""
    db_path, p_ver, p_frame, p_conn = _patched_backfill(
        tmp_path, prod_version="2"
    )
    with p_ver, p_frame, p_conn:
        backfill_minutes()

    conn = get_connection(db_path)
    try:
        out = MINUTES_PREDICTION.load(conn)
    finally:
        conn.close()
    assert set(out["season"].to_list()) == {"2024-25", CURRENT_SEASON}
    assert set(out["model_version"].to_list()) == {"2"}


def test_backfill_minutes_skips_historic_when_version_matches(
    tmp_path,
) -> None:
    """Matching version leaves historic rows untouched, refreshes current."""
    db_path, p_ver, p_frame, p_conn = _patched_backfill(
        tmp_path, prod_version="2"
    )
    # Pre-seed historic season with a sentinel expected_minutes at version 2.
    seed_conn = get_connection(db_path)
    try:
        seeded = pl.DataFrame(
            [
                {
                    "season": "2024-25",
                    "gw": 1,
                    "element": 5,
                    "opponent": 12,
                    "p_zero": 0.0,
                    "p_partial": 0.0,
                    "p_sixty_plus": 1.0,
                    "expected_minutes": 999.0,
                    "model_version": "2",
                }
            ]
        )
        MINUTES_PREDICTION.upsert_current(seed_conn, seeded, "2024-25")
    finally:
        seed_conn.close()

    with p_ver, p_frame, p_conn:
        backfill_minutes()

    conn = get_connection(db_path)
    try:
        out = MINUTES_PREDICTION.load(conn)
    finally:
        conn.close()
    historic = out.filter(pl.col("season") == "2024-25")
    # Untouched sentinel proves historic was not rewritten.
    assert historic["expected_minutes"].to_list() == [999.0]
    # Current season still scored.
    assert out.filter(pl.col("season") == CURRENT_SEASON).height == 1


def test_backfill_minutes_rebuilds_historic_on_version_change(
    tmp_path,
) -> None:
    """A new production version triggers a full historic rewrite."""
    db_path, p_ver, p_frame, p_conn = _patched_backfill(
        tmp_path, prod_version="3"
    )
    seed_conn = get_connection(db_path)
    try:
        seeded = pl.DataFrame(
            [
                {
                    "season": "2024-25",
                    "gw": 1,
                    "element": 5,
                    "opponent": 12,
                    "p_zero": 0.0,
                    "p_partial": 0.0,
                    "p_sixty_plus": 1.0,
                    "expected_minutes": 999.0,
                    "model_version": "2",
                }
            ]
        )
        MINUTES_PREDICTION.upsert_current(seed_conn, seeded, "2024-25")
    finally:
        seed_conn.close()

    with p_ver, p_frame, p_conn:
        backfill_minutes()

    conn = get_connection(db_path)
    try:
        out = MINUTES_PREDICTION.load(conn)
    finally:
        conn.close()
    assert set(out["model_version"].to_list()) == {"3"}
    historic = out.filter(pl.col("season") == "2024-25")
    # Sentinel overwritten by a fresh score.
    assert historic["expected_minutes"].to_list() != [999.0]


def test_num_features_includes_history_and_cold_start() -> None:
    """The history block is actually wired into the model's feature list."""
    from fantasy_football.features.history import (
        COLD_START_FEATURES,
        HISTORY_FEATURES,
    )
    from fantasy_football.modelling.minutes import NUM_FEATURES

    # is_pl_newcomer and is_promoted_club are booleans that live in
    # BOOL_FEATURES instead (passthrough, not median-imputed/scaled) -- see
    # make_pipeline. days_since_team_join is deliberately excluded from the
    # model (see the comment on NUM_FEATURES): it is 100% null in most
    # training seasons and null for every 2026-27 row. Every other
    # history/cold-start feature is continuous and must land in NUM_FEATURES.
    excluded = {"is_pl_newcomer", "is_promoted_club", "days_since_team_join"}
    for feature in HISTORY_FEATURES + COLD_START_FEATURES:
        if feature in excluded:
            continue
        assert feature in NUM_FEATURES, f"{feature} missing from NUM_FEATURES"

    assert "days_since_team_join" not in NUM_FEATURES
    assert "avg_minutes_rolling_5" in NUM_FEATURES
    assert "games_played_this_season" in NUM_FEATURES


def test_build_feature_frame_emits_every_declared_feature() -> None:
    """Every name in FEATURES exists as a column on the built frame."""
    from datetime import datetime

    from fantasy_football.modelling.minutes import (
        FEATURES,
        build_feature_frame,
    )

    player_week = pl.DataFrame(
        {
            "season": ["2023-24", "2023-24"],
            "gw": [1, 2],
            "element": [10, 10],
            "position": ["MID", "MID"],
            "team": ["Liverpool", "Liverpool"],
            "value": [125, 125],
            "minutes": [90, 80],
        }
    )
    availability = pl.DataFrame(
        {
            "season": ["2023-24", "2023-24"],
            "gw": [1, 2],
            "element": [10, 10],
            "chance_of_playing_this_round": [100, 100],
        }
    )
    player_match = pl.DataFrame(
        {
            "season": ["2023-24", "2023-24"],
            "gw": [1, 2],
            "element": [10, 10],
            "opponent": [3, 4],
            "minutes": [90, 80],
            "total_points": [8, 5],
        }
    )
    player_season = pl.DataFrame(
        {
            "season": ["2023-24"],
            "element": [10],
            "player_code": [111],
            "birth_date": [date(1992, 6, 15)],
            "team_join_date": [date(2017, 7, 1)],
        },
        schema_overrides={"birth_date": pl.Date, "team_join_date": pl.Date},
    )
    team_fixture = pl.DataFrame(
        {
            "season": ["2023-24"],
            "gw": [1],
            "team": ["Liverpool"],
            "is_home": [True],
            "opposition": ["Arsenal"],
            "kickoff_time": [datetime(2023, 8, 12, 15, 0)],
        }
    )

    out = build_feature_frame(
        player_week, availability, player_match, player_season, team_fixture
    )

    for feature in FEATURES:
        assert feature in out.columns, f"{feature} missing from feature frame"
