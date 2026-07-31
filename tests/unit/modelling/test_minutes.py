import logging
from datetime import date, datetime, timedelta
from unittest import mock

import numpy as np
import polars as pl
import pytest_mock
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
    warn_unidentified_snapshot,
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
    ``build_model_frame``. Also doubles as the rolling-minutes match
    stream inside ``build_feature_frame``, so it carries ``kickoff_time``.
    """
    return pl.DataFrame(
        {
            "season": ["2022-23"] * 3,
            "gw": [1, 1, 1],
            "element": [1, 2, 3],
            "kickoff_time": [datetime(2022, 8, 6, 15, 0)] * 3,
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
    score_forward_minutes,
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


def test_get_production_model_returns_none_without_an_alias(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """The documented no-alias behaviour must actually hold."""
    client = mocker.Mock()
    client.get_model_version_by_alias.side_effect = MlflowException("no alias")
    mocker.patch("mlflow.tracking.MlflowClient", return_value=client)
    mocker.patch("mlflow.set_tracking_uri")

    assert get_production_model() is None


def test_score_forward_minutes_noops_without_an_alias(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """No production model means no forward rows and no exception."""
    mocker.patch(
        "fantasy_football.modelling.minutes.get_production_model",
        return_value=None,
    )
    assemble = mocker.patch(
        "fantasy_football.modelling.minutes.assemble_model_frame"
    )

    score_forward_minutes()

    assemble.assert_not_called()


import duckdb  # noqa: E402

from fantasy_football.constants import CURRENT_SEASON  # noqa: E402
from fantasy_football.modelling.minutes import (
    _score_and_store,  # noqa: E402
    backfill_minutes,  # noqa: E402
)
from fantasy_football.storage.database import get_connection  # noqa: E402
from fantasy_football.storage.tables import (  # noqa: E402
    MINUTES_PREDICTION,
    PLAYER_SNAPSHOT,
    PLAYER_WEEK,
    TEAM_FIXTURE,
)


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
                    "prediction_kind": "backfill",
                    "snapshot_captured_at": None,
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
                    "prediction_kind": "backfill",
                    "snapshot_captured_at": None,
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


def test_score_forward_minutes_freezes_already_played_gameweeks(
    tmp_path, mocker: pytest_mock.MockerFixture
) -> None:
    """A played gameweek's stored forecast survives a fresh forward score.

    GW1 has been played (player_week has a row for it), so
    ``from_gw = last_played_gw + 1 == 2``. A forward row is pre-seeded at
    GW1 -- below that floor -- as a sentinel. If ``replace_partition``
    were called without ``gw_from`` (or with the wrong value), the delete
    would remove every stored forward row for the season before
    inserting only the fresh GW2 rows, destroying the GW1 sentinel. This
    test fails under that regression and passes only when the freeze
    rule is actually honoured.
    """
    db_path = tmp_path / "forward.duckdb"
    connection = get_connection(db_path)
    try:
        PLAYER_WEEK.append(
            connection,
            pl.DataFrame(
                {
                    "season": [CURRENT_SEASON],
                    "gw": [1],
                    "element": [5],
                    "name": ["Test Player"],
                    "position": ["MID"],
                    "team": ["Arsenal"],
                    "bonus": [0],
                    "minutes": [90],
                    "round": [1],
                    "total_points": [6],
                    "value": [100],
                }
            ),
        )
        TEAM_FIXTURE.append(
            connection,
            pl.DataFrame(
                {
                    "season": [CURRENT_SEASON, CURRENT_SEASON],
                    "gw": [1, 2],
                    "team": ["Arsenal", "Arsenal"],
                    "is_home": [True, False],
                    "opposition": ["Everton", "Chelsea"],
                    "kickoff_time": [
                        datetime(2026, 8, 15, 15, 0),
                        datetime(2026, 8, 22, 15, 0),
                    ],
                }
            ),
        )
        PLAYER_SNAPSHOT.append(
            connection,
            pl.DataFrame(
                {
                    "season": [CURRENT_SEASON],
                    "captured_at": [datetime(2026, 8, 20, 9, 0)],
                    "element": [5],
                    "value": [100],
                    "team": ["Arsenal"],
                    "position": ["MID"],
                    "chance_of_playing_this_round": [100],
                }
            ),
        )
        # Sentinel: a forward row already stored for GW1 -- below the
        # floor -- must survive untouched.
        MINUTES_PREDICTION.append(
            connection,
            pl.DataFrame(
                {
                    "season": [CURRENT_SEASON],
                    "gw": [1],
                    "element": [5],
                    "opponent": [8],
                    "p_zero": [0.0],
                    "p_partial": [0.0],
                    "p_sixty_plus": [1.0],
                    "expected_minutes": [999.0],
                    "model_version": ["1"],
                    "prediction_kind": ["forward"],
                    "snapshot_captured_at": [datetime(2026, 8, 13, 9, 0)],
                }
            ),
        )
    finally:
        connection.close()

    chelsea = SimpleNamespace(id=8, code=8, name="Chelsea", short_name="CHE")
    everton = SimpleNamespace(id=7, code=7, name="Everton", short_name="EVE")
    mocker.patch(
        "fantasy_football.modelling.minutes.get_production_model",
        return_value=("2", _ConstantModel()),
    )
    mocker.patch(
        "fantasy_football.modelling.minutes.get_connection",
        side_effect=lambda: get_connection(db_path),
    )
    fpl_api = mocker.Mock()
    fpl_api.get_teams.return_value = [chelsea, everton]
    mocker.patch(
        "fantasy_football.modelling.minutes.FplAPI", return_value=fpl_api
    )

    score_forward_minutes()

    conn = get_connection(db_path)
    try:
        out = MINUTES_PREDICTION.load(conn)
    finally:
        conn.close()

    gw1 = out.filter(pl.col("gw") == 1)
    # The sentinel below the floor was never touched.
    assert gw1["expected_minutes"].to_list() == [999.0]
    assert gw1["model_version"].to_list() == ["1"]
    # GW2 -- the unplayed fixture -- was freshly scored.
    gw2 = out.filter(pl.col("gw") == 2)
    assert gw2.height == 1
    assert gw2["model_version"].to_list() == ["2"]
    assert gw2["prediction_kind"].to_list() == ["forward"]


def _model_frame_fixture(season: str = "2025-26") -> pl.DataFrame:
    """Build a minimal model frame: identifiers plus every FEATURES column."""
    rows = {
        "season": [season, season],
        "gw": [1, 1],
        "element": [5, 6],
        "opponent": [12, 12],
    }
    for feature in FEATURES:
        rows[feature] = [1.0, 2.0] if feature != "position" else ["MID", "FWD"]
    return pl.DataFrame(rows)


def test_backfill_rows_are_stamped_as_backfill(
    db: duckdb.DuckDBPyConnection, mocker: pytest_mock.MockerFixture
) -> None:
    """Backfilled rows must be distinguishable from forward forecasts."""
    frame = _model_frame_fixture()
    model = mocker.Mock()
    model.predict_proba.return_value = np.tile(
        [0.1, 0.2, 0.7], (frame.height, 1)
    )
    model.classes_ = MINUTES_BUCKETS

    _score_and_store(db, frame, "2025-26", model, "3")

    stored = MINUTES_PREDICTION.load(db)
    assert set(stored["prediction_kind"].to_list()) == {"backfill"}
    assert stored["snapshot_captured_at"].null_count() == stored.height


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
            "kickoff_time": [
                datetime(2023, 8, 12, 15, 0),
                datetime(2023, 8, 19, 15, 0),
            ],
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


def _forward_stream_player_week(gws: list[int]) -> pl.DataFrame:
    """Player-week rows for one player across the given gameweeks."""
    n = len(gws)
    return pl.DataFrame(
        {
            "season": ["2026-27"] * n,
            "gw": gws,
            "element": [1] * n,
            "name": ["A"] * n,
            "position": ["MID"] * n,
            "team": ["Arsenal"] * n,
            "bonus": [0] * n,
            "minutes": [None] * n,
            "round": gws,
            "total_points": [None] * n,
            "value": [70] * n,
        },
        schema_overrides={"minutes": pl.Int64, "total_points": pl.Int64},
    )


def test_build_feature_frame_freezes_rolling_minutes_across_forward_gws() -> (
    None
):
    """Forward gameweeks keep the frozen window at an early *and* a late gw.

    The stream mixes three played matches with eleven unplayed fixtures.
    A shifted window over the combined stream decays after the fifth
    unplayed fixture and goes null soon after, so GW6+ would silently lose
    ``avg_minutes_rolling_5`` in a pre-season run -- the exact case forward
    scoring exists to serve. Both GW5 (early) and GW14 (late) must carry
    the mean of the played matches.
    """
    played_gws = [1, 2, 3]
    forward_gws = list(range(4, 15))
    base = datetime(2026, 8, 1, 15, 0)
    played_match = pl.DataFrame(
        {
            "season": ["2026-27"] * 3,
            "gw": played_gws,
            "element": [1] * 3,
            "opponent": [7, 8, 9],
            "kickoff_time": [
                base + timedelta(weeks=i) for i in range(len(played_gws))
            ],
            "minutes": [60, 90, 45],
            "total_points": [3, 6, 1],
        }
    )
    forward_fixtures = pl.DataFrame(
        {
            "season": ["2026-27"] * len(forward_gws),
            "gw": forward_gws,
            "element": [1] * len(forward_gws),
            "opponent": [10] * len(forward_gws),
            "kickoff_time": [
                base + timedelta(weeks=3 + i) for i in range(len(forward_gws))
            ],
            "minutes": [None] * len(forward_gws),
        },
        schema_overrides={"minutes": pl.Int64},
    )
    player_week = pl.concat(
        [
            _forward_stream_player_week(played_gws).with_columns(
                pl.Series("minutes", [60, 90, 45], dtype=pl.Int64)
            ),
            _forward_stream_player_week(forward_gws),
        ],
        how="vertical",
    )
    availability = pl.DataFrame(
        {
            "season": ["2026-27"],
            "gw": [1],
            "element": [1],
            "chance_of_playing_this_round": [100],
        }
    )
    player_season = pl.DataFrame(
        {
            "season": ["2026-27"],
            "element": [1],
            "player_code": [999],
            "birth_date": [date(1995, 1, 1)],
            "team_join_date": [date(2020, 1, 1)],
        },
        schema_overrides={"birth_date": pl.Date, "team_join_date": pl.Date},
    )
    team_fixture = pl.DataFrame(
        {
            "season": ["2026-27", "2025-26"],
            "gw": [1, 1],
            "team": ["Arsenal", "Arsenal"],
        }
    )

    frame = build_feature_frame(
        player_week,
        availability,
        played_match,
        player_season,
        team_fixture,
        forward_fixtures=forward_fixtures,
    )

    by_gw = {
        row["gw"]: row["avg_minutes_rolling_5"]
        for row in frame.iter_rows(named=True)
    }
    # (60 + 90 + 45) / 3 = 65.0, frozen at the last played match.
    assert by_gw[5] == 65.0  # early forward gameweek
    assert by_gw[14] == 65.0  # late forward gameweek
    assert all(by_gw[gw] == 65.0 for gw in forward_gws)


def test_warn_unidentified_snapshot_reports_the_coverage_gap(caplog) -> None:
    """Snapshot elements with no player_season row are counted and logged."""
    snapshot = pl.DataFrame(
        {
            "season": ["2026-27"] * 3,
            "element": [1, 2, 3],
        }
    )
    player_season = pl.DataFrame(
        {
            "season": ["2026-27", "2026-27"],
            "element": [1, 2],
            "player_code": [111, None],
        },
        schema_overrides={"player_code": pl.Int64},
    )

    with caplog.at_level(logging.WARNING):
        missing = warn_unidentified_snapshot(
            snapshot, player_season, "2026-27"
        )

    # Element 2 has a row but a null player_code, so it is unusable too.
    assert missing == 2
    assert "2 of 3" in caplog.text


def test_warn_unidentified_snapshot_silent_when_fully_covered(
    caplog,
) -> None:
    """Full identity coverage logs nothing and reports zero."""
    snapshot = pl.DataFrame({"season": ["2026-27"], "element": [1]})
    player_season = pl.DataFrame(
        {"season": ["2026-27"], "element": [1], "player_code": [111]}
    )

    with caplog.at_level(logging.WARNING):
        missing = warn_unidentified_snapshot(
            snapshot, player_season, "2026-27"
        )

    assert missing == 0
    assert caplog.text == ""
