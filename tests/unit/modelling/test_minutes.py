from unittest import mock

import numpy as np
import polars as pl
from sklearn.pipeline import Pipeline

from fantasy_football.constants import MLFLOW_TRACKING_URI
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


def test_build_feature_frame_has_one_row_per_player_week() -> None:
    """Feature frame keeps the player-week grain and the feature columns."""
    frame = build_feature_frame(_player_week(), _availability())

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
    feature_frame = build_feature_frame(_player_week(), _availability())
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
    classes = MINUTES_BUCKETS  # ["0_minutes", "1_to_59_minutes", "60_minutes_plus"]
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
            "position": rng.choice(["GK", "DEF", "MID", "FWD"], n),
            "minutes_bucket": rng.choice(MINUTES_BUCKETS, n),
        }
    )

    pipe = make_pipeline()
    pipe.fit(df.select(FEATURES).to_pandas(), df["minutes_bucket"].to_list())
    proba = pipe.predict_proba(df.select(FEATURES).to_pandas())

    assert proba.shape == (n, len(MINUTES_BUCKETS))


def _synthetic_model_df(seasons: list[str], per_season: int = 60) -> pl.DataFrame:
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
                    "position": rng.choice(["DEF", "MID", "FWD"]),
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


def test_run_minutes_model_returns_empty_when_one_season() -> None:
    """With a single season there are no folds; returns {} without MLflow."""
    df = _synthetic_model_df(["2022-23"])

    with (
        mock.patch(
            "fantasy_football.modelling.minutes.assemble_model_frame",
            return_value=df,
        ),
        mock.patch("fantasy_football.modelling.minutes.mlflow") as mlflow_mock,
    ):
        agg = run_minutes_model()

    assert agg == {}
    mlflow_mock.start_run.assert_not_called()


def test_build_model_frame_double_gameweek_yields_two_rows() -> None:
    """A DGW (two opponents same gw/element) produces two model-frame rows."""
    feature_frame = build_feature_frame(_player_week(), _availability())

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
