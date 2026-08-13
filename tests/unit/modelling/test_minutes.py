import json
from datetime import date, datetime, timedelta
from typing import cast

import duckdb
import numpy as np
import polars as pl
import pytest
from sklearn.pipeline import Pipeline

from fantasy_football.modelling.folds import (
    Fold,
    FoldTestKey,
    SeasonFoldStrategy,
)
from fantasy_football.modelling.metrics import MinutesMetrics
from fantasy_football.modelling.minutes import (
    BUCKET_PARTIAL,
    BUCKET_SIXTY_PLUS,
    BUCKET_ZERO,
    FEATURES,
    MINUTES_BUCKETS,
    MINUTES_SPEC,
    MinutesPredictor,
    boundary_metrics,
    build_feature_frame,
    build_model_frame,
    create_minutes_bucket,
    make_pipeline,
    score_minutes,
)
from fantasy_football.storage.tables import (
    MINUTES_PREDICTION,
    PLAYER_AVAILABILITY,
    PLAYER_MATCH,
    PLAYER_SEASON,
    PLAYER_WEEK,
    TEAM_FIXTURE,
    TEST_MINUTES_PREDICTION,
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


def test_boundary_metrics_perfect_predictions() -> None:
    """Confident, correct probabilities give ~0 loss and AUC 1.0."""
    classes = (
        MINUTES_BUCKETS  # ["0_minutes", "1_to_59_minutes", "60_minutes_plus"]
    )
    y_true = [BUCKET_ZERO, BUCKET_SIXTY_PLUS]
    true_minutes = [0.0, 90.0]
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
    true_minutes = [90.0, 75.0]
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
        for index in range(per_season):
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
                    # Match-grain keys, carried so scored rows can be
                    # stored the way the real model frame allows.
                    "gw": index + 1,
                    "element": index + 1,
                    "opponent": 1,
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


class _StubModel:
    """Minimal stand-in for a fitted pipeline: fixed classes and proba."""

    def __init__(self, classes: list[str], proba: np.ndarray) -> None:
        self.classes_ = np.asarray(classes)
        self._proba = proba

    def predict_proba(self, _x: object) -> np.ndarray:
        return self._proba


def _stub_model(classes: list[str], proba: np.ndarray) -> Pipeline:
    """Return a ``_StubModel`` typed as the ``Pipeline`` callers declare.

    ``score_minutes`` and ``build_prediction_rows`` annotate their model as
    a fitted ``Pipeline`` but only ever touch ``predict_proba`` and
    ``classes_``, both of which the stub provides. Stating that once here
    keeps the call sites free of type-checker suppressions.
    """
    return cast(Pipeline, _StubModel(classes, proba))


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
    model = _stub_model(
        [BUCKET_ZERO, BUCKET_PARTIAL, BUCKET_SIXTY_PLUS], proba
    )

    out = score_minutes(_scoring_frame(), model)

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
    model = _stub_model([BUCKET_ZERO, BUCKET_SIXTY_PLUS], proba)
    frame = _scoring_frame().head(1)

    out = score_minutes(frame, model)

    assert out["p_partial"].to_list() == [0.0]
    assert out["p_zero"].to_list() == [0.3]
    assert out["p_sixty_plus"].to_list() == [0.7]
    assert out["expected_minutes"].to_list() == [0.7 * 75]


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


SEEDED_SEASON = "2024-25"


def _seed_minutes_tables(connection: duckdb.DuckDBPyConnection) -> None:
    """Write the five source tables ``MinutesPredictor`` reads.

    Two Arsenal midfielders over two played gameweeks, plus an unplayed
    GW3 fixture for the forward-data path to reach.
    """
    kickoffs = [datetime(2024, 8, 17, 15, 0), datetime(2024, 8, 24, 15, 0)]
    PLAYER_WEEK.append(
        connection,
        pl.DataFrame(
            {
                "season": [SEEDED_SEASON] * 4,
                "gw": [1, 1, 2, 2],
                "element": [5, 6, 5, 6],
                "name": ["A", "B", "A", "B"],
                "position": ["MID"] * 4,
                "team": ["Arsenal"] * 4,
                "bonus": [0] * 4,
                "minutes": [90, 0, 75, 20],
                "round": [1, 1, 2, 2],
                "total_points": [6, 0, 5, 1],
                "value": [70, 50, 70, 50],
            }
        ),
    )
    PLAYER_MATCH.append(
        connection,
        pl.DataFrame(
            {
                "season": [SEEDED_SEASON] * 4,
                "gw": [1, 1, 2, 2],
                "element": [5, 6, 5, 6],
                "opponent": [12, 12, 13, 13],
                "is_home": [True, True, False, False],
                "minutes": [90, 0, 75, 20],
                "total_points": [6, 0, 5, 1],
                "kickoff_time": [kickoffs[0]] * 2 + [kickoffs[1]] * 2,
            }
        ),
    )
    PLAYER_AVAILABILITY.append(
        connection,
        pl.DataFrame(
            {
                "season": [SEEDED_SEASON] * 4,
                "gw": [1, 1, 2, 2],
                "element": [5, 6, 5, 6],
                "chance_of_playing_this_round": [100, 75, 100, 100],
            }
        ),
    )
    PLAYER_SEASON.append(
        connection,
        pl.DataFrame(
            {
                "season": [SEEDED_SEASON] * 2,
                "element": [5, 6],
                "player_code": [101, 102],
                "web_name": ["A", "B"],
                "first_name": ["Player", "Player"],
                "second_name": ["A", "B"],
                "position": ["MID", "MID"],
                "team_code": [3, 3],
                "region": [1, 1],
                "birth_date": [date(1995, 1, 1)] * 2,
                "team_join_date": [date(2020, 1, 1)] * 2,
            },
            schema_overrides={
                "birth_date": pl.Date,
                "team_join_date": pl.Date,
            },
        ),
    )
    TEAM_FIXTURE.append(
        connection,
        pl.DataFrame(
            {
                "season": [SEEDED_SEASON] * 3,
                "gw": [1, 2, 3],
                "team": ["Arsenal"] * 3,
                "is_home": [True, False, True],
                "opposition": ["Everton", "Chelsea", "Fulham"],
                "kickoff_time": [*kickoffs, datetime(2024, 8, 31, 15, 0)],
            }
        ),
    )


@pytest.fixture
def predictor(db: duckdb.DuckDBPyConnection) -> MinutesPredictor:
    """Return a ``MinutesPredictor`` bound to the seeded temporary database."""
    _seed_minutes_tables(db)
    return MinutesPredictor(
        experiment_name="test-minutes",
        params={"model": "logistic_regression"},
        model_spec=MINUTES_SPEC,
        connection=db,
        fold_strategy=SeasonFoldStrategy(),
    )


def test_build_training_data_returns_a_scorable_model_frame(
    predictor: MinutesPredictor,
) -> None:
    """Training data comes back at match grain with the target and features."""
    frame = predictor.build_training_data()

    # One row per stored player_match row -- the target's grain.
    assert frame.height == 4
    for column in ["season", "gw", "element", "opponent", "minutes_bucket"]:
        assert column in frame.columns
    for feature in FEATURES:
        assert feature in frame.columns
    buckets = {
        (row["element"], row["gw"]): row["minutes_bucket"]
        for row in frame.iter_rows(named=True)
    }
    assert buckets[(5, 1)] == BUCKET_SIXTY_PLUS
    assert buckets[(6, 1)] == BUCKET_ZERO
    assert buckets[(6, 2)] == BUCKET_PARTIAL


def test_build_forward_data_gives_every_forward_fixture_features(
    predictor: MinutesPredictor,
) -> None:
    """Each unplayed fixture gets one feature row, keyed to its opponent."""
    forward_fixtures = pl.DataFrame(
        {
            "season": [SEEDED_SEASON] * 2,
            "gw": [3, 3],
            "element": [5, 6],
            "opponent": [14, 14],
            "kickoff_time": [datetime(2024, 8, 31, 15, 0)] * 2,
            "minutes": [None, None],
            "position": ["MID", "MID"],
            "team": ["Arsenal", "Arsenal"],
            "value": [70, 50],
            "chance_of_playing_this_round": [100, 100],
        },
        schema_overrides={"minutes": pl.Int64},
    )

    frame = predictor.build_forward_data(forward_fixtures)

    assert frame.height == 2
    assert frame["gw"].to_list() == [3, 3]
    assert frame["opponent"].to_list() == [14, 14]
    for feature in FEATURES:
        assert feature in frame.columns
    # The rolling window is frozen at the last played match, not null.
    rolling = {
        row["element"]: row["avg_minutes_rolling_5"]
        for row in frame.iter_rows(named=True)
    }
    assert rolling[5] == 82.5  # (90 + 75) / 2


def test_fit_predict_fold_returns_minutes_metrics(
    predictor: MinutesPredictor,
) -> None:
    """One fold scores into a MinutesMetrics carrying every boundary metric."""
    df = _synthetic_model_df(["2022-23", "2023-24"])
    fold = Fold(
        train=df.filter(pl.col("season") == "2022-23"),
        test=df.filter(pl.col("season") == "2023-24"),
        test_key=FoldTestKey(season="2023-24"),
    )

    result = predictor.fit_predict_fold(fold)

    assert isinstance(result.metrics, MinutesMetrics)
    scores = result.metrics.as_dict()
    for key in [
        "logloss_appear",
        "brier_appear",
        "logloss_60",
        "brier_60",
        "e_min_mae",
    ]:
        assert key in scores
    # Learnable signal -> better-than-chance appearance separation.
    assert scores["logloss_appear"] < 0.69


def test_train_final_fits_on_all_rows(predictor: MinutesPredictor) -> None:
    """train_final returns a fitted pipeline that predicts probabilities."""
    df = _synthetic_model_df(["2022-23", "2023-24"])

    model = predictor.train_final(df)

    assert isinstance(model, Pipeline)
    proba = model.predict_proba(df.select(FEATURES).to_pandas())
    assert proba.shape[0] == df.height


def test_build_prediction_rows_matches_the_stored_table(
    predictor: MinutesPredictor,
) -> None:
    """Rows come back in exactly the table's column order, ready to store.

    ``build_prediction_rows`` ends in ``.select(table.columns)``, so a
    column added to ``MINUTES_PREDICTION`` -- or dropped from
    ``score_minutes`` -- breaks here rather than at write time.
    """
    frame = _scoring_frame()
    model = _stub_model(
        MINUTES_BUCKETS, np.tile([0.1, 0.2, 0.7], (frame.height, 1))
    )

    rows = predictor.build_prediction_rows(frame, model, "7", "backfill")

    assert rows.columns == MINUTES_PREDICTION.columns
    assert rows["model_version"].to_list() == ["7", "7"]
    assert rows["prediction_kind"].to_list() == ["backfill", "backfill"]
    assert rows["snapshot_captured_at"].null_count() == rows.height


def test_build_prediction_rows_are_storable(
    predictor: MinutesPredictor, db: duckdb.DuckDBPyConnection
) -> None:
    """The rows survive a real write, proving the schema actually lines up."""
    frame = _scoring_frame()
    model = _stub_model(
        MINUTES_BUCKETS, np.tile([0.1, 0.2, 0.7], (frame.height, 1))
    )

    rows = predictor.build_prediction_rows(frame, model, "7", "forward")
    MINUTES_PREDICTION.append(db, rows)

    stored = MINUTES_PREDICTION.load(db)
    assert stored.height == 2
    expected = 0.2 * 30 + 0.7 * 75
    assert stored["expected_minutes"].to_list() == [expected, expected]


def test_fit_predict_fold_returns_storable_predictions(
    predictor: MinutesPredictor,
) -> None:
    """Scored rows carry both actuals and the model's inputs as JSON."""
    df = _synthetic_model_df(["2022-23", "2023-24"])
    test = df.filter(pl.col("season") == "2023-24")
    fold = Fold(
        train=df.filter(pl.col("season") == "2022-23"),
        test=test,
        test_key=FoldTestKey(season="2023-24"),
    )

    predictions = predictor.fit_predict_fold(fold).predictions

    assert predictions.height == test.height
    assert predictions.columns == [
        column
        for column in TEST_MINUTES_PREDICTION.columns
        if column != "run_id"
    ]
    assert predictions["actual_bucket"].to_list() == (
        test["minutes_bucket"].to_list()
    )
    assert predictions["actual_minutes"].to_list() == test["minutes"].to_list()
    assert sorted(json.loads(predictions["features"][0])) == sorted(FEATURES)


def test_store_fold_predictions_writes_the_minutes_table(
    predictor: MinutesPredictor,
) -> None:
    """A minutes run's scored rows land in its own evaluation table."""
    df = _synthetic_model_df(["2022-23", "2023-24"])
    fold = Fold(
        train=df.filter(pl.col("season") == "2022-23"),
        test=df.filter(pl.col("season") == "2023-24"),
        test_key=FoldTestKey(season="2023-24"),
    )
    results, _ = predictor.cross_validate([fold])

    predictor.store_fold_predictions(results, "run-1")

    stored = TEST_MINUTES_PREDICTION.load(predictor.connection)
    assert stored.height == fold.test.height
    assert stored["run_id"].unique().to_list() == ["run-1"]
