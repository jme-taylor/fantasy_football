from datetime import datetime, timedelta
from types import SimpleNamespace

import duckdb
import numpy as np
import polars as pl
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

from fantasy_football.features.views import register_feature_views
from fantasy_football.modelling import defender
from fantasy_football.modelling.folds import gameweek_folds
from fantasy_football.storage.tables import (
    BACKFILL_KIND,
    FORWARD_KIND,
    MINUTES_PREDICTION,
    PLAYER_MATCH,
    PLAYER_MATCH_OPTA,
    PLAYER_SEASON,
    PLAYER_SNAPSHOT,
    PLAYER_WEEK,
    POINTS_PREDICTION,
    TABLES,
    TEAM_FIXTURE,
)

SEASON = "2025-26"
FIRST_KICKOFF = datetime(2025, 8, 16, 14, 0)

# Real FCI_SLUG_TO_FPL entries, reused from tests/unit/features/test_team_form.py
# and test_match_form.py so the club-name join actually resolves.
UNITED, ARSENAL, SPURS, CHELSEA = "Man Utd", "Arsenal", "Spurs", "Chelsea"
SLUGS = {
    UNITED: "manchester-united",
    ARSENAL: "arsenal",
    SPURS: "tottenham-hotspur",
    CHELSEA: "chelsea",
}
# fpl_team_id only maps a team name to an id when some player_match row
# has that team as its opponent, so every club needs at least one
# player_match row naming it as the opposition -- see the extra Spurs and
# Arsenal player_match rows below, which exist solely to mint the Man Utd
# and Chelsea mappings.
TEAM_IDS = {UNITED: 1, SPURS: 2, ARSENAL: 3, CHELSEA: 4}


@pytest.fixture
def connection():
    """Yield an in-memory DuckDB connection with every table created."""
    conn = duckdb.connect(":memory:")
    for table in TABLES:
        conn.execute(table.ddl)
    yield conn
    conn.close()


def _append(table, connection, rows: list[dict]) -> None:
    """Append dict rows, filling unstated columns with nulls."""
    table.append(connection, table.conform(pl.DataFrame(rows)))


def _opta_row(
    season: str,
    gw: int,
    element: int,
    match_id: str,
    xg: float,
    goals: int,
    conceded: int,
) -> dict:
    """Build a minimal player_match_opta row for one side of a fixture."""
    return {
        "season": season,
        "gw": gw,
        "element": element,
        "match_id": match_id,
        "competition": "prem",
        "minutes_played": 90,
        "xg": xg,
        "goals": goals,
        "team_goals_conceded": conceded,
    }


def _seed_defender_frame(connection: duckdb.DuckDBPyConnection) -> None:
    """Seed a minimal but real row set distinguishing own from opposition.

    Two prior-gameweek fixtures give Man Utd and Arsenal each their own,
    deliberately different, rolling defensive/attacking record before the
    gw2 fixture between them (our target row, element 1, a Man Utd
    defender). Man Utd's own xg_against/goals_against/clean_sheet in gw1
    (vs Spurs) and Arsenal's own xg_for/goals_for in gw1 (vs Chelsea) are
    picked to be unambiguously distinct, so a transposed own/opposition
    join is caught rather than coincidentally passing.
    """
    gw1_kickoff = FIRST_KICKOFF
    gw2_kickoff = FIRST_KICKOFF + timedelta(days=7)
    utd_v_spurs = f"25-26-prem-{SLUGS[UNITED]}-vs-{SLUGS[SPURS]}"
    arsenal_v_chelsea = f"25-26-prem-{SLUGS[ARSENAL]}-vs-{SLUGS[CHELSEA]}"
    utd_v_arsenal = f"25-26-prem-{SLUGS[UNITED]}-vs-{SLUGS[ARSENAL]}"

    _append(
        PLAYER_SEASON,
        connection,
        [{"season": SEASON, "element": 1, "position": "DEF"}],
    )
    _append(
        PLAYER_WEEK,
        connection,
        [
            {
                "season": SEASON,
                "gw": 1,
                "element": 1,
                "position": "DEF",
                "team": UNITED,
            },
            {
                "season": SEASON,
                "gw": 1,
                "element": 2,
                "position": "MID",
                "team": SPURS,
            },
            {
                "season": SEASON,
                "gw": 1,
                "element": 3,
                "position": "MID",
                "team": ARSENAL,
            },
            {
                "season": SEASON,
                "gw": 1,
                "element": 4,
                "position": "MID",
                "team": CHELSEA,
            },
            {
                "season": SEASON,
                "gw": 2,
                "element": 1,
                "position": "DEF",
                "team": UNITED,
            },
            {
                "season": SEASON,
                "gw": 2,
                "element": 3,
                "position": "MID",
                "team": ARSENAL,
            },
        ],
    )
    _append(
        PLAYER_MATCH,
        connection,
        [
            {
                "season": SEASON,
                "gw": 1,
                "element": 1,
                "opponent": TEAM_IDS[SPURS],
                "is_home": True,
                "minutes": 90,
                "kickoff_time": gw1_kickoff,
            },
            {
                "season": SEASON,
                "gw": 2,
                "element": 1,
                "opponent": TEAM_IDS[ARSENAL],
                "is_home": True,
                "minutes": 90,
                "kickoff_time": gw2_kickoff,
            },
            # These two exist only to mint the Man Utd and Chelsea
            # team-id mappings in fpl_team_id (see the TEAM_IDS comment
            # above) -- they play no other role in the test.
            {
                "season": SEASON,
                "gw": 1,
                "element": 2,
                "opponent": TEAM_IDS[UNITED],
                "is_home": False,
                "minutes": 90,
                "kickoff_time": gw1_kickoff,
            },
            {
                "season": SEASON,
                "gw": 1,
                "element": 3,
                "opponent": TEAM_IDS[CHELSEA],
                "is_home": True,
                "minutes": 90,
                "kickoff_time": gw1_kickoff,
            },
        ],
    )
    _append(
        TEAM_FIXTURE,
        connection,
        [
            {
                "season": SEASON,
                "gw": 1,
                "team": UNITED,
                "is_home": True,
                "opposition": SPURS,
                "kickoff_time": gw1_kickoff,
            },
            {
                "season": SEASON,
                "gw": 1,
                "team": SPURS,
                "is_home": False,
                "opposition": UNITED,
                "kickoff_time": gw1_kickoff,
            },
            {
                "season": SEASON,
                "gw": 1,
                "team": ARSENAL,
                "is_home": True,
                "opposition": CHELSEA,
                "kickoff_time": gw1_kickoff,
            },
            {
                "season": SEASON,
                "gw": 1,
                "team": CHELSEA,
                "is_home": False,
                "opposition": ARSENAL,
                "kickoff_time": gw1_kickoff,
            },
            {
                "season": SEASON,
                "gw": 2,
                "team": UNITED,
                "is_home": True,
                "opposition": ARSENAL,
                "kickoff_time": gw2_kickoff,
            },
            {
                "season": SEASON,
                "gw": 2,
                "team": ARSENAL,
                "is_home": False,
                "opposition": UNITED,
                "kickoff_time": gw2_kickoff,
            },
        ],
    )
    _append(
        PLAYER_MATCH_OPTA,
        connection,
        [
            # gw1: Man Utd 2-0 Spurs. Man Utd keep a clean sheet, facing
            # 0.3 xG against.
            _opta_row(SEASON, 1, 1, utd_v_spurs, 2.0, 2, 0),
            _opta_row(SEASON, 1, 2, utd_v_spurs, 0.3, 0, 2),
            # gw1: Arsenal 1-1 Chelsea. Arsenal's own attacking record
            # (1.0 xG for) is what must land in the *opposition* columns
            # of the gw2 fixture below, not the own-team columns.
            _opta_row(SEASON, 1, 3, arsenal_v_chelsea, 1.0, 1, 1),
            _opta_row(SEASON, 1, 4, arsenal_v_chelsea, 0.5, 1, 1),
            # gw2: Man Utd vs Arsenal, the target row. Values here are
            # irrelevant to the rolling columns -- the window excludes
            # the current match by construction.
            _opta_row(SEASON, 2, 1, utd_v_arsenal, 1.5, 1, 1),
            _opta_row(SEASON, 2, 3, utd_v_arsenal, 0.8, 1, 1),
        ],
    )


def test_features_exclude_keys_and_target() -> None:
    """Neither the target nor a key column leaks into FEATURES."""
    assert defender.TARGET not in defender.FEATURES
    for key in defender.KEY_COLUMNS:
        assert key not in defender.FEATURES


def test_model_frame_sql_filters_to_defenders_and_played_matches() -> None:
    """The SQL restricts to defenders and rows with recorded minutes."""
    sql = defender.model_frame_sql()
    assert "s.position = 'DEF'" in sql
    assert "m.minutes IS NOT NULL" in sql


def test_build_model_frame_returns_keys_target_and_features(
    connection,
) -> None:
    """The frame's columns are keys, then target, then features."""
    frame = defender.build_model_frame(connection)
    expected = defender.KEY_COLUMNS + [defender.TARGET] + defender.FEATURES
    assert frame.columns == expected


def test_pipeline_imputes_and_has_no_scaler() -> None:
    """The pipeline median-imputes into a random forest, with no scaler."""
    pipe = defender.make_pipeline()
    steps = dict(pipe.named_steps)
    assert isinstance(steps["impute"], SimpleImputer)
    assert steps["impute"].strategy == "median"
    assert isinstance(steps["model"], RandomForestRegressor)
    assert steps["model"].n_estimators == defender.N_ESTIMATORS
    assert not any("scale" in name for name in steps)


def test_pipeline_fits_and_predicts_with_nulls() -> None:
    """The pipeline fits and predicts on a frame containing null features."""
    pipe = defender.make_pipeline()
    x = pl.DataFrame(
        {name: [1.0, None, 3.0] for name in defender.FEATURES}
    ).to_pandas()
    y = [1.0, 2.0, 3.0]
    pipe.fit(x, y)
    assert len(pipe.predict(x)) == 3


def test_own_team_and_opposition_form_are_not_swapped(connection) -> None:
    """The gw2 row carries Man Utd's own record, not Arsenal's.

    A cartesian fan-out is caught by the row count; a transposed
    own/opposition join (``own.team = opp_id.team`` instead of
    ``own.team = pw.team``, and vice versa) is caught because Man Utd's
    and Arsenal's gw1 records are deliberately different -- if the two
    sides were swapped, this would fail rather than coincidentally pass.
    """
    _seed_defender_frame(connection)
    frame = defender.build_model_frame(connection)
    row = frame.filter(pl.col("gw") == 2)

    assert row.height == 1

    # Man Utd's own gw1 record (vs Spurs): 0.3 xG conceded, 0 goals
    # conceded, a clean sheet kept.
    assert row["xg_against_rolling_5"].item() == pytest.approx(0.3)
    assert row["goals_against_rolling_5"].item() == pytest.approx(0.0)
    assert row["clean_sheet_rolling_5"].item() == pytest.approx(1.0)

    # Arsenal's own gw1 record (vs Chelsea): 1.0 xG for, 1 goal for.
    assert row["xg_for_rolling_5"].item() == pytest.approx(1.0)
    assert row["goals_for_rolling_5"].item() == pytest.approx(1.0)


def _synthetic_frame(
    n_gws: int = 14, n_players: int = 12, season: str = "2026-27"
) -> pl.DataFrame:
    """Build a synthetic model frame spanning several gameweeks."""
    rows = []
    for gw in range(1, n_gws + 1):
        for element in range(1, n_players + 1):
            rows.append(
                {
                    "season": season,
                    "gw": gw,
                    "element": element,
                    "opponent": (element % 5) + 1,
                    defender.TARGET: float(element % 7),
                    **{
                        name: float(element + gw) for name in defender.FEATURES
                    },
                }
            )
    return pl.DataFrame(rows).select(
        defender.KEY_COLUMNS + [defender.TARGET] + defender.FEATURES
    )


def test_fold_metrics_reports_every_expected_metric():
    """fold_metrics returns exactly the five expected metric keys."""
    test_df = _synthetic_frame(n_gws=1)
    predicted = [1.0] * test_df.height
    metrics = defender.fold_metrics(test_df, predicted)
    assert set(metrics) == {
        "mae",
        "rmse",
        "skill_score",
        "spearman",
        "precision_at_k",
    }


def test_fold_metrics_perfect_predictions_are_zero_error():
    """Exact predictions give mae == 0.0 and rmse == 0.0, not swapped."""
    test_df = _synthetic_frame(n_gws=1)
    predicted = test_df[defender.TARGET].to_list()
    metrics = defender.fold_metrics(test_df, predicted)
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0


def test_fold_metrics_constant_mean_prediction_has_zero_skill():
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
    test_df = _synthetic_frame(n_gws=1)
    mean_target = float(np.mean(test_df[defender.TARGET].to_list()))
    predicted = [mean_target] * test_df.height
    metrics = defender.fold_metrics(test_df, predicted)
    assert metrics["mae"] == pytest.approx(1.5)
    assert metrics["rmse"] == pytest.approx(1.7795130420052185)
    assert metrics["skill_score"] == pytest.approx(0.0)


def test_fold_metrics_perfect_and_reversed_ranking_spearman():
    """A perfectly ordered prediction scores spearman == 1.0.

    Feeding the actual target values back in as the "prediction"
    guarantees an identical ordering (so it is perfectly rank-
    correlated with itself) regardless of repeats in the target, and
    negating them reverses every pairwise order.
    """
    test_df = _synthetic_frame(n_gws=1)
    actual = test_df[defender.TARGET].to_list()
    perfect = defender.fold_metrics(test_df, actual)
    assert perfect["spearman"] == pytest.approx(1.0)

    reversed_pred = [-value for value in actual]
    reversed_metrics = defender.fold_metrics(test_df, reversed_pred)
    assert reversed_metrics["spearman"] == pytest.approx(-1.0)


def test_cross_validate_returns_per_fold_and_aggregates():
    """cross_validate returns one metric dict per fold plus aggregates."""
    model_df = _synthetic_frame()
    folds = gameweek_folds(
        list(zip(model_df["season"], model_df["gw"])), min_train_gws=10
    )
    per_fold, agg = defender.cross_validate(model_df, folds)
    assert len(per_fold) == len(folds) == 4
    assert "mae_mean" in agg
    assert "mae_std" in agg


def test_cross_validate_handles_no_folds():
    """No folds yields empty per-fold and aggregate results."""
    per_fold, agg = defender.cross_validate(_synthetic_frame(), [])
    assert per_fold == []
    assert agg == {}


def test_train_final_fits_on_every_row():
    """train_final fits a pipeline that predicts every row in the frame."""
    model_df = _synthetic_frame()
    pipe = defender.train_final(model_df)
    predicted = pipe.predict(model_df.select(defender.FEATURES).to_pandas())
    assert len(predicted) == model_df.height


def test_run_defender_model_logs_and_registers(mocker):
    """run_defender_model logs metrics and registers the model to MLflow."""
    model_df = _synthetic_frame()
    mocker.patch.object(defender, "get_connection")
    mocker.patch.object(defender, "build_model_frame", return_value=model_df)
    mocker.patch.object(defender.mlflow, "set_tracking_uri")
    mocker.patch.object(defender.mlflow, "set_experiment")
    mocker.patch.object(defender.mlflow, "start_run")
    mocker.patch.object(defender.mlflow, "log_params")
    mocker.patch.object(defender.mlflow, "log_metric")
    log_metrics = mocker.patch.object(defender.mlflow, "log_metrics")
    log_model = mocker.patch.object(defender.mlflow.sklearn, "log_model")

    agg = defender.run_defender_model()

    assert "mae_mean" in agg
    log_metrics.assert_called_once()
    assert (
        log_model.call_args.kwargs["registered_model_name"]
        == defender.REGISTERED_MODEL
    )


def test_score_defender_points_shapes_rows_for_storage():
    """score_defender_points shapes rows for POINTS_PREDICTION storage."""
    frame = _synthetic_frame(n_gws=1, n_players=3)
    pipe = defender.train_final(frame)
    scored = defender.score_defender_points(frame, pipe, "4", BACKFILL_KIND)

    assert scored.columns == POINTS_PREDICTION.columns
    assert scored.height == frame.height
    assert scored["position"].unique().to_list() == [defender.POSITION]
    assert scored["model_version"].unique().to_list() == ["4"]
    assert scored["prediction_kind"].unique().to_list() == [BACKFILL_KIND]


def test_backfill_writes_nothing_without_a_production_alias(mocker):
    """No production alias: backfill returns quietly without a connection."""
    mocker.patch.object(defender, "get_production_model", return_value=None)
    connection = mocker.patch.object(defender, "get_connection")

    defender.backfill_defender_points()

    connection.assert_not_called()


def test_backfill_rescores_the_current_season_every_run(mocker):
    """The current season is always re-scored, with no historic seasons."""
    frame = _synthetic_frame(n_gws=2, n_players=3)
    pipe = defender.train_final(frame)
    mocker.patch.object(
        defender, "get_production_model", return_value=("4", pipe)
    )
    mocker.patch.object(defender, "build_model_frame", return_value=frame)
    conn = duckdb.connect(":memory:")
    for table in TABLES:
        conn.execute(table.ddl)
    mocker.patch.object(defender, "get_connection", return_value=conn)
    mocker.patch.object(defender, "CURRENT_SEASON", "2026-27")
    # backfill_defender_points closes the connection it is handed in its
    # finally block; the test owns this one and needs it open afterwards
    # to read back what was stored, so close is stubbed out here.
    mocker.patch.object(duckdb.DuckDBPyConnection, "close")

    defender.backfill_defender_points()

    stored = conn.sql("SELECT * FROM points_prediction").pl()
    assert stored.height == frame.height
    assert stored["prediction_kind"].unique().to_list() == [BACKFILL_KIND]
    assert stored["model_version"].unique().to_list() == ["4"]


def _seed_points_prediction(
    connection,
    frame: pl.DataFrame,
    season: str,
    model_version: str,
    predicted_points: float = 999.0,
) -> None:
    """Seed a sentinel backfill row per (season) key in ``frame``."""
    sentinel = (
        frame.filter(pl.col("season") == season)
        .select(defender.KEY_COLUMNS)
        .with_columns(
            position=pl.lit(defender.POSITION),
            predicted_points=pl.lit(predicted_points),
            model_version=pl.lit(model_version),
            prediction_kind=pl.lit(BACKFILL_KIND),
        )
        .select(POINTS_PREDICTION.columns)
    )
    POINTS_PREDICTION.append(connection, sentinel)


def _two_season_frame() -> pl.DataFrame:
    """Combine a historic and a current season into one model frame."""
    historic = _synthetic_frame(n_gws=2, n_players=3, season="2025-26")
    current = _synthetic_frame(n_gws=2, n_players=3, season="2026-27")
    return pl.concat([historic, current], how="vertical")


def _backfill_test_connection(mocker, frame, prod_version):
    """Wire up a real in-memory connection and patch backfill's dependencies."""
    pipe = defender.train_final(frame)
    mocker.patch.object(
        defender,
        "get_production_model",
        return_value=(prod_version, pipe),
    )
    mocker.patch.object(defender, "build_model_frame", return_value=frame)
    conn = duckdb.connect(":memory:")
    for table in TABLES:
        conn.execute(table.ddl)
    mocker.patch.object(defender, "get_connection", return_value=conn)
    mocker.patch.object(defender, "CURRENT_SEASON", "2026-27")
    mocker.patch.object(duckdb.DuckDBPyConnection, "close")
    return conn


def test_backfill_rescores_historic_season_when_nothing_stored(mocker):
    """A historic season with no stored rows is scored on this run."""
    frame = _two_season_frame()
    conn = _backfill_test_connection(mocker, frame, prod_version="4")

    defender.backfill_defender_points()

    stored = conn.sql("SELECT * FROM points_prediction").pl()
    historic = stored.filter(pl.col("season") == "2025-26")
    assert (
        historic.height == frame.filter(pl.col("season") == "2025-26").height
    )
    assert historic["model_version"].unique().to_list() == ["4"]


def test_backfill_rescores_historic_season_on_version_change(mocker):
    """A historic season stored under a stale version is rewritten."""
    frame = _two_season_frame()
    conn = _backfill_test_connection(mocker, frame, prod_version="4")
    _seed_points_prediction(
        conn, frame, "2025-26", model_version="3", predicted_points=999.0
    )

    defender.backfill_defender_points()

    stored = conn.sql("SELECT * FROM points_prediction").pl()
    historic = stored.filter(pl.col("season") == "2025-26")
    assert (
        historic.height == frame.filter(pl.col("season") == "2025-26").height
    )
    assert historic["model_version"].unique().to_list() == ["4"]
    assert 999.0 not in historic["predicted_points"].to_list()


def test_backfill_skips_historic_season_already_at_production_version(
    mocker,
):
    """A complete, up-to-date historic season is left untouched.

    The current season is still re-scored regardless of what the gate
    decides about historic seasons.
    """
    frame = _two_season_frame()
    conn = _backfill_test_connection(mocker, frame, prod_version="4")
    _seed_points_prediction(
        conn, frame, "2025-26", model_version="4", predicted_points=999.0
    )

    defender.backfill_defender_points()

    stored = conn.sql("SELECT * FROM points_prediction").pl()
    historic = stored.filter(pl.col("season") == "2025-26")
    # Untouched sentinel proves the historic partition was not rewritten.
    assert historic["predicted_points"].unique().to_list() == [999.0]
    assert historic["model_version"].unique().to_list() == ["4"]

    current = stored.filter(pl.col("season") == "2026-27")
    assert current.height == frame.filter(pl.col("season") == "2026-27").height
    assert current["model_version"].unique().to_list() == ["4"]


# --- Forward scoring for unplayed fixtures ---------------------------

GW1_KICKOFF = FIRST_KICKOFF
GW2_KICKOFF = FIRST_KICKOFF + timedelta(days=7)
GW3_KICKOFF = FIRST_KICKOFF + timedelta(days=14)


class _ConstantPointsModel:
    """Predicts the same number for every row handed to it."""

    def __init__(self, points: float = 4.5) -> None:
        self.points = points

    def predict(self, x) -> np.ndarray:
        """Return ``points`` once per input row."""
        return np.full(len(x), self.points)


def _stub_view(connection, name: str, frame: pl.DataFrame) -> None:
    """Replace a registered feature view with a hand-built frame."""
    connection.register(f"{name}_stub", frame)
    connection.execute(
        f"CREATE OR REPLACE TEMP VIEW {name} AS SELECT * FROM {name}_stub"
    )


def _stub_player_form(connection, rows: dict) -> pl.DataFrame:
    """Stub ``player_match_form_inclusive``, null-filling absent columns."""
    frame = pl.DataFrame(rows).with_columns(
        [
            pl.lit(None, dtype=pl.Float64).alias(name)
            for name in defender._PLAYER_FORM_COLUMNS
            if name not in rows
        ]
    )
    _stub_view(connection, "player_match_form_inclusive", frame)
    return frame


def _stub_team_form(connection, rows: dict) -> pl.DataFrame:
    """Stub ``team_match_form_inclusive``, null-filling absent columns."""
    columns = defender._OWN_TEAM_COLUMNS + defender._OPPOSITION_COLUMNS
    frame = pl.DataFrame(rows).with_columns(
        [
            pl.lit(None, dtype=pl.Float64).alias(name)
            for name in columns
            if name not in rows
        ]
    )
    _stub_view(connection, "team_match_form_inclusive", frame)
    return frame


def _forward_fixture(
    gw: int = 3,
    element: int = 1,
    opponent: int = TEAM_IDS[ARSENAL],
    kickoff: datetime = GW3_KICKOFF,
    team: str = UNITED,
    season: str = SEASON,
    position: str = "DEF",
) -> dict:
    """Build one row in ``build_forward_fixtures`` shape."""
    return {
        "season": season,
        "gw": gw,
        "element": element,
        "opponent": opponent,
        "kickoff_time": kickoff,
        "minutes": None,
        "position": position,
        "team": team,
        "value": 50,
        "chance_of_playing_this_round": 100,
    }


def _forward_frame(rows: list[dict]) -> pl.DataFrame:
    """Type a list of :func:`_forward_fixture` rows like the real thing."""
    return pl.DataFrame(
        rows,
        schema={
            "season": pl.Utf8,
            "gw": pl.Int64,
            "element": pl.Int64,
            "opponent": pl.Int64,
            "kickoff_time": pl.Datetime("us"),
            "minutes": pl.Int64,
            "position": pl.Utf8,
            "team": pl.Utf8,
            "value": pl.Int64,
            "chance_of_playing_this_round": pl.Int64,
        },
    )


def _seed_forward_history(connection) -> None:
    """Seed two played gameweeks plus an unplayed gw3 Man Utd v Arsenal.

    Every club's gw1 and gw2 figures are deliberately different, so the
    inclusive rolling value (the mean of both matches) is distinct from
    the exclusive one (gw1 alone). Derivations, all per club:

    * element 1 (a Man Utd defender) records 0.1 xG in gw1 and 0.9 in
      gw2 over 90 minutes each: inclusive 0.5 per 90, exclusive 0.1.
    * Man Utd concede 1.0 xG and 2 goals in gw1, then 0.2 xG and none in
      gw2: inclusive 0.6 / 1.0 / 0.5 clean-sheet rate, exclusive
      1.0 / 2.0 / 0.0.
    * Arsenal make 1.0 xG and 1 goal in gw1, then 3.0 and 3 in gw2:
      inclusive 2.0 / 2.0, exclusive 1.0 / 1.0.
    """
    utd_v_spurs = f"25-26-prem-{SLUGS[UNITED]}-vs-{SLUGS[SPURS]}"
    ars_v_chelsea = f"25-26-prem-{SLUGS[ARSENAL]}-vs-{SLUGS[CHELSEA]}"
    utd_v_chelsea = f"25-26-prem-{SLUGS[UNITED]}-vs-{SLUGS[CHELSEA]}"
    ars_v_spurs = f"25-26-prem-{SLUGS[ARSENAL]}-vs-{SLUGS[SPURS]}"
    clubs = {1: UNITED, 2: SPURS, 3: ARSENAL, 4: CHELSEA}

    _append(
        PLAYER_SEASON,
        connection,
        [
            {
                "season": SEASON,
                "element": element,
                "position": "DEF" if element == 1 else "MID",
            }
            for element in clubs
        ],
    )
    _append(
        PLAYER_WEEK,
        connection,
        [
            {
                "season": SEASON,
                "gw": gw,
                "element": element,
                "position": "DEF" if element == 1 else "MID",
                "team": team,
            }
            for gw in (1, 2)
            for element, team in clubs.items()
        ],
    )
    # (gw, element, opponent id, is_home, kickoff)
    legs = [
        (1, 1, TEAM_IDS[SPURS], True, GW1_KICKOFF),
        (1, 2, TEAM_IDS[UNITED], False, GW1_KICKOFF),
        (1, 3, TEAM_IDS[CHELSEA], True, GW1_KICKOFF),
        (1, 4, TEAM_IDS[ARSENAL], False, GW1_KICKOFF),
        (2, 1, TEAM_IDS[CHELSEA], True, GW2_KICKOFF),
        (2, 4, TEAM_IDS[UNITED], False, GW2_KICKOFF),
        (2, 3, TEAM_IDS[SPURS], True, GW2_KICKOFF),
        (2, 2, TEAM_IDS[ARSENAL], False, GW2_KICKOFF),
    ]
    _append(
        PLAYER_MATCH,
        connection,
        [
            {
                "season": SEASON,
                "gw": gw,
                "element": element,
                "opponent": opponent,
                "is_home": is_home,
                "minutes": 90,
                "kickoff_time": kickoff,
            }
            for gw, element, opponent, is_home, kickoff in legs
        ],
    )
    # (gw, team, is_home, opposition, kickoff)
    fixtures = [
        (1, UNITED, True, SPURS, GW1_KICKOFF),
        (1, SPURS, False, UNITED, GW1_KICKOFF),
        (1, ARSENAL, True, CHELSEA, GW1_KICKOFF),
        (1, CHELSEA, False, ARSENAL, GW1_KICKOFF),
        (2, UNITED, True, CHELSEA, GW2_KICKOFF),
        (2, CHELSEA, False, UNITED, GW2_KICKOFF),
        (2, ARSENAL, True, SPURS, GW2_KICKOFF),
        (2, SPURS, False, ARSENAL, GW2_KICKOFF),
        (3, UNITED, True, ARSENAL, GW3_KICKOFF),
        (3, ARSENAL, False, UNITED, GW3_KICKOFF),
    ]
    _append(
        TEAM_FIXTURE,
        connection,
        [
            {
                "season": SEASON,
                "gw": gw,
                "team": team,
                "is_home": is_home,
                "opposition": opposition,
                "kickoff_time": kickoff,
            }
            for gw, team, is_home, opposition, kickoff in fixtures
        ],
    )
    _append(
        PLAYER_MATCH_OPTA,
        connection,
        [
            _opta_row(SEASON, 1, 1, utd_v_spurs, 0.1, 0, 2),
            _opta_row(SEASON, 1, 2, utd_v_spurs, 1.0, 2, 0),
            _opta_row(SEASON, 1, 3, ars_v_chelsea, 1.0, 1, 1),
            _opta_row(SEASON, 1, 4, ars_v_chelsea, 0.5, 1, 1),
            _opta_row(SEASON, 2, 1, utd_v_chelsea, 0.9, 1, 0),
            _opta_row(SEASON, 2, 4, utd_v_chelsea, 0.2, 0, 1),
            _opta_row(SEASON, 2, 3, ars_v_spurs, 3.0, 3, 0),
            _opta_row(SEASON, 2, 2, ars_v_spurs, 0.4, 0, 3),
        ],
    )


def test_forward_frame_takes_the_most_recent_appearance(connection) -> None:
    """The as-of join takes the latest appearance before the fixture."""
    register_feature_views(connection)
    _stub_player_form(
        connection,
        {
            "season": [SEASON, SEASON],
            "element": [1, 1],
            "kickoff_time": [GW1_KICKOFF, GW2_KICKOFF],
            "xg_per90_rolling_5": [0.1, 0.9],
        },
    )
    frame = defender.build_forward_feature_frame(
        connection, _forward_frame([_forward_fixture()])
    )

    assert frame["xg_per90_rolling_5"].to_list() == [0.9]


def test_forward_frame_covers_every_rostered_defender(connection) -> None:
    """Every rostered defender gets a row, form or no form."""
    register_feature_views(connection)
    frame = defender.build_forward_feature_frame(
        connection,
        _forward_frame(
            [_forward_fixture(element=1), _forward_fixture(element=2)]
        ),
    )

    assert sorted(frame["element"].to_list()) == [1, 2]
    assert frame.columns == defender.KEY_COLUMNS + defender.FEATURES


def test_forward_scoring_skips_without_a_production_alias(mocker) -> None:
    """No production alias: forward scoring never opens a connection."""
    mocker.patch.object(defender, "get_production_model", return_value=None)
    connection = mocker.patch.object(defender, "get_connection")

    defender.score_forward_defender_points()

    connection.assert_not_called()


def test_forward_frame_uses_the_inclusive_player_form_view(
    connection,
) -> None:
    """Player form includes the most recent appearance, not just before it.

    Element 1 recorded 0.1 xG in gw1 and 0.9 in gw2 over 90 minutes
    each. The exclusive view -- which the training frame reads -- carries
    0.1 on the gw2 row, because its window stops one match short. The
    inclusive view carries 0.5, the mean over both. As-of joining the
    exclusive view here would make the gw3 forecast a match stale, so
    this test fails against ``player_match_form`` and passes only
    against ``player_match_form_inclusive``.
    """
    _seed_forward_history(connection)
    register_feature_views(connection)

    frame = defender.build_forward_feature_frame(
        connection, _forward_frame([_forward_fixture()])
    )

    assert frame["xg_per90_rolling_5"].item() == pytest.approx(0.5)


def test_forward_frame_uses_the_inclusive_team_form_views(connection) -> None:
    """Both teams' form includes their most recent match.

    Man Utd's own record over gw1 and gw2 is 0.6 xG against, 1.0 goals
    against and a 0.5 clean-sheet rate; the exclusive view would report
    gw1 alone, 1.0 / 2.0 / 0.0. Arsenal's attacking record is 2.0 xG for
    and 2.0 goals for inclusive, 1.0 / 1.0 exclusive. Every one of the
    five numbers differs between the two views, so this fails if either
    as-of join reaches for the exclusive view -- and the own/opposition
    values are distinct from each other, so a transposed join fails too.
    """
    _seed_forward_history(connection)
    register_feature_views(connection)

    frame = defender.build_forward_feature_frame(
        connection, _forward_frame([_forward_fixture()])
    )

    assert frame["xg_against_rolling_5"].item() == pytest.approx(0.6)
    assert frame["goals_against_rolling_5"].item() == pytest.approx(1.0)
    assert frame["clean_sheet_rolling_5"].item() == pytest.approx(0.5)
    assert frame["xg_for_rolling_5"].item() == pytest.approx(2.0)
    assert frame["goals_for_rolling_5"].item() == pytest.approx(2.0)


def test_forward_frame_carries_is_home_from_the_fixture(connection) -> None:
    """``is_home`` is resolved from team_fixture, not left null.

    It is a model feature, so a null here would be median-imputed on
    every live row and the model would never see who is at home.
    """
    _seed_forward_history(connection)
    register_feature_views(connection)

    home = defender.build_forward_feature_frame(
        connection, _forward_frame([_forward_fixture()])
    )
    away = defender.build_forward_feature_frame(
        connection,
        _forward_frame(
            [
                _forward_fixture(
                    element=3, team=ARSENAL, opponent=TEAM_IDS[UNITED]
                )
            ]
        ),
    )

    assert home["is_home"].item() == 1.0
    assert away["is_home"].item() == 0.0


def test_forward_frame_matches_each_player_to_his_own_last_appearance(
    connection,
) -> None:
    """Interleaved appearances still resolve per player, not globally.

    Element 1's appearances straddle element 2's in time, so a join that
    lost the per-player grouping -- or sorted only globally -- would hand
    at least one of them the other's row.
    """
    register_feature_views(connection)
    _stub_player_form(
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
            "xg_per90_rolling_5": [0.1, 5.0, 0.2, 6.0],
        },
    )

    frame = defender.build_forward_feature_frame(
        connection,
        _forward_frame(
            [_forward_fixture(element=1), _forward_fixture(element=2)]
        ),
    ).sort("element")

    assert frame["xg_per90_rolling_5"].to_list() == [0.2, 6.0]


def _stub_two_season_form(connection) -> None:
    """Stub form and id views with a club present in two seasons.

    Arsenal and Chelsea both appear in 2024-25 and 2025-26, and the FPL
    ids are reshuffled between the two: id 1 is Chelsea in 2024-25 and
    Arsenal in 2025-26. Any join that resolves the opposition by id
    without the season -- or that resolves it by id at all -- can pick
    up the wrong club's form.
    """
    _stub_team_form(
        connection,
        {
            "season": ["2024-25", "2024-25", SEASON, SEASON],
            "team": [ARSENAL, CHELSEA, ARSENAL, CHELSEA],
            "kickoff_time": [
                GW1_KICKOFF - timedelta(days=365),
                GW1_KICKOFF - timedelta(days=364),
                GW1_KICKOFF,
                GW2_KICKOFF,
            ],
            "xg_for_rolling_5": [9.0, 9.5, 0.5, 3.0],
            "goals_for_rolling_5": [9.0, 9.5, 0.5, 3.0],
        },
    )
    _stub_view(
        connection,
        "fpl_team_id",
        pl.DataFrame(
            {
                "season": ["2024-25", "2024-25", SEASON, SEASON],
                "team_id": [1, 4, 1, 4],
                "team": [CHELSEA, ARSENAL, ARSENAL, CHELSEA],
            }
        ),
    )


def test_forward_frame_resolves_the_opponent_by_name_not_reused_id(
    connection,
) -> None:
    """Opposition form is Arsenal's, even though id 1 was Chelsea's.

    Chelsea played more recently than Arsenal, so a join that pools both
    clubs under FPL id 1 -- which is what dropping ``season`` from the
    ``fpl_team_id`` join does -- as-of matches Chelsea's 3.0 rather than
    Arsenal's 0.5. Resolving the opponent by name through ``team_fixture``
    cannot make that mistake.
    """
    _seed_forward_history(connection)
    register_feature_views(connection)
    _stub_two_season_form(connection)

    frame = defender.build_forward_feature_frame(
        connection, _forward_frame([_forward_fixture()])
    )

    assert frame["xg_for_rolling_5"].item() == pytest.approx(0.5)
    assert frame["goals_for_rolling_5"].item() == pytest.approx(0.5)


def test_forward_frame_has_exactly_one_row_per_fixture(connection) -> None:
    """Three input fixtures give three output rows, keys unchanged.

    The seeded form carries a club in two seasons, so a join that
    multiplies form rows by season would show up here as extra rows.
    """
    _seed_forward_history(connection)
    register_feature_views(connection)
    _stub_two_season_form(connection)
    forward = _forward_frame(
        [
            _forward_fixture(element=1),
            _forward_fixture(element=2),
            _forward_fixture(
                element=3, team=ARSENAL, opponent=TEAM_IDS[UNITED]
            ),
        ]
    )

    frame = defender.build_forward_feature_frame(connection, forward)

    assert frame.height == forward.height
    assert sorted(frame.select(defender.KEY_COLUMNS).rows()) == sorted(
        forward.select(defender.KEY_COLUMNS).rows()
    )


def test_forward_frame_gives_a_double_gameweek_two_rows(connection) -> None:
    """One defender with two gw3 fixtures gets two rows, not one or four."""
    _seed_forward_history(connection)
    register_feature_views(connection)
    second_kickoff = GW3_KICKOFF + timedelta(days=3)
    _append(
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

    frame = defender.build_forward_feature_frame(
        connection,
        _forward_frame(
            [
                _forward_fixture(),
                _forward_fixture(
                    opponent=TEAM_IDS[SPURS], kickoff=second_kickoff
                ),
            ]
        ),
    )

    assert frame.height == 2
    assert sorted(frame["opponent"].to_list()) == sorted(
        [TEAM_IDS[SPURS], TEAM_IDS[ARSENAL]]
    )
    assert sorted(frame["is_home"].to_list()) == [0.0, 1.0]


def test_forward_frame_takes_only_forward_minutes_predictions(
    connection,
) -> None:
    """Backfill minutes rows for the same key are ignored, not joined.

    ``prediction_kind`` is part of the minutes primary key, so both kinds
    can sit on one fixture. Joining without the filter would either take
    the backfill value or fan the row out.
    """
    _seed_forward_history(connection)
    register_feature_views(connection)
    for kind, expected in ((BACKFILL_KIND, 11.0), (FORWARD_KIND, 77.0)):
        _append(
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

    frame = defender.build_forward_feature_frame(
        connection, _forward_frame([_forward_fixture()])
    )

    assert frame.height == 1
    assert frame["expected_minutes"].item() == pytest.approx(77.0)


def test_score_forward_defender_points_stores_and_freezes(mocker) -> None:
    """Forward rows land for gw3 while the gw2 forecast stays frozen.

    gw1 and gw2 have been played, so ``from_gw`` is 3 and the gw2
    sentinel sits below the delete floor. Losing ``gw_from`` on
    ``replace_partition`` would wipe it.
    """
    conn = duckdb.connect(":memory:")
    for table in TABLES:
        conn.execute(table.ddl)
    _seed_forward_history(conn)
    _append(
        PLAYER_SNAPSHOT,
        conn,
        [
            {
                "season": SEASON,
                "captured_at": GW2_KICKOFF + timedelta(days=1),
                "element": 1,
                "value": 50,
                "team": UNITED,
                "position": "DEF",
                "chance_of_playing_this_round": 100,
            },
            {
                "season": SEASON,
                "captured_at": GW2_KICKOFF + timedelta(days=1),
                "element": 3,
                "value": 60,
                "team": ARSENAL,
                "position": "MID",
                "chance_of_playing_this_round": 100,
            },
        ],
    )
    POINTS_PREDICTION.append(
        conn,
        pl.DataFrame(
            {
                "season": [SEASON],
                "gw": [2],
                "element": [1],
                "opponent": [TEAM_IDS[CHELSEA]],
                "position": ["DEF"],
                "predicted_points": [999.0],
                "model_version": ["1"],
                "prediction_kind": [FORWARD_KIND],
            }
        ),
    )

    mocker.patch.object(
        defender,
        "get_production_model",
        return_value=("9", _ConstantPointsModel()),
    )
    mocker.patch.object(defender, "get_connection", return_value=conn)
    mocker.patch.object(defender, "CURRENT_SEASON", SEASON)
    mocker.patch.object(duckdb.DuckDBPyConnection, "close")
    api = mocker.Mock()
    api.get_teams.return_value = [
        SimpleNamespace(id=team_id, name=name)
        for name, team_id in TEAM_IDS.items()
    ]
    mocker.patch.object(defender, "FplAPI", return_value=api)

    defender.score_forward_defender_points()

    stored = POINTS_PREDICTION.load(conn)
    frozen = stored.filter(pl.col("gw") == 2)
    assert frozen["predicted_points"].to_list() == [999.0]
    assert frozen["model_version"].to_list() == ["1"]

    fresh = stored.filter(pl.col("gw") == 3)
    # Only the defender is scored -- the Arsenal midfielder is dropped.
    assert fresh.select("element", "opponent").rows() == [
        (1, TEAM_IDS[ARSENAL])
    ]
    assert fresh["predicted_points"].to_list() == [4.5]
    assert fresh["prediction_kind"].to_list() == [FORWARD_KIND]
    assert fresh["model_version"].to_list() == ["9"]
    assert fresh["position"].to_list() == [defender.POSITION]


def test_forward_frame_never_inherits_a_previous_holder_of_the_id(
    connection,
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
    _stub_player_form(
        connection,
        {
            "season": ["2024-25"],
            "element": [1],
            "kickoff_time": [GW1_KICKOFF - timedelta(days=365)],
            "xg_per90_rolling_5": [7.0],
        },
    )

    frame = defender.build_forward_feature_frame(
        connection, _forward_frame([_forward_fixture()])
    )

    assert frame["xg_per90_rolling_5"].to_list() == [None]


def test_forward_frame_scores_through_a_fitted_pipeline(connection) -> None:
    """The forward frame's dtypes survive a real fitted pipeline.

    Two of the form columns arrive from DuckDB as ``Decimal``, not
    ``Float64``, and reach pandas as ``object``. A frame that builds
    cleanly but cannot be predicted on is no use, so this exercises the
    real pipeline rather than a stand-in model.
    """
    _seed_forward_history(connection)
    register_feature_views(connection)
    frame = defender.build_forward_feature_frame(
        connection, _forward_frame([_forward_fixture()])
    )
    pipe = defender.train_final(_synthetic_frame(n_gws=2, n_players=3))

    scored = defender.score_defender_points(frame, pipe, "1", FORWARD_KIND)

    assert scored.columns == POINTS_PREDICTION.columns
    assert scored.height == 1
    assert scored["predicted_points"].dtype == pl.Float64
    assert scored["predicted_points"].item() is not None
