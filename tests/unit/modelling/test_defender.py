import duckdb
import polars as pl
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

from fantasy_football.modelling import defender
from fantasy_football.storage.tables import TABLES


@pytest.fixture
def connection():
    """Yield an in-memory DuckDB connection with every table created."""
    conn = duckdb.connect(":memory:")
    for table in TABLES:
        conn.execute(table.ddl)
    yield conn
    conn.close()


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
