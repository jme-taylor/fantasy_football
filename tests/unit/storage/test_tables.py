"""Tests for the table specs, including DDL equivalence."""

import duckdb
import polars as pl
import pytest

from fantasy_football.storage import database, engine
from fantasy_football.storage.tables import (
    MINUTES_PREDICTION,
    PLAYER_AVAILABILITY,
    PLAYER_MATCH,
    PLAYER_MATCH_FPL,
    PLAYER_MATCH_OPTA,
    PLAYER_SEASON,
    PLAYER_SNAPSHOT,
    PLAYER_WEEK,
    TABLES,
    TEAM_FIXTURE,
    TEST_MINUTES_PREDICTION,
    TEST_POINTS_PREDICTION,
    gkp_to_gk,
    propagate_static_columns,
)

# Frozen copies of the DDL as it stood before the descriptor refactor.
# These are a golden record: if a spec change alters the built schema,
# this test fails and the change must be deliberate.
_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS player_week (
    season VARCHAR NOT NULL,
    gw BIGINT NOT NULL,
    element BIGINT NOT NULL,
    name VARCHAR,
    position VARCHAR,
    team VARCHAR,
    bonus BIGINT,
    minutes BIGINT,
    round BIGINT,
    total_points BIGINT,
    value BIGINT,
    PRIMARY KEY (season, gw, element)
)
"""

_CREATE_TEAM_FIXTURE_TABLE = """
CREATE TABLE IF NOT EXISTS team_fixture (
    season VARCHAR NOT NULL,
    gw BIGINT NOT NULL,
    team VARCHAR NOT NULL,
    is_home BOOLEAN,
    opposition VARCHAR NOT NULL,
    kickoff_time TIMESTAMP,
    PRIMARY KEY (season, gw, team, opposition)
)
"""

_CREATE_PLAYER_MATCH_TABLE = """
CREATE TABLE IF NOT EXISTS player_match (
    season VARCHAR NOT NULL,
    gw BIGINT NOT NULL,
    element BIGINT NOT NULL,
    opponent BIGINT NOT NULL,
    is_home BOOLEAN,
    minutes BIGINT,
    total_points BIGINT,
    kickoff_time TIMESTAMP,
    PRIMARY KEY (season, gw, element, opponent)
)
"""

_CREATE_PLAYER_AVAILABILITY_TABLE = """
CREATE TABLE IF NOT EXISTS player_availability (
    season VARCHAR NOT NULL,
    gw BIGINT NOT NULL,
    element BIGINT NOT NULL,
    chance_of_playing_this_round BIGINT,
    PRIMARY KEY (season, gw, element)
)
"""

_CREATE_MINUTES_PREDICTION_TABLE = """
CREATE TABLE IF NOT EXISTS minutes_prediction (
    season VARCHAR NOT NULL,
    gw BIGINT NOT NULL,
    element BIGINT NOT NULL,
    opponent BIGINT NOT NULL,
    p_zero DOUBLE,
    p_partial DOUBLE,
    p_sixty_plus DOUBLE,
    expected_minutes DOUBLE,
    model_version VARCHAR,
    prediction_kind VARCHAR NOT NULL,
    snapshot_captured_at TIMESTAMP,
    PRIMARY KEY (season, gw, element, opponent, prediction_kind)
)
"""

_CREATE_PLAYER_SEASON_TABLE = """
CREATE TABLE IF NOT EXISTS player_season (
    season VARCHAR NOT NULL,
    element BIGINT NOT NULL,
    player_code BIGINT,
    web_name VARCHAR,
    first_name VARCHAR,
    second_name VARCHAR,
    position VARCHAR,
    team_code BIGINT,
    birth_date DATE,
    region BIGINT,
    team_join_date DATE,
    PRIMARY KEY (season, element)
)
"""

_CREATE_PLAYER_SNAPSHOT_TABLE = """
CREATE TABLE IF NOT EXISTS player_snapshot (
    season VARCHAR NOT NULL,
    captured_at TIMESTAMP NOT NULL,
    element BIGINT NOT NULL,
    value BIGINT,
    team VARCHAR,
    position VARCHAR,
    chance_of_playing_this_round BIGINT,
    PRIMARY KEY (season, captured_at, element)
)
"""

# Each spec paired with the legacy DDL string it must reproduce.
LEGACY_DDL = [
    (PLAYER_WEEK, _CREATE_TABLE),
    (TEAM_FIXTURE, _CREATE_TEAM_FIXTURE_TABLE),
    (PLAYER_MATCH, _CREATE_PLAYER_MATCH_TABLE),
    (PLAYER_AVAILABILITY, _CREATE_PLAYER_AVAILABILITY_TABLE),
    (MINUTES_PREDICTION, _CREATE_MINUTES_PREDICTION_TABLE),
    (PLAYER_SEASON, _CREATE_PLAYER_SEASON_TABLE),
    (PLAYER_SNAPSHOT, _CREATE_PLAYER_SNAPSHOT_TABLE),
]


def table_info(ddl: str, name: str) -> list[tuple]:
    """Build a table from ``ddl`` and return DuckDB's own description.

    ``PRAGMA table_info`` reports column order, name, DuckDB type,
    nullability, and primary-key membership -- everything the two DDLs
    must agree on. Comparing what DuckDB builds is far less brittle than
    comparing SQL text.

    Parameters
    ----------
    ddl : str
        A create statement.
    name : str
        The table the statement creates.

    Returns
    -------
    list[tuple]
        One row per column, in ordinal position.
    """
    connection = duckdb.connect(":memory:")
    try:
        connection.execute(ddl)
        return connection.execute(f"PRAGMA table_info('{name}')").fetchall()
    finally:
        connection.close()


@pytest.mark.parametrize(
    ("table", "legacy"), LEGACY_DDL, ids=lambda v: getattr(v, "name", "")
)
def test_generated_ddl_matches_legacy_ddl(table, legacy):
    """Every generated DDL builds the same table as the legacy string."""
    assert table_info(table.ddl, table.name) == table_info(legacy, table.name)
    assert f"PRIMARY KEY ({', '.join(table.primary_key)})" in legacy


def test_tables_tuple_covers_every_spec():
    """``TABLES`` holds all fourteen specs, so loops cannot miss one."""
    assert len(TABLES) == 14
    assert {t.name for t in TABLES} == {
        "player_week",
        "team_fixture",
        "player_match",
        "player_match_fpl",
        "player_match_opta",
        "player_availability",
        "minutes_prediction",
        "points_component",
        "points_prediction",
        "player_season",
        "player_snapshot",
        "test_points_prediction",
        "test_minutes_prediction",
        "test_conceding_prediction",
    }


def test_gkp_to_gk_collapses_the_legacy_label():
    """The legacy GKP position label becomes GK."""
    frame = pl.DataFrame({"position": ["GKP", "DEF", "GK"]})
    assert gkp_to_gk(frame)["position"].to_list() == ["GK", "DEF", "GK"]


def test_propagate_static_columns_fills_across_seasons():
    """A static attribute seen once fills every season for that code."""
    frame = pl.DataFrame(
        {
            "player_code": [1, 1],
            "first_name": [None, "Bukayo"],
            "second_name": [None, "Saka"],
            "birth_date": [None, None],
            "region": [None, None],
        }
    )
    result = propagate_static_columns(frame)
    assert result["first_name"].to_list() == ["Bukayo", "Bukayo"]


def test_propagate_static_columns_leaves_null_codes_alone():
    """Rows with no player_code have no identity to propagate along."""
    frame = pl.DataFrame(
        {
            "player_code": [None, None],
            "first_name": [None, "Someone"],
            "second_name": [None, None],
            "birth_date": [None, None],
            "region": [None, None],
        }
    )
    result = propagate_static_columns(frame)
    assert result["first_name"].to_list() == [None, "Someone"]


def minimal_row(table) -> pl.DataFrame:
    """Build a one-row frame satisfying a table's NOT NULL columns.

    Primary-key columns across all six tables are either Utf8 or Int64,
    so those two placeholders suffice. Every other column is null.

    Parameters
    ----------
    table : Table
        The spec to build a row for.

    Returns
    -------
    pl.DataFrame
        A single row shaped to the table's schema.
    """
    values = {}
    for column, dtype in table.schema.items():
        if column not in table.primary_key:
            values[column] = [None]
        elif dtype == pl.Utf8:
            values[column] = ["2024-25"]
        else:
            values[column] = [1]
    return pl.DataFrame(values, schema=table.schema)


def test_reset_database_clears_every_table(db):
    """Reset empties every table, not a hand-listed subset."""
    for table in TABLES:
        table.upsert_current(db, minimal_row(table), "2024-25")
        assert not table.load(db).is_empty()

    database.reset_database(db)

    for table in TABLES:
        assert table.load(db).is_empty(), f"{table.name} not cleared"


def test_player_match_fpl_is_registered():
    """The Vaastav match table is in TABLES so it gets created and reset."""
    assert PLAYER_MATCH_FPL in TABLES


def test_player_match_opta_is_registered():
    """The FCI match table is in TABLES so it gets created and reset."""
    assert PLAYER_MATCH_OPTA in TABLES


def test_player_match_fpl_is_keyed_on_fixture():
    """Fixture, not opponent: it exists in every season, unlike opponent."""
    assert PLAYER_MATCH_FPL.primary_key == (
        "season",
        "gw",
        "element",
        "fixture",
    )


def test_player_match_opta_is_keyed_on_match_id():
    """FCI identifies fixtures by slug, so match_id completes the key."""
    assert PLAYER_MATCH_OPTA.primary_key == (
        "season",
        "gw",
        "element",
        "match_id",
    )


def test_new_match_tables_do_not_disturb_player_match():
    """The pre-existing thin table keeps its own name and schema."""
    assert PLAYER_MATCH.name == "player_match"
    assert PLAYER_MATCH.name not in {
        PLAYER_MATCH_FPL.name,
        PLAYER_MATCH_OPTA.name,
    }


def test_player_match_fpl_carries_the_rich_columns():
    """The columns the thin player_match table discards are stored here."""
    for column in (
        "goals_scored",
        "assists",
        "bps",
        "ict_index",
        "expected_goals",
        "defensive_contribution",
        "value",
        "selected",
    ):
        assert column in PLAYER_MATCH_FPL.schema


def test_player_match_opta_carries_the_opta_columns():
    """The FCI stat set is stored, including the wholly-null ones."""
    for column in (
        "xg",
        "xgot",
        "goals_prevented",
        "aerial_duels_won",
        "defensive_contributions",
        "corners",
        "competition",
        "distance_covered",
    ):
        assert column in PLAYER_MATCH_OPTA.schema


def test_both_new_tables_have_executable_ddl(db):
    """DuckDB creates both tables from their specs."""
    for table in (PLAYER_MATCH_FPL, PLAYER_MATCH_OPTA):
        stored = engine.table_columns(db, table.name)
        assert stored == set(table.columns)


def test_points_prediction_is_registered():
    """The points_prediction table is in TABLES so it gets created and reset."""
    from fantasy_football.storage.tables import POINTS_PREDICTION, TABLES

    assert POINTS_PREDICTION in TABLES


def test_points_prediction_round_trips(db):
    """A points_prediction row round-trips through the table's DDL."""
    from fantasy_football.storage.tables import (
        FORWARD_KIND,
        POINTS_PREDICTION,
    )

    frame = pl.DataFrame(
        {
            "season": ["2026-27"],
            "gw": [5],
            "element": [101],
            "opponent": [7],
            "position": ["DEF"],
            "predicted_points": [4.25],
            "model_version": ["3"],
            "prediction_kind": [FORWARD_KIND],
        }
    )
    POINTS_PREDICTION.replace_partition(
        db,
        frame,
        equals={"season": "2026-27", "prediction_kind": FORWARD_KIND},
    )

    stored = POINTS_PREDICTION.load(db)
    assert stored.height == 1
    assert stored["predicted_points"][0] == 4.25


def test_points_prediction_versions_filters_by_season(db):
    """``points_prediction_versions`` restricts to the given seasons."""
    from fantasy_football.storage.tables import (
        BACKFILL_KIND,
        POINTS_PREDICTION,
        points_prediction_versions,
    )

    frame = pl.DataFrame(
        {
            "season": ["2025-26", "2026-27"],
            "gw": [1, 1],
            "element": [1, 1],
            "opponent": [2, 2],
            "position": ["DEF", "DEF"],
            "predicted_points": [1.0, 2.0],
            "model_version": ["1", "2"],
            "prediction_kind": [BACKFILL_KIND, BACKFILL_KIND],
        }
    )
    POINTS_PREDICTION.replace_partition(
        db, frame, equals={"prediction_kind": BACKFILL_KIND}
    )

    assert points_prediction_versions(db) == {"1", "2"}
    assert points_prediction_versions(db, seasons=["2025-26"]) == {"1"}


def test_points_prediction_versions_empty_seasons_returns_empty_set(
    db,
) -> None:
    """An empty ``seasons`` filter returns an empty set without raising.

    A naive ``WHERE season IN (...)`` built from an empty list is
    invalid SQL (``IN ()``); this must short-circuit before reaching
    DuckDB rather than crash. Seeding a row first proves the empty
    result comes from the empty filter, not an empty table.
    """
    from fantasy_football.storage.tables import (
        BACKFILL_KIND,
        POINTS_PREDICTION,
        points_prediction_versions,
    )

    frame = pl.DataFrame(
        {
            "season": ["2025-26"],
            "gw": [1],
            "element": [1],
            "opponent": [2],
            "position": ["DEF"],
            "predicted_points": [1.0],
            "model_version": ["1"],
            "prediction_kind": [BACKFILL_KIND],
        }
    )
    POINTS_PREDICTION.replace_partition(
        db, frame, equals={"prediction_kind": BACKFILL_KIND}
    )

    assert points_prediction_versions(db, seasons=[]) == set()


def test_minutes_prediction_versions_empty_seasons_returns_empty_set(
    db,
) -> None:
    """An empty ``seasons`` filter returns an empty set without raising.

    Mirrors the points_prediction case: an empty ``IN ()`` clause is
    invalid SQL, so an empty ``seasons`` list must short-circuit before
    any query runs. Seeding a row first proves the empty result comes
    from the empty filter, not an empty table.
    """
    from fantasy_football.storage.tables import (
        BACKFILL_KIND,
        MINUTES_PREDICTION,
        minutes_prediction_versions,
    )

    frame = pl.DataFrame(
        {
            "season": ["2025-26"],
            "gw": [1],
            "element": [1],
            "opponent": [2],
            "p_zero": [0.1],
            "p_partial": [0.2],
            "p_sixty_plus": [0.7],
            "expected_minutes": [70.0],
            "model_version": ["1"],
            "prediction_kind": [BACKFILL_KIND],
            "snapshot_captured_at": [None],
        }
    )
    MINUTES_PREDICTION.replace_partition(
        db, frame, equals={"prediction_kind": BACKFILL_KIND}
    )

    assert minutes_prediction_versions(db, seasons=[]) == set()


# --- Evaluation prediction tables -------------------------------------


def _eval_points_row(run_id: str, predicted: float) -> pl.DataFrame:
    """One stored evaluation prediction for a defender."""
    return pl.DataFrame(
        {
            "run_id": [run_id],
            "season": ["2025-26"],
            "gw": [1],
            "element": [1],
            "opponent": [2],
            "position": ["DEF"],
            "predicted_points": [predicted],
            "actual_points": [4.0],
            "features": ['{"is_home": 1.0}'],
        }
    )


def test_evaluation_tables_are_registered() -> None:
    """Database creation and reset loop over TABLES, so these must be in it."""
    assert TEST_POINTS_PREDICTION in TABLES
    assert TEST_MINUTES_PREDICTION in TABLES


def test_evaluation_predictions_accumulate_across_runs(db) -> None:
    """Two runs' predictions for one fixture coexist rather than overwrite."""
    TEST_POINTS_PREDICTION.append(db, _eval_points_row("run-a", 3.0))
    TEST_POINTS_PREDICTION.append(db, _eval_points_row("run-b", 5.0))
    stored = TEST_POINTS_PREDICTION.load(db)
    assert stored.height == 2
    assert stored["run_id"].to_list() == ["run-a", "run-b"]


def test_evaluation_predictions_reject_a_repeated_run(db) -> None:
    """One run stores each fixture once, so a duplicate is a bug."""
    TEST_POINTS_PREDICTION.append(db, _eval_points_row("run-a", 3.0))
    with pytest.raises(duckdb.ConstraintException):
        TEST_POINTS_PREDICTION.append(db, _eval_points_row("run-a", 4.0))


def test_evaluation_features_are_readable_back_out_of_json(db) -> None:
    """Features survive the round trip as queryable JSON."""
    TEST_POINTS_PREDICTION.append(db, _eval_points_row("run-a", 3.0))
    value = db.sql(
        "SELECT json_extract(features, '$.is_home') FROM "
        f"{TEST_POINTS_PREDICTION.name}"
    ).fetchone()
    assert value is not None
    assert float(value[0]) == 1.0
