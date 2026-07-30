"""Tests for the six table specs, including DDL equivalence."""

import duckdb
import polars as pl
import pytest

from fantasy_football.storage import database
from fantasy_football.storage.tables import (
    MINUTES_PREDICTION,
    PLAYER_AVAILABILITY,
    PLAYER_MATCH,
    PLAYER_SEASON,
    PLAYER_WEEK,
    TABLES,
    TEAM_FIXTURE,
    gkp_to_gk,
    propagate_static_columns,
)

# Each spec paired with the legacy DDL string it must reproduce.
LEGACY_DDL = [
    (PLAYER_WEEK, database._CREATE_TABLE),
    (TEAM_FIXTURE, database._CREATE_TEAM_FIXTURE_TABLE),
    (PLAYER_MATCH, database._CREATE_PLAYER_MATCH_TABLE),
    (PLAYER_AVAILABILITY, database._CREATE_PLAYER_AVAILABILITY_TABLE),
    (MINUTES_PREDICTION, database._CREATE_MINUTES_PREDICTION_TABLE),
    (PLAYER_SEASON, database._CREATE_PLAYER_SEASON_TABLE),
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


def test_tables_tuple_covers_every_spec():
    """``TABLES`` holds all six specs, so loops cannot miss one."""
    assert len(TABLES) == 6
    assert {t.name for t in TABLES} == {
        "player_week",
        "team_fixture",
        "player_match",
        "player_availability",
        "minutes_prediction",
        "player_season",
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
    """Reset empties all six tables, not a hand-listed subset."""
    for table in TABLES:
        table.upsert_current(db, minimal_row(table), "2024-25")
        assert not table.load(db).is_empty()

    database.reset_database(db)

    for table in TABLES:
        assert table.load(db).is_empty(), f"{table.name} not cleared"
