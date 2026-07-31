"""Tests for the DuckDB engine primitives."""

from collections.abc import Iterator

import duckdb
import polars as pl
import pytest

from fantasy_football.storage import engine

DDL = """
CREATE TABLE IF NOT EXISTS widget (
    season VARCHAR NOT NULL,
    element BIGINT NOT NULL,
    label VARCHAR,
    PRIMARY KEY (season, element)
)
"""


@pytest.fixture
def conn() -> Iterator[duckdb.DuckDBPyConnection]:
    """Yield an in-memory connection with the ``widget`` table created."""
    connection = duckdb.connect(":memory:")
    connection.execute(DDL)
    yield connection
    connection.close()


def frame(rows: list[tuple]) -> pl.DataFrame:
    """Build a widget-shaped frame from ``(season, element, label)`` rows."""
    return pl.DataFrame(
        rows, schema=["season", "element", "label"], orient="row"
    )


def test_insert_frame_round_trips(conn):
    """A frame inserted through the engine reads back unchanged."""
    engine.insert_frame(conn, "widget", frame([("2024-25", 1, "a")]))
    result = engine.select(
        conn, "widget", ["season", "element", "label"], ["season", "element"]
    )
    assert result.rows() == [("2024-25", 1, "a")]


def test_insert_frame_unregisters_view_on_failure(conn):
    """A failed insert still unregisters its temporary view."""
    bad = pl.DataFrame({"nonsense": [1]})
    with pytest.raises(Exception):
        engine.insert_frame(conn, "widget", bad)
    with pytest.raises(Exception):
        conn.execute("SELECT * FROM incoming_widget")


def test_select_applies_column_order_and_sort(conn):
    """``select`` returns the requested columns in the requested order."""
    engine.insert_frame(
        conn, "widget", frame([("2024-25", 2, "b"), ("2024-25", 1, "a")])
    )
    result = engine.select(conn, "widget", ["element", "season"], ["element"])
    assert result.columns == ["element", "season"]
    assert result["element"].to_list() == [1, 2]


def test_delete_season_removes_only_that_season(conn):
    """``delete_season`` leaves other seasons untouched."""
    engine.insert_frame(
        conn, "widget", frame([("2024-25", 1, "a"), ("2025-26", 1, "b")])
    )
    engine.delete_season(conn, "widget", "2024-25")
    result = engine.select(conn, "widget", ["season"], ["season"])
    assert result["season"].to_list() == ["2025-26"]


def test_distinct_returns_unique_values_including_null(conn):
    """``distinct`` does not filter nulls, matching legacy behaviour."""
    engine.insert_frame(
        conn, "widget", frame([("2024-25", 1, "a"), ("2024-25", 2, None)])
    )
    assert engine.distinct(conn, "widget", "label") == {"a", None}


def test_drop_then_create_empties_the_table(conn):
    """``drop`` followed by ``create`` yields an empty table."""
    engine.insert_frame(conn, "widget", frame([("2024-25", 1, "a")]))
    engine.drop(conn, "widget")
    engine.create(conn, DDL)
    result = engine.select(conn, "widget", ["season"], ["season"])
    assert result.is_empty()
