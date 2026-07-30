"""Tests for the Table descriptor and its DDL generation."""

import duckdb
import polars as pl
import pytest

from fantasy_football.storage.table import Table, duckdb_type

WIDGET = Table(
    name="widget",
    schema={
        "season": pl.Utf8,
        "element": pl.Int64,
        "label": pl.Utf8,
        "ratio": pl.Float64,
    },
    primary_key=("season", "element"),
    order_by=("season", "element"),
)


def test_columns_derive_from_schema_order():
    """``columns`` is the schema's keys in declaration order."""
    assert WIDGET.columns == ["season", "element", "label", "ratio"]


def test_ddl_marks_primary_key_columns_not_null():
    """Primary-key columns are NOT NULL; others are nullable."""
    assert "season VARCHAR NOT NULL" in WIDGET.ddl
    assert "element BIGINT NOT NULL" in WIDGET.ddl
    assert "label VARCHAR," in WIDGET.ddl
    assert "label VARCHAR NOT NULL" not in WIDGET.ddl


def test_ddl_declares_the_primary_key():
    """The generated DDL carries a composite PRIMARY KEY clause."""
    assert "PRIMARY KEY (season, element)" in WIDGET.ddl


def test_ddl_is_executable_and_builds_the_expected_table():
    """DuckDB accepts the generated DDL and builds the right columns."""
    connection = duckdb.connect(":memory:")
    try:
        connection.execute(WIDGET.ddl)
        info = connection.execute("PRAGMA table_info('widget')").fetchall()
    finally:
        connection.close()
    assert [row[1] for row in info] == WIDGET.columns
    assert [row[2] for row in info] == [
        "VARCHAR",
        "BIGINT",
        "VARCHAR",
        "DOUBLE",
    ]


def test_ddl_is_idempotent_via_if_not_exists():
    """Executing the DDL twice does not raise."""
    connection = duckdb.connect(":memory:")
    try:
        connection.execute(WIDGET.ddl)
        connection.execute(WIDGET.ddl)
    finally:
        connection.close()


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (pl.Utf8, "VARCHAR"),
        (pl.Int64, "BIGINT"),
        (pl.Boolean, "BOOLEAN"),
        (pl.Float64, "DOUBLE"),
        (pl.Date, "DATE"),
        (pl.Datetime("us"), "TIMESTAMP"),
    ],
)
def test_duckdb_type_maps_every_dtype_in_use(dtype, expected):
    """Each Polars dtype the project stores maps to a DuckDB type."""
    assert duckdb_type(dtype) == expected


def test_duckdb_type_rejects_an_unmapped_dtype():
    """An unmapped dtype raises rather than emitting invalid SQL."""
    with pytest.raises(ValueError, match="No DuckDB type"):
        duckdb_type(pl.List(pl.Int64))
