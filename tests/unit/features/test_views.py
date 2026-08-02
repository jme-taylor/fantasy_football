"""Tests for the single entry point onto the feature views."""

import duckdb
import polars as pl
import pytest

from fantasy_football.features.views import register_feature_views
from fantasy_football.storage.tables import TABLES


@pytest.fixture
def connection():
    """Return an in-memory connection with every table schema created."""
    conn = duckdb.connect(":memory:")
    for table in TABLES:
        conn.execute(table.ddl)
    yield conn
    conn.close()


def test_registers_every_view_in_dependency_order(connection):
    """Every lookup and form view is registered on the connection."""
    register_feature_views(connection)

    names = {
        row[0]
        for row in connection.execute(
            "SELECT view_name FROM duckdb_views()"
        ).fetchall()
    }
    for view in (
        "fci_slug_map",
        "fpl_team_id",
        "opta_match",
        "player_match_form",
        "team_match_form",
    ):
        assert view in names


def test_views_are_queryable_when_sources_are_empty(connection):
    """The views return zero rows rather than erroring on empty sources."""
    register_feature_views(connection)

    frame = connection.sql("SELECT * FROM player_match_form").pl()
    assert isinstance(frame, pl.DataFrame)
    assert frame.height == 0
