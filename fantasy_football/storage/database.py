"""DuckDB storage for player-week data — the single source of truth.

This module is the only place that touches duckdb. Extractors hand it Polars
frames; downstream code reads frames back. The ``player_week`` table holds one
row per (player, gameweek) across every season, keyed on
``(season, gw, element)``.
"""

import logging
from pathlib import Path

import duckdb
import polars as pl

from fantasy_football.constants import DATABASE_PATH

logger = logging.getLogger(__name__)

# Canonical player-week column order. Both the table and every frame written
# to or read from it use exactly these columns in this order.
PLAYER_WEEK_COLUMNS: list[str] = [
    "season",
    "gw",
    "element",
    "name",
    "position",
    "team",
    "bonus",
    "minutes",
    "round",
    "total_points",
    "value",
]

# Polars dtypes the incoming frames are pinned to before insertion, so a value
# read as Float64 in one source and Int64 in another lands consistently.
PLAYER_WEEK_SCHEMA: dict[str, pl.DataType] = {
    "season": pl.Utf8,
    "gw": pl.Int64,
    "element": pl.Int64,
    "name": pl.Utf8,
    "position": pl.Utf8,
    "team": pl.Utf8,
    "bonus": pl.Int64,
    "minutes": pl.Int64,
    "round": pl.Int64,
    "total_points": pl.Int64,
    "value": pl.Int64,
}

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


def get_connection(
    db_path: Path | None = None,
) -> duckdb.DuckDBPyConnection:
    """Open the duckdb database, creating the file and schema if absent.

    Parameters
    ----------
    db_path : Path | None, optional
        Path to the database file. Defaults to ``DATABASE_PATH``.

    Returns
    -------
    duckdb.DuckDBPyConnection
        An open connection with the ``player_week`` table guaranteed to exist.
        The caller is responsible for closing the connection.
    """
    path = db_path or DATABASE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(path))
    connection.execute(_CREATE_TABLE)
    return connection
