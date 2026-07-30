"""DuckDB storage for player-week data — the single source of truth.

This module owns the DuckDB connection lifecycle. Table schemas live as
declarative ``Table`` specs in ``tables.py``; those specs are built from the
low-level DDL/DML descriptors in ``table.py``, which in turn sit on the
DuckDB primitives in ``engine.py``. This module just wires the two lifecycle
entry points -- opening a connection and resetting the database -- to
whatever tables ``tables.TABLES`` declares, so adding a table never requires
touching this file.
"""

import logging
from pathlib import Path

import duckdb

from fantasy_football.constants import DATABASE_PATH
from fantasy_football.storage import engine
from fantasy_football.storage.tables import TABLES

logger = logging.getLogger(__name__)


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
        An open connection with every table in ``TABLES`` guaranteed to
        exist. The caller is responsible for closing the connection.
    """
    path = db_path or DATABASE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(path))
    for table in TABLES:
        engine.create(connection, table.ddl)
    return connection


def reset_database(connection: duckdb.DuckDBPyConnection) -> None:
    """Drop and recreate every table in ``TABLES`` (full-rebuild escape hatch).

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        An open connection.
    """
    for table in TABLES:
        engine.drop(connection, table.name)
        engine.create(connection, table.ddl)
