"""DuckDB connection lifecycle for the storage layer's nine tables.

This module owns opening the database connection and resetting it. Table
schemas live as declarative ``Table`` specs in ``tables.py``; those specs
are built from the low-level DDL/DML descriptors in ``table.py``, which in
turn sit on the DuckDB primitives in ``engine.py``. This module just wires
the two lifecycle entry points -- opening a connection and resetting the
database -- to whatever tables ``tables.TABLES`` declares, so adding a
table never requires touching this file.
"""

from pathlib import Path

import duckdb

from fantasy_football.constants import DATABASE_PATH
from fantasy_football.storage import engine
from fantasy_football.storage.tables import TABLES


def get_connection(
    db_path: Path | None = None,
    check_drift: bool = True,
) -> duckdb.DuckDBPyConnection:
    """Open the duckdb database, creating the file and schema if absent.

    Parameters
    ----------
    db_path : Path | None, optional
        Path to the database file. Defaults to ``DATABASE_PATH``.
    check_drift : bool, optional
        When True (the default), verify the stored schema still matches the
        table specs and raise if it does not. Pass False when the caller is
        about to drop and recreate every table anyway -- a rebuild is the
        cure for drift, so it must not be blocked by the check.

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
    if check_drift:
        check_schema_drift(connection)
    return connection


def check_schema_drift(connection: duckdb.DuckDBPyConnection) -> None:
    """Fail loudly when a stored table is missing declared columns.

    ``CREATE TABLE IF NOT EXISTS`` is a no-op against a table that already
    exists, so adding a column to a ``Table`` spec leaves an older database
    file behind. The first read then dies with a DuckDB Binder Error naming
    only the column. This raises first, naming the table and the remedy.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        An open connection with the DDL already applied.

    Raises
    ------
    RuntimeError
        If any table lacks a column its spec declares.
    """
    drifted: list[str] = []
    for table in TABLES:
        stored = engine.table_columns(connection, table.name)
        missing = [column for column in table.columns if column not in stored]
        if missing:
            drifted.append(f"{table.name} (missing {', '.join(missing)})")
    if drifted:
        raise RuntimeError(
            "Stored schema is behind the table specs: "
            + "; ".join(drifted)
            + ". CREATE TABLE IF NOT EXISTS cannot add columns to an "
            "existing table -- run main(rebuild=True) to rebuild the "
            "database from scratch."
        )


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
