import duckdb
import polars as pl


def create(connection: duckdb.DuckDBPyConnection, ddl: str) -> None:
    """Execute a ``CREATE TABLE IF NOT EXISTS`` statement.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        An open connection.
    ddl : str
        The full create statement.
    """
    connection.execute(ddl)


def drop(connection: duckdb.DuckDBPyConnection, table_name: str) -> None:
    """Drop a table if it exists.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        An open connection.
    table_name : str
        The table to drop.
    """
    connection.execute(f"DROP TABLE IF EXISTS {table_name}")


def table_columns(
    connection: duckdb.DuckDBPyConnection, table_name: str
) -> set[str]:
    """Return the column names a table actually has in the database.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        An open connection.
    table_name : str
        The table to inspect.

    Returns
    -------
    set[str]
        The stored column names; empty when the table does not exist.
    """
    rows = connection.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = ?",
        [table_name],
    ).fetchall()
    return {row[0] for row in rows}


def insert_frame(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    frame: pl.DataFrame,
) -> None:
    """Append a frame to a table via a temporary Arrow view.

    The frame is registered zero-copy, inserted, and unregistered even if
    the insert raises.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        An open connection.
    table_name : str
        The destination table.
    frame : pl.DataFrame
        Rows already shaped to the table's columns and dtypes.
    """
    view = f"incoming_{table_name}"
    connection.register(view, frame.to_arrow())
    try:
        connection.execute(f"INSERT INTO {table_name} SELECT * FROM {view}")
    finally:
        connection.unregister(view)


def delete_season(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    season: str,
) -> None:
    """Delete every row belonging to one season.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        An open connection.
    table_name : str
        The table to delete from.
    season : str
        The season whose rows are removed.
    """
    connection.execute(f"DELETE FROM {table_name} WHERE season = ?", [season])


def delete_where(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    equals: dict[str, object],
    gw_from: int | None = None,
) -> None:
    """Delete rows matching equality predicates and an optional gw floor.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        An open connection.
    table_name : str
        The table to delete from.
    equals : dict[str, object]
        Column-to-value equality predicates, combined with AND.
    gw_from : int | None, optional
        When given, also require ``gw >= gw_from``. Defaults to None.
    """
    clauses = [f"{column} = ?" for column in equals]
    params: list[object] = list(equals.values())
    if gw_from is not None:
        clauses.append("gw >= ?")
        params.append(gw_from)
    where = " AND ".join(clauses)
    connection.execute(f"DELETE FROM {table_name} WHERE {where}", params)


def select(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    columns: list[str],
    order_by: tuple[str, ...],
) -> pl.DataFrame:
    """Read columns from a table into a Polars frame.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        An open connection.
    table_name : str
        The table to read.
    columns : list[str]
        Columns to select, in the order they should appear.
    order_by : tuple[str, ...]
        Columns to sort by.

    Returns
    -------
    pl.DataFrame
        The selected rows.
    """
    column_list = ", ".join(columns)
    order_list = ", ".join(order_by)
    return connection.execute(
        f"SELECT {column_list} FROM {table_name} " f"ORDER BY {order_list}"
    ).pl()


def distinct(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    column: str,
) -> set:
    """Return the distinct values of one column.

    Nulls are not filtered out; callers that need them removed do so
    themselves.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        An open connection.
    table_name : str
        The table to read.
    column : str
        The column whose distinct values are returned.

    Returns
    -------
    set
        Distinct values, possibly including ``None``.
    """
    rows = connection.execute(
        f"SELECT DISTINCT {column} FROM {table_name}"
    ).fetchall()
    return {row[0] for row in rows}
