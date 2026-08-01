import logging
from collections.abc import Callable
from dataclasses import dataclass

import duckdb
import polars as pl

from fantasy_football.storage import engine

logger = logging.getLogger(__name__)

_DUCKDB_TYPES: dict[pl.DataType, str] = {
    pl.Utf8: "VARCHAR",
    pl.Int64: "BIGINT",
    pl.Boolean: "BOOLEAN",
    pl.Float64: "DOUBLE",
    pl.Date: "DATE",
    pl.Datetime: "TIMESTAMP",
}


def duckdb_type(dtype: pl.DataType) -> str:
    """Map a Polars dtype onto its DuckDB column type.

    Parameters
    ----------
    dtype : pl.DataType
        The Polars dtype declared in a table's schema.

    Returns
    -------
    str
        The equivalent DuckDB column type.

    Raises
    ------
    ValueError
        If the dtype has no mapping, rather than emitting invalid SQL.
    """
    # Parametrised dtypes such as ``pl.Datetime("us")`` compare equal to
    # their bare class but hash differently, so they would miss the dict.
    try:
        return _DUCKDB_TYPES[dtype.base_type()]
    except KeyError:
        raise ValueError(
            f"No DuckDB type mapped for Polars dtype {dtype!r}"
        ) from None


@dataclass(frozen=True, eq=False)
class Table:
    """A declarative description of one stored table.

    ``eq=False`` is paired with ``frozen=True`` because ``schema`` is a
    dict and therefore unhashable. The nine specs are module-level
    singletons, so identity equality is the correct semantics.

    Attributes
    ----------
    name : str
        The table name in DuckDB.
    schema : dict[str, pl.DataType]
        Column order and dtypes. The single source of truth for both.
    primary_key : tuple[str, ...]
        Columns forming the primary key. These become NOT NULL.
    order_by : tuple[str, ...]
        Columns rows are sorted by when read.
    normalise : Callable | None
        Optional pre-write fixup applied before column selection.
    enrich : Callable | None
        Optional post-read transform applied after selection.
    """

    name: str
    schema: dict[str, pl.DataType]
    primary_key: tuple[str, ...]
    order_by: tuple[str, ...]
    normalise: Callable[[pl.DataFrame], pl.DataFrame] | None = None
    enrich: Callable[[pl.DataFrame], pl.DataFrame] | None = None

    @property
    def columns(self) -> list[str]:
        """Return the canonical column order.

        Returns
        -------
        list[str]
            The schema's keys, in declaration order.
        """
        return list(self.schema)

    @property
    def ddl(self) -> str:
        """Return the ``CREATE TABLE IF NOT EXISTS`` statement.

        Primary-key columns are declared NOT NULL; every other column is
        nullable.

        Returns
        -------
        str
            The full create statement.
        """
        lines = []
        for column, dtype in self.schema.items():
            suffix = " NOT NULL" if column in self.primary_key else ""
            lines.append(f"    {column} {duckdb_type(dtype)}{suffix}")
        lines.append(f"    PRIMARY KEY ({', '.join(self.primary_key)})")
        body = ",\n".join(lines)
        return f"CREATE TABLE IF NOT EXISTS {self.name} (\n{body}\n)"

    def conform(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Add any schema columns the frame lacks, as typed nulls.

        Source data is ragged across seasons: Vaastav published
        ``key_passes`` only from 2016-17 to 2018-19 and ``expected_goals``
        only from 2022-23, and FCI added ``corners`` in 2026-27. Without
        this, ``coerce``'s ``select`` raises on any season missing a
        column. Filling with a typed null lets every season reduce to one
        schema, and the null is honest: the stat was not published, which
        is not the same as zero. Which seasons those are is recorded in
        ``storage/coverage.py``.

        Extra columns are deliberately left in place; ``coerce`` drops
        them, and ``unknown_columns`` reports them first.

        Parameters
        ----------
        frame : pl.DataFrame
            A frame carrying some subset of this table's columns.

        Returns
        -------
        pl.DataFrame
            The frame plus a typed-null column for each missing one.
        """
        missing = [
            pl.lit(None, dtype=dtype).alias(column)
            for column, dtype in self.schema.items()
            if column not in frame.columns
        ]
        if not missing:
            return frame
        if frame.width == 0:
            # A zero-column frame has no column for ``with_columns`` to
            # take its height from, so each ``pl.lit(None)`` broadcasts
            # to a single row -- a phantom (1, N) frame rather than the
            # empty one the caller expects. Build the fully-typed empty
            # frame directly instead.
            return pl.DataFrame(schema=self.schema)
        return frame.with_columns(missing)

    def unknown_columns(self, frame: pl.DataFrame) -> list[str]:
        """Return frame columns this table's schema does not declare.

        ``coerce`` drops undeclared columns silently, which is how a new
        upstream stat disappears without anyone noticing. Loaders call
        this and log the result so the next addition is visible.

        Parameters
        ----------
        frame : pl.DataFrame
            A frame as read from the source.

        Returns
        -------
        list[str]
            Undeclared column names, sorted.
        """
        return sorted(set(frame.columns) - set(self.schema))

    def coerce(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Narrow a frame to this table's columns, order, and dtypes.

        Any ``normalise`` hook runs first. Extra source columns are
        dropped.

        Parameters
        ----------
        frame : pl.DataFrame
            A frame carrying at least this table's columns.

        Returns
        -------
        pl.DataFrame
            The frame reduced to the canonical shape.
        """
        shaped = frame if self.normalise is None else self.normalise(frame)
        return shaped.select(self.columns).cast(self.schema, strict=False)

    def load(
        self, connection: "duckdb.DuckDBPyConnection | None" = None
    ) -> pl.DataFrame:
        """Return the whole table as a Polars frame.

        Parameters
        ----------
        connection : duckdb.DuckDBPyConnection | None, optional
            An open connection. When None, one is opened against
            ``DATABASE_PATH`` and closed before returning.

        Returns
        -------
        pl.DataFrame
            All rows, in ``columns`` order, sorted by ``order_by``.
        """
        # Imported here because database.py imports tables.py, which
        # imports this module; a top-level import would be circular.
        from fantasy_football.storage.database import get_connection

        owns_connection = connection is None
        conn = connection or get_connection()
        try:
            frame = engine.select(conn, self.name, self.columns, self.order_by)
        finally:
            if owns_connection:
                conn.close()
        if self.enrich is None or frame.is_empty():
            return frame
        return self.enrich(frame)

    def seasons_present(
        self, connection: "duckdb.DuckDBPyConnection"
    ) -> set[str]:
        """Return the seasons already stored in this table.

        Parameters
        ----------
        connection : duckdb.DuckDBPyConnection
            An open connection.

        Returns
        -------
        set[str]
            Distinct ``season`` values currently stored.
        """
        return engine.distinct(connection, self.name, "season")

    def write_immutable(
        self,
        connection: "duckdb.DuckDBPyConnection",
        frame: pl.DataFrame,
        season: str,
    ) -> None:
        """Insert a completed season, but only if not already stored.

        Completed seasons never change, so an existing season is left
        untouched and the frame is discarded.

        Parameters
        ----------
        connection : duckdb.DuckDBPyConnection
            An open connection.
        frame : pl.DataFrame
            Rows for a single season.
        season : str
            The season these rows belong to.
        """
        if season in self.seasons_present(connection):
            logger.info(
                "%s season %s already present; skipping immutable load.",
                self.name,
                season,
            )
            return
        shaped = self.coerce(frame)
        engine.insert_frame(connection, self.name, shaped)
        logger.info(
            "Inserted %d %s rows for immutable season %s",
            shaped.height,
            self.name,
            season,
        )

    def append(
        self,
        connection: "duckdb.DuckDBPyConnection",
        frame: pl.DataFrame,
    ) -> None:
        """Insert rows without deleting anything first.

        For append-only tables where each write is a new partition of the
        primary key rather than a correction of an existing one.

        Parameters
        ----------
        connection : duckdb.DuckDBPyConnection
            An open connection.
        frame : pl.DataFrame
            The rows to insert.
        """
        shaped = self.coerce(frame)
        engine.insert_frame(connection, self.name, shaped)
        logger.info("Appended %d %s rows", shaped.height, self.name)

    def replace_partition(
        self,
        connection: "duckdb.DuckDBPyConnection",
        frame: pl.DataFrame,
        equals: dict[str, object],
        gw_from: int | None = None,
    ) -> None:
        """Replace one partition's rows, leaving every other row alone.

        Deletes rows matching every predicate in ``equals`` and, when
        ``gw_from`` is given, ``gw >= gw_from``; then inserts ``frame``.
        Rows below ``gw_from`` survive untouched, which is what lets a
        forecast for an already-played gameweek stay frozen.

        Parameters
        ----------
        connection : duckdb.DuckDBPyConnection
            An open connection.
        frame : pl.DataFrame
            The replacement rows.
        equals : dict[str, object]
            Column-to-value equality predicates identifying the partition.
        gw_from : int | None, optional
            Lower gameweek bound on the delete. Defaults to None, meaning
            the whole partition is replaced.
        """
        shaped = self.coerce(frame)
        engine.delete_where(connection, self.name, equals, gw_from)
        engine.insert_frame(connection, self.name, shaped)
        logger.info(
            "Replaced %d %s rows for %s",
            shaped.height,
            self.name,
            equals,
        )

    def upsert_current(
        self,
        connection: "duckdb.DuckDBPyConnection",
        frame: pl.DataFrame,
        season: str,
    ) -> None:
        """Replace all stored rows for ``season`` with a fresh frame.

        Deleting then re-inserting guarantees late corrections overwrite
        cleanly and new gameweeks are added, while rows that vanished
        upstream do not linger. Other seasons are untouched.

        Parameters
        ----------
        connection : duckdb.DuckDBPyConnection
            An open connection.
        frame : pl.DataFrame
            The season-to-date rows.
        season : str
            The season being refreshed.
        """
        self.replace_partition(connection, frame, {"season": season})
