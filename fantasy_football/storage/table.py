"""Engine-agnostic table descriptors.

A ``Table`` declares everything the storage layer needs to know about one
stored table: its name, its column order and dtypes, its primary key, its
sort order, and any per-table quirks. Operations are expressed against
this declaration rather than hand-written per table.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass

import polars as pl

logger = logging.getLogger(__name__)


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
    if dtype == pl.Utf8:
        return "VARCHAR"
    if dtype == pl.Int64:
        return "BIGINT"
    if dtype == pl.Boolean:
        return "BOOLEAN"
    if dtype == pl.Float64:
        return "DOUBLE"
    if dtype == pl.Date:
        return "DATE"
    if dtype == pl.Datetime:
        return "TIMESTAMP"
    raise ValueError(f"No DuckDB type mapped for Polars dtype {dtype!r}")


@dataclass(frozen=True, eq=False)
class Table:
    """A declarative description of one stored table.

    ``eq=False`` is paired with ``frozen=True`` because ``schema`` is a
    dict and therefore unhashable. The six specs are module-level
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
        nullable. This rule reproduces all six legacy DDL strings exactly.

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
