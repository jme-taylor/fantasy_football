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


# Canonical team-fixture column order. Both the table and every frame written
# to or read from it use exactly these columns in this order.
TEAM_FIXTURE_COLUMNS: list[str] = [
    "season",
    "gw",
    "team",
    "is_home",
    "opposition",
    "kickoff_time",
]

# Polars dtypes incoming fixture frames are pinned to before insertion.
TEAM_FIXTURE_SCHEMA: dict[str, pl.DataType] = {
    "season": pl.Utf8,
    "gw": pl.Int64,
    "team": pl.Utf8,
    "is_home": pl.Boolean,
    "opposition": pl.Utf8,
    "kickoff_time": pl.Datetime("us"),
}

_CREATE_TEAM_FIXTURE_TABLE = """
CREATE TABLE IF NOT EXISTS team_fixture (
    season VARCHAR NOT NULL,
    gw BIGINT NOT NULL,
    team VARCHAR NOT NULL,
    is_home BOOLEAN,
    opposition VARCHAR NOT NULL,
    kickoff_time TIMESTAMP,
    PRIMARY KEY (season, gw, team, opposition)
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
        An open connection with the ``player_week`` and ``team_fixture`` tables
        guaranteed to exist. The caller is responsible for closing the
        connection.
    """
    path = db_path or DATABASE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(path))
    connection.execute(_CREATE_TABLE)
    connection.execute(_CREATE_TEAM_FIXTURE_TABLE)
    connection.execute(_CREATE_PLAYER_MATCH_TABLE)
    connection.execute(_CREATE_PLAYER_AVAILABILITY_TABLE)
    connection.execute(_CREATE_MINUTES_PREDICTION_TABLE)
    return connection


def coerce_player_week(frame: pl.DataFrame) -> pl.DataFrame:
    """Normalise an incoming frame to the canonical player-week shape.

    Collapses the legacy ``GKP`` position label to ``GK``, selects exactly
    ``PLAYER_WEEK_COLUMNS`` (ignoring any extra source columns), and pins the
    dtypes to ``PLAYER_WEEK_SCHEMA``.

    Parameters
    ----------
    frame : pl.DataFrame
        A frame already carrying every name in ``PLAYER_WEEK_COLUMNS``.

    Returns
    -------
    pl.DataFrame
        The frame reduced to the canonical columns, order, and dtypes.
    """
    return (
        frame.with_columns(
            pl.when(pl.col("position") == "GKP")
            .then(pl.lit("GK"))
            .otherwise(pl.col("position"))
            .alias("position")
        )
        .select(PLAYER_WEEK_COLUMNS)
        .cast(PLAYER_WEEK_SCHEMA, strict=False)
    )


def coerce_team_fixture(frame: pl.DataFrame) -> pl.DataFrame:
    """Normalise an incoming frame to the canonical team-fixture shape.

    Selects exactly ``TEAM_FIXTURE_COLUMNS`` (ignoring any extra source
    columns) and pins the dtypes to ``TEAM_FIXTURE_SCHEMA``.

    Parameters
    ----------
    frame : pl.DataFrame
        A frame already carrying every name in ``TEAM_FIXTURE_COLUMNS``.

    Returns
    -------
    pl.DataFrame
        The frame reduced to the canonical columns, order, and dtypes.
    """
    return frame.select(TEAM_FIXTURE_COLUMNS).cast(
        TEAM_FIXTURE_SCHEMA, strict=False
    )


def load_team_fixture(
    connection: duckdb.DuckDBPyConnection | None = None,
) -> pl.DataFrame:
    """Return the entire ``team_fixture`` table as a Polars frame.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection | None, optional
        An open connection. When None, a connection to ``DATABASE_PATH`` is
        opened and closed inside this call.

    Returns
    -------
    pl.DataFrame
        All team-fixture rows, columns in ``TEAM_FIXTURE_COLUMNS`` order,
        sorted by ``(season, gw, team)``.
    """
    owns_connection = connection is None
    conn = connection or get_connection()
    try:
        return conn.execute(
            "SELECT season, gw, team, is_home, opposition, kickoff_time "
            "FROM team_fixture ORDER BY season, gw, team"
        ).pl()
    finally:
        if owns_connection:
            conn.close()


def seasons_present(connection: duckdb.DuckDBPyConnection) -> set[str]:
    """Return the set of seasons already stored in ``player_week``.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        An open connection.

    Returns
    -------
    set[str]
        Distinct ``season`` values currently in the table.
    """
    rows = connection.execute(
        "SELECT DISTINCT season FROM player_week"
    ).fetchall()
    return {row[0] for row in rows}


def write_immutable_season(
    connection: duckdb.DuckDBPyConnection,
    frame: pl.DataFrame,
    season: str,
) -> None:
    """Insert a completed season's rows, but only if it is not already stored.

    Completed (historic and bridge) seasons never change, so an existing season
    is left untouched and the frame is discarded.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        An open connection.
    frame : pl.DataFrame
        Rows for a single season, carrying every ``PLAYER_WEEK_COLUMNS`` name.
    season : str
        The season these rows belong to.
    """
    if season in seasons_present(connection):
        logger.info(
            "Season %s already present; skipping immutable load.", season
        )
        return
    shaped = coerce_player_week(frame)
    connection.register("incoming_player_week", shaped.to_arrow())
    try:
        connection.execute(
            "INSERT INTO player_week SELECT * FROM incoming_player_week"
        )
    finally:
        connection.unregister("incoming_player_week")
    logger.info(
        "Inserted %d rows for immutable season %s", shaped.height, season
    )


def upsert_current_season(
    connection: duckdb.DuckDBPyConnection,
    frame: pl.DataFrame,
    season: str,
) -> None:
    """Replace all stored rows for ``season`` with a freshly fetched frame.

    The current season is re-fetched season-to-date each run. Deleting the
    season's existing rows and re-inserting guarantees late corrections (bonus,
    minutes) overwrite cleanly and brand-new gameweeks are added, while rows
    that vanished upstream do not linger. Other seasons are untouched.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        An open connection.
    frame : pl.DataFrame
        The season-to-date rows, carrying every ``PLAYER_WEEK_COLUMNS`` name.
    season : str
        The current season being refreshed.
    """
    shaped = coerce_player_week(frame)
    connection.register("incoming_player_week", shaped.to_arrow())
    try:
        connection.execute(
            "DELETE FROM player_week WHERE season = ?", [season]
        )
        connection.execute(
            "INSERT INTO player_week SELECT * FROM incoming_player_week"
        )
    finally:
        connection.unregister("incoming_player_week")
    logger.info(
        "Upserted %d rows for current season %s", shaped.height, season
    )


def load_player_week(
    connection: duckdb.DuckDBPyConnection | None = None,
) -> pl.DataFrame:
    """Return the entire ``player_week`` table as a Polars frame.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection | None, optional
        An open connection. When None, a connection to ``DATABASE_PATH`` is
        opened and closed inside this call.

    Returns
    -------
    pl.DataFrame
        All player-week rows, columns in ``PLAYER_WEEK_COLUMNS`` order, sorted
        by ``(season, gw, element)``.
    """
    owns_connection = connection is None
    conn = connection or get_connection()
    try:
        return conn.execute(
            "SELECT season, gw, element, name, position, team, bonus, "
            "minutes, round, total_points, value FROM player_week "
            "ORDER BY season, gw, element"
        ).pl()
    finally:
        if owns_connection:
            conn.close()


def fixture_seasons_present(connection: duckdb.DuckDBPyConnection) -> set[str]:
    """Return the set of seasons already stored in ``team_fixture``.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        An open connection.

    Returns
    -------
    set[str]
        Distinct ``season`` values currently in the table.
    """
    rows = connection.execute(
        "SELECT DISTINCT season FROM team_fixture"
    ).fetchall()
    return {row[0] for row in rows}


def write_immutable_fixtures(
    connection: duckdb.DuckDBPyConnection,
    frame: pl.DataFrame,
    season: str,
) -> None:
    """Insert a completed season's fixtures, but only if not already stored.

    Completed seasons' fixtures never change, so an existing season is left
    untouched and the frame is discarded.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        An open connection.
    frame : pl.DataFrame
        Rows for a single season, carrying every ``TEAM_FIXTURE_COLUMNS`` name.
    season : str
        The season these rows belong to.
    """
    if season in fixture_seasons_present(connection):
        logger.info(
            "Fixtures for season %s already present; skipping.", season
        )
        return
    shaped = coerce_team_fixture(frame)
    connection.register("incoming_team_fixture", shaped.to_arrow())
    try:
        connection.execute(
            "INSERT INTO team_fixture SELECT * FROM incoming_team_fixture"
        )
    finally:
        connection.unregister("incoming_team_fixture")
    logger.info(
        "Inserted %d fixture rows for immutable season %s",
        shaped.height,
        season,
    )


def upsert_current_fixtures(
    connection: duckdb.DuckDBPyConnection,
    frame: pl.DataFrame,
    season: str,
) -> None:
    """Replace all stored fixtures for ``season`` with a freshly fetched frame.

    The current season's fixtures are re-fetched each run (kickoff times and
    new gameweeks can change). Deleting then re-inserting the season keeps the
    table consistent with the upstream schedule. Other seasons are untouched.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        An open connection.
    frame : pl.DataFrame
        The season's fixtures, carrying every ``TEAM_FIXTURE_COLUMNS`` name.
    season : str
        The current season being refreshed.
    """
    shaped = coerce_team_fixture(frame)
    connection.register("incoming_team_fixture", shaped.to_arrow())
    try:
        connection.execute(
            "DELETE FROM team_fixture WHERE season = ?", [season]
        )
        connection.execute(
            "INSERT INTO team_fixture SELECT * FROM incoming_team_fixture"
        )
    finally:
        connection.unregister("incoming_team_fixture")
    logger.info(
        "Upserted %d fixture rows for current season %s", shaped.height, season
    )


# Canonical player-match column order. One row per (player, fixture); the table
# and every frame written to or read from it use exactly these columns in this
# order.
PLAYER_MATCH_COLUMNS: list[str] = [
    "season",
    "gw",
    "element",
    "opponent",
    "is_home",
    "minutes",
    "total_points",
]

# Polars dtypes incoming player-match frames are pinned to before insertion.
PLAYER_MATCH_SCHEMA: dict[str, pl.DataType] = {
    "season": pl.Utf8,
    "gw": pl.Int64,
    "element": pl.Int64,
    "opponent": pl.Int64,
    "is_home": pl.Boolean,
    "minutes": pl.Int64,
    "total_points": pl.Int64,
}

_CREATE_PLAYER_MATCH_TABLE = """
CREATE TABLE IF NOT EXISTS player_match (
    season VARCHAR NOT NULL,
    gw BIGINT NOT NULL,
    element BIGINT NOT NULL,
    opponent BIGINT NOT NULL,
    is_home BOOLEAN,
    minutes BIGINT,
    total_points BIGINT,
    PRIMARY KEY (season, gw, element, opponent)
)
"""


def coerce_player_match(frame: pl.DataFrame) -> pl.DataFrame:
    """Reduce a frame to the canonical player-match columns, order, and dtypes.

    Selects exactly ``PLAYER_MATCH_COLUMNS`` (ignoring any extra source
    columns) and pins the dtypes to ``PLAYER_MATCH_SCHEMA``.
    """
    return frame.select(PLAYER_MATCH_COLUMNS).cast(
        PLAYER_MATCH_SCHEMA, strict=False
    )


def player_match_seasons_present(
    connection: duckdb.DuckDBPyConnection,
) -> set[str]:
    """Return the set of seasons already stored in ``player_match``."""
    rows = connection.execute(
        "SELECT DISTINCT season FROM player_match"
    ).fetchall()
    return {row[0] for row in rows}


def write_immutable_player_match(
    connection: duckdb.DuckDBPyConnection,
    frame: pl.DataFrame,
    season: str,
) -> None:
    """Insert a completed season's player-match rows, only if not already stored."""
    if season in player_match_seasons_present(connection):
        logger.info(
            "Player-match season %s already present; skipping.", season
        )
        return
    shaped = coerce_player_match(frame)
    connection.register("incoming_player_match", shaped.to_arrow())
    try:
        connection.execute(
            "INSERT INTO player_match SELECT * FROM incoming_player_match"
        )
    finally:
        connection.unregister("incoming_player_match")
    logger.info(
        "Inserted %d player-match rows for immutable season %s",
        shaped.height,
        season,
    )


def upsert_current_player_match(
    connection: duckdb.DuckDBPyConnection,
    frame: pl.DataFrame,
    season: str,
) -> None:
    """Replace all stored player-match rows for ``season`` with a fresh frame."""
    shaped = coerce_player_match(frame)
    connection.register("incoming_player_match", shaped.to_arrow())
    try:
        connection.execute(
            "DELETE FROM player_match WHERE season = ?", [season]
        )
        connection.execute(
            "INSERT INTO player_match SELECT * FROM incoming_player_match"
        )
    finally:
        connection.unregister("incoming_player_match")
    logger.info(
        "Upserted %d player-match rows for current season %s",
        shaped.height,
        season,
    )


def load_player_match(
    connection: duckdb.DuckDBPyConnection | None = None,
) -> pl.DataFrame:
    """Return the entire ``player_match`` table as a Polars frame.

    Rows are ordered by ``(season, gw, element, opponent)``.
    """
    owns_connection = connection is None
    conn = connection or get_connection()
    try:
        return conn.execute(
            "SELECT season, gw, element, opponent, is_home, minutes, "
            "total_points FROM player_match "
            "ORDER BY season, gw, element, opponent"
        ).pl()
    finally:
        if owns_connection:
            conn.close()


# Canonical player-availability column order. One row per (season, gw, element);
# chance_of_playing_this_round is FPL's point-in-time availability percentage
# for that gameweek (null coalesced to 100 at extraction time).
PLAYER_AVAILABILITY_COLUMNS: list[str] = [
    "season",
    "gw",
    "element",
    "chance_of_playing_this_round",
]

# Polars dtypes incoming player-availability frames are pinned to before insert.
PLAYER_AVAILABILITY_SCHEMA: dict[str, pl.DataType] = {
    "season": pl.Utf8,
    "gw": pl.Int64,
    "element": pl.Int64,
    "chance_of_playing_this_round": pl.Int64,
}

_CREATE_PLAYER_AVAILABILITY_TABLE = """
CREATE TABLE IF NOT EXISTS player_availability (
    season VARCHAR NOT NULL,
    gw BIGINT NOT NULL,
    element BIGINT NOT NULL,
    chance_of_playing_this_round BIGINT,
    PRIMARY KEY (season, gw, element)
)
"""


def coerce_player_availability(frame: pl.DataFrame) -> pl.DataFrame:
    """Reduce a frame to the canonical player-availability columns and dtypes."""
    return frame.select(PLAYER_AVAILABILITY_COLUMNS).cast(
        PLAYER_AVAILABILITY_SCHEMA, strict=False
    )


def player_availability_seasons_present(
    connection: duckdb.DuckDBPyConnection,
) -> set[str]:
    """Return the set of seasons already stored in ``player_availability``."""
    rows = connection.execute(
        "SELECT DISTINCT season FROM player_availability"
    ).fetchall()
    return {row[0] for row in rows}


def write_immutable_player_availability(
    connection: duckdb.DuckDBPyConnection,
    frame: pl.DataFrame,
    season: str,
) -> None:
    """Insert a completed season's availability rows, only if not already stored."""
    if season in player_availability_seasons_present(connection):
        logger.info(
            "Player-availability season %s already present; skipping.", season
        )
        return
    shaped = coerce_player_availability(frame)
    connection.register("incoming_player_availability", shaped.to_arrow())
    try:
        connection.execute(
            "INSERT INTO player_availability "
            "SELECT * FROM incoming_player_availability"
        )
    finally:
        connection.unregister("incoming_player_availability")
    logger.info(
        "Inserted %d player-availability rows for immutable season %s",
        shaped.height,
        season,
    )


def upsert_current_player_availability(
    connection: duckdb.DuckDBPyConnection,
    frame: pl.DataFrame,
    season: str,
) -> None:
    """Replace all stored availability rows for ``season`` with a fresh frame."""
    shaped = coerce_player_availability(frame)
    connection.register("incoming_player_availability", shaped.to_arrow())
    try:
        connection.execute(
            "DELETE FROM player_availability WHERE season = ?", [season]
        )
        connection.execute(
            "INSERT INTO player_availability "
            "SELECT * FROM incoming_player_availability"
        )
    finally:
        connection.unregister("incoming_player_availability")
    logger.info(
        "Upserted %d player-availability rows for current season %s",
        shaped.height,
        season,
    )


def load_player_availability(
    connection: duckdb.DuckDBPyConnection | None = None,
) -> pl.DataFrame:
    """Return the entire ``player_availability`` table as a Polars frame.

    Rows are ordered by ``(season, gw, element)``.
    """
    owns_connection = connection is None
    conn = connection or get_connection()
    try:
        return conn.execute(
            "SELECT season, gw, element, chance_of_playing_this_round "
            "FROM player_availability ORDER BY season, gw, element"
        ).pl()
    finally:
        if owns_connection:
            conn.close()


# Canonical minutes-prediction column order. One row per match (season, gw, element, opponent).
MINUTES_PREDICTION_COLUMNS: list[str] = [
    "season",
    "gw",
    "element",
    "opponent",
    "p_zero",
    "p_partial",
    "p_sixty_plus",
    "expected_minutes",
    "model_version",
]

# Polars dtypes incoming minutes-prediction frames are pinned to before insert.
MINUTES_PREDICTION_SCHEMA: dict[str, pl.DataType] = {
    "season": pl.Utf8,
    "gw": pl.Int64,
    "element": pl.Int64,
    "opponent": pl.Int64,
    "p_zero": pl.Float64,
    "p_partial": pl.Float64,
    "p_sixty_plus": pl.Float64,
    "expected_minutes": pl.Float64,
    "model_version": pl.Utf8,
}

_CREATE_MINUTES_PREDICTION_TABLE = """
CREATE TABLE IF NOT EXISTS minutes_prediction (
    season VARCHAR NOT NULL,
    gw BIGINT NOT NULL,
    element BIGINT NOT NULL,
    opponent BIGINT NOT NULL,
    p_zero DOUBLE,
    p_partial DOUBLE,
    p_sixty_plus DOUBLE,
    expected_minutes DOUBLE,
    model_version VARCHAR,
    PRIMARY KEY (season, gw, element, opponent)
)
"""


def coerce_minutes_prediction(frame: pl.DataFrame) -> pl.DataFrame:
    """Reduce a frame to the canonical minutes-prediction columns and dtypes.

    Selects exactly ``MINUTES_PREDICTION_COLUMNS`` (ignoring any extra source
    columns) and pins the dtypes to ``MINUTES_PREDICTION_SCHEMA``.

    Parameters
    ----------
    frame : pl.DataFrame
        A frame to coerce.

    Returns
    -------
    pl.DataFrame
        The coerced frame.
    """
    return frame.select(MINUTES_PREDICTION_COLUMNS).cast(
        MINUTES_PREDICTION_SCHEMA, strict=False
    )


# TODO(JT): Look into the best practice for this, would you keep the prev version predictions?
def upsert_minutes_prediction(
    connection: duckdb.DuckDBPyConnection,
    frame: pl.DataFrame,
    season: str,
) -> None:
    """Replace all stored minutes-prediction rows for ``season`` with a frame.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        An open connection.
    frame : pl.DataFrame
        A frame to upsert.
    season : str
        The season to upsert.
    """
    shaped = coerce_minutes_prediction(frame)
    connection.register("incoming_minutes_prediction", shaped.to_arrow())
    try:
        connection.execute(
            "DELETE FROM minutes_prediction WHERE season = ?", [season]
        )
        connection.execute(
            "INSERT INTO minutes_prediction "
            "SELECT * FROM incoming_minutes_prediction"
        )
    finally:
        connection.unregister("incoming_minutes_prediction")
    logger.info(
        "Upserted %d minutes-prediction rows for season %s",
        shaped.height,
        season,
    )


def load_minutes_prediction(
    connection: duckdb.DuckDBPyConnection | None = None,
) -> pl.DataFrame:
    """Return the entire ``minutes_prediction`` table as a Polars frame.

    Rows are ordered by ``(season, gw, element, opponent)``.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection | None, optional
        An open connection. Defaults to ``get_connection()``.

    Returns
    -------
    pl.DataFrame
        The entire ``minutes_prediction`` table as a Polars frame.
    """
    owns_connection = connection is None
    conn = connection or get_connection()
    try:
        return conn.execute(
            "SELECT " + ", ".join(MINUTES_PREDICTION_COLUMNS) + " "
            "FROM minutes_prediction "
            "ORDER BY season, gw, element, opponent"
        ).pl()
    finally:
        if owns_connection:
            conn.close()


def minutes_prediction_seasons_present(
    connection: duckdb.DuckDBPyConnection,
) -> set[str]:
    """Return the set of seasons already stored in ``minutes_prediction``.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        An open connection.

    Returns
    -------
    set[str]
        The set of seasons already stored in ``minutes_prediction``.
    """
    rows = connection.execute(
        "SELECT DISTINCT season FROM minutes_prediction"
    ).fetchall()
    return {row[0] for row in rows}


def minutes_prediction_versions(
    connection: duckdb.DuckDBPyConnection,
    seasons: list[str] | None = None,
) -> set[str]:
    """Return the distinct ``model_version`` values stored in the table.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        An open connection.
    seasons : list[str] | None, optional
        When given, restrict to these seasons (used to gate the historic
        backfill on the versions already stored for historic seasons).
    """
    if seasons is None:
        rows = connection.execute(
            "SELECT DISTINCT model_version FROM minutes_prediction"
        ).fetchall()
    else:
        placeholders = ", ".join("?" for _ in seasons)
        rows = connection.execute(
            "SELECT DISTINCT model_version FROM minutes_prediction "
            f"WHERE season IN ({placeholders})",
            seasons,
        ).fetchall()
    return {row[0] for row in rows if row[0] is not None}


def reset_database(connection: duckdb.DuckDBPyConnection) -> None:
    """Drop and recreate the ``player_week`` and ``team_fixture`` tables (full-rebuild escape hatch).

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        An open connection.
    """
    connection.execute("DROP TABLE IF EXISTS player_week")
    connection.execute(_CREATE_TABLE)
    connection.execute("DROP TABLE IF EXISTS team_fixture")
    connection.execute(_CREATE_TEAM_FIXTURE_TABLE)
    connection.execute("DROP TABLE IF EXISTS player_match")
    connection.execute(_CREATE_PLAYER_MATCH_TABLE)
    connection.execute("DROP TABLE IF EXISTS player_availability")
    connection.execute(_CREATE_PLAYER_AVAILABILITY_TABLE)
    connection.execute("DROP TABLE IF EXISTS minutes_prediction")
    connection.execute(_CREATE_MINUTES_PREDICTION_TABLE)
