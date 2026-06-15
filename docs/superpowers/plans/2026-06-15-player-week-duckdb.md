# Player-week data into duckdb Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move all player-week data into a single embedded duckdb database that becomes the source of truth, with incremental-by-default ingestion (immutable seasons loaded once, current season upserted each run).

**Architecture:** A new `fantasy_football/storage/database.py` module owns every duckdb interaction (connection, schema, reads, writes). The Vaastav (`DataExtractor`) and FCI (`FciExtractor`) extractors keep their existing Polars reshaping logic but change their *sink* from CSV files to DB write functions. `load_gw_data` reads the whole `player_week` table from the DB instead of stitching CSVs. `cleaned_merged_seasons.csv` and the per-season `merged_gw.csv` files are retired.

**Tech Stack:** Python, Polars, duckdb, pytest, pytest-mock.

---

## File Structure

- **Create** `fantasy_football/storage/__init__.py` — empty package marker.
- **Create** `fantasy_football/storage/database.py` — duckdb owner: connection, schema, `coerce_player_week`, `seasons_present`, `write_immutable_season`, `upsert_current_season`, `load_player_week`, `reset_database`.
- **Create** `tests/storage/__init__.py`, `tests/storage/test_database.py` — unit tests for the storage module.
- **Modify** `fantasy_football/constants.py` — add `DATABASE_PATH`.
- **Modify** `pyproject.toml` — add `duckdb` dependency.
- **Modify** `.gitignore` — explicit `data/*.duckdb` line (documentation of intent).
- **Modify** `fantasy_football/extraction/extractor.py` — add `_read_csv`, `load_immutable_seasons`, `_historic_loaded`; retire `save_all_data_files`.
- **Modify** `fantasy_football/extraction/fci.py` — `build_current_season_merged_gw` upserts to the DB instead of writing CSV.
- **Modify** `fantasy_football/features/transformation.py` — `load_gw_data` reads from the DB; delete `_load_season_merged_gw` and CSV reads.
- **Modify** `main.py` — replace `download_all_data` flag with `rebuild`; open/reset/close the connection; route extraction through the DB.
- **Modify** `tests/extraction/test_extractor.py`, `tests/extraction/test_fci.py`, `tests/features/test_transformation.py` — update to the new DB-backed behavior.

---

## Task 1: Dependency, constant, and gitignore

**Files:**
- Modify: `pyproject.toml`
- Modify: `fantasy_football/constants.py`
- Modify: `.gitignore`

- [ ] **Step 1: Add the duckdb dependency**

In `pyproject.toml`, find the `dependencies` array containing `"polars>=0.20.3,<0.21",` and add a duckdb entry alongside it:

```toml
    "duckdb>=1.0,<2",
```

- [ ] **Step 2: Sync the environment**

Run: `uv sync`
Expected: completes successfully, `duckdb` installed into `.venv`.

- [ ] **Step 3: Add the DATABASE_PATH constant**

In `fantasy_football/constants.py`, immediately after the line `MLFLOW_DB_PATH = MODELS_FOLDER.joinpath("mlflow.db")`, add:

```python
# Single-file duckdb database; source of truth for player-week data.
# Lives under the gitignored data/ folder.
DATABASE_PATH = DATA_FOLDER.joinpath("fantasy_football.duckdb")
```

- [ ] **Step 4: Add an explicit gitignore line**

In `.gitignore`, under the existing data section (the lines `# data folder + other local data` / `data/`), add an explicit entry documenting intent:

```gitignore
data/*.duckdb
```

- [ ] **Step 5: Verify duckdb imports**

Run: `uv run python -c "import duckdb; print(duckdb.__version__)"`
Expected: prints a version `1.x.x` with no error.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock fantasy_football/constants.py .gitignore
git commit -m "build: add duckdb dependency and DATABASE_PATH constant"
```

---

## Task 2: Connection and schema

**Files:**
- Create: `fantasy_football/storage/__init__.py`
- Create: `fantasy_football/storage/database.py`
- Create: `tests/storage/__init__.py`
- Create: `tests/storage/test_database.py`

- [ ] **Step 1: Create the package markers**

Create `fantasy_football/storage/__init__.py` as an empty file.
Create `tests/storage/__init__.py` as an empty file.

- [ ] **Step 2: Write the failing test**

Create `tests/storage/test_database.py`:

```python
from pathlib import Path

import polars as pl
import pytest

from fantasy_football.storage.database import get_connection


def test_get_connection_creates_player_week_table(tmp_path: Path) -> None:
    """get_connection creates the DB file and an empty player_week table."""
    db_path = tmp_path / "test.duckdb"
    connection = get_connection(db_path)
    try:
        columns = [
            row[0]
            for row in connection.execute("DESCRIBE player_week").fetchall()
        ]
    finally:
        connection.close()

    assert db_path.exists()
    assert columns == [
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


def test_get_connection_is_idempotent(tmp_path: Path) -> None:
    """Opening the same DB twice does not error or duplicate the schema."""
    db_path = tmp_path / "test.duckdb"
    first = get_connection(db_path)
    first.close()
    second = get_connection(db_path)
    try:
        count = second.execute("SELECT COUNT(*) FROM player_week").fetchone()[0]
    finally:
        second.close()
    assert count == 0
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run pytest tests/storage/test_database.py -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError` for `get_connection`.

- [ ] **Step 4: Write the implementation**

Create `fantasy_football/storage/database.py`:

```python
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
    gw INTEGER NOT NULL,
    element INTEGER NOT NULL,
    name VARCHAR,
    position VARCHAR,
    team VARCHAR,
    bonus INTEGER,
    minutes INTEGER,
    round INTEGER,
    total_points INTEGER,
    value INTEGER,
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
    """
    path = db_path or DATABASE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(path))
    connection.execute(_CREATE_TABLE)
    return connection
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/storage/test_database.py -v`
Expected: both tests PASS.

- [ ] **Step 6: Commit**

```bash
git add fantasy_football/storage tests/storage
git commit -m "feat(storage): add duckdb connection and player_week schema"
```

---

## Task 3: `coerce_player_week` pure helper

**Files:**
- Modify: `fantasy_football/storage/database.py`
- Test: `tests/storage/test_database.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/storage/test_database.py`:

```python
from fantasy_football.storage.database import (  # noqa: E402
    PLAYER_WEEK_COLUMNS,
    coerce_player_week,
)


def test_coerce_player_week_normalises_gkp_and_selects_columns() -> None:
    """GKP collapses to GK; output is exactly PLAYER_WEEK_COLUMNS in order."""
    raw = pl.DataFrame(
        {
            "season": ["2020-21"],
            "gw": [1],
            "element": [1],
            "name": ["Player1"],
            "position": ["GKP"],
            "team": ["Arsenal"],
            "bonus": [1],
            "minutes": [90],
            "round": [1],
            "total_points": [6],
            "value": [50],
            "unused_extra": ["drop me"],
        }
    )

    result = coerce_player_week(raw)

    assert result.columns == PLAYER_WEEK_COLUMNS
    assert result["position"].to_list() == ["GK"]


def test_coerce_player_week_pins_dtypes() -> None:
    """Numeric columns arriving as Float64 are cast to Int64."""
    raw = pl.DataFrame(
        {
            "season": ["2020-21"],
            "gw": [1],
            "element": [1],
            "name": ["Player1"],
            "position": ["GK"],
            "team": ["Arsenal"],
            "bonus": [1.0],
            "minutes": [90.0],
            "round": [1],
            "total_points": [6],
            "value": [50],
        }
    )

    result = coerce_player_week(raw)

    assert result.schema["bonus"] == pl.Int64
    assert result.schema["minutes"] == pl.Int64
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/storage/test_database.py -k coerce -v`
Expected: FAIL with `ImportError` for `coerce_player_week`.

- [ ] **Step 3: Write the implementation**

In `fantasy_football/storage/database.py`, add after `get_connection`:

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/storage/test_database.py -k coerce -v`
Expected: both `coerce` tests PASS.

- [ ] **Step 5: Commit**

```bash
git add fantasy_football/storage/database.py tests/storage/test_database.py
git commit -m "feat(storage): add coerce_player_week shaping helper"
```

---

## Task 4: `seasons_present` and `write_immutable_season`

**Files:**
- Modify: `fantasy_football/storage/database.py`
- Test: `tests/storage/test_database.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/storage/test_database.py`:

```python
from fantasy_football.storage.database import (  # noqa: E402
    seasons_present,
    write_immutable_season,
)


def _historic_row(season: str, element: int, name: str) -> pl.DataFrame:
    """Build a one-row player-week frame for a given season/player."""
    return pl.DataFrame(
        {
            "season": [season],
            "gw": [1],
            "element": [element],
            "name": [name],
            "position": ["DEF"],
            "team": ["Arsenal"],
            "bonus": [1],
            "minutes": [90],
            "round": [1],
            "total_points": [6],
            "value": [50],
        }
    )


def test_seasons_present_reflects_writes(tmp_path: Path) -> None:
    """seasons_present returns the distinct seasons inserted so far."""
    connection = get_connection(tmp_path / "t.duckdb")
    try:
        assert seasons_present(connection) == set()
        write_immutable_season(
            connection, _historic_row("2020-21", 1, "P1"), "2020-21"
        )
        assert seasons_present(connection) == {"2020-21"}
    finally:
        connection.close()


def test_write_immutable_season_is_noop_when_present(tmp_path: Path) -> None:
    """Re-writing an existing season does not change or duplicate its rows."""
    connection = get_connection(tmp_path / "t.duckdb")
    try:
        write_immutable_season(
            connection, _historic_row("2020-21", 1, "Original"), "2020-21"
        )
        # Attempt to overwrite the same season with different data.
        write_immutable_season(
            connection, _historic_row("2020-21", 1, "Changed"), "2020-21"
        )
        rows = connection.execute(
            "SELECT name FROM player_week WHERE season = '2020-21'"
        ).fetchall()
    finally:
        connection.close()

    assert rows == [("Original",)]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/storage/test_database.py -k "seasons_present or immutable" -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write the implementation**

In `fantasy_football/storage/database.py`, add:

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/storage/test_database.py -k "seasons_present or immutable" -v`
Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add fantasy_football/storage/database.py tests/storage/test_database.py
git commit -m "feat(storage): add seasons_present and immutable-season write"
```

---

## Task 5: `upsert_current_season`

**Files:**
- Modify: `fantasy_football/storage/database.py`
- Test: `tests/storage/test_database.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/storage/test_database.py`:

```python
from fantasy_football.storage.database import (  # noqa: E402
    upsert_current_season,
)


def _current_row(
    gw: int, element: int, total_points: int
) -> dict[str, object]:
    """Build one current-season player-week record as a dict of lists' values."""
    return {
        "season": "2025-26",
        "gw": gw,
        "element": element,
        "name": "Player",
        "position": "MID",
        "team": "Arsenal",
        "bonus": 0,
        "minutes": 90,
        "round": gw,
        "total_points": total_points,
        "value": 75,
    }


def _frame(records: list[dict[str, object]]) -> pl.DataFrame:
    """Build a player-week frame from a list of record dicts."""
    return pl.DataFrame(records)


def test_upsert_overwrites_changed_row_and_inserts_new_gw(
    tmp_path: Path,
) -> None:
    """A later upsert corrects an existing GW's value and adds the next GW."""
    connection = get_connection(tmp_path / "t.duckdb")
    try:
        # First pull: GW1 provisional points.
        upsert_current_season(
            connection, _frame([_current_row(1, 1, 5)]), "2025-26"
        )
        # Second pull: GW1 corrected to 7, GW2 newly available.
        upsert_current_season(
            connection,
            _frame([_current_row(1, 1, 7), _current_row(2, 1, 9)]),
            "2025-26",
        )
        rows = connection.execute(
            "SELECT gw, total_points FROM player_week "
            "WHERE season = '2025-26' ORDER BY gw"
        ).fetchall()
    finally:
        connection.close()

    assert rows == [(1, 7), (2, 9)]


def test_upsert_does_not_touch_other_seasons(tmp_path: Path) -> None:
    """Upserting the current season leaves stored prior seasons intact."""
    connection = get_connection(tmp_path / "t.duckdb")
    try:
        write_immutable_season(
            connection, _historic_row("2020-21", 1, "Old"), "2020-21"
        )
        upsert_current_season(
            connection, _frame([_current_row(1, 1, 5)]), "2025-26"
        )
        assert seasons_present(connection) == {"2020-21", "2025-26"}
    finally:
        connection.close()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/storage/test_database.py -k upsert -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write the implementation**

In `fantasy_football/storage/database.py`, add:

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/storage/test_database.py -k upsert -v`
Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add fantasy_football/storage/database.py tests/storage/test_database.py
git commit -m "feat(storage): add current-season upsert"
```

---

## Task 6: `load_player_week` and `reset_database`

**Files:**
- Modify: `fantasy_football/storage/database.py`
- Test: `tests/storage/test_database.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/storage/test_database.py`:

```python
from fantasy_football.storage import database  # noqa: E402
from fantasy_football.storage.database import (  # noqa: E402
    load_player_week,
    reset_database,
)


def test_load_player_week_returns_all_rows_ordered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """load_player_week opens the default DB and returns every row, ordered."""
    db_path = tmp_path / "t.duckdb"
    monkeypatch.setattr(database, "DATABASE_PATH", db_path)
    connection = get_connection(db_path)
    write_immutable_season(
        connection, _historic_row("2020-21", 2, "Older"), "2020-21"
    )
    upsert_current_season(
        connection, _frame([_current_row(1, 1, 5)]), "2025-26"
    )
    connection.close()

    result = load_player_week()

    assert result.columns == PLAYER_WEEK_COLUMNS
    assert result["season"].to_list() == ["2020-21", "2025-26"]


def test_reset_database_empties_the_table(tmp_path: Path) -> None:
    """reset_database drops all rows but leaves a usable empty table."""
    connection = get_connection(tmp_path / "t.duckdb")
    try:
        write_immutable_season(
            connection, _historic_row("2020-21", 1, "P1"), "2020-21"
        )
        reset_database(connection)
        assert seasons_present(connection) == set()
    finally:
        connection.close()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/storage/test_database.py -k "load_player_week or reset" -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write the implementation**

In `fantasy_football/storage/database.py`, add:

```python
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


def reset_database(connection: duckdb.DuckDBPyConnection) -> None:
    """Drop and recreate the ``player_week`` table (full-rebuild escape hatch).

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        An open connection.
    """
    connection.execute("DROP TABLE IF EXISTS player_week")
    connection.execute(_CREATE_TABLE)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/storage/test_database.py -v`
Expected: every test in the file PASSES.

- [ ] **Step 5: Commit**

```bash
git add fantasy_football/storage/database.py tests/storage/test_database.py
git commit -m "feat(storage): add load_player_week reader and reset_database"
```

---

## Task 7: FCI extractor upserts to the DB

**Files:**
- Modify: `fantasy_football/extraction/fci.py:278-310` (the `build_current_season_merged_gw` method)
- Test: `tests/extraction/test_fci.py:166-201`

- [ ] **Step 1: Update the failing test**

In `tests/extraction/test_fci.py`, replace the entire
`test_build_current_season_merged_gw_writes_contract_columns` function
(lines 166-201) with:

```python
def test_build_current_season_merged_gw_upserts_to_db(
    mocker: MockerFixture, tmp_path
) -> None:
    """End-to-end build upserts player-week rows readable via load_player_week."""
    from fantasy_football.storage.database import (
        PLAYER_WEEK_COLUMNS,
        get_connection,
        load_player_week,
    )

    extractor = FciExtractor(
        api_client=GitHubAPIClient(
            api_key="k", owner="o", repo="r", branch="main"
        ),
        fpl_api=mocker.Mock(),
    )
    mocker.patch.object(
        extractor,
        "fetch_season_frames",
        return_value=(SNAPSHOTS, MATCHSTATS, PLAYERS),
    )
    mocker.patch.object(
        extractor, "_team_code_to_name", return_value=TEAM_CODE_TO_NAME
    )

    connection = get_connection(tmp_path / "t.duckdb")
    try:
        extractor.build_current_season_merged_gw("2025-26", connection)
        stored = load_player_week(connection)
    finally:
        connection.close()

    assert stored.columns == PLAYER_WEEK_COLUMNS
    assert stored.height == 4
    assert stored["season"].unique().to_list() == ["2025-26"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/extraction/test_fci.py -k upserts -v`
Expected: FAIL — `build_current_season_merged_gw` does not accept a `connection` argument.

- [ ] **Step 3: Update the implementation**

In `fantasy_football/extraction/fci.py`, add this import near the other
`from fantasy_football...` imports at the top of the file:

```python
from fantasy_football.storage.database import (
    coerce_player_week,
    upsert_current_season,
)
```

Then replace the entire `build_current_season_merged_gw` method (lines 278-310)
with:

```python
    def build_current_season_merged_gw(
        self, short_season: str, connection: "DuckDBPyConnection"
    ) -> pl.DataFrame:
        """Build current-season player-week rows and upsert them into the DB.

        Parameters
        ----------
        short_season : str
            Short-form season string, e.g. ``"2025-26"``.
        connection : duckdb.DuckDBPyConnection
            Open connection to the player-week database.

        Returns
        -------
        pl.DataFrame
            The coerced player-week frame that was upserted.
        """
        long_season = season_short_to_long(short_season)
        snapshots, matchstats, players = self.fetch_season_frames(long_season)
        merged = build_merged_gw(
            snapshots, matchstats, players, self._team_code_to_name()
        )
        player_week = merged.rename({"GW": "gw"}).with_columns(
            pl.lit(short_season).alias("season")
        )
        upsert_current_season(connection, player_week, short_season)
        logger.info(
            "Upserted %d FCI rows for %s", player_week.height, short_season
        )
        return coerce_player_week(player_week)
```

Add the type-only import for the annotation at the top of the file, under a
`TYPE_CHECKING` guard (add the `typing` import if not already present):

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/extraction/test_fci.py -v`
Expected: all tests PASS (the `build_merged_gw` tests are unchanged and still pass).

- [ ] **Step 5: Commit**

```bash
git add fantasy_football/extraction/fci.py tests/extraction/test_fci.py
git commit -m "feat(extraction): FCI current-season build upserts to duckdb"
```

---

## Task 8: Vaastav extractor loads immutable seasons into the DB

**Files:**
- Modify: `fantasy_football/extraction/extractor.py` (add imports, `_read_csv`, `_historic_loaded`, `load_immutable_seasons`; remove `save_all_data_files`)
- Test: `tests/extraction/test_extractor.py:234-250` (replace the `save_all_data_files` test)

- [ ] **Step 1: Write the failing tests**

In `tests/extraction/test_extractor.py`, replace the function
`test_save_all_data_files_downloads_aggregate_and_bridge_seasons`
(lines 234-250) with:

```python
def test_historic_loaded_detects_seasons_beyond_current_and_bridge() -> None:
    """A season outside current/bridge means the historic aggregate is loaded."""
    from fantasy_football.extraction.extractor import _historic_loaded

    assert not _historic_loaded(set(), "2025-26", ["2024-25"])
    assert not _historic_loaded({"2025-26", "2024-25"}, "2025-26", ["2024-25"])
    assert _historic_loaded({"2020-21"}, "2025-26", ["2024-25"])


def test_load_immutable_seasons_loads_aggregate_and_bridge(
    mocker: MockerFixture, mock_data_extractor: DataExtractor, tmp_path
) -> None:
    """Aggregate (split by season) and bridge seasons are inserted once."""
    from fantasy_football.storage.database import (
        get_connection,
        seasons_present,
    )

    aggregate = pl.DataFrame(
        {
            "season_x": ["2019-20", "2020-21"],
            "name": ["A", "B"],
            "position": ["GKP", "DEF"],
            "team_x": ["Arsenal", "Chelsea"],
            "bonus": [1, 2],
            "element": [1, 2],
            "minutes": [90, 90],
            "round": [1, 1],
            "total_points": [6, 8],
            "value": [50, 55],
            "GW": [1, 1],
        }
    )
    bridge = pl.DataFrame(
        {
            "name": ["C"],
            "position": ["MID"],
            "team": ["Leeds"],
            "bonus": [3],
            "element": [3],
            "minutes": [90],
            "round": [5],
            "total_points": [9],
            "value": [60],
            "GW": [5],
        }
    )

    def fake_read_csv(path: str) -> pl.DataFrame:
        return bridge if "2024-25" in path else aggregate

    mocker.patch.object(
        mock_data_extractor, "_read_csv", side_effect=fake_read_csv
    )

    connection = get_connection(tmp_path / "t.duckdb")
    try:
        mock_data_extractor.load_immutable_seasons(
            connection, "2025-26", bridge_seasons=["2024-25"]
        )
        present = seasons_present(connection)
    finally:
        connection.close()

    assert present == {"2019-20", "2020-21", "2024-25"}


def test_load_immutable_seasons_skips_when_already_present(
    mocker: MockerFixture, mock_data_extractor: DataExtractor, tmp_path
) -> None:
    """With historic + bridge already stored, no downloads happen."""
    from fantasy_football.storage.database import (
        get_connection,
        write_immutable_season,
    )

    seeded = pl.DataFrame(
        {
            "season": ["2020-21", "2024-25"],
            "gw": [1, 1],
            "element": [1, 2],
            "name": ["A", "C"],
            "position": ["DEF", "MID"],
            "team": ["Arsenal", "Leeds"],
            "bonus": [1, 2],
            "minutes": [90, 90],
            "round": [1, 1],
            "total_points": [6, 8],
            "value": [50, 60],
        }
    )
    read_csv = mocker.patch.object(mock_data_extractor, "_read_csv")

    connection = get_connection(tmp_path / "t.duckdb")
    try:
        write_immutable_season(
            connection, seeded.filter(pl.col("season") == "2020-21"), "2020-21"
        )
        write_immutable_season(
            connection, seeded.filter(pl.col("season") == "2024-25"), "2024-25"
        )
        mock_data_extractor.load_immutable_seasons(
            connection, "2025-26", bridge_seasons=["2024-25"]
        )
    finally:
        connection.close()

    read_csv.assert_not_called()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/extraction/test_extractor.py -k "historic_loaded or immutable_seasons" -v`
Expected: FAIL with `ImportError` / `AttributeError` for the new symbols.

- [ ] **Step 3: Update the implementation**

In `fantasy_football/extraction/extractor.py`:

Add these imports at the top, alongside the existing imports:

```python
import io

import polars as pl

from fantasy_football.constants import VASTAAV_BRIDGE_SEASONS
from fantasy_football.storage.database import (
    seasons_present,
    write_immutable_season,
)
```

(`VASTAAV_BRIDGE_SEASONS` is already imported in this module — keep the single
existing import line; do not duplicate it.)

Add this module-level function after the imports (before the classes):

```python
def _historic_loaded(
    present: set[str], current_season: str, bridge_seasons: list[str]
) -> bool:
    """Return True if the historic aggregate has already been loaded.

    The aggregate covers many older seasons and is loaded once. We infer it is
    loaded when any stored season falls outside the current season and the
    enumerable bridge seasons.

    Parameters
    ----------
    present : set[str]
        Seasons already stored in the DB.
    current_season : str
        The current season (sourced from FCI, not the aggregate).
    bridge_seasons : list[str]
        Vaastav bridge seasons handled by their own per-season check.

    Returns
    -------
    bool
        True if historic seasons are present, False otherwise.
    """
    return bool(set(present) - {current_season} - set(bridge_seasons))
```

Add a `_read_csv` method and a `load_immutable_seasons` method to
`DataExtractor`, and delete the existing `save_all_data_files` method entirely.
The replacement methods:

```python
    def _read_csv(self, file_path: str) -> pl.DataFrame:
        """Download a Vaastav repo CSV into a Polars DataFrame in memory.

        Parameters
        ----------
        file_path : str
            Repo-relative path of the CSV.

        Returns
        -------
        pl.DataFrame
            The parsed CSV.
        """
        url = self.api_client.get_raw_file_url(file_path)
        response = requests.get(url)
        response.raise_for_status()
        return pl.read_csv(io.BytesIO(response.content))

    def load_immutable_seasons(
        self,
        connection: "DuckDBPyConnection",
        current_season: str,
        bridge_seasons: list[str] | None = None,
    ) -> None:
        """Load completed seasons (aggregate + bridge) into the DB if absent.

        Bridge seasons are checked individually. The historic aggregate is
        downloaded and split by season only when no historic seasons are yet
        stored, so a normal run touches the network only for already-missing
        data.

        Parameters
        ----------
        connection : duckdb.DuckDBPyConnection
            Open connection to the player-week database.
        current_season : str
            The current season, excluded from the historic-loaded check.
        bridge_seasons : list[str] | None, optional
            Vaastav bridge seasons. Defaults to ``VASTAAV_BRIDGE_SEASONS``.
        """
        seasons = (
            bridge_seasons
            if bridge_seasons is not None
            else VASTAAV_BRIDGE_SEASONS
        )
        present = seasons_present(connection)

        for season in seasons:
            if season in present:
                continue
            bridge = self._read_csv(f"data/{season}/gws/merged_gw.csv")
            shaped = bridge.rename({"GW": "gw"}).with_columns(
                pl.lit(season).alias("season")
            )
            write_immutable_season(connection, shaped, season)

        if not _historic_loaded(present, current_season, seasons):
            aggregate = self._read_csv(self.HISTORIC_FILE).rename(
                {"season_x": "season", "team_x": "team", "GW": "gw"}
            )
            for season in sorted(aggregate["season"].unique().to_list()):
                slice_ = aggregate.filter(pl.col("season") == season)
                write_immutable_season(connection, slice_, season)
```

Add the type-only import at the top of the file (add the `TYPE_CHECKING` block
if not present):

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/extraction/test_extractor.py -v`
Expected: all tests PASS (the GitHub API client tests are unchanged).

- [ ] **Step 5: Commit**

```bash
git add fantasy_football/extraction/extractor.py tests/extraction/test_extractor.py
git commit -m "feat(extraction): load immutable seasons into duckdb"
```

---

## Task 9: `load_gw_data` reads from the DB

**Files:**
- Modify: `fantasy_football/features/transformation.py:1-13` (imports), `:60-123` (`load_gw_data`), `:231` (call site)
- Test: `tests/features/test_transformation.py` (rewrite the `load_gw_data` and `create_rolling_points_data` tests)

- [ ] **Step 1: Rewrite the failing tests**

In `tests/features/test_transformation.py`, delete these five functions
entirely:
- `test_load_gw_data` (lines 46-125)
- `test_load_gw_data_includes_bridge_season` (lines 165-198)
- `test_load_gw_data_skips_absent_bridge_season` (lines 200-235)
- `test_load_gw_data_missing_previous_seasons_file` (lines 238-267)
- `test_load_gw_data_missing_current_season_file` (lines 270-299)
- `test_load_gw_data_schema_drift_in_previous_seasons` (lines 302-345)

Also delete the now-unused helpers `_write_aggregate` (lines 128-143) and
`_write_season_merged_gw` (lines 146-162).

Add this import near the top of the file (with the other imports):

```python
from fantasy_football.storage import database
from fantasy_football.storage.database import (
    get_connection,
    upsert_current_season,
    write_immutable_season,
)
```

Add these new tests in place of the deleted `load_gw_data` tests:

```python
def _seed_player_week_db(db_path: Path) -> None:
    """Seed a temp DB with one historic (GKP) season and one current season."""
    historic = pl.DataFrame(
        {
            "season": ["2020-21", "2020-21"],
            "gw": [1, 2],
            "element": [1, 1],
            "name": ["Player1", "Player1"],
            "position": ["GKP", "GKP"],
            "team": ["Arsenal", "Arsenal"],
            "bonus": [1, 2],
            "minutes": [90, 90],
            "round": [1, 2],
            "total_points": [6, 8],
            "value": [50, 50],
        }
    )
    current = pl.DataFrame(
        {
            "season": ["2025-26", "2025-26"],
            "gw": [1, 2],
            "element": [2, 2],
            "name": ["Player2", "Player2"],
            "position": ["GK", "GK"],
            "team": ["Chelsea", "Chelsea"],
            "bonus": [1, 2],
            "minutes": [90, 90],
            "round": [1, 2],
            "total_points": [7, 9],
            "value": [45, 45],
        }
    )
    connection = get_connection(db_path)
    try:
        write_immutable_season(connection, historic, "2020-21")
        upsert_current_season(connection, current, "2025-26")
    finally:
        connection.close()


def test_load_gw_data_reads_all_seasons_from_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """load_gw_data returns every stored season with GKP collapsed to GK."""
    db_path = tmp_path / "t.duckdb"
    monkeypatch.setattr(database, "DATABASE_PATH", db_path)
    _seed_player_week_db(db_path)

    result = load_gw_data()

    assert isinstance(result, pl.DataFrame)
    assert set(result["season"].unique().to_list()) == {"2020-21", "2025-26"}
    assert result.filter(pl.col("position") == "GKP").height == 0
    assert result.filter(pl.col("position") == "GK").height == 4
    assert result.filter(pl.col("name") == "Player1")[
        "team"
    ].unique().to_list() == ["Arsenal"]
```

Then rewrite `test_create_rolling_points_data_respects_rolling_window`
(lines 553-628) to seed the DB instead of CSV files:

```python
@pytest.mark.parametrize("rolling_window", [2, 3, 5])
def test_create_rolling_points_data_respects_rolling_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rolling_window: int,
) -> None:
    """Verify the output column name and values track the rolling_window arg.

    Parameters
    ----------
    tmp_path : Path
        A temporary directory path provided by pytest.
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch fixture used to redirect the DB and output folder.
    rolling_window : int
        The rolling window size to test.
    """
    db_path = tmp_path / "t.duckdb"
    transformed_data = tmp_path / "transformed"
    current_season = "2025-26"
    monkeypatch.setattr(database, "DATABASE_PATH", db_path)
    monkeypatch.setattr(
        data_transformation, "TRANSFORMED_DATA_FOLDER", transformed_data
    )
    assert not transformed_data.exists()

    historic = pl.DataFrame(
        {
            "season": ["2020-21"] * 6,
            "gw": [1, 2, 3, 4, 5, 6],
            "element": [1] * 6,
            "name": ["Player1"] * 6,
            "position": ["GK"] * 6,
            "team": ["Arsenal"] * 6,
            "bonus": [0, 1, 2, 0, 1, 2],
            "minutes": [90] * 6,
            "round": [1, 2, 3, 4, 5, 6],
            "total_points": [2, 4, 6, 8, 10, 12],
            "value": [50] * 6,
        }
    )
    connection = get_connection(db_path)
    try:
        write_immutable_season(connection, historic, "2020-21")
    finally:
        connection.close()

    create_rolling_points_data(current_season, rolling_window=rolling_window)

    result = pl.read_csv(transformed_data / "rolling_points.csv")
    expected_column = rolling_column_name("total_points", rolling_window)
    assert expected_column in result.columns

    player1_tail_points = (
        result.filter(pl.col("name") == "Player1")
        .sort(["season", "gw"])
        .tail(rolling_window)["total_points"]
        .to_list()
    )
    expected_mean = sum(player1_tail_points) / rolling_window
    actual = result.filter(pl.col("name") == "Player1").sort(["season", "gw"])[
        expected_column
    ][-1]
    assert actual == pytest.approx(expected_mean)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/features/test_transformation.py -k "load_gw_data or rolling_points_data" -v`
Expected: FAIL — `load_gw_data` still takes a required `current_season` arg / still reads CSVs.

- [ ] **Step 3: Update the implementation**

In `fantasy_football/features/transformation.py`:

Replace the imports block (lines 1-13) so the constants no longer used here are
dropped and the DB reader is imported. `DATA_FOLDER` stays (it builds
`TRANSFORMED_DATA_FOLDER`); `RAW_DATA_FOLDER` and `VASTAAV_BRIDGE_SEASONS` go.
Use exactly this imports block:

```python
import logging

import polars as pl

from fantasy_football.constants import DATA_FOLDER, ROLLING_WINDOW
from fantasy_football.storage.database import load_player_week

logger = logging.getLogger(__name__)

TRANSFORMED_DATA_FOLDER = DATA_FOLDER.joinpath("transformed")

KNOWN_POSITIONS: tuple[str, ...] = ("GK", "DEF", "MID", "FWD")
```

(The `RAW_DATA_FOLDER` and `VASTAAV_BRIDGE_SEASONS` module-level names and
imports are removed, since nothing in this module uses them after the rewrite.)

Delete the `_load_season_merged_gw` helper entirely (lines 24-58).

Replace the whole `load_gw_data` function (lines 60-123) with:

```python
def load_gw_data() -> pl.DataFrame:
    """Load all player-week data across every season from the database.

    Reads the entire ``player_week`` table — historic, bridge, and current
    seasons — which the extraction step populated. Positions are already
    normalised (``GKP`` collapsed to ``GK``) at write time.

    Returns
    -------
    pl.DataFrame
        One row per (player, gameweek) for all seasons, with a ``season`` and
        ``gw`` column.
    """
    return load_player_week()
```

Update the single call site inside `create_rolling_points_data` (was line 231)
from `gw_data = load_gw_data(current_season)` to:

```python
    gw_data = load_gw_data()
```

(Leave `create_rolling_points_data`'s own `current_season` parameter as-is — it
remains part of the public signature and its callers are unchanged.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/features/test_transformation.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add fantasy_football/features/transformation.py tests/features/test_transformation.py
git commit -m "feat(features): load_gw_data reads player-week data from duckdb"
```

---

## Task 10: Wire `main.py` to the DB with a `rebuild` flag

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Update imports**

In `main.py`, add the storage import near the other
`from fantasy_football...` imports:

```python
from fantasy_football.storage.database import get_connection, reset_database
```

- [ ] **Step 2: Update `update_current_season`**

Replace the `update_current_season` function with a version that takes the
connection and routes FCI seasons to the upsert:

```python
def update_current_season(season: str, connection: "DuckDBPyConnection") -> None:
    """Refresh the current season's player-week rows from the correct source.

    Parameters
    ----------
    season : str
        Short-form season string, e.g. ``"2025-26"``.
    connection : duckdb.DuckDBPyConnection
        Open connection to the player-week database.
    """
    if source_for_season(season) == DataSource.FCI:
        FciExtractor().build_current_season_merged_gw(season, connection)
    else:
        raise NotImplementedError(
            f"Current-season ingestion for the Vaastav-sourced season "
            f"{season} is not supported; current seasons come from FCI."
        )
```

Add the type-only import at the top of `main.py` (add the `TYPE_CHECKING`
block):

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection
```

- [ ] **Step 3: Update `main` signature and extraction flow**

Replace the `main` function signature and the extraction portion of its body.
Change the signature from `download_all_data: bool = False` to
`rebuild: bool = False`, and replace the block:

```python
    configure_logging()
    if download_all_data:
        DataExtractor().save_all_data_files()
    update_current_season(CURRENT_SEASON)
```

with:

```python
    configure_logging()
    connection = get_connection()
    try:
        if rebuild:
            reset_database(connection)
        DataExtractor().load_immutable_seasons(connection, CURRENT_SEASON)
        update_current_season(CURRENT_SEASON, connection)
    finally:
        connection.close()
```

Update the `main` docstring's `download_all_data` parameter entry to describe
`rebuild`:

```python
    rebuild : bool, optional
        When True, drops and reloads every season from scratch (full refresh /
        recovery escape hatch). Defaults to False, which loads only missing
        immutable seasons and upserts the current season.
```

- [ ] **Step 4: Update the `__main__` block**

Replace the final block:

```python
if __name__ == "__main__":
    main(download_all_data=True, team_file="data/dummy_team.json")
```

with:

```python
if __name__ == "__main__":
    main(team_file="data/dummy_team.json")
```

- [ ] **Step 5: Verify the module imports and type-checks**

Run: `uv run python -c "import main"`
Expected: no error.

Run: `uv run mypy main.py fantasy_football/storage fantasy_football/extraction fantasy_football/features/transformation.py`
Expected: no type errors.

- [ ] **Step 6: Run the full test suite**

Run: `uv run pytest -q`
Expected: all tests PASS.

- [ ] **Step 7: Lint**

Run: `uv run ruff check fantasy_football main.py tests`
Expected: no errors (fix any unused-import warnings from removed CSV code).

- [ ] **Step 8: Commit**

```bash
git add main.py
git commit -m "feat: wire main to duckdb ingestion with rebuild flag"
```

---

## Task 11: End-to-end smoke verification

**Files:** none (manual verification)

- [ ] **Step 1: Fresh-build the database**

Run: `uv run python -c "from main import main; main(rebuild=True)"`
Expected: completes; logs show immutable seasons inserted and current season upserted; `data/fantasy_football.duckdb` exists.

(Requires `GITHUB_API_KEY` in `.env` and network access. If unavailable in the
execution environment, skip this task and note it.)

- [ ] **Step 2: Confirm incremental run does not re-download history**

Run: `uv run python -c "from main import main; main()"`
Expected: completes; logs show immutable seasons skipped ("already present"), only the current season upserted.

- [ ] **Step 3: Query the DB to confirm contents**

Run: `uv run python -c "from fantasy_football.storage.database import load_player_week; df = load_player_week(); print(df.shape); print(sorted(df['season'].unique().to_list()))"`
Expected: a non-trivial row count and a list of seasons including `2025-26` and historic seasons.

- [ ] **Step 4: Confirm retired CSVs are no longer required**

Run: `uv run pytest -q`
Expected: all tests pass without any `cleaned_merged_seasons.csv` present.

---

## Self-Review

**Spec coverage:**
- duckdb as source of truth for all player-week data → Tasks 2-6, 9. ✓
- `cleaned_merged_seasons.csv` retired → Tasks 8 (aggregate loaded into DB, split by season), 9 (CSV reads deleted). ✓
- Incremental default; no full rewrite each run → Task 8 (`load_immutable_seasons` skips present seasons; historic loaded once via `_historic_loaded`), Task 5/7 (current-season upsert), Task 10 (`rebuild` flag is opt-in). ✓
- Option C upsert keyed on `(season, gw, element)` → Task 5 (delete-then-insert within season; PK enforces the key). ✓
- Immutable seasons via download-once (approach A) → Task 8. ✓
- Schema option B (downstream subset + `value`) → Task 2 schema, Task 3 coerce. ✓
- `value` mapped for both sources → Tasks 7, 8. ✓
- `load_gw_data` reads from DB → Task 9. ✓
- `main.py` `download_all_data` → `rebuild` → Task 10. ✓
- DB gitignored, explicit line → Task 1. ✓
- Queryable by external SQL tools → inherent to duckdb file; verified in Task 11 Step 3. ✓
- Tests against temp-file duckdb, round-trip test → Tasks 2-7, 9. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code. ✓

**Type consistency:** `get_connection`, `seasons_present`, `write_immutable_season`, `upsert_current_season`, `load_player_week`, `reset_database`, `coerce_player_week`, `_historic_loaded`, `load_immutable_seasons`, and `build_current_season_merged_gw(season, connection)` signatures match across their definitions (Tasks 2-8) and call sites (Tasks 9-10). `PLAYER_WEEK_COLUMNS` order is identical in the schema DDL, `coerce_player_week`, `load_player_week`, and the assertions. ✓
