# Team Fixture Table Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `team_fixture` duckdb table holding one row per team, per fixture, per gameweek for all available seasons, populated during ingestion alongside `player_week`.

**Architecture:** Two source adapters (Vaastav per-season `fixtures.csv` for historic seasons; FPL `/api/fixtures/` for the current live season) feed one shared transform that explodes each FPL-shaped fixture into two team-perspective rows. Storage mirrors the existing `player_week` immutable-historic / upsert-current pattern. An orchestration function iterates the seasons already in `player_week`, routes each by data source, and writes fixtures.

**Tech Stack:** Python 3.12, Polars, DuckDB, Pydantic, pytest. Test runner: `uv run pytest`.

**Reference spec:** `docs/superpowers/specs/2026-06-17-team-fixture-table-design.md`

---

## File Structure

- **Create** `fantasy_football/extraction/fixtures.py` — fixture source adapters + shared transform + `load_fixtures` orchestration.
- **Modify** `fantasy_football/storage/database.py` — add `team_fixture` schema, coercion, create/reset wiring, write/upsert/read helpers.
- **Modify** `main.py` — call `load_fixtures` inside the existing connection block.
- **Create** `tests/unit/extraction/test_fixtures.py` — adapter + transform + orchestration tests.
- **Modify/Create** `tests/unit/storage/test_database.py` — team_fixture storage tests (create if absent).

### Canonical `team_fixture` schema (used everywhere)

| column | duckdb type | polars dtype |
| --- | --- | --- |
| `season` | VARCHAR | `pl.Utf8` |
| `gw` | BIGINT | `pl.Int64` |
| `team` | VARCHAR | `pl.Utf8` |
| `is_home` | BOOLEAN | `pl.Boolean` |
| `opposition` | VARCHAR | `pl.Utf8` |
| `kickoff_time` | TIMESTAMP | `pl.Datetime("us")` (naive UTC) |

Primary key: `(season, gw, team, opposition)`.

---

## Task 1: team_fixture table — schema, coercion, reader, create/reset wiring

**Files:**
- Modify: `fantasy_football/storage/database.py`
- Test: `tests/unit/storage/test_database.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/storage/test_database.py` (if the directory `tests/unit/storage/` does not exist, create it; no `__init__.py` is needed — the existing test suite uses none).

```python
"""Unit tests for the team_fixture duckdb storage helpers."""

from datetime import datetime

import duckdb
import polars as pl
import pytest

from fantasy_football.storage.database import (
    TEAM_FIXTURE_COLUMNS,
    coerce_team_fixture,
    get_connection,
    load_team_fixture,
    reset_database,
)


def _conn(tmp_path) -> duckdb.DuckDBPyConnection:
    return get_connection(tmp_path / "test.duckdb")


def _fixture_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": ["2023-24", "2023-24"],
            "gw": [1, 1],
            "team": ["Arsenal", "Chelsea"],
            "is_home": [True, False],
            "opposition": ["Chelsea", "Arsenal"],
            "kickoff_time": [
                datetime(2023, 8, 11, 19, 0),
                datetime(2023, 8, 11, 19, 0),
            ],
        }
    )


def test_get_connection_creates_team_fixture_table(tmp_path) -> None:
    """get_connection creates the team_fixture table."""
    conn = _conn(tmp_path)
    try:
        names = {
            row[0]
            for row in conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'team_fixture'"
            ).fetchall()
        }
        assert names == set(TEAM_FIXTURE_COLUMNS)
    finally:
        conn.close()


def test_coerce_team_fixture_selects_and_orders_columns(tmp_path) -> None:
    """coerce_team_fixture reduces a frame to the canonical columns/order."""
    frame = _fixture_frame().with_columns(pl.lit("extra").alias("junk"))
    shaped = coerce_team_fixture(frame)
    assert shaped.columns == TEAM_FIXTURE_COLUMNS


def test_load_team_fixture_round_trips(tmp_path) -> None:
    """A frame inserted directly is read back via load_team_fixture."""
    conn = _conn(tmp_path)
    try:
        conn.register("incoming", coerce_team_fixture(_fixture_frame()).to_arrow())
        conn.execute("INSERT INTO team_fixture SELECT * FROM incoming")
        conn.unregister("incoming")
        out = load_team_fixture(conn)
        assert out.columns == TEAM_FIXTURE_COLUMNS
        assert out.height == 2
        assert set(out["team"].to_list()) == {"Arsenal", "Chelsea"}
    finally:
        conn.close()


def test_reset_database_drops_team_fixture_rows(tmp_path) -> None:
    """reset_database recreates an empty team_fixture table."""
    conn = _conn(tmp_path)
    try:
        conn.register("incoming", coerce_team_fixture(_fixture_frame()).to_arrow())
        conn.execute("INSERT INTO team_fixture SELECT * FROM incoming")
        conn.unregister("incoming")
        reset_database(conn)
        assert load_team_fixture(conn).height == 0
    finally:
        conn.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/storage/test_database.py -q`
Expected: FAIL with `ImportError` (e.g. `cannot import name 'TEAM_FIXTURE_COLUMNS'`).

- [ ] **Step 3: Add the schema, create SQL, coercion, and reader to `database.py`**

In `fantasy_football/storage/database.py`, add `import polars as pl` is already present. After the `_CREATE_TABLE` block (around line 66) add:

```python
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
```

- [ ] **Step 4: Wire the new table into `get_connection` and `reset_database`**

In `get_connection`, after `connection.execute(_CREATE_TABLE)` (line 88) add:

```python
    connection.execute(_CREATE_TEAM_FIXTURE_TABLE)
```

In `reset_database`, after the two existing player_week statements add:

```python
    connection.execute("DROP TABLE IF EXISTS team_fixture")
    connection.execute(_CREATE_TEAM_FIXTURE_TABLE)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/storage/test_database.py -q`
Expected: PASS (4 passed).

- [ ] **Step 6: Commit**

```bash
git add fantasy_football/storage/database.py tests/unit/storage/test_database.py
git commit -m "feat(storage): add team_fixture table schema, coercion, reader"
```

---

## Task 2: team_fixture write helpers — immutable + upsert + seasons-present

**Files:**
- Modify: `fantasy_football/storage/database.py`
- Test: `tests/unit/storage/test_database.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/storage/test_database.py`:

```python
from fantasy_football.storage.database import (  # noqa: E402
    fixture_seasons_present,
    upsert_current_fixtures,
    write_immutable_fixtures,
)


def test_write_immutable_fixtures_inserts_once(tmp_path) -> None:
    """write_immutable_fixtures inserts a new season but skips a present one."""
    conn = _conn(tmp_path)
    try:
        write_immutable_fixtures(conn, _fixture_frame(), "2023-24")
        assert load_team_fixture(conn).height == 2
        # Second call with different data is ignored — season already present.
        changed = _fixture_frame().with_columns(pl.lit("Spurs").alias("team"))
        write_immutable_fixtures(conn, changed, "2023-24")
        teams = set(load_team_fixture(conn)["team"].to_list())
        assert teams == {"Arsenal", "Chelsea"}
    finally:
        conn.close()


def test_upsert_current_fixtures_replaces_season(tmp_path) -> None:
    """upsert_current_fixtures deletes then reinserts the season's rows."""
    conn = _conn(tmp_path)
    try:
        upsert_current_fixtures(conn, _fixture_frame(), "2023-24")
        replacement = pl.DataFrame(
            {
                "season": ["2023-24"],
                "gw": [2],
                "team": ["Arsenal"],
                "is_home": [True],
                "opposition": ["Spurs"],
                "kickoff_time": [datetime(2023, 8, 19, 15, 0)],
            }
        )
        upsert_current_fixtures(conn, replacement, "2023-24")
        out = load_team_fixture(conn)
        assert out.height == 1
        assert out.row(0, named=True)["opposition"] == "Spurs"
    finally:
        conn.close()


def test_fixture_seasons_present(tmp_path) -> None:
    """fixture_seasons_present returns distinct stored seasons."""
    conn = _conn(tmp_path)
    try:
        assert fixture_seasons_present(conn) == set()
        write_immutable_fixtures(conn, _fixture_frame(), "2023-24")
        assert fixture_seasons_present(conn) == {"2023-24"}
    finally:
        conn.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/storage/test_database.py -q`
Expected: FAIL with `ImportError` for `write_immutable_fixtures`.

- [ ] **Step 3: Add the write/upsert/present helpers to `database.py`**

Append to `fantasy_football/storage/database.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/storage/test_database.py -q`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add fantasy_football/storage/database.py tests/unit/storage/test_database.py
git commit -m "feat(storage): add team_fixture write/upsert/present helpers"
```

---

## Task 3: shared fixture transform

**Files:**
- Create: `fantasy_football/extraction/fixtures.py`
- Test: `tests/unit/extraction/test_fixtures.py`

The transform takes already-normalised fixtures (FPL-shaped) and a team id→name
map and explodes each fixture into two team-perspective rows. Both adapters
(Tasks 4 and 5) feed it.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/extraction/test_fixtures.py` (create the `tests/unit/extraction/` directory if absent):

```python
"""Unit tests for fixture extraction adapters and transform."""

from datetime import datetime

import polars as pl
import pytest

from fantasy_football.extraction.fixtures import fixtures_to_team_rows


def test_transform_emits_two_rows_per_fixture() -> None:
    """Each fixture becomes a home row and an away row."""
    teams = {1: "Arsenal", 2: "Chelsea"}
    fixtures = [(1, "2023-08-11T19:00:00Z", 1, 2)]
    result = fixtures_to_team_rows(fixtures, teams, season="2023-24")
    assert result.height == 2
    home = result.filter(pl.col("team") == "Arsenal").row(0, named=True)
    away = result.filter(pl.col("team") == "Chelsea").row(0, named=True)
    assert home["opposition"] == "Chelsea"
    assert home["is_home"] is True
    assert away["opposition"] == "Arsenal"
    assert away["is_home"] is False
    for row in (home, away):
        assert row["season"] == "2023-24"
        assert row["gw"] == 1
        assert row["kickoff_time"] == datetime(2023, 8, 11, 19, 0)


def test_transform_kickoff_is_naive_utc() -> None:
    """kickoff_time is stored as a naive UTC datetime (no timezone)."""
    teams = {1: "Arsenal", 2: "Chelsea"}
    fixtures = [(1, "2023-08-11T19:00:00Z", 1, 2)]
    result = fixtures_to_team_rows(fixtures, teams, season="2023-24")
    assert result.schema["kickoff_time"] == pl.Datetime("us")
    assert result["kickoff_time"][0].tzinfo is None


def test_transform_handles_double_gameweek() -> None:
    """A team with two fixtures in one gw yields two rows."""
    teams = {1: "Arsenal", 2: "Chelsea", 3: "Spurs"}
    fixtures = [
        (29, "2026-03-14T15:00:00Z", 1, 2),
        (29, "2026-03-17T19:45:00Z", 1, 3),
    ]
    result = fixtures_to_team_rows(fixtures, teams, season="2025-26")
    arsenal = result.filter(pl.col("team") == "Arsenal").sort("kickoff_time")
    assert arsenal.height == 2
    assert arsenal["opposition"].to_list() == ["Chelsea", "Spurs"]


def test_transform_empty_returns_typed_empty_frame() -> None:
    """No fixtures yields an empty frame with the canonical schema."""
    result = fixtures_to_team_rows([], {1: "Arsenal"}, season="2023-24")
    assert result.is_empty()
    assert result.columns == [
        "season",
        "gw",
        "team",
        "is_home",
        "opposition",
        "kickoff_time",
    ]


def test_transform_unknown_team_id_raises() -> None:
    """An unknown team id raises KeyError."""
    teams = {1: "Arsenal"}  # id 2 missing
    fixtures = [(1, "2023-08-11T19:00:00Z", 1, 2)]
    with pytest.raises(KeyError):
        fixtures_to_team_rows(fixtures, teams, season="2023-24")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/extraction/test_fixtures.py -q`
Expected: FAIL with `ModuleNotFoundError: fantasy_football.extraction.fixtures`.

- [ ] **Step 3: Create the module with the transform**

Create `fantasy_football/extraction/fixtures.py`:

```python
"""Build the ``team_fixture`` table from true fixture lists.

Historic seasons come from Vaastav's per-season ``fixtures.csv``; the current
live season comes from the FPL ``/api/fixtures/`` endpoint. Both sources are
FPL-shaped (gameweek, ISO kickoff string, home/away team ids), so a single
transform explodes each fixture into two team-perspective rows. See
``docs/superpowers/specs/2026-06-17-team-fixture-table-design.md``.
"""

import logging
from datetime import datetime, timezone

import polars as pl

from fantasy_football.storage.database import TEAM_FIXTURE_SCHEMA

logger = logging.getLogger(__name__)

# (gameweek, ISO kickoff string, home team id, away team id)
NormalisedFixture = tuple[int, str, int, int]


def _parse_kickoff(kickoff: str) -> datetime:
    """Parse an ISO kickoff string to a naive UTC datetime.

    Parameters
    ----------
    kickoff : str
        ISO 8601 timestamp, e.g. ``"2023-08-11T19:00:00Z"``.

    Returns
    -------
    datetime
        The instant in UTC with the timezone stripped, so it maps to a duckdb
        ``TIMESTAMP`` (tz-naive) column.
    """
    parsed = datetime.fromisoformat(kickoff.replace("Z", "+00:00"))
    return parsed.astimezone(timezone.utc).replace(tzinfo=None)


def fixtures_to_team_rows(
    fixtures: list[NormalisedFixture],
    teams_by_id: dict[int, str],
    season: str,
) -> pl.DataFrame:
    """Explode FPL-shaped fixtures into two team-perspective rows each.

    Parameters
    ----------
    fixtures : list[NormalisedFixture]
        ``(gw, kickoff_iso, home_team_id, away_team_id)`` tuples.
    teams_by_id : dict[int, str]
        Maps a team id (as used in ``fixtures``) to its display name.
    season : str
        Short-form season string, e.g. ``"2023-24"``.

    Returns
    -------
    pl.DataFrame
        One row per team per fixture, columns and dtypes per
        ``TEAM_FIXTURE_SCHEMA``. Double gameweeks naturally produce two rows
        for the affected team.

    Raises
    ------
    KeyError
        If a fixture references a team id absent from ``teams_by_id``.
    """
    rows: list[dict] = []
    for gw, kickoff_iso, home_id, away_id in fixtures:
        kickoff = _parse_kickoff(kickoff_iso)
        home = teams_by_id[home_id]
        away = teams_by_id[away_id]
        rows.append(
            {
                "season": season,
                "gw": gw,
                "team": home,
                "is_home": True,
                "opposition": away,
                "kickoff_time": kickoff,
            }
        )
        rows.append(
            {
                "season": season,
                "gw": gw,
                "team": away,
                "is_home": False,
                "opposition": home,
                "kickoff_time": kickoff,
            }
        )
    if not rows:
        return pl.DataFrame(schema=TEAM_FIXTURE_SCHEMA)
    return pl.DataFrame(rows).cast(TEAM_FIXTURE_SCHEMA, strict=False)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/extraction/test_fixtures.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add fantasy_football/extraction/fixtures.py tests/unit/extraction/test_fixtures.py
git commit -m "feat(fixtures): shared fixture-to-team-rows transform"
```

---

## Task 4: current-season adapter (FPL API)

**Files:**
- Modify: `fantasy_football/extraction/fixtures.py`
- Test: `tests/unit/extraction/test_fixtures.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/extraction/test_fixtures.py`:

```python
from unittest.mock import MagicMock  # noqa: E402

from fantasy_football.extraction.fixtures import build_current_fixtures  # noqa: E402
from fantasy_football.fpl_types import (  # noqa: E402
    FplFixture,
    FplFixtures,
    FplTeamInfo,
)


def _team(id: int, name: str) -> FplTeamInfo:
    return FplTeamInfo(
        id=id, code=id * 10, name=name, short_name=name[:3].upper()
    )


def _fix(event: int, team_h: int, team_a: int, kickoff: str) -> FplFixture:
    return FplFixture(
        code=event * 100 + team_h,
        event=event,
        finished=False,
        id=event * 100 + team_h,
        kickoff_time=kickoff,
        team_a=team_a,
        team_a_difficulty=3,
        team_h=team_h,
        team_h_difficulty=3,
    )


def _api(teams, fixtures) -> MagicMock:
    api = MagicMock()
    api.get_teams.return_value = teams
    api.get_fixtures.return_value = FplFixtures(fixtures=fixtures)
    return api


def test_build_current_fixtures_from_api() -> None:
    """build_current_fixtures maps FPL fixtures into team_fixture rows."""
    teams = [_team(1, "Arsenal"), _team(2, "Chelsea")]
    fixtures = [_fix(1, 1, 2, "2025-08-16T15:00:00Z")]
    result = build_current_fixtures("2025-26", _api(teams, fixtures))
    assert result.height == 2
    home = result.filter(pl.col("is_home")).row(0, named=True)
    assert home["team"] == "Arsenal"
    assert home["opposition"] == "Chelsea"
    assert home["gw"] == 1
    assert home["season"] == "2025-26"


def test_build_current_fixtures_empty_when_no_fixtures() -> None:
    """No fixtures yields an empty canonical frame."""
    result = build_current_fixtures("2025-26", _api([_team(1, "Arsenal")], []))
    assert result.is_empty()
    assert "kickoff_time" in result.columns
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/extraction/test_fixtures.py -q`
Expected: FAIL with `ImportError` for `build_current_fixtures`.

- [ ] **Step 3: Add the adapter**

Append to `fantasy_football/extraction/fixtures.py` (add the `FplAPI` import to the existing import block at the top):

```python
from fantasy_football.extraction.fpl import FplAPI
```

Then add the function:

```python
def build_current_fixtures(season: str, api: FplAPI) -> pl.DataFrame:
    """Build ``team_fixture`` rows for the current season from the FPL API.

    Parameters
    ----------
    season : str
        Short-form season string for the live season, e.g. ``"2025-26"``.
    api : FplAPI
        The FPL API client. ``get_fixtures`` already drops fixtures with no
        gameweek or kickoff time.

    Returns
    -------
    pl.DataFrame
        Team-fixture rows in the canonical schema.
    """
    teams_by_id = {team.id: team.name for team in api.get_teams()}
    fixtures: list[NormalisedFixture] = [
        (fixture.event, fixture.kickoff_time, fixture.team_h, fixture.team_a)
        for fixture in api.get_fixtures().fixtures
    ]
    return fixtures_to_team_rows(fixtures, teams_by_id, season)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/extraction/test_fixtures.py -q`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add fantasy_football/extraction/fixtures.py tests/unit/extraction/test_fixtures.py
git commit -m "feat(fixtures): current-season adapter from FPL API"
```

---

## Task 5: historic-season adapter (Vaastav)

**Files:**
- Modify: `fantasy_football/extraction/fixtures.py`
- Test: `tests/unit/extraction/test_fixtures.py`

Vaastav's `data/{season}/fixtures.csv` is FPL-shaped (`event`, `kickoff_time`,
`team_h`, `team_a`) with season-local team ids resolved via that season's
`data/{season}/teams.csv` (`id`, `name`). Some fixtures may have a blank
`event`/`kickoff_time` (postponed/unscheduled) and must be dropped. Files are
fetched with the existing `DataExtractor._read_csv` (same GitHub auth + raw-url
path already used for player-week CSVs).

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/extraction/test_fixtures.py`:

```python
from fantasy_football.extraction.fixtures import build_vaastav_fixtures  # noqa: E402


def _vaastav_extractor(fixtures_df, teams_df) -> MagicMock:
    """A DataExtractor stub whose _read_csv routes by path suffix."""
    extractor = MagicMock()

    def _read_csv(path: str):
        if path.endswith("teams.csv"):
            return teams_df
        if path.endswith("fixtures.csv"):
            return fixtures_df
        raise AssertionError(f"unexpected path {path}")

    extractor._read_csv.side_effect = _read_csv
    return extractor


def test_build_vaastav_fixtures_maps_local_team_ids() -> None:
    """Season-local team ids are resolved via teams.csv."""
    teams_df = pl.DataFrame({"id": [6, 13], "name": ["Arsenal", "Liverpool"]})
    fixtures_df = pl.DataFrame(
        {
            "event": [1],
            "kickoff_time": ["2023-08-11T19:00:00Z"],
            "team_h": [6],
            "team_a": [13],
        }
    )
    extractor = _vaastav_extractor(fixtures_df, teams_df)
    result = build_vaastav_fixtures("2023-24", extractor)
    assert result.height == 2
    home = result.filter(pl.col("is_home")).row(0, named=True)
    assert home["team"] == "Arsenal"
    assert home["opposition"] == "Liverpool"
    assert home["season"] == "2023-24"
    extractor._read_csv.assert_any_call("data/2023-24/teams.csv")
    extractor._read_csv.assert_any_call("data/2023-24/fixtures.csv")


def test_build_vaastav_fixtures_drops_unscheduled_rows() -> None:
    """Rows with a null event or blank kickoff_time are dropped."""
    teams_df = pl.DataFrame({"id": [6, 13], "name": ["Arsenal", "Liverpool"]})
    fixtures_df = pl.DataFrame(
        {
            "event": [1, None],
            "kickoff_time": ["2023-08-11T19:00:00Z", None],
            "team_h": [6, 6],
            "team_a": [13, 13],
        }
    )
    extractor = _vaastav_extractor(fixtures_df, teams_df)
    result = build_vaastav_fixtures("2023-24", extractor)
    assert result.height == 2  # only the one scheduled fixture, exploded
    assert result["gw"].unique().to_list() == [1]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/extraction/test_fixtures.py -q`
Expected: FAIL with `ImportError` for `build_vaastav_fixtures`.

- [ ] **Step 3: Add the adapter**

Append to `fantasy_football/extraction/fixtures.py`. Add the import for the
extractor at the top of the file (after the `FplAPI` import):

```python
from fantasy_football.extraction.extractor import DataExtractor
```

Then add the function:

```python
def build_vaastav_fixtures(
    season: str, extractor: DataExtractor
) -> pl.DataFrame:
    """Build ``team_fixture`` rows for a historic season from Vaastav.

    Reads the season's ``fixtures.csv`` and ``teams.csv`` from the Vaastav
    repo. Fixtures with a missing gameweek or kickoff time (postponed /
    unscheduled) are dropped. Team ids are season-local and resolved via that
    season's ``teams.csv``.

    Parameters
    ----------
    season : str
        Short-form season string, e.g. ``"2023-24"``.
    extractor : DataExtractor
        Provides ``_read_csv`` to download a repo CSV into a Polars frame.

    Returns
    -------
    pl.DataFrame
        Team-fixture rows in the canonical schema.
    """
    teams_df = extractor._read_csv(f"data/{season}/teams.csv")
    teams_by_id = dict(
        zip(teams_df["id"].to_list(), teams_df["name"].to_list())
    )
    fixtures_df = extractor._read_csv(f"data/{season}/fixtures.csv")
    scheduled = fixtures_df.filter(
        pl.col("event").is_not_null()
        & pl.col("kickoff_time").is_not_null()
        & (pl.col("kickoff_time") != "")
    )
    fixtures: list[NormalisedFixture] = [
        (int(row["event"]), row["kickoff_time"], int(row["team_h"]), int(row["team_a"]))
        for row in scheduled.iter_rows(named=True)
    ]
    return fixtures_to_team_rows(fixtures, teams_by_id, season)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/extraction/test_fixtures.py -q`
Expected: PASS (9 passed).

- [ ] **Step 5: Commit**

```bash
git add fantasy_football/extraction/fixtures.py tests/unit/extraction/test_fixtures.py
git commit -m "feat(fixtures): historic-season adapter from Vaastav"
```

---

## Task 6: orchestration — load_fixtures

**Files:**
- Modify: `fantasy_football/extraction/fixtures.py`
- Test: `tests/unit/extraction/test_fixtures.py`

`load_fixtures` iterates the seasons already in `player_week` (the canonical
season list), skips any already in `team_fixture`, and routes each:
Vaastav historic seasons are built and written immutably; the current live
season is upserted from the FPL API. A non-current FCI season has no kickoff
source (FCI lacks kickoff times) and is logged and skipped — Vaastav backfills
it once `VASTAAV_LAST_SEASON` advances. A failed Vaastav fetch (e.g. HTTP 404
for a season with no fixtures.csv) is logged and skipped, not fatal.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/extraction/test_fixtures.py`:

```python
import duckdb  # noqa: E402

from fantasy_football.extraction.fixtures import load_fixtures  # noqa: E402
from fantasy_football.storage.database import (  # noqa: E402
    fixture_seasons_present,
    get_connection,
    load_team_fixture,
)


def _seed_player_week(conn, seasons) -> None:
    for i, season in enumerate(seasons):
        conn.execute(
            "INSERT INTO player_week (season, gw, element) VALUES (?, ?, ?)",
            [season, 1, i + 1],
        )


def test_load_fixtures_routes_historic_and_current(tmp_path, monkeypatch) -> None:
    """Historic seasons use Vaastav; the current season uses the API."""
    conn = get_connection(tmp_path / "t.duckdb")
    try:
        _seed_player_week(conn, ["2023-24", "2025-26"])

        def fake_vaastav(season, extractor):
            return fixtures_to_team_rows(
                [(1, "2023-08-11T19:00:00Z", 1, 2)],
                {1: "Arsenal", 2: "Chelsea"},
                season,
            )

        def fake_current(season, api):
            return fixtures_to_team_rows(
                [(1, "2025-08-16T15:00:00Z", 1, 2)],
                {1: "Arsenal", 2: "Chelsea"},
                season,
            )

        monkeypatch.setattr(
            "fantasy_football.extraction.fixtures.build_vaastav_fixtures",
            fake_vaastav,
        )
        monkeypatch.setattr(
            "fantasy_football.extraction.fixtures.build_current_fixtures",
            fake_current,
        )

        load_fixtures(conn, "2025-26", api=MagicMock(), extractor=MagicMock())

        assert fixture_seasons_present(conn) == {"2023-24", "2025-26"}
        assert load_team_fixture(conn).height == 4
    finally:
        conn.close()


def test_load_fixtures_skips_seasons_already_present(tmp_path, monkeypatch) -> None:
    """A season already in team_fixture is not rebuilt."""
    conn = get_connection(tmp_path / "t.duckdb")
    try:
        _seed_player_week(conn, ["2023-24"])
        calls = []
        monkeypatch.setattr(
            "fantasy_football.extraction.fixtures.build_vaastav_fixtures",
            lambda season, extractor: (
                calls.append(season)
                or fixtures_to_team_rows(
                    [(1, "2023-08-11T19:00:00Z", 1, 2)],
                    {1: "A", 2: "B"},
                    season,
                )
            ),
        )
        monkeypatch.setattr(
            "fantasy_football.extraction.fixtures.build_current_fixtures",
            lambda season, api: fixtures_to_team_rows([], {}, season),
        )
        load_fixtures(conn, "2099-00", api=MagicMock(), extractor=MagicMock())
        load_fixtures(conn, "2099-00", api=MagicMock(), extractor=MagicMock())
        assert calls == ["2023-24"]  # built once, skipped on the second run
    finally:
        conn.close()


def test_load_fixtures_vaastav_fetch_failure_is_skipped(tmp_path, monkeypatch) -> None:
    """A Vaastav fetch error logs and skips rather than aborting the run."""
    conn = get_connection(tmp_path / "t.duckdb")
    try:
        _seed_player_week(conn, ["2016-17", "2023-24"])

        def flaky(season, extractor):
            if season == "2016-17":
                raise requests.HTTPError("404")
            return fixtures_to_team_rows(
                [(1, "2023-08-11T19:00:00Z", 1, 2)],
                {1: "A", 2: "B"},
                season,
            )

        monkeypatch.setattr(
            "fantasy_football.extraction.fixtures.build_vaastav_fixtures", flaky
        )
        monkeypatch.setattr(
            "fantasy_football.extraction.fixtures.build_current_fixtures",
            lambda season, api: fixtures_to_team_rows([], {}, season),
        )
        load_fixtures(conn, "2099-00", api=MagicMock(), extractor=MagicMock())
        assert fixture_seasons_present(conn) == {"2023-24"}
    finally:
        conn.close()
```

Add the import `import requests` near the top of the test file (with the other
imports) for the failure test.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/extraction/test_fixtures.py -q`
Expected: FAIL with `ImportError` for `load_fixtures`.

- [ ] **Step 3: Add the orchestration function**

Append to `fantasy_football/extraction/fixtures.py`. Add these imports at the
top of the file (with the existing imports):

```python
import requests

from fantasy_football.extraction.seasons import DataSource, source_for_season
from fantasy_football.storage.database import (
    fixture_seasons_present,
    seasons_present,
    upsert_current_fixtures,
    write_immutable_fixtures,
)
```

(Keep the existing `from fantasy_football.storage.database import TEAM_FIXTURE_SCHEMA`
line — either merge it into this combined import or leave it; both work.)

Then add the function:

```python
def load_fixtures(
    connection: "duckdb.DuckDBPyConnection",
    current_season: str,
    *,
    api: FplAPI | None = None,
    extractor: DataExtractor | None = None,
) -> None:
    """Populate ``team_fixture`` for every season present in ``player_week``.

    Seasons already in ``team_fixture`` are skipped. Each remaining season is
    routed by data source: Vaastav historic seasons are built and written
    immutably; the current live season is upserted from the FPL API. A
    non-current FCI-era season has no kickoff source and is skipped (Vaastav
    backfills it once it falls within the Vaastav range). A Vaastav fetch
    failure for one season is logged and skipped, not fatal.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        Open connection to the database.
    current_season : str
        The current live season, e.g. ``"2025-26"``.
    api : FplAPI | None, optional
        FPL API client for the current season. Defaults to a new ``FplAPI``.
    extractor : DataExtractor | None, optional
        Vaastav CSV reader. Defaults to a new ``DataExtractor``.
    """
    api = api or FplAPI()
    extractor = extractor or DataExtractor()
    already = fixture_seasons_present(connection)

    for season in sorted(seasons_present(connection)):
        if season in already or season == current_season:
            continue
        if source_for_season(season) != DataSource.VAASTAV:
            logger.warning(
                "No fixture source for non-current FCI season %s; skipping.",
                season,
            )
            continue
        try:
            frame = build_vaastav_fixtures(season, extractor)
        except requests.HTTPError as exc:
            logger.warning(
                "Could not fetch Vaastav fixtures for %s (%s); skipping.",
                season,
                exc,
            )
            continue
        write_immutable_fixtures(connection, frame, season)

    # Always refresh the current season (kickoff times and new gameweeks can
    # change), matching the player-week upsert behaviour.
    frame = build_current_fixtures(current_season, api)
    upsert_current_fixtures(connection, frame, current_season)
```

Note: add `import duckdb` is **not** required at runtime (only used in the type
hint string `"duckdb.DuckDBPyConnection"`), but add a `TYPE_CHECKING` import to
mirror the codebase style:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection
```

and change the annotation to `connection: "DuckDBPyConnection"`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/extraction/test_fixtures.py -q`
Expected: PASS (12 passed).

- [ ] **Step 5: Run the full unit suite to check for regressions**

Run: `uv run pytest tests/unit -q`
Expected: PASS (all existing tests plus the new ones).

- [ ] **Step 6: Commit**

```bash
git add fantasy_football/extraction/fixtures.py tests/unit/extraction/test_fixtures.py
git commit -m "feat(fixtures): load_fixtures orchestration with source routing"
```

---

## Task 7: wire load_fixtures into the ingestion run

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Add the import**

In `main.py`, add to the imports near the other extraction imports (after line 8):

```python
from fantasy_football.extraction.fixtures import load_fixtures
```

- [ ] **Step 2: Call load_fixtures inside the connection block**

In `main`, inside the `try` block, after `update_current_season(CURRENT_SEASON, connection)` (line 80) and before the `finally`, add:

```python
        load_fixtures(connection, CURRENT_SEASON)
```

The block now reads:

```python
    try:
        if rebuild:
            reset_database(connection)
        DataExtractor().load_immutable_seasons(connection, CURRENT_SEASON)
        update_current_season(CURRENT_SEASON, connection)
        load_fixtures(connection, CURRENT_SEASON)
    finally:
        connection.close()
```

- [ ] **Step 3: Verify the import graph and unit suite still pass**

Run: `uv run python -c "import main"`
Expected: no error (no circular import).

Run: `uv run pytest tests/unit -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add main.py
git commit -m "feat: populate team_fixture during the ingestion run"
```

---

## Task 8: end-to-end smoke check (manual, network)

**Files:** none (verification only)

- [ ] **Step 1: Run a real ingestion against a throwaway database**

This hits live GitHub + FPL endpoints, so it needs `GITHUB_API_KEY` in `.env`.
Run a short Python snippet that builds fixtures into a temp DB and inspects it:

```bash
uv run python - <<'PY'
from fantasy_football.constants import CURRENT_SEASON
from fantasy_football.extraction.extractor import DataExtractor
from fantasy_football.extraction.fixtures import load_fixtures
from fantasy_football.storage.database import get_connection, load_team_fixture
from pathlib import Path

conn = get_connection(Path("/tmp/ff_fixtures_smoke.duckdb"))
# Seed a couple of historic seasons + current into player_week so load_fixtures
# has a season list to work from.
for s in ("2023-24", CURRENT_SEASON):
    conn.execute(
        "INSERT INTO player_week (season, gw, element) VALUES (?, 1, 1)", [s]
    )
load_fixtures(conn, CURRENT_SEASON)
df = load_team_fixture(conn)
print(df.shape)
print(df.head(6))
print("seasons:", sorted(df["season"].unique().to_list()))
# Spot-check: a known 2023-24 opening fixture should be present, two rows.
print(df.filter((df["season"] == "2023-24") & (df["gw"] == 1)).head(4))
conn.close()
Path("/tmp/ff_fixtures_smoke.duckdb").unlink(missing_ok=True)
PY
```

Expected: a non-empty frame; `seasons` includes `2023-24` and the current
season; each `(season, gw, team)` for a single-fixture gw has exactly one row
per team; `kickoff_time` values are populated timestamps.

- [ ] **Step 2: Confirm and note any anomalies**

If the current season has no scheduled fixtures yet (off-season), the current
season may contribute zero rows — that is acceptable. Record the row count and
season list in the task notes.

---

## Self-Review Notes

- **Spec coverage:** schema (Task 1), separate-extraction sourcing (Tasks 3–5),
  immutable-historic / upsert-current storage (Task 2), season routing &
  orchestration (Task 6), ingestion wiring (Task 7), testing incl.
  double-gameweek and home/away explosion (Tasks 3–6). Coverage note (FCI
  non-current gap) handled by the warn-and-skip branch in Task 6.
- **Out of scope (unchanged):** `features/fixtures.py` → `fixtures_enriched.csv`.
- **Type consistency:** `NormalisedFixture` tuple, `TEAM_FIXTURE_COLUMNS`,
  `TEAM_FIXTURE_SCHEMA`, and helper names (`fixtures_to_team_rows`,
  `build_current_fixtures`, `build_vaastav_fixtures`, `load_fixtures`,
  `write_immutable_fixtures`, `upsert_current_fixtures`,
  `fixture_seasons_present`, `load_team_fixture`) are used identically across
  tasks.
