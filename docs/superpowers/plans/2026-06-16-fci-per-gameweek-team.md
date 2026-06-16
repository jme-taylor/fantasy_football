# FCI per-gameweek team Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record each player's *actual* club per gameweek (fixing mid-season transfers like Semenyo/Guehi) by reading per-gameweek team from the `Randdalf/fplcache` bootstrap-static time-series instead of FCI's static `team_code`.

**Architecture:** A new `FplCacheExtractor` downloads the fplcache snapshot active at each gameweek's deadline and emits a `(gw, element, team_code)` table. `build_merged_gw` joins that table for the per-gameweek `team`, falling back to FCI's static `team_code` only when a player is absent from a snapshot. Tests are split into `tests/unit` (run in CI) and `tests/integration` (network, deselected by default).

**Tech Stack:** Python 3.12, Polars, `requests`, stdlib `lzma`/`json`/`datetime`, DuckDB, pytest, ruff (line-length 79, NumPy-style docstrings required by the `D` ruleset), uv.

---

## File Structure

- **Create** `fantasy_football/extraction/fplcache.py` — `FplCacheExtractor`: fetch fplcache snapshots, resolve gameweek deadlines, select the snapshot per gameweek, emit `(gw, element, team_code)`.
- **Modify** `fantasy_football/extraction/fci.py` — `build_merged_gw` gains a `player_gw_team` parameter and joins it for `team`; `FciExtractor` builds and passes that frame.
- **Create** `tests/unit/extraction/test_fplcache.py` — unit tests for `FplCacheExtractor`.
- **Create** `tests/integration/test_transfer_team.py` — network regression test for Semenyo/Guehi.
- **Move** every existing `tests/<pkg>/...` to `tests/unit/<pkg>/...`.
- **Modify** `pyproject.toml` — pytest config (testpaths, `integration` marker, default deselect).
- **Modify** `.github/workflows/ci.yml` — clarifying comment (no behaviour change).
- **Modify** `README.md` — document the unit/integration split and how to run each.

---

## Task 1: Restructure tests into unit/integration and configure pytest

**Files:**
- Move: `tests/extraction/`, `tests/features/`, `tests/modelling/`, `tests/optimisation/`, `tests/storage/` → under `tests/unit/`
- Create: `tests/unit/__init__.py`, `tests/integration/__init__.py`
- Modify: `pyproject.toml`
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Move existing test packages under `tests/unit/`**

```bash
cd /Users/jamie/personal/fantasy_football
find tests -name '__pycache__' -type d -exec rm -rf {} +
mkdir -p tests/unit tests/integration
git mv tests/extraction tests/unit/extraction
git mv tests/features tests/unit/features
git mv tests/modelling tests/unit/modelling
git mv tests/optimisation tests/unit/optimisation
git mv tests/storage tests/unit/storage
touch tests/unit/__init__.py tests/integration/__init__.py
```

- [ ] **Step 2: Add pytest config to `pyproject.toml`**

Append this section to `pyproject.toml` (after the `[tool.ruff.format]` block):

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "integration: hits live network data sources; deselected by default",
]
addopts = "-m 'not integration'"
```

- [ ] **Step 3: Run the suite to confirm the move and default deselect work**

Run: `uv run pytest -q`
Expected: PASS — every existing test is collected from `tests/unit/...`; no integration tests run yet.

- [ ] **Step 4: Add a clarifying comment to CI**

In `.github/workflows/ci.yml`, change the Pytest step to:

```yaml
      - name: Pytest
        # addopts in pyproject deselects @pytest.mark.integration, so CI runs unit tests only.
        run: uv run pytest
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "test: split tests into unit/integration with default deselect"
```

---

## Task 2: `FplCacheExtractor` — download and decode a snapshot

**Files:**
- Create: `fantasy_football/extraction/fplcache.py`
- Test: `tests/unit/extraction/test_fplcache.py`

- [ ] **Step 1: Write the failing test**

```python
import io
import json
import lzma

import polars as pl
from pytest_mock import MockerFixture

from fantasy_football.extraction.extractor import GitHubAPIClient
from fantasy_football.extraction.fplcache import FplCacheExtractor


def _make_extractor() -> FplCacheExtractor:
    return FplCacheExtractor(
        api_client=GitHubAPIClient(
            api_key="k", owner="Randdalf", repo="fplcache", branch="main"
        )
    )


def test_read_snapshot_decodes_lzma_json(mocker: MockerFixture) -> None:
    """_read_snapshot downloads, LZMA-decompresses and JSON-parses a file."""
    payload = {"elements": [{"id": 1, "team_code": 43}]}
    compressed = lzma.compress(json.dumps(payload).encode())

    extractor = _make_extractor()
    response = mocker.Mock()
    response.content = compressed
    response.raise_for_status = mocker.Mock()
    mocker.patch(
        "fantasy_football.extraction.fplcache.requests.get",
        return_value=response,
    )

    result = extractor._read_snapshot("cache/2025/8/16/1250.json.xz")
    assert result == payload
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/extraction/test_fplcache.py::test_read_snapshot_decodes_lzma_json -v`
Expected: FAIL with `ModuleNotFoundError: fantasy_football.extraction.fplcache`.

- [ ] **Step 3: Write minimal implementation**

Create `fantasy_football/extraction/fplcache.py`:

```python
"""Read per-gameweek team from the Randdalf/fplcache bootstrap time-series.

FCI records only a player's final club, so mid-season transfers are wrong for
pre-transfer gameweeks. ``Randdalf/fplcache`` snapshots FPL's bootstrap-static
endpoint four times a day; the snapshot active at a gameweek's deadline gives
the player's actual club that week. ``FplCacheExtractor`` selects that snapshot
per gameweek and emits a ``(gw, element, team_code)`` table.
"""

import io
import json
import logging
import lzma

import polars as pl
import requests

from fantasy_football.extraction.extractor import GitHubAPIClient

logger = logging.getLogger(__name__)


class FplCacheExtractor:
    """Resolve each gameweek's player->team_code map from fplcache snapshots."""

    def __init__(self, api_client: GitHubAPIClient | None = None) -> None:
        """Initialise the extractor.

        Parameters
        ----------
        api_client : GitHubAPIClient | None, optional
            Client pointed at the fplcache repo. Defaults to a new client for
            ``Randdalf/fplcache`` on ``main``.
        """
        self.api_client = api_client or GitHubAPIClient(
            owner="Randdalf", repo="fplcache", branch="main"
        )

    def _read_snapshot(self, path: str) -> dict:
        """Download an LZMA-compressed JSON snapshot and parse it.

        Parameters
        ----------
        path : str
            Repo-relative path of the ``.json.xz`` snapshot.

        Returns
        -------
        dict
            The parsed bootstrap-static object.
        """
        url = self.api_client.get_raw_file_url(path)
        response = requests.get(url)
        response.raise_for_status()
        raw = lzma.open(io.BytesIO(response.content)).read()
        return json.loads(raw)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/extraction/test_fplcache.py::test_read_snapshot_decodes_lzma_json -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add fantasy_football/extraction/fplcache.py tests/unit/extraction/test_fplcache.py
git commit -m "feat(fplcache): read and decode bootstrap snapshots"
```

---

## Task 3: `FplCacheExtractor` — list directories and find the latest snapshot

**Files:**
- Modify: `fantasy_football/extraction/fplcache.py`
- Test: `tests/unit/extraction/test_fplcache.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/extraction/test_fplcache.py`:

```python
def test_latest_snapshot_path_walks_descending(
    mocker: MockerFixture,
) -> None:
    """_latest_snapshot_path picks max year/month/day/time numerically."""
    extractor = _make_extractor()

    listings = {
        "cache": [{"name": "2024"}, {"name": "2025"}],
        "cache/2025": [{"name": "8"}, {"name": "12"}, {"name": "2"}],
        "cache/2025/12": [{"name": "9"}, {"name": "26"}],
        "cache/2025/12/26": [
            {"name": "0202.json.xz"},
            {"name": "1833.json.xz"},
            {"name": "1250.json.xz"},
        ],
    }
    mocker.patch.object(
        extractor.api_client,
        "get_file_details",
        side_effect=lambda path: listings[path],
    )

    assert (
        extractor._latest_snapshot_path()
        == "cache/2025/12/26/1833.json.xz"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/extraction/test_fplcache.py::test_latest_snapshot_path_walks_descending -v`
Expected: FAIL with `AttributeError: ... no attribute '_latest_snapshot_path'`.

- [ ] **Step 3: Write minimal implementation**

Add these methods to `FplCacheExtractor`:

```python
    def _list_names(self, path: str) -> list[str]:
        """List the entry names directly under a repo directory.

        Parameters
        ----------
        path : str
            Repo-relative directory path.

        Returns
        -------
        list[str]
            The ``name`` of each entry in the directory.
        """
        return [entry["name"] for entry in self.api_client.get_file_details(path)]

    def _latest_snapshot_path(self) -> str:
        """Return the path of the newest snapshot in the cache.

        Returns
        -------
        str
            Repo-relative path of the most recent ``.json.xz`` snapshot.
        """
        year = max(self._list_names("cache"), key=int)
        month = max(self._list_names(f"cache/{year}"), key=int)
        day = max(self._list_names(f"cache/{year}/{month}"), key=int)
        time = max(self._list_names(f"cache/{year}/{month}/{day}"))
        return f"cache/{year}/{month}/{day}/{time}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/extraction/test_fplcache.py::test_latest_snapshot_path_walks_descending -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add fantasy_football/extraction/fplcache.py tests/unit/extraction/test_fplcache.py
git commit -m "feat(fplcache): locate the latest snapshot via directory listings"
```

---

## Task 4: `FplCacheExtractor` — gameweek deadlines and per-gameweek snapshot selection

**Files:**
- Modify: `fantasy_football/extraction/fplcache.py`
- Test: `tests/unit/extraction/test_fplcache.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/extraction/test_fplcache.py`:

```python
from datetime import datetime, timezone


def test_event_deadlines_parses_from_latest_snapshot(
    mocker: MockerFixture,
) -> None:
    """event_deadlines reads the events array from the latest snapshot."""
    extractor = _make_extractor()
    mocker.patch.object(
        extractor, "_latest_snapshot_path", return_value="cache/x.json.xz"
    )
    mocker.patch.object(
        extractor,
        "_read_snapshot",
        return_value={
            "events": [
                {"id": 1, "deadline_time": "2025-08-15T17:30:00Z"},
                {"id": 2, "deadline_time": "2025-08-22T17:30:00Z"},
            ]
        },
    )

    deadlines = extractor.event_deadlines()
    assert deadlines[1] == datetime(2025, 8, 15, 17, 30, tzinfo=timezone.utc)
    assert deadlines[2] == datetime(2025, 8, 22, 17, 30, tzinfo=timezone.utc)


def test_snapshot_path_for_picks_first_at_or_after_deadline(
    mocker: MockerFixture,
) -> None:
    """The chosen snapshot is the earliest one at or after the deadline."""
    extractor = _make_extractor()
    mocker.patch.object(
        extractor,
        "_list_names",
        return_value=[
            "0202.json.xz",
            "0636.json.xz",
            "1250.json.xz",
            "1833.json.xz",
        ],
    )
    deadline = datetime(2025, 8, 16, 13, 0, tzinfo=timezone.utc)
    assert (
        extractor._snapshot_path_for(deadline)
        == "cache/2025/8/16/1833.json.xz"
    )


def test_snapshot_path_for_rolls_to_next_day(mocker: MockerFixture) -> None:
    """When no snapshot follows the deadline that day, roll to the next day."""
    extractor = _make_extractor()

    def fake_list(path: str) -> list[str]:
        if path == "cache/2025/8/16":
            return ["0202.json.xz", "0636.json.xz"]
        if path == "cache/2025/8/17":
            return ["0205.json.xz", "0640.json.xz"]
        return []

    mocker.patch.object(extractor, "_list_names", side_effect=fake_list)
    deadline = datetime(2025, 8, 16, 17, 30, tzinfo=timezone.utc)
    assert (
        extractor._snapshot_path_for(deadline)
        == "cache/2025/8/17/0205.json.xz"
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/extraction/test_fplcache.py -k "event_deadlines or snapshot_path_for" -v`
Expected: FAIL with `AttributeError` for `event_deadlines` / `_snapshot_path_for`.

- [ ] **Step 3: Write minimal implementation**

Add to the imports at the top of `fplcache.py`:

```python
from datetime import datetime, timedelta, timezone
```

Add these methods to `FplCacheExtractor`:

```python
    # How many days past a deadline to search for a snapshot before giving up.
    _MAX_LOOKAHEAD_DAYS = 7

    def event_deadlines(self) -> dict[int, datetime]:
        """Map gameweek id to its deadline datetime (UTC).

        Returns
        -------
        dict[int, datetime]
            ``{event_id: deadline_time}`` parsed from the latest snapshot.
        """
        snapshot = self._read_snapshot(self._latest_snapshot_path())
        return {
            event["id"]: datetime.fromisoformat(
                event["deadline_time"].replace("Z", "+00:00")
            )
            for event in snapshot["events"]
        }

    def _snapshot_path_for(self, deadline: datetime) -> str:
        """Return the path of the first snapshot at or after a deadline.

        Parameters
        ----------
        deadline : datetime
            Timezone-aware gameweek deadline.

        Returns
        -------
        str
            Repo-relative path of the chosen snapshot.

        Raises
        ------
        ValueError
            If no snapshot is found within the lookahead window.
        """
        for offset in range(self._MAX_LOOKAHEAD_DAYS):
            day = deadline.date() + timedelta(days=offset)
            directory = f"cache/{day.year}/{day.month}/{day.day}"
            try:
                names = sorted(self._list_names(directory))
            except Exception:
                continue
            for name in names:
                stamp = name.removesuffix(".json.xz")
                taken = datetime(
                    day.year,
                    day.month,
                    day.day,
                    int(stamp[:2]),
                    int(stamp[2:]),
                    tzinfo=timezone.utc,
                )
                if taken >= deadline:
                    return f"{directory}/{name}"
        raise ValueError(f"No fplcache snapshot found at or after {deadline}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/extraction/test_fplcache.py -k "event_deadlines or snapshot_path_for" -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add fantasy_football/extraction/fplcache.py tests/unit/extraction/test_fplcache.py
git commit -m "feat(fplcache): resolve deadlines and per-gameweek snapshot"
```

---

## Task 5: `FplCacheExtractor.build_player_gw_team`

**Files:**
- Modify: `fantasy_football/extraction/fplcache.py`
- Test: `tests/unit/extraction/test_fplcache.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/extraction/test_fplcache.py`:

```python
def test_build_player_gw_team_emits_per_gw_team_code(
    mocker: MockerFixture,
) -> None:
    """A player's team_code reflects the snapshot active each gameweek."""
    extractor = _make_extractor()
    mocker.patch.object(
        extractor,
        "event_deadlines",
        return_value={
            1: datetime(2025, 8, 15, 17, 30, tzinfo=timezone.utc),
            2: datetime(2025, 8, 22, 17, 30, tzinfo=timezone.utc),
        },
    )
    mocker.patch.object(
        extractor,
        "_snapshot_path_for",
        side_effect=lambda d: f"cache/{d.day}.json.xz",
    )
    snapshots = {
        "cache/15.json.xz": {"elements": [{"id": 82, "team_code": 91}]},
        "cache/22.json.xz": {"elements": [{"id": 82, "team_code": 43}]},
    }
    mocker.patch.object(
        extractor, "_read_snapshot", side_effect=lambda p: snapshots[p]
    )

    result = extractor.build_player_gw_team([1, 2]).sort("gw")
    assert result.columns == ["gw", "element", "team_code"]
    assert result.to_dicts() == [
        {"gw": 1, "element": 82, "team_code": 91},
        {"gw": 2, "element": 82, "team_code": 43},
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/extraction/test_fplcache.py::test_build_player_gw_team_emits_per_gw_team_code -v`
Expected: FAIL with `AttributeError: ... 'build_player_gw_team'`.

- [ ] **Step 3: Write minimal implementation**

Add this method to `FplCacheExtractor`:

```python
    def build_player_gw_team(self, gameweeks: list[int]) -> pl.DataFrame:
        """Build the per-gameweek player team_code table.

        Parameters
        ----------
        gameweeks : list[int]
            Gameweek numbers to resolve.

        Returns
        -------
        pl.DataFrame
            One row per ``(gw, element)`` with columns ``gw, element,
            team_code`` (all ``Int64``), where ``element`` is the FPL element
            id (== FCI ``player_id``).
        """
        deadlines = self.event_deadlines()
        frames: list[pl.DataFrame] = []
        for gw in gameweeks:
            path = self._snapshot_path_for(deadlines[gw])
            snapshot = self._read_snapshot(path)
            frames.append(
                pl.DataFrame(
                    {
                        "gw": gw,
                        "element": [e["id"] for e in snapshot["elements"]],
                        "team_code": [
                            e["team_code"] for e in snapshot["elements"]
                        ],
                    },
                    schema={
                        "gw": pl.Int64,
                        "element": pl.Int64,
                        "team_code": pl.Int64,
                    },
                )
            )
        return pl.concat(frames, how="vertical")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/extraction/test_fplcache.py::test_build_player_gw_team_emits_per_gw_team_code -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add fantasy_football/extraction/fplcache.py tests/unit/extraction/test_fplcache.py
git commit -m "feat(fplcache): build per-gameweek player team_code table"
```

---

## Task 6: `build_merged_gw` joins per-gameweek team with static fallback

**Files:**
- Modify: `fantasy_football/extraction/fci.py:74-152`
- Test: `tests/unit/extraction/test_fci.py`

- [ ] **Step 1: Update existing tests to the new signature and add the fallback test**

In `tests/unit/extraction/test_fci.py`, after the `PLAYERS` / `TEAM_CODE_TO_NAME` constants, add a per-gameweek team frame and a transfer fixture:

```python
# Per-gameweek team_code: player 2 is at Arsenal (3) in GW1, Man City (43) GW2.
PLAYER_GW_TEAM = pl.DataFrame(
    {
        "gw": [1, 1, 2, 2],
        "element": [1, 2, 1, 2],
        "team_code": [3, 3, 3, 43],
    },
    schema={"gw": pl.Int64, "element": pl.Int64, "team_code": pl.Int64},
)
```

Update **every** existing `build_merged_gw(...)` call in this file to pass
`PLAYER_GW_TEAM` as the fourth argument (before `TEAM_CODE_TO_NAME`). The five
call sites are in: `test_build_merged_gw_columns_and_shape`,
`test_build_merged_gw_maps_core_fields`,
`test_build_merged_gw_sums_double_gameweek_minutes`,
`test_build_merged_gw_bonus_is_event_level_diff`,
`test_build_merged_gw_fills_missing_minutes_with_zero`. For example:

```python
result = build_merged_gw(
    SNAPSHOTS, MATCHSTATS, PLAYERS, PLAYER_GW_TEAM, TEAM_CODE_TO_NAME
)
```

Then add two new tests:

```python
def test_build_merged_gw_uses_per_gameweek_team() -> None:
    """team comes from the per-gameweek snapshot, not the static team_code."""
    result = build_merged_gw(
        SNAPSHOTS, MATCHSTATS, PLAYERS, PLAYER_GW_TEAM, TEAM_CODE_TO_NAME
    )
    # Player 2 (static team_code 43 = Man City) was at Arsenal in GW1.
    p2_gw1 = result.filter(
        (pl.col("element") == 2) & (pl.col("GW") == 1)
    ).row(0, named=True)
    assert p2_gw1["team"] == "Arsenal"
    p2_gw2 = result.filter(
        (pl.col("element") == 2) & (pl.col("GW") == 2)
    ).row(0, named=True)
    assert p2_gw2["team"] == "Man City"


def test_build_merged_gw_falls_back_to_static_team() -> None:
    """A player absent from the per-gameweek table keeps the static team."""
    # Drop player 1 from the per-gameweek table entirely.
    partial = PLAYER_GW_TEAM.filter(pl.col("element") != 1)
    result = build_merged_gw(
        SNAPSHOTS, MATCHSTATS, PLAYERS, partial, TEAM_CODE_TO_NAME
    )
    p1_gw1 = result.filter(
        (pl.col("element") == 1) & (pl.col("GW") == 1)
    ).row(0, named=True)
    assert p1_gw1["team"] == "Arsenal"  # static team_code 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/extraction/test_fci.py -k build_merged_gw -v`
Expected: FAIL — the new fourth positional argument is rejected (`TypeError`) until the signature changes.

- [ ] **Step 3: Update `build_merged_gw`**

In `fantasy_football/extraction/fci.py`, change the signature and docstring of
`build_merged_gw` to add `player_gw_team` before `team_code_to_name`:

```python
def build_merged_gw(
    snapshots: pl.DataFrame,
    matchstats: pl.DataFrame,
    players: pl.DataFrame,
    player_gw_team: pl.DataFrame,
    team_code_to_name: dict[int, str],
) -> pl.DataFrame:
```

Add to the Parameters section of its docstring (after the ``players`` entry):

```
    player_gw_team : pl.DataFrame
        Per-gameweek team table from fplcache. Must contain ``gw, element,
        team_code``; ``element`` is the FPL element id (== ``id``).
```

Replace the `merged = (...)` assignment (the block joining `players`,
`team_map`, and `minutes`) with this version, which resolves `team_code`
per-gameweek first and falls back to the static value:

```python
    per_gw_team = player_gw_team.rename(
        {"element": "id", "team_code": "team_code_gw"}
    )

    merged = (
        snap.join(
            players.select("player_id", "position", "team_code"),
            left_on="id",
            right_on="player_id",
            how="left",
            coalesce=True,
        )
        .rename({"team_code": "team_code_static"})
        .join(per_gw_team, on=["gw", "id"], how="left", coalesce=True)
        .sort(["id", "gw"])
        .with_columns(
            pl.col("team_code_gw")
            .fill_null(strategy="forward")
            .fill_null(strategy="backward")
            .over("id")
            .alias("team_code_filled")
        )
        .with_columns(
            pl.coalesce(["team_code_filled", "team_code_static"]).alias(
                "team_code"
            )
        )
        .join(team_map, on="team_code", how="left", coalesce=True)
        .join(
            minutes,
            left_on=["gw", "id"],
            right_on=["gw", "player_id"],
            how="left",
            coalesce=True,
        )
        .with_columns(
            (pl.col("first_name") + " " + pl.col("second_name")).alias("name"),
            pl.col("position").replace(FCI_POSITION_TO_VAASTAV),
            pl.col("id").alias("element"),
            pl.col("minutes").fill_null(0).cast(pl.Int64),
            pl.col("gw").alias("round"),
            pl.col("event_points").alias("total_points"),
            pl.col("gw").alias("GW"),
            (pl.col("now_cost") * 10).round(0).cast(pl.Int64).alias("value"),
            pl.col("event_bonus").cast(pl.Int64).alias("bonus"),
        )
    )
    return merged.select(MERGED_GW_COLUMNS)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/extraction/test_fci.py -k build_merged_gw -v`
Expected: PASS (all build_merged_gw tests, including the two new ones).

- [ ] **Step 5: Commit**

```bash
git add fantasy_football/extraction/fci.py tests/unit/extraction/test_fci.py
git commit -m "feat(fci): source per-gameweek team from fplcache with fallback"
```

---

## Task 7: Wire `FciExtractor` to build and pass the per-gameweek team table

**Files:**
- Modify: `fantasy_football/extraction/fci.py:158-315`
- Test: `tests/unit/extraction/test_fci.py`

- [ ] **Step 1: Update the end-to-end test to provide a per-gameweek team table**

In `tests/unit/extraction/test_fci.py`, update
`test_build_current_season_merged_gw_upserts_to_db` so the extractor's
fplcache lookup is mocked. Replace its body's mock setup with:

```python
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
    mocker.patch.object(
        extractor.fpl_cache, "build_player_gw_team", return_value=PLAYER_GW_TEAM
    )
    mocker.patch.object(extractor, "list_gameweeks", return_value=[1, 2])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/extraction/test_fci.py::test_build_current_season_merged_gw_upserts_to_db -v`
Expected: FAIL with `AttributeError: 'FciExtractor' object has no attribute 'fpl_cache'`.

- [ ] **Step 3: Update `FciExtractor`**

In `fantasy_football/extraction/fci.py`, add the import near the other
extraction imports:

```python
from fantasy_football.extraction.fplcache import FplCacheExtractor
```

In `FciExtractor.__init__`, add a `fpl_cache` parameter and attribute. Change
the signature to:

```python
    def __init__(
        self,
        api_client: GitHubAPIClient | None = None,
        fpl_api: FplAPI | None = None,
        fpl_cache: FplCacheExtractor | None = None,
    ) -> None:
```

Add to its docstring Parameters (after ``fpl_api``):

```
        fpl_cache : FplCacheExtractor | None, optional
            Source of per-gameweek team_code. Defaults to a new
            ``FplCacheExtractor``.
```

And at the end of `__init__`, after `self.raw_data_folder = RAW_DATA_FOLDER`:

```python
        self.fpl_cache = fpl_cache or FplCacheExtractor()
```

Then update `build_current_season_merged_gw` to build and pass the table.
Replace its body (from the `long_season = ...` line to the `merged = ...`
call) with:

```python
        long_season = season_short_to_long(short_season)
        snapshots, matchstats, players = self.fetch_season_frames(long_season)
        gameweeks = self.list_gameweeks(long_season)
        player_gw_team = self.fpl_cache.build_player_gw_team(gameweeks)
        merged = build_merged_gw(
            snapshots,
            matchstats,
            players,
            player_gw_team,
            self._team_code_to_name(),
        )
```

- [ ] **Step 4: Run the full extraction test module**

Run: `uv run pytest tests/unit/extraction/test_fci.py -v`
Expected: PASS (all tests, including the end-to-end upsert).

- [ ] **Step 5: Commit**

```bash
git add fantasy_football/extraction/fci.py tests/unit/extraction/test_fci.py
git commit -m "feat(fci): wire FciExtractor to fplcache per-gameweek team"
```

---

## Task 8: Synthetic regression test for Semenyo/Guehi transfers

**Files:**
- Test: `tests/unit/extraction/test_transfer_regression.py` (create)

- [ ] **Step 1: Write the regression test**

Create `tests/unit/extraction/test_transfer_regression.py`:

```python
"""Regression: transferred players keep their real club per gameweek.

Antoine Semenyo and Marc Guehi both transfer to Man City mid-season in this
synthetic fixture. Before the fix, ``team`` came from a static final club and
read Man City for every gameweek. The per-gameweek fplcache team_code must show
their pre-transfer clubs early and Man City only after the transfer.
"""

import polars as pl

from fantasy_football.extraction.fci import build_merged_gw

# Semenyo (element 82) and Guehi (element 200) across GW1 and GW2.
SNAPSHOTS = pl.DataFrame(
    {
        "gw": [1, 1, 2, 2],
        "id": [82, 200, 82, 200],
        "first_name": ["Antoine", "Marc", "Antoine", "Marc"],
        "second_name": ["Semenyo", "Guehi", "Semenyo", "Guehi"],
        "now_cost": [7.0, 5.0, 7.1, 5.1],
        "event_points": [6, 2, 8, 5],
        "bonus": [1, 0, 1, 1],
    }
)
MATCHSTATS = pl.DataFrame(
    {
        "gw": [1, 1, 2, 2],
        "player_id": [82, 200, 82, 200],
        "minutes_played": [90, 90, 90, 90],
    }
)
# Static club is the FINAL club (Man City, 43) for both -- the bug source.
PLAYERS = pl.DataFrame(
    {
        "player_id": [82, 200],
        "position": ["Midfielder", "Defender"],
        "team_code": [43, 43],
    }
)
# Per-gameweek truth: Bournemouth (91) / Crystal Palace (31) in GW1; City GW2.
PLAYER_GW_TEAM = pl.DataFrame(
    {
        "gw": [1, 1, 2, 2],
        "element": [82, 200, 82, 200],
        "team_code": [91, 31, 43, 43],
    },
    schema={"gw": pl.Int64, "element": pl.Int64, "team_code": pl.Int64},
)
TEAM_CODE_TO_NAME = {
    43: "Man City",
    91: "Bournemouth",
    31: "Crystal Palace",
}


def test_transferred_players_keep_real_club_per_gameweek() -> None:
    """Semenyo/Guehi show pre-transfer clubs in GW1, Man City in GW2."""
    result = build_merged_gw(
        SNAPSHOTS, MATCHSTATS, PLAYERS, PLAYER_GW_TEAM, TEAM_CODE_TO_NAME
    ).sort(["element", "GW"])

    teams = {
        (row["name"], row["GW"]): row["team"]
        for row in result.iter_rows(named=True)
    }
    assert teams[("Antoine Semenyo", 1)] == "Bournemouth"
    assert teams[("Antoine Semenyo", 2)] == "Man City"
    assert teams[("Marc Guehi", 1)] == "Crystal Palace"
    assert teams[("Marc Guehi", 2)] == "Man City"

    # Guard against the old bug: not uniformly Man City.
    assert result.filter(pl.col("team") == "Man City").height == 2
```

- [ ] **Step 2: Run the test to verify it passes**

Run: `uv run pytest tests/unit/extraction/test_transfer_regression.py -v`
Expected: PASS (the fix from Task 6 makes this pass immediately).

- [ ] **Step 3: Commit**

```bash
git add tests/unit/extraction/test_transfer_regression.py
git commit -m "test: regression for per-gameweek team on transferred players"
```

---

## Task 9: Network integration test (deselected by default)

**Files:**
- Test: `tests/integration/test_transfer_team.py` (create)

- [ ] **Step 1: Write the integration test**

Create `tests/integration/test_transfer_team.py`:

```python
"""Integration: real fplcache data gives transferred players their real club.

Hits the live Randdalf/fplcache and FPL APIs. Deselected by default; run with
``uv run pytest -m integration``.
"""

import pytest

from fantasy_football.extraction.fplcache import FplCacheExtractor

# FPL element ids and pre-transfer team_codes for the 2025-26 season.
SEMENYO_ELEMENT = 82
GUEHI_ELEMENT = 200
BOURNEMOUTH_CODE = 91
CRYSTAL_PALACE_CODE = 31
MAN_CITY_CODE = 43


@pytest.mark.integration
def test_gw1_team_is_pre_transfer_club() -> None:
    """In GW1, Semenyo/Guehi are at their original clubs, not Man City."""
    extractor = FplCacheExtractor()
    table = extractor.build_player_gw_team([1])

    semenyo = table.filter(
        (table["gw"] == 1) & (table["element"] == SEMENYO_ELEMENT)
    )
    guehi = table.filter(
        (table["gw"] == 1) & (table["element"] == GUEHI_ELEMENT)
    )

    assert semenyo.height == 1
    assert guehi.height == 1
    semenyo_code = semenyo["team_code"][0]
    guehi_code = guehi["team_code"][0]
    assert semenyo_code != MAN_CITY_CODE
    assert guehi_code != MAN_CITY_CODE
    assert semenyo_code == BOURNEMOUTH_CODE
    assert guehi_code == CRYSTAL_PALACE_CODE
```

- [ ] **Step 2: Verify it is deselected by default**

Run: `uv run pytest tests/integration/test_transfer_team.py -v`
Expected: `1 deselected` (no network call, no failure).

- [ ] **Step 3: Run it explicitly to confirm it works against live data**

Run: `uv run pytest tests/integration/test_transfer_team.py -m integration -v`
Expected: PASS (requires `GITHUB_API_KEY` in the environment for directory listings, and network access). If element ids differ from the live data, update the constants to match the verified 2025-26 ids.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_transfer_team.py
git commit -m "test(integration): live fplcache transfer-team check (opt-in)"
```

---

## Task 10: Update README

**Files:**
- Modify: `README.md` (the "`tests/` mirrors this structure" line near line 40, and the testing section near lines 134-141)

- [ ] **Step 1: Update the test-structure description**

Replace the line `` `tests/` mirrors this structure. `` with:

```markdown
`tests/unit/` mirrors this structure and runs in CI. `tests/integration/`
holds tests that hit live network data sources (FPL / fplcache); they are
marked `@pytest.mark.integration` and deselected by default.
```

- [ ] **Step 2: Update the testing commands section**

Replace the testing code block (currently `uv run pytest` plus the ruff/ty
commands) with:

```markdown
Run the unit suite and linters with `uv`:

```bash
uv run pytest                  # unit tests only (integration deselected)
uv run pytest -m integration   # live network integration tests
uv run ruff check .
uv run ruff format --check .
uv run ty check
```
```

- [ ] **Step 3: Verify the docs render and commit**

Run: `uv run pytest -q`
Expected: PASS (unit suite green; integration deselected).

```bash
git add README.md
git commit -m "docs: document unit/integration test split"
```

---

## Final verification

- [ ] **Run the full unit suite, linters, and type check**

Run:
```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run ty check
```
Expected: all green; integration tests deselected.

- [ ] **Run the integration test explicitly (network + `GITHUB_API_KEY`)**

Run: `uv run pytest -m integration -v`
Expected: PASS against live data.
