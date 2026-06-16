# FCI per-gameweek team — fixing mid-season transfers

**Date:** 2026-06-16
**Status:** Approved (design)

## Problem

`build_merged_gw` (in `fantasy_football/extraction/fci.py`) sources each
player-week's `team` from `players.csv`'s `team_code` — a single, *current*
team per player. Every per-gameweek FCI snapshot (`players.csv`,
`player_gameweek_stats.csv`) is likewise overwritten with the player's *final*
team.

The consequence: a player who transfers clubs mid-season is recorded as playing
for their later club for the *entire* season. Confirmed against live data —
even **GW1**'s `players.csv` lists Antoine Semenyo (and Marc Guehi) at Man City
(code 43), which is wrong for their pre-transfer gameweeks.

There is no per-gameweek team field anywhere in the FCI player files, and
`playermatchstats.csv` has no team or home/away flag. So the correct team per
gameweek cannot be read from FCI alone, and we do not want to infer it.

## Data source: `Randdalf/fplcache`

[`Randdalf/fplcache`](https://github.com/Randdalf/fplcache) is a time-series
cache of FPL's `bootstrap-static` endpoint, snapshotted four times a day across
the whole season. Because each snapshot records every player's team *as of that
moment*, a snapshot taken during a given gameweek gives the player's **actual**
club that week — deterministically, with no inference.

Verified facts about the cache:

- **Layout:** `cache/{year}/{month}/{day}/{HHMM}.json.xz`, where `month` and
  `day` are **not** zero-padded (e.g. `cache/2025/8/16/1250.json.xz`). Files are
  LZMA-compressed JSON (stdlib `lzma` decodes them).
- **Coverage:** four snapshots/day, full 2025-26 season present.
- **`elements[]`** (players) carry: `id` (FPL element id — equals FCI's
  `player_id` within a season), `code` (stable cross-season id, equals FCI's
  leading id column, e.g. 437730 for Semenyo), `team` (season-local team id
  1–20), `team_code` (stable team code — **same namespace** as FCI's
  `players.csv` `team_code` and our `FplAPI.get_teams().code`), `element_type`,
  `web_name`.
- **`events[]`** carry each gameweek's `id` and `deadline_time` (38 entries).
- **`teams[]`** carry `id`, `code`, `name`, `short_name`.

Example: on `2025-08-16`, Semenyo's element has `team_code` 91 (Bournemouth);
after his transfer a later snapshot shows Man City. The fix reads the right one
for each gameweek.

## Design

### Approach

For each gameweek, select the fplcache snapshot active at that gameweek's
deadline, and read every player's `team_code` from it. This produces a
`(gw, element, team_code)` table reflecting the actual club that week. We
replace the static `players.team_code` join in `build_merged_gw` with this
per-gameweek table. Fully deterministic; handles any number of transfers.

(Considered and rejected: a single late snapshot — reproduces the current bug;
two pre/post-January snapshots — fragile special-casing; inferring from
`matches.csv` home/away — explicitly out of scope.)

### Components

1. **`fantasy_football/extraction/fplcache.py` — `FplCacheExtractor`**
   - Fetches snapshots from `Randdalf/fplcache`. Uses the GitHub contents API to
     list the snapshot times available on a given day, and raw URLs
     (`https://raw.githubusercontent.com/Randdalf/fplcache/main/<path>`) to
     download. Accepts an optional `api_key` (same pattern as `GitHubAPIClient`)
     so directory listings use authenticated requests and avoid the
     unauthenticated rate limit.
   - Reads `events` deadlines once (from any recent snapshot).
   - For each requested gameweek `g`, selects the **first snapshot with timestamp
     ≥ `deadline_time(g)`** (the team is locked at the deadline, so any snapshot
     in `[deadline(g), deadline(g+1))` is consistent; "first at/after deadline"
     is the simplest deterministic rule). Decompresses with `lzma`, parses JSON.
   - Returns a tidy Polars frame `(gw, element, team_code)`, where `element` is
     the FPL element `id` (matches FCI's `player_id` for the within-season join).
   - Public surface (indicative):
     - `event_deadlines() -> dict[int, datetime]`
     - `snapshot_for_gameweek(gw, deadline) -> dict` (selection + download + parse)
     - `build_player_gw_team(gameweeks: list[int]) -> pl.DataFrame`

2. **`build_merged_gw` (`fci.py`)** — gains a `player_gw_team: pl.DataFrame`
   parameter `(gw, element, team_code)`. It joins this on `(gw, element)` for
   `team_code` instead of joining the static `players.team_code`. `position`
   still comes from `players.csv`; the `team_code → name` mapping
   (`team_code_to_name`) is unchanged.

3. **`FciExtractor` (`fci.py`)** — `fetch_season_frames` /
   `build_current_season_merged_gw` instantiate `FplCacheExtractor`, build the
   per-gameweek team frame for the season's gameweeks, and pass it into
   `build_merged_gw`.

### Data flow

`list_gameweeks` → fplcache `events` deadlines → per-gw deadline snapshot →
`(gw, element, team_code)` → joined into `build_merged_gw` → correct `team` per
player-gameweek → `upsert_current_season`.

### Edge cases

- **Snapshot selection:** first snapshot with timestamp ≥ `deadline_time(g)`.
  Snapshots run 4×/day, so the gap is ≤6h. The final gameweek has no `g+1`; the
  rule still works (just picks the first snapshot after the last deadline).
- **Player missing from a snapshot** (not yet registered): the left join leaves
  `team_code` null. Fallback order: forward/back-fill `team_code` within the
  player's gw-ordered sequence; if still null, fall back to `players.csv`
  `team_code`. Every row ends up populated.
- **Double gameweeks:** team is keyed per `(gw, element)`, so a player's two
  matches in one gameweek share one team — consistent with how minutes already
  collapse to one row.
- **Missing snapshot file at a date:** widen to the nearest available time that
  day, then the next day; raise a clear error only if a whole gameweek window
  has no snapshot.
- **Join key:** FCI `player_id` == FPL element `id` within a season, and
  fplcache elements carry the same `id`, so we join on `id`. The stable `code`
  is present in both if cross-season robustness is ever needed.

## Testing

### Test layout restructure

- Move existing tests from `tests/<pkg>/...` to **`tests/unit/<pkg>/...`**,
  preserving the package-mirrored subfolders. Add `tests/integration/` for the
  network test. Add `__init__.py` files where needed.
- Register an `integration` marker in `pyproject.toml` and set
  `addopts = "-m 'not integration'"` with `testpaths = ["tests"]`. Integration
  tests are decorated `@pytest.mark.integration`, so they are deselected by
  default everywhere — local `uv run pytest` and CI both run unit-only.
- `ci.yml`'s `uv run pytest` then runs unit-only with no change; add a clarifying
  comment.
- Run the integration suite on demand with `uv run pytest -m integration`.
- Update `README.md`: replace "`tests/` mirrors this structure" to describe
  `tests/unit` (mirrors the package) vs `tests/integration` (network, deselected
  by default), and add the `-m integration` command to the testing section.

### Regression test (the core ask) — `tests/unit`

Unit-test the team-assembly with **synthetic** snapshots for two players named
**Semenyo** and **Guehi**: early-gameweek snapshots place them at Bournemouth /
Crystal Palace, later snapshots at Man City. Assert each player's `team` flips
at the right gameweek and is **not** uniformly Man City. Reproduces and locks
the bug fix, no network.

### Unit tests — `tests/unit`

- `FplCacheExtractor`: deadline → snapshot selection; missing-player fallback;
  `lzma` + JSON parsing (mock the download).
- Update existing `build_merged_gw` tests to pass the new `player_gw_team`
  argument.

### Integration test — `tests/integration` (`@pytest.mark.integration`)

Hits real fplcache + FCI for the 2025-26 season and asserts Semenyo's and
Guehi's GW1 team is their pre-transfer club (Bournemouth / Crystal Palace),
not Man City. Deselected by default; runnable with `-m integration`.

## Out of scope

- Adding `opponent` / `is_home` columns to `merged_gw` (the schema stores only
  `team`).
- Backfilling historic Vaastav seasons (their data already has correct
  per-gameweek teams).
- Any inference-based team assignment.
