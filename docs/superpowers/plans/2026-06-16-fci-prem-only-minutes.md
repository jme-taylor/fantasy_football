# FCI Prem-Only Minutes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop FCI minutes being inflated by cup/European matches by counting only Premier League appearances.

**Architecture:** FCI's `playermatchstats.csv` carries one row per match across all competitions. `build_merged_gw` sums `minutes_played` over `(gw, player_id)` indiscriminately, double-counting cup/Euro games. The fix filters to Premier League matches (identified by the `match_id` prefix `<yy>-<yy>-prem-`) before the sum. Because `fetch_season_frames` currently projects `match_id` away, the column must first be retained.

**Tech Stack:** Python, Polars, pytest, DuckDB.

**Spec:** `docs/superpowers/specs/2026-06-16-fci-prem-only-minutes-design.md`

---

## File Structure

- Modify: `fantasy_football/extraction/fci.py`
  - Add `match_id` to `MATCHSTATS_SCHEMA` so `fetch_season_frames` keeps it.
  - Add a Premier-League filter in `build_merged_gw` before the minutes aggregation.
  - Update the `build_merged_gw` matchstats docstring to require `match_id`.
- Modify (tests): `tests/unit/extraction/test_fci.py`
  - Make the shared `MATCHSTATS` fixture use realistic `match_id` values.
  - Add a test that non-prem matches are excluded.

---

### Task 1: Make existing tests use realistic match_ids and add the exclusion test

This locks in the desired behaviour before touching `fci.py`. The shared
fixture's placeholder `match_id`s (`"m1"`, `"m2"`, ...) are replaced with
Premier-League-shaped ids so the existing double-gameweek test doubles as the
"real PL double gameweek still sums" regression, and a new test asserts a cup
match is excluded.

**Files:**
- Test: `tests/unit/extraction/test_fci.py:21-29` (replace `MATCHSTATS`)
- Test: `tests/unit/extraction/test_fci.py` (add one test function)

- [ ] **Step 1: Replace the shared `MATCHSTATS` fixture**

Replace the current definition (lines 21-29) with PL-shaped match ids. Player 2
still plays twice in GW2, and both are Premier League matches, so the sum to 110
remains the correct "real double gameweek" expectation.

```python
MATCHSTATS = pl.DataFrame(
    {
        "gw": [1, 1, 2, 2, 2],
        "player_id": [1, 2, 1, 2, 2],  # player 2 plays twice in GW2 (real PL DGW)
        "match_id": [
            "25-26-prem-arsenal-vs-chelsea",
            "25-26-prem-arsenal-vs-chelsea",
            "25-26-prem-arsenal-vs-spurs",
            "25-26-prem-man-city-vs-spurs",
            "25-26-prem-man-city-vs-wolves",
        ],
        "minutes_played": [90, 90, 90, 80, 30],
    },
    schema_overrides={"gw": pl.Int32},
)
```

- [ ] **Step 2: Add the non-prem exclusion test**

Append this test function to `tests/unit/extraction/test_fci.py` (next to the
other `build_merged_gw` tests, e.g. after
`test_build_merged_gw_sums_double_gameweek_minutes`):

```python
def test_build_merged_gw_excludes_non_prem_match_minutes() -> None:
    """Cup / European minutes are not summed into the gameweek total."""
    matchstats = pl.DataFrame(
        {
            "gw": [1, 1],
            "player_id": [1, 1],
            "match_id": [
                "25-26-prem-arsenal-vs-chelsea",
                "25-26-efl-cup-arsenal-vs-brighton",
            ],
            "minutes_played": [90, 90],
        },
        schema_overrides={"gw": pl.Int32},
    )
    result = build_merged_gw(
        SNAPSHOTS, matchstats, PLAYERS, PLAYER_GW_TEAM, TEAM_CODE_TO_NAME
    )
    raya_gw1 = result.filter(
        (pl.col("element") == 1) & (pl.col("GW") == 1)
    ).row(0, named=True)
    assert raya_gw1["minutes"] == 90  # PL only, not 180
```

- [ ] **Step 3: Run the tests to verify the new one fails and the DGW test passes**

Run: `.venv/bin/pytest tests/unit/extraction/test_fci.py -v`

Expected:
- `test_build_merged_gw_excludes_non_prem_match_minutes` FAILS with
  `assert 180 == 90` (filter not yet implemented — cup minutes still summed).
- `test_build_merged_gw_sums_double_gameweek_minutes` PASSES (both GW2 matches
  are prem, still sum to 110).

- [ ] **Step 4: Commit**

```bash
git add tests/unit/extraction/test_fci.py
git commit -m "test(fci): require prem-only minutes, realistic match_ids"
```

---

### Task 2: Retain `match_id` and filter minutes to Premier League matches

**Files:**
- Modify: `fantasy_football/extraction/fci.py:69-72` (`MATCHSTATS_SCHEMA`)
- Modify: `fantasy_football/extraction/fci.py:88-93` (matchstats docstring)
- Modify: `fantasy_football/extraction/fci.py:110-114` (minutes aggregation)

- [ ] **Step 1: Add `match_id` to `MATCHSTATS_SCHEMA`**

`fetch_season_frames` projects matchstats down to `list(MATCHSTATS_SCHEMA)`, so
`match_id` must be in the schema or it is dropped before `build_merged_gw` runs.
Replace the current `MATCHSTATS_SCHEMA` (lines 69-72):

```python
MATCHSTATS_SCHEMA: dict[str, pl.DataType] = {
    "player_id": pl.Int64,
    "minutes_played": pl.Int64,
    "match_id": pl.Utf8,
}
```

- [ ] **Step 2: Update the matchstats docstring in `build_merged_gw`**

Replace the `matchstats` parameter description (lines 88-93) so the required
columns include `match_id`:

```python
    matchstats : pl.DataFrame
        Concatenated ``playermatchstats`` with a ``gw`` column. Must contain
        ``gw, player_id, minutes_played, match_id``. ``match_id`` encodes the
        competition as ``<yy>-<yy>-<comp>-<home>-vs-<away>``; only Premier
        League rows (``prem`` competition token) count towards FPL minutes.
```

- [ ] **Step 3: Filter to Premier League matches before summing minutes**

Replace the minutes aggregation (lines 110-114). The anchored regex matches the
`<yy>-<yy>-prem-` prefix so a team name containing "prem" can never match, and
only Premier League minutes are summed (real PL double gameweeks still sum;
non-prem cup/Euro matches are dropped):

```python
    # Per-GW minutes: sum across Premier League matches only so double
    # gameweeks accumulate but cup/European fixtures (also present in
    # playermatchstats) do not inflate the total. ``match_id`` encodes the
    # competition as ``<yy>-<yy>-<comp>-...``; ``prem`` is the only PL token.
    # Cast ``gw`` to Int64 so the join key matches the fplcache per-GW team
    # table regardless of the caller's source dtype (production emits Int32).
    minutes = (
        matchstats.filter(
            pl.col("match_id").str.contains(r"^\d{2}-\d{2}-prem-")
        )
        .group_by(["gw", "player_id"])
        .agg(pl.col("minutes_played").sum().alias("minutes"))
        .with_columns(pl.col("gw").cast(pl.Int64))
    )
```

- [ ] **Step 4: Run the FCI tests to verify they pass**

Run: `.venv/bin/pytest tests/unit/extraction/test_fci.py -v`

Expected: all tests PASS, including
`test_build_merged_gw_excludes_non_prem_match_minutes` (now 90) and
`test_build_merged_gw_sums_double_gameweek_minutes` (still 110).

- [ ] **Step 5: Run the full unit suite to check nothing else broke**

Run: `.venv/bin/pytest tests/unit -q`

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add fantasy_football/extraction/fci.py
git commit -m "fix(fci): count only Premier League minutes per gameweek"
```

---

### Task 3: Verify the fix against live FCI data for Guéhi

A non-automated sanity check that the real symptom is resolved (uses the network;
read-only, no DB writes).

**Files:** none (verification only).

- [ ] **Step 1: Build merged_gw from live FCI frames and check Guéhi GW15/17**

Run this script (downloads FCI GW1-17, runs the real adapter path for minutes):

```bash
.venv/bin/python - <<'PY'
import requests, polars as pl, io
base = ('https://raw.githubusercontent.com/olbauday/FPL-Core-Insights'
        '/main/data/2025-2026/By Gameweek')
frames = []
for gw in range(1, 18):
    url = f'{base}/GW{gw}/playermatchstats.csv'.replace(' ', '%20')
    r = requests.get(url)
    if r.status_code != 200:
        continue
    d = (pl.read_csv(io.BytesIO(r.content))
         .select('player_id', 'minutes_played', 'match_id')
         .with_columns(pl.lit(gw).alias('gw')))
    frames.append(d)
matchstats = pl.concat(frames, how='diagonal')
minutes = (
    matchstats.filter(pl.col('match_id').str.contains(r'^\d{2}-\d{2}-prem-'))
    .group_by(['gw', 'player_id'])
    .agg(pl.col('minutes_played').sum().alias('minutes'))
)
guehi = minutes.filter(pl.col('player_id') == 260).sort('gw')
print(guehi)
assert guehi.filter(pl.col('gw') == 15)['minutes'][0] == 90
assert guehi.filter(pl.col('gw') == 17)['minutes'][0] == 90
print('OK: GW15 and GW17 are 90 minutes')
PY
```

Expected: prints Guéhi's per-gameweek minutes with GW15 = 90 and GW17 = 90, and
`OK: GW15 and GW17 are 90 minutes`.

---

## Notes

- **Backfill:** none required. `upsert_current_season` deletes and re-inserts the
  whole current season each run, so the next ingestion overwrites the bad rows.
- **Scope:** points and bonus are unaffected (sourced from snapshots). Historic
  Vaastav seasons use a separate one-row-per-fixture collapse path and are
  untouched.
