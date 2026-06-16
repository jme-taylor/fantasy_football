# FCI minutes: restrict to Premier League matches

**Date:** 2026-06-16
**Status:** Approved, ready for implementation
**Branch:** jt/data_ingestion

## Problem

The current season (2025-26 onwards) is sourced from FPL Core Insights (FCI).
`build_merged_gw` in `fantasy_football/extraction/fci.py` reconstructs
Vaastav-shaped per-gameweek rows, computing per-gameweek minutes by summing
`minutes_played` over `(gw, player_id)` from each gameweek's
`playermatchstats.csv`.

That file contains **one row per match across all competitions** — not just the
Premier League. FPL minutes count Premier League appearances only, so any player
whose club also played an EFL Cup or European fixture (Champions / Europa /
Conference League) in the same gameweek window has those minutes summed on top
of their PL minutes, inflating the stored value.

### Evidence

Marc Guéhi (element 260), Crystal Palace, 2025-26 — the reported symptom:

| GW | PL match (minutes)              | Extra match counted (minutes)        | Stored | Correct |
|----|---------------------------------|--------------------------------------|--------|---------|
| 15 | `prem-fulham-vs-crystal...` (90)| `conference-league-crystal...` (90)  | 180    | 90      |
| 17 | `prem-leeds-united-vs-cry...`(90)| `efl-cup-crystal-palace-v...` (90)  | 180    | 90      |
| 16 | `prem-crystal-palace-vs-m...`(90)| `conference-league-kups...` (26)    | 116    | 90      |
| 4  | `prem-...` (90)                  | `efl-cup-millwall-vs-crystal...` (61)| 151    | 90      |

Verified for GW1–17: every inflated gameweek (4, 6, 9, 10, 12, 15, 16, 17) is
explained exactly by a non-PL match being summed in.

### Blast radius

This is systemic, not Guéhi-specific. Across GW1–17, **993 of 5,697**
player-gameweek rows have more than one match and therefore inflated minutes.
Non-PL match rows by competition: EFL Cup 654, Champions League 496, Europa 138,
Conference 75. It affects every club in European or domestic-cup competition.

`total_points` (from snapshots' `event_points`) and `bonus` (cumulative-diff
from snapshots) are FPL figures and remain correct. **Only `minutes` is wrong.**

## Fix

Filter `matchstats` to Premier League rows before the minutes aggregation, using
the only available competition signal: the `match_id` prefix.

FCI exposes no dedicated competition column. The `match_id` format
`<yy>-<yy>-<comp>-<home>-vs-<away>` is 100% consistent (249/249 distinct match
ids in GW1–17 conform), and the competition token set is exactly
`prem / efl / champions / europa / conference`. `prem` is the only PL token.

### Change

In `build_merged_gw` (`fantasy_football/extraction/fci.py`), at the minutes step
(currently lines 110–114), insert an anchored-prefix filter before the
`group_by`:

```python
minutes = (
    matchstats.filter(pl.col("match_id").str.contains(r"^\d{2}-\d{2}-prem-"))
    .group_by(["gw", "player_id"])
    .agg(pl.col("minutes_played").sum().alias("minutes"))
    .with_columns(pl.col("gw").cast(pl.Int64))
)
```

- The regex is anchored on the `<yy>-<yy>-<comp>-` prefix so a team name
  containing the substring "prem" can never match.
- Players with no PL match in a gameweek fall out of the minutes join and get
  `minutes = 0` via the existing `fill_null(0)` at line 174 — no behaviour
  change there.
- Real Premier League double gameweeks still sum correctly: two `-prem-` matches
  in one gameweek aggregate to 180, as they should.

### Why this location

The filter lives in the pure `build_merged_gw` adapter, so it is exercised by
fast unit tests with no IO — consistent with the existing adapter tests.

## Testing

Extend the `matchstats` fixture in `tests/unit/extraction/test_fci.py`:

1. A player with one `-prem-` match plus a non-prem cup match in the same
   gameweek — assert stored `minutes` reflects the PL match only.
2. A player with two `-prem-` matches in one gameweek — assert they still sum
   (double-gameweek regression).

## Backfill

None required. `upsert_current_season` deletes and re-inserts the entire current
season on every run, so the next ingestion overwrites the bad rows with the
corrected minutes.

## Out of scope

- Any runtime sanity-check / validation guard (regression test only).
- Changes to points, bonus, or team-resolution logic.
- Historic Vaastav seasons, which are one-row-per-fixture and collapsed via their
  own `_collapse_double_gameweeks` path — unaffected.
