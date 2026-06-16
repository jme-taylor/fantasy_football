# Team Fixture Table — Design

**Date:** 2026-06-17
**Branch:** jt/availability_features
**Status:** Approved

## Goal

Add a new duckdb table, `team_fixture`, populated during ingestion (alongside the
existing `player_week` build). It holds one row **per team, per fixture, per
gameweek**, for all available seasons.

### Schema

| column         | type      | notes                                          |
| -------------- | --------- | ---------------------------------------------- |
| `season`       | VARCHAR   | short form, e.g. `2023-24`                     |
| `gw`           | BIGINT    | gameweek, from fixture `event`                 |
| `team`         | VARCHAR   | team name, consistent with `player_week.team`  |
| `is_home`      | BOOLEAN   | true if `team` is the home side                |
| `opposition`   | VARCHAR   | opponent team name                             |
| `kickoff_time` | TIMESTAMP | full UTC timestamp (date derivable downstream) |

**Primary key:** `(season, gw, team, opposition)` — unique even in
double-gameweeks, since a team cannot play the same opponent twice in one gw.

## Why a separate extraction path (not derived from player rows)

The `player_week` pipeline flows through "merged_gw"-shaped player frames. For the
current FCI-era season those frames **collapse double-gameweeks**, so per-fixture
granularity is lost. We therefore source fixtures from **true fixture lists**, not
from player rows.

## Sourcing

Two source adapters feed one shared transform. Routing mirrors the existing
`player_week` historic/current split.

- **Historic seasons** (Vaastav folder seasons, ~2016-17 → last completed):
  - `https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/fixtures.csv`
    — FPL-shaped: `code, event, finished, id, kickoff_time, team_a, team_h, ...`
  - `https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/teams.csv`
    — maps **season-local** team ids (1–20, alphabetical) → `name`.
  - Written **immutably** (once), like `write_immutable_season`.

- **Current live season** (2025-26): the FPL `/api/fixtures/` endpoint, already used
  by `features/fixtures.py` (`team_h`, `team_a`, `event`, `kickoff_time`), with team
  id→name via `FplAPI.get_teams()`.
  - **Upserted** each run (delete + reinsert the season), like
    `upsert_current_season`.

Both sources are FPL-shaped, so one transform handles both: each fixture is
**exploded into two team-perspective rows** (home + away). Double-gameweeks
naturally produce two distinct rows per team. This is the per-fixture granularity
the FCI player-stats collapse cannot provide.

When the live season finishes, Vaastav backfills its season folder, so the historic
(immutable) path takes over with no data loss and no need to snapshot the live API.

### Coverage note

"All seasons" means all seasons with an available Vaastav `fixtures.csv` folder
(~2016-17 onward) plus the current live season. Earlier seasons present in
`cleaned_merged_seasons.csv` but lacking a fixtures folder are out of scope.

## Code structure

- **`fantasy_football/extraction/fixtures.py`** (new) — exposes a function that
  returns a Polars frame in the `team_fixture` schema for a given season, choosing
  the Vaastav vs FPL-API adapter by season. Contains the shared
  fixture→two-team-rows transform.
- **`fantasy_football/storage/database.py`** — add `TEAM_FIXTURE_COLUMNS`, a
  `CREATE TABLE IF NOT EXISTS team_fixture` (PK `(season, gw, team, opposition)`), a
  coercion helper, and `write_immutable_fixtures` / `upsert_current_fixtures`
  mirroring the existing player-week functions.
- **`main.py`** — call fixtures ingestion immediately after player-week ingestion,
  reusing the same season routing (immutable historic, upsert current).

### Out of scope

The existing `features/fixtures.py` → `data/transformed/fixtures_enriched.csv`
(current season only, CSV, `kickoff_date`) stays as-is. The new table is the
durable, all-seasons duckdb equivalent. Retiring the CSV is a later, separate
change.

## Testing

Unit tests with small sample CSV inputs:

- **Vaastav adapter**: season-local team id→name mapping via `teams.csv`;
  fixture→two-row explosion; correct `is_home`/`opposition`/`kickoff_time`.
- **FPL-API adapter**: same explosion from API-shaped fixtures.
- **Double-gameweek**: a team with two fixtures in one gw yields two rows with
  distinct oppositions.
- **Schema/dtype coercion**: columns and types match `TEAM_FIXTURE_COLUMNS`.
