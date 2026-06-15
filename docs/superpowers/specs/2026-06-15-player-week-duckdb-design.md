# Player-week data into duckdb — design

**Date:** 2026-06-15
**Status:** Approved, ready for implementation planning
**Branch:** `jt/data_ingestion`

## Summary

Move all player-week data (one row per player per gameweek, across every
season) out of CSV files and into a single embedded duckdb database. The
duckdb table becomes the **single source of truth** for player-week data;
`load_gw_data` reads from it instead of stitching CSVs together. The historic
aggregate `cleaned_merged_seasons.csv` is retired entirely.

A normal `main.py` run is **incremental**: missing immutable seasons are loaded
once, and the current season is re-fetched and upserted. The DB is not rewritten
wholesale on every run.

Other project data (fixtures, teams, etc.) stays as CSV for now — only
player-week data moves to the DB. The DB schema can be widened later when a
feature needs it (YAGNI).

## Storage choice: duckdb

duckdb is the right tool here:

- Embedded, single-file, zero-server — just a file under the (already
  gitignored) `data/` folder.
- Native Arrow/Polars bridge — slots into the existing Polars pipeline with no
  dataframe-shape changes; the *sink* changes, not the reshaping logic.
- Columnar OLAP engine matched to the access pattern (scan a wide player-week
  table, filter by season/GW, aggregate).
- SQL `INSERT … ON CONFLICT` upserts — exactly what the current-season refresh
  needs.
- Portable: the `.duckdb` file is queryable by any duckdb client (CLI, DBeaver,
  DataGrip, DuckDB UI) — open read-only to query while the pipeline runs.

Alternatives considered and rejected:

- **SQLite** — embedded and already used for MLflow, but row-oriented (OLTP) and
  no native Arrow/Polars bridge; slower analytical scans and more marshalling.
- **Parquet files (no DB)** — columnar and readable by duckdb/Polars, but would
  require hand-rolling partition layout, "which GWs exist" checks, and upserts
  as file-rewrite logic, and loses ad-hoc SQL. Revisit only if the data outgrows
  one machine (it never will). We can always `COPY` a table out to Parquet if a
  portable artifact is ever wanted.
- **Postgres / server DB** — overkill for a single-user local analytics project.

## Architecture & module boundary

New module `fantasy_football/storage/database.py` is the only place that touches
duckdb. Public functions:

- `get_connection()` — opens (creating if absent)
  `data/fantasy_football.duckdb` and ensures the `player_week` table exists via
  idempotent `CREATE TABLE IF NOT EXISTS`.
- `seasons_present()` — returns the set of seasons already in the table; drives
  "load if absent" for immutable seasons.
- `write_immutable_season(df, season)` — inserts a completed season's rows only
  if that season is not already present; no-op otherwise.
- `upsert_current_season(df, season)` — replaces the current season's rows via
  `INSERT … ON CONFLICT (season, gw, element) DO UPDATE`.
- `load_player_week()` — returns the whole table as a Polars frame; the single
  entry point `load_gw_data` calls.

The extractors (`DataExtractor`, `FciExtractor`) stop writing player-week CSVs
and hand their Polars frames to these write functions instead. The reshaping
logic (`build_merged_gw`, the Vaastav parsing) is untouched — only the sink
changes.

## Schema

Table `player_week`, types pinned rather than inferred:

| column | type | notes |
|---|---|---|
| `season` | VARCHAR | e.g. `2025-26` |
| `gw` | INTEGER | gameweek (the old `GW`/`round`) |
| `element` | INTEGER | FPL player id |
| `name` | VARCHAR | |
| `position` | VARCHAR | GK/DEF/MID/FWD |
| `team` | VARCHAR | |
| `bonus` | INTEGER | per-GW bonus |
| `minutes` | INTEGER | |
| `round` | INTEGER | kept alongside `gw` to match current shape |
| `total_points` | INTEGER | |
| `value` | INTEGER | player price |

**Primary key / conflict target: `(season, gw, element)`** — makes the upsert
and the "already present" checks work.

The `value` column (option B) is the one addition over what `load_gw_data`
consumes today. Both sources already produce it (FCI derives it; the Vaastav
aggregate carries it), so there are no backfill gaps. Wider future-proofing
columns are deliberately excluded for now.

## Write flow (default `main.py` run, incremental)

1. `get_connection()` ensures the table exists.
2. **Immutable seasons** (historic aggregate + Vaastav bridge seasons): for each
   season *not* in `seasons_present()`, download from Vaastav, reshape to the
   schema, `write_immutable_season`. Already-present seasons are skipped — no
   download.
3. **Current season** (FCI): always re-fetch season-to-date, reshape,
   `upsert_current_season` keyed on `(season, gw, element)`. Late bonus/minutes
   corrections overwrite cleanly; brand-new GWs are inserted.

A normal run touches the network only for the current season (cheap, matches
today's behavior) and writes only changed/new current-season rows plus any
missing historic season.

Details:

- **Historic aggregate → seasons:** `cleaned_merged_seasons.csv` is one file
  covering many seasons. Split it by its `season` column on import so each
  season is tracked independently in `seasons_present()` — the "load if absent"
  check is per-season, not all-or-nothing.
- **`value` for historic rows:** the Vaastav aggregate carries `value`, mapped
  straight in.

## Downstream reads

`load_gw_data` in `features/transformation.py` collapses its three-CSV stitch
into a single `df = load_player_week()` call (the frame is already shaped as
`season, gw, element, …`). The `_load_season_merged_gw` helper and the
`cleaned_merged_seasons.csv` / bridge-file reads are deleted. Everything
downstream (`create_rolling_points_data`, fixtures, elo, prediction,
optimisation) is unchanged because it consumes the frame, not the files.

## `main.py` flags

- Default run = **incremental** (the write flow above). No flag needed for the
  common case.
- `download_all_data` is repurposed to **`rebuild: bool = False`** — when
  `True`, drops and reloads every season (full refresh / recovery escape hatch).
- The `__main__` block uses `main(rebuild=False, ...)` so the default invocation
  is the cheap incremental path.

## Retirements

- `DataExtractor.save_all_data_files` and the CSV-writing path in
  `FciExtractor.build_current_season_merged_gw` stop writing player-week CSVs;
  their reshaping logic is preserved and rewired to the DB sink.
- `cleaned_merged_seasons.csv` and `data/raw/<season>/gws/merged_gw.csv` are no
  longer produced or read.

## Gitignore

The DB lives at `data/fantasy_football.duckdb`. `data/` is already gitignored,
so it is covered; add an explicit `data/*.duckdb` line as documentation of
intent.

## Testing

- `database.py` unit tests against a temp-file duckdb (real engine, throwaway
  `tmp_path`): table creation is idempotent; `write_immutable_season` is a no-op
  when the season exists; `upsert_current_season` overwrites a changed row and
  inserts a new GW; `seasons_present` reflects writes.
- Extractor tests: existing pure reshapers (`build_merged_gw`) keep their tests;
  add a round-trip test that a built frame survives DB write → `load_player_week`
  read unchanged.
- `dummy_team.json` / optimisation tests: unaffected.
