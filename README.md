# Fantasy Football

A personal project to automatically pick my Fantasy Premier League (FPL) team. The rough plan is:

* Use historic player and points data to build models that predict future points scored.
* Predict points for upcoming gameweeks.
* Run an optimisation algorithm on top of those predictions to select the best squad, starting XI, captain, and transfers.

## Current State

This is an early-stage work in progress. `main.py` is the entry point that
refreshes the current season's data into a DuckDB store, derives feature
tables, trains the minutes-played model, predicts points, and runs the
optimiser. The package is organised by domain:

```text
fantasy_football/
├── constants.py        — shared constants (current season, paths, model tunables, MLflow config)
├── logging_config.py   — logging setup
├── fpl_types.py        — Pydantic types describing FPL API responses
├── storage/            — persistence
│   └── database.py     — DuckDB store; the single source of truth for the
│                          player_week, team_fixture, player_match and
│                          player_availability tables (create/coerce/upsert/load)
├── extraction/         — ingest raw data into the DuckDB store
│   ├── fpl.py          — wrappers around the live FPL API (players, teams, fixtures, per-fixture history)
│   ├── player_match.py — build the current season's per-fixture player_match rows from the FPL API
│   ├── extractor.py    — download the frozen Vaastav historic dataset (player_week + player_match); GitHub API client
│   ├── fci.py          — download current-season data from FPL Core Insights and reshape to player_week rows
│   ├── fplcache.py     — read per-gameweek team and chance-of-playing from the Randdalf/fplcache snapshots
│   ├── availability.py — backfill/upsert the player_availability table
│   ├── fixtures.py     — load team-fixture rows (historic Vaastav, current FPL API) into team_fixture
│   └── seasons.py      — season-string conversions and data-source routing
├── features/           — derive model-ready features
│   ├── transformation.py — transform player_week into the rolling/feature dataset
│   ├── fixtures.py     — build the enriched fixtures table used as model features
│   ├── elo.py          — build team Elo ratings (scraped via ScraperFC ClubElo)
│   ├── valuation.py    — team-value share and positional value rank features
│   └── availability.py — rolling minutes, chance-of-playing and positional-availability features
├── modelling/          — train, predict, evaluate
│   ├── models.py       — per-position points models behind a shared interface
│   ├── minutes.py      — end-to-end minutes-played classifier (features, CV, MLflow)
│   ├── prediction.py   — apply the models to produce per-(player, gameweek) predictions
│   ├── metrics.py      — regression metrics (skill score, Spearman, precision@k, MAE, RMSE, Poisson deviance)
│   └── evaluation.py   — replay historical gameweeks one step ahead and log to MLflow
└── optimisation/       — build the plan
    ├── optimiser.py    — linear-programming optimiser → squad, XI, captain, transfers
    ├── plan_report.py  — format the optimiser output into readable decisions
    └── team_input.py   — load and resolve a carried-in squad
```

`tests/unit/` mirrors this structure and runs in CI. `tests/integration/`
holds tests that hit live network data sources (FPL / fplcache); they are
marked `@pytest.mark.integration` and deselected by default.

## Storage

The source of truth is a single-file DuckDB database at
`data/fantasy_football.duckdb` (gitignored). `storage/database.py` is the only
module that touches DuckDB: extractors hand it Polars frames, downstream code
reads frames back. It owns four tables:

* **`player_week`** — one row per `(season, gw, element)`, collapsed across
  double gameweeks. The substrate for the rolling-points features and the
  per-position points models.
* **`team_fixture`** — one row per `(season, gw, team, opposition)` with
  home/away and kickoff time.
* **`player_match`** — one row per `(season, gw, element, opponent)` (minutes
  and points only), **not** collapsed across double gameweeks. The training
  substrate for the minutes-played model.
* **`player_availability`** — one row per `(season, gw, element)` capturing
  FPL's point-in-time `chance_of_playing_this_round`.

Completed (immutable) seasons are inserted once and skipped thereafter; the
current season is upserted (delete-then-insert) every run so late corrections,
new gameweeks, and injury news refresh cleanly. `main(rebuild=True)` drops and
reloads everything as a recovery escape hatch.

A handful of derived feature tables (`rolling_points.csv`,
`fixtures_enriched.csv`, `team_elo.csv`, `predictions.csv`) are still written as
CSVs under `data/transformed/`.

## Data sources

Historic data (through the 2024-25 season) comes from the
[Vaastav FPL repository](https://github.com/vaastav/Fantasy-Premier-League),
which stopped weekly updates as of 2025/26. From the 2025-26 season onwards,
current data comes from [FPL Core Insights](https://github.com/olbauday/FPL-Core-Insights).
The cutoff is controlled by `VASTAAV_LAST_SEASON` and
`FPL_CORE_INSIGHTS_FIRST_SEASON` in `constants.py`. FCI stores per-gameweek
snapshots in a different shape, so it is reconstructed into the canonical
`player_week` layout — keeping the rest of the pipeline unchanged.

The `player_match` table's historic rows (≤2024-25) come from the same Vaastav
per-fixture files; current-season rows (2025-26) come from the FPL API
`element-summary` endpoint, which reports native per-fixture minutes and points.

> **⚠️ Time-sensitive backfill:** the live FPL API only exposes the **current**
> season's per-fixture history via `element-summary`. Once the API rolls over
> to the next season (typically late July/August), the previous season's
> per-fixture data is no longer retrievable there. Run the current-season
> `player_match` backfill before the rollover; historic Vaastav data has no
> such deadline.

The `player_availability` table's `chance_of_playing_this_round` is FPL's
0/25/50/75 percentage signal; `null` (no injury doubt) is stored as `100`. The
source is the [Randdalf/fplcache](https://github.com/Randdalf/fplcache)
bootstrap snapshot at each gameweek's deadline — the same source used to derive
per-gameweek `team_code`, because FCI only records a player's final club and so
gets mid-season transfers wrong for pre-transfer gameweeks. Coverage starts
from `2022-23` (fplcache's earliest snapshots, set by `FPLCACHE_FIRST_SEASON`).

## Minutes-played model

`modelling/minutes.py` trains a single 3-class classifier that predicts each
player's minutes bucket for an upcoming match: benched (`0_minutes`), partial
(`1_to_59_minutes`), or a full-ish shift (`60_minutes_plus`). It is scored on
the two decision boundaries the downstream points models care about —
probability of *any* appearance and probability of a *60+ minute* appearance —
cross-validated with an expanding window over seasons, then refit on all
seasons and logged to MLflow (experiment `minutes_played_classification`).

Its features are assembled from the `player_match`, `player_week` and
`player_availability` tables: rolling minutes, FPL chance-of-playing,
positional availability (fit same-club, same-position rivals), team-value share
and positional value rank. The model runs on every data refresh from `main`,
but a modelling failure is logged and swallowed so it can never block
prediction and optimisation.

## Roadmap

The high-level milestones are:

* [X] Fetch and download a basic dataset of all weekly data
* [X] Transform it into a rolling points dataset
* [X] Build a simple model of rolling points weighted by opponent strength and home/away
* [X] Build an optimisation algorithm on top of the predicted points
* [X] Format the optimiser output into concrete team / transfer decisions
* [X] Get a new datasource for future/current data now that Vastaav has sunsetted their project
* [X] Backtest against a mid-season gameweek and ensure all decisions respect FPL rules
* [X] Migrate the data layer from Polars + CSVs to a DuckDB store
* [~] Define metrics for evaluating model quality (per-position models + MLflow evaluation)
  * [X] Choose regression metrics suited to low, zero-inflated FPL points (skill score, Spearman, precision@k, MAE, RMSE, Poisson deviance)
  * [X] Stand up MLflow tracking with simple start/stop scripts
  * [X] Baseline-score each position's model using the current rolling-points formula
  * [X] Wire evaluation logging into the main pipeline run (`main(evaluate=True)`)
* [~] Build a minutes-played model to feed appearance probabilities into the points models
  * [X] Per-fixture `player_match` table and point-in-time `player_availability` table
  * [X] Availability / valuation features
  * [X] 3-class classifier with expanding-window CV, scored on appearance and 60+ boundaries, logged to MLflow
  * [ ] Feed its appearance probabilities into the per-position points models
* [ ] Dig into the worst-performing position and investigate its scoring errors
* [ ] Identify and incorporate additional features to improve the model

The current baseline (the rolling-points formula, scored per position over all
historical gameweeks) only narrowly beats predicting each player's recent
average — its value is in *ranking* players (Spearman ≈ 0.6–0.8 by position)
rather than predicting exact point totals. Improving on that baseline is the
focus of the next milestones.

### Looking further ahead

Shortened from `notepad.md`, the longer-term plan once the minutes model is
integrated:

* **Minutes model** — add a random and an informed baseline, then evaluate the
  full model's scoring impact before wiring its appearance probabilities into
  the per-position points models.
* **Points models** — research where the per-position models are weakest, find
  features that address it, and measure the lift.
* **Operations** — retrain every model weekly on historic data and test for
  drift; richer dummy-data fixtures in `conftest.py`; code-coverage metrics and
  CI checks.
* **Application** — a Streamlit frontend (pick a team from the bootstrap or
  load one by FPL ID) and API endpoints; possibly a reinforcement-learning
  agent as an alternative to the pure optimiser.

Open questions still being worked through include how to handle double
gameweeks, chips (wildcard, triple captain, etc.), promoted teams and new
players, managerial changes, disciplinary suspensions, mid-season transfers,
and backtesting how good the optimiser actually is.

## Installation

This project uses Python 3.12 and [uv](https://docs.astral.sh/uv/) for dependency and environment management.

1. Clone the repository and `cd` into it.
2. Install Python 3.12 (e.g. via [pyenv](https://github.com/pyenv/pyenv) or `uv python install 3.12`).
3. Install dependencies:

   ```bash
   uv sync
   ```

   This will create a `.venv/` and install both runtime and dev dependencies from `uv.lock`.

A `GITHUB_API_KEY` (in a `.env` file) is required to download the Vaastav,
FCI, and fplcache datasets via the GitHub API.

## Usage

Run the full pipeline via `uv`:

```bash
# Refresh the current season, train the minutes model, predict and optimise
uv run python main.py
```

`main()` accepts a few keyword arguments:

* `rebuild` (default `False`) — drop and reload every season from scratch
  (full refresh / recovery escape hatch). The default loads only missing
  immutable seasons and upserts the current season.
* `team_file` — path to a name-authored team JSON; optimisation then carries
  in that squad from its gameweek instead of free-building.
* `evaluate` (default `False`) — run the rolling-origin model evaluation
  (logging metrics to MLflow) before predicting and optimising.

### Evaluating the models with MLflow

Each position (GK, DEF, MID, FWD) has its own points model. The evaluation
harness replays every historical gameweek one step ahead, compares each
model's predictions to the actual points scored, and records the metrics in
MLflow — one experiment per position (`gk-points-model`, `def-points-model`,
`mid-points-model`, `fwd-points-model`). The minutes model logs to its own
experiment (`minutes_played_classification`).

Run an evaluation (this writes runs to a local SQLite store at
`models/mlflow.db`):

```bash
uv run python -m fantasy_football.modelling.evaluation
```

Browse the results in the MLflow UI using the helper scripts:

```bash
./.bin/start_mlflow.sh   # launches the UI at http://127.0.0.1:5050
./.bin/stop_mlflow.sh    # stops it again
```

`start_mlflow.sh` first clears any process already on the port, so re-running
it restarts the UI cleanly. All MLflow data lives under the `models/` folder,
which is gitignored — only the scripts are tracked.

## Development

Run the unit suite and linters with `uv`:

```bash
uv run pytest                  # unit tests only (integration deselected)
uv run pytest -m integration   # live network integration tests
uv run ruff check .
uv run ruff format --check .
uv run ty check
```

## Resources

Historic data is sourced from the [Fantasy Premier League](https://github.com/vaastav/Fantasy-Premier-League)
repository by Vaastav Anand (through 2024-25). Current-season data (2025-26+)
is sourced from [FPL Core Insights](https://github.com/olbauday/FPL-Core-Insights),
and point-in-time team and availability snapshots from
[Randdalf/fplcache](https://github.com/Randdalf/fplcache). All three are
accessed via the GitHub API using `GITHUB_API_KEY`. Team Elo ratings are
scraped from ClubElo via [ScraperFC](https://github.com/oseymour/ScraperFC).
