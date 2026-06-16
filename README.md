# Fantasy Football

A personal project to automatically pick my Fantasy Premier League (FPL) team. The rough plan is:

* Use historic player and points data to build models that predict future points scored.
* Predict points for upcoming gameweeks.
* Run an optimisation algorithm on top of those predictions to select the best squad, starting XI, captain, and transfers.

## Current State

This is an early-stage work in progress. `main.py` is the entry point that
refreshes the current season's data, transforms it, predicts points, and runs
the optimiser. The package is organised by domain:

```text
fantasy_football/
├── constants.py        — shared constants such as the current season
├── logging_config.py   — logging setup
├── fpl_types.py        — Pydantic types describing FPL API responses
├── extraction/         — ingest raw data
│   ├── fpl.py          — wrappers around the live FPL API (players, teams, fixtures)
│   ├── fci.py          — download current-season data from FPL Core Insights and reshape to merged_gw.csv
│   ├── extractor.py    — download the frozen Vaastav historic dataset
│   └── seasons.py      — season-string conversions and data-source routing
├── features/           — derive model-ready features
│   ├── transformation.py — transform raw data into rolling/feature datasets
│   ├── fixtures.py     — build the enriched fixtures table used as model features
│   └── elo.py          — build team Elo ratings
├── modelling/          — train, predict, evaluate
│   ├── models.py       — per-position points models behind a shared interface
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

## Data sources

Historic data (through the 2024-25 season) comes from the
[Vaastav FPL repository](https://github.com/vaastav/Fantasy-Premier-League),
which stopped weekly updates as of 2025/26. From the 2025-26 season onwards,
current data comes from [FPL Core Insights](https://github.com/olbauday/FPL-Core-Insights).
The cutoff is controlled by `VASTAAV_LAST_SEASON` and
`FPL_CORE_INSIGHTS_FIRST_SEASON` in `constants.py`. FCI stores per-gameweek
snapshots in a different shape, so it is reconstructed into Vaastav's
`merged_gw.csv` layout — keeping the rest of the pipeline unchanged.

## Roadmap

The high-level milestones are:

* [X] Fetch and download a basic dataset of all weekly data
* [X] Transform it into a rolling points dataset
* [X] Build a simple model of rolling points weighted by opponent strength and home/away
* [X] Build an optimisation algorithm on top of the predicted points
* [X] Format the optimiser output into concrete team / transfer decisions
* [X] Get a new datasource for future/current data now that Vastaav has sunsetted their project
* [X] Backtest against a mid-season gameweek and ensure all decisions respect FPL rules
* [~] Define metrics for evaluating model quality (per-position models + MLflow evaluation)
  * [X] Choose regression metrics suited to low, zero-inflated FPL points (skill score, Spearman, precision@k, MAE, RMSE, Poisson deviance)
  * [X] Stand up MLflow tracking with simple start/stop scripts
  * [X] Baseline-score each position's model using the current rolling-points formula
  * [ ] Wire evaluation logging into the main pipeline run
* [ ] Dig into the worst-performing position and investigate its scoring errors
* [ ] Identify and incorporate additional features to improve the model

The current baseline (the rolling-points formula, scored per position over all
historical gameweeks) only narrowly beats predicting each player's recent
average — its value is in *ranking* players (Spearman ≈ 0.6–0.8 by position)
rather than predicting exact point totals. Improving on that baseline is the
focus of the next milestones.

Open questions still being worked through include how to handle double gameweeks, chips (wildcard, triple captain, etc.), promoted teams and new players, managerial changes, and disciplinary suspensions. On the application side I'm considering moving from Polars + CSVs to DuckDB, adding proper logging, exploring reinforcement learning as an alternative to a pure optimiser, and adding code-coverage and CI checks.

## Installation

This project uses Python 3.12 and [uv](https://docs.astral.sh/uv/) for dependency and environment management.

1. Clone the repository and `cd` into it.
2. Install Python 3.12 (e.g. via [pyenv](https://github.com/pyenv/pyenv) or `uv python install 3.12`).
3. Install dependencies:

   ```bash
   uv sync
   ```

   This will create a `.venv/` and install both runtime and dev dependencies from `uv.lock`.

## Usage

Run the data download via `uv`:

```bash
# Update only the current season's data (default)
uv run python main.py

# Download all historical seasons
uv run python main.py  # then set download_all_data=True in main()
```

### Evaluating the models with MLflow

Each position (GK, DEF, MID, FWD) has its own points model. The evaluation
harness replays every historical gameweek one step ahead, compares each
model's predictions to the actual points scored, and records the metrics in
MLflow — one experiment per position (`gk-points-model`, `def-points-model`,
`mid-points-model`, `fwd-points-model`).

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
is sourced from [FPL Core Insights](https://github.com/olbauday/FPL-Core-Insights).
Both are accessed via the GitHub API using `GITHUB_API_KEY`.
