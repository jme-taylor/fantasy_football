# Fantasy Football

A personal project to automatically pick my Fantasy Premier League (FPL) team. The rough plan is:

* Use historic player and points data to build models that predict future points scored.
* Predict points for upcoming gameweeks.
* Run an optimisation algorithm on top of those predictions to select the best squad, starting XI, captain, and transfers.

## Current State

This is an early-stage work in progress. The repository currently contains:

* `main.py` — entry point that refreshes the current season's data from the correct source (FCI for 2025-26+, Vaastav for historic seasons), transforms it, predicts points, and runs the optimiser.
* `fantasy_football/data_extraction.py` — downloads the frozen Vaastav historic dataset (`cleaned_merged_seasons.csv`).
* `fantasy_football/fci_extraction.py` — downloads current-season data from FPL Core Insights and reconstructs it into Vaastav's `merged_gw.csv` shape.
* `fantasy_football/seasons.py` — season-string conversions and data-source routing.
* `fantasy_football/data_transformation.py` — functions for transforming raw data into rolling/feature datasets.
* `fantasy_football/fpl.py` — wrappers around the live FPL API (players, teams, fixtures).
* `fantasy_football/fpl_types.py` — Pydantic types describing FPL API responses.
* `fantasy_football/constants.py` — shared constants such as the current season.
* `tests/` — unit tests for the modules above.

Earlier per-position prediction models and the linear-programming optimisation prototype have been removed while the data pipeline is being rebuilt. They will be reintroduced once the underlying data and feature pipeline is stable.

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
* [ ] Backtest against a mid-season gameweek and ensure all decisions respect FPL rules
* [ ] Define metrics for evaluating model quality
* [ ] Identify and incorporate additional features to improve the model

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

## Development

Run the test suite and linters with `uv`:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy fantasy_football
```

## Resources

Historic data is sourced from the [Fantasy Premier League](https://github.com/vaastav/Fantasy-Premier-League)
repository by Vaastav Anand (through 2024-25). Current-season data (2025-26+)
is sourced from [FPL Core Insights](https://github.com/olbauday/FPL-Core-Insights).
Both are accessed via the GitHub API using `GITHUB_API_KEY`.
