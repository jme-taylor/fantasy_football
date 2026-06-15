# Fantasy Football Package Reorganisation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganise the 17 flat modules in `fantasy_football/` into four domain sub-packages (`extraction`, `features`, `modelling`, `optimisation`) plus a shared root, mirror the structure in `tests/`, and update the README — with no behaviour change.

**Architecture:** Pure refactor. Each domain is moved as one task: `git mv` the source and test files (renaming where they'd stutter or carry a redundant prefix), then a repo-wide find-replace of that domain's dotted import paths (`fantasy_football.<module> import` → `fantasy_football.<package>.<module> import`), then auto-fix import ordering and run the full check suite. Because every reference in the repo is a `from ... import` statement (verified: no `patch("fantasy_football...")` string targets exist), the find-replace is reliable. Domains are done in dependency order so each commit leaves a green suite.

**Tech Stack:** Python 3.12, uv, pytest, ruff (rules F/I/D), ty, hatchling.

---

## Reference: full import-path mapping

Every module's old → new dotted path. Modules that stay at root are unchanged.

| Old path | New path |
|---|---|
| `fantasy_football.data_extraction` | `fantasy_football.extraction.extractor` |
| `fantasy_football.fci_extraction` | `fantasy_football.extraction.fci` |
| `fantasy_football.fpl` | `fantasy_football.extraction.fpl` |
| `fantasy_football.seasons` | `fantasy_football.extraction.seasons` |
| `fantasy_football.data_transformation` | `fantasy_football.features.transformation` |
| `fantasy_football.fixtures` | `fantasy_football.features.fixtures` |
| `fantasy_football.elo` | `fantasy_football.features.elo` |
| `fantasy_football.models` | `fantasy_football.modelling.models` |
| `fantasy_football.prediction` | `fantasy_football.modelling.prediction` |
| `fantasy_football.evaluation` | `fantasy_football.modelling.evaluation` |
| `fantasy_football.metrics` | `fantasy_football.modelling.metrics` |
| `fantasy_football.optimisation` | `fantasy_football.optimisation.optimiser` |
| `fantasy_football.plan_report` | `fantasy_football.optimisation.plan_report` |
| `fantasy_football.team_input` | `fantasy_football.optimisation.team_input` |
| `fantasy_football.constants` | *(unchanged — root)* |
| `fantasy_football.logging_config` | *(unchanged — root)* |
| `fantasy_football.fpl_types` | *(unchanged — root)* |

**Why the patterns include ` import`:** matching `fantasy_football.fpl import` (not bare `fantasy_football.fpl`) avoids corrupting `fantasy_football.fpl_types`, which shares the `fpl` prefix. The macOS/BSD `sed -i ''` syntax is used throughout.

---

## Task 0: Establish a green baseline

**Files:** none (verification only)

- [ ] **Step 1: Confirm the suite is green before touching anything**

Run: `uv run pytest -q && uv run ruff check . && uv run ty check`
Expected: all pass. If anything fails here, stop and report — the reorg must start from green so later failures are attributable to the move.

---

## Task 1: `extraction/` package

**Files:**
- Create: `fantasy_football/extraction/__init__.py`
- Move: `fantasy_football/data_extraction.py` → `fantasy_football/extraction/extractor.py`
- Move: `fantasy_football/fci_extraction.py` → `fantasy_football/extraction/fci.py`
- Move: `fantasy_football/fpl.py` → `fantasy_football/extraction/fpl.py`
- Move: `fantasy_football/seasons.py` → `fantasy_football/extraction/seasons.py`
- Move tests: `tests/test_data_extraction.py` → `tests/extraction/test_extractor.py`, `tests/test_fci_extraction.py` → `tests/extraction/test_fci.py`, `tests/test_fpl_api.py` → `tests/extraction/test_fpl.py`, `tests/test_seasons.py` → `tests/extraction/test_seasons.py`

- [ ] **Step 1: Create the package directories**

```bash
mkdir -p fantasy_football/extraction tests/extraction
```

- [ ] **Step 2: Create the package `__init__.py` with a docstring**

Create `fantasy_football/extraction/__init__.py` (ruff `D104` requires the docstring):

```python
"""Ingest raw FPL data from the FPL API, FCI, and Vaastav sources."""
```

- [ ] **Step 3: Move source and test files with `git mv` (preserves history)**

```bash
git mv fantasy_football/data_extraction.py  fantasy_football/extraction/extractor.py
git mv fantasy_football/fci_extraction.py   fantasy_football/extraction/fci.py
git mv fantasy_football/fpl.py              fantasy_football/extraction/fpl.py
git mv fantasy_football/seasons.py          fantasy_football/extraction/seasons.py
git mv tests/test_data_extraction.py        tests/extraction/test_extractor.py
git mv tests/test_fci_extraction.py         tests/extraction/test_fci.py
git mv tests/test_fpl_api.py                tests/extraction/test_fpl.py
git mv tests/test_seasons.py                tests/extraction/test_seasons.py
```

- [ ] **Step 4: Rewrite this domain's import paths repo-wide**

```bash
grep -rlE "fantasy_football\.(data_extraction|fci_extraction|fpl|seasons) import" fantasy_football tests main.py | xargs sed -i '' \
  -e 's/fantasy_football\.data_extraction import/fantasy_football.extraction.extractor import/g' \
  -e 's/fantasy_football\.fci_extraction import/fantasy_football.extraction.fci import/g' \
  -e 's/fantasy_football\.fpl import/fantasy_football.extraction.fpl import/g' \
  -e 's/fantasy_football\.seasons import/fantasy_football.extraction.seasons import/g'
```

- [ ] **Step 5: Assert no stale references remain**

Run: `grep -rnE "fantasy_football\.(data_extraction|fci_extraction|fpl|seasons) import" fantasy_football tests main.py`
Expected: no output (exit 1). Any line printed is a missed reference — fix it before continuing. (`fantasy_football.fpl_types` should NOT appear; confirm it is untouched.)

- [ ] **Step 6: Fix import ordering and formatting**

Reordered paths change isort grouping, so let ruff repair it:

Run: `uv run ruff check --fix . && uv run ruff format .`
Expected: ruff reports fixes applied (or "All checks passed").

- [ ] **Step 7: Run the full check suite**

Run: `uv run pytest -q && uv run ruff check . && uv run ty check`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "refactor: move extraction modules into fantasy_football/extraction"
```

---

## Task 2: `features/` package

**Files:**
- Create: `fantasy_football/features/__init__.py`
- Move: `fantasy_football/data_transformation.py` → `fantasy_football/features/transformation.py`
- Move: `fantasy_football/fixtures.py` → `fantasy_football/features/fixtures.py`
- Move: `fantasy_football/elo.py` → `fantasy_football/features/elo.py`
- Move tests: `tests/test_data_transformation.py` → `tests/features/test_transformation.py`, `tests/test_fixtures.py` → `tests/features/test_fixtures.py`, `tests/test_elo.py` → `tests/features/test_elo.py`

- [ ] **Step 1: Create the package directories**

```bash
mkdir -p fantasy_football/features tests/features
```

- [ ] **Step 2: Create the package `__init__.py` with a docstring**

Create `fantasy_football/features/__init__.py`:

```python
"""Derive model-ready features: rolling points, enriched fixtures, team Elo."""
```

- [ ] **Step 3: Move source and test files with `git mv`**

```bash
git mv fantasy_football/data_transformation.py fantasy_football/features/transformation.py
git mv fantasy_football/fixtures.py            fantasy_football/features/fixtures.py
git mv fantasy_football/elo.py                 fantasy_football/features/elo.py
git mv tests/test_data_transformation.py       tests/features/test_transformation.py
git mv tests/test_fixtures.py                  tests/features/test_fixtures.py
git mv tests/test_elo.py                       tests/features/test_elo.py
```

- [ ] **Step 4: Rewrite this domain's import paths repo-wide**

```bash
grep -rlE "fantasy_football\.(data_transformation|fixtures|elo) import" fantasy_football tests main.py | xargs sed -i '' \
  -e 's/fantasy_football\.data_transformation import/fantasy_football.features.transformation import/g' \
  -e 's/fantasy_football\.fixtures import/fantasy_football.features.fixtures import/g' \
  -e 's/fantasy_football\.elo import/fantasy_football.features.elo import/g'
```

- [ ] **Step 5: Assert no stale references remain**

Run: `grep -rnE "fantasy_football\.(data_transformation|fixtures|elo) import" fantasy_football tests main.py`
Expected: no output (exit 1).

- [ ] **Step 6: Fix import ordering and formatting**

Run: `uv run ruff check --fix . && uv run ruff format .`
Expected: fixes applied or all checks passed.

- [ ] **Step 7: Run the full check suite**

Run: `uv run pytest -q && uv run ruff check . && uv run ty check`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "refactor: move feature modules into fantasy_football/features"
```

---

## Task 3: `modelling/` package

**Files:**
- Create: `fantasy_football/modelling/__init__.py`
- Move: `fantasy_football/models.py` → `fantasy_football/modelling/models.py`
- Move: `fantasy_football/prediction.py` → `fantasy_football/modelling/prediction.py`
- Move: `fantasy_football/evaluation.py` → `fantasy_football/modelling/evaluation.py`
- Move: `fantasy_football/metrics.py` → `fantasy_football/modelling/metrics.py`
- Move tests: `tests/test_models.py` → `tests/modelling/test_models.py`, `tests/test_prediction.py` → `tests/modelling/test_prediction.py`, `tests/test_evaluation.py` → `tests/modelling/test_evaluation.py`, `tests/test_metrics.py` → `tests/modelling/test_metrics.py`

- [ ] **Step 1: Create the package directories**

```bash
mkdir -p fantasy_football/modelling tests/modelling
```

- [ ] **Step 2: Create the package `__init__.py` with a docstring**

Create `fantasy_football/modelling/__init__.py`:

```python
"""Per-position points models: train, predict, and evaluate."""
```

- [ ] **Step 3: Move source and test files with `git mv`**

```bash
git mv fantasy_football/models.py     fantasy_football/modelling/models.py
git mv fantasy_football/prediction.py fantasy_football/modelling/prediction.py
git mv fantasy_football/evaluation.py fantasy_football/modelling/evaluation.py
git mv fantasy_football/metrics.py    fantasy_football/modelling/metrics.py
git mv tests/test_models.py           tests/modelling/test_models.py
git mv tests/test_prediction.py       tests/modelling/test_prediction.py
git mv tests/test_evaluation.py       tests/modelling/test_evaluation.py
git mv tests/test_metrics.py          tests/modelling/test_metrics.py
```

- [ ] **Step 4: Rewrite this domain's import paths repo-wide**

```bash
grep -rlE "fantasy_football\.(models|prediction|evaluation|metrics) import" fantasy_football tests main.py | xargs sed -i '' \
  -e 's/fantasy_football\.models import/fantasy_football.modelling.models import/g' \
  -e 's/fantasy_football\.prediction import/fantasy_football.modelling.prediction import/g' \
  -e 's/fantasy_football\.evaluation import/fantasy_football.modelling.evaluation import/g' \
  -e 's/fantasy_football\.metrics import/fantasy_football.modelling.metrics import/g'
```

- [ ] **Step 5: Assert no stale references remain**

Run: `grep -rnE "fantasy_football\.(models|prediction|evaluation|metrics) import" fantasy_football tests main.py`
Expected: no output (exit 1).

- [ ] **Step 6: Fix import ordering and formatting**

Run: `uv run ruff check --fix . && uv run ruff format .`
Expected: fixes applied or all checks passed.

- [ ] **Step 7: Run the full check suite**

Run: `uv run pytest -q && uv run ruff check . && uv run ty check`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "refactor: move modelling modules into fantasy_football/modelling"
```

---

## Task 4: `optimisation/` package

**Files:**
- Create: `fantasy_football/optimisation/__init__.py`
- Move: `fantasy_football/optimisation.py` → `fantasy_football/optimisation/optimiser.py`
- Move: `fantasy_football/plan_report.py` → `fantasy_football/optimisation/plan_report.py`
- Move: `fantasy_football/team_input.py` → `fantasy_football/optimisation/team_input.py`
- Move tests: `tests/test_optimisation.py` → `tests/optimisation/test_optimiser.py`, `tests/test_plan_report.py` → `tests/optimisation/test_plan_report.py`, `tests/test_team_input.py` → `tests/optimisation/test_team_input.py`

> **Note:** the source file `optimisation.py` and the new directory `optimisation/` are distinct names on disk (one has the `.py` suffix), so they coexist without conflict — no special handling needed.

- [ ] **Step 1: Create the package directories**

```bash
mkdir -p fantasy_football/optimisation tests/optimisation
```

- [ ] **Step 2: Create the package `__init__.py` with a docstring**

Create `fantasy_football/optimisation/__init__.py`:

```python
"""Build the squad, starting XI, captain, and transfer plan from predictions."""
```

- [ ] **Step 3: Move the source and test files with `git mv`**

```bash
git mv fantasy_football/optimisation.py fantasy_football/optimisation/optimiser.py
git mv fantasy_football/plan_report.py fantasy_football/optimisation/plan_report.py
git mv fantasy_football/team_input.py  fantasy_football/optimisation/team_input.py
git mv tests/test_optimisation.py      tests/optimisation/test_optimiser.py
git mv tests/test_plan_report.py       tests/optimisation/test_plan_report.py
git mv tests/test_team_input.py        tests/optimisation/test_team_input.py
```

- [ ] **Step 4: Rewrite this domain's import paths repo-wide**

```bash
grep -rlE "fantasy_football\.(optimisation|plan_report|team_input) import" fantasy_football tests main.py | xargs sed -i '' \
  -e 's/fantasy_football\.optimisation import/fantasy_football.optimisation.optimiser import/g' \
  -e 's/fantasy_football\.plan_report import/fantasy_football.optimisation.plan_report import/g' \
  -e 's/fantasy_football\.team_input import/fantasy_football.optimisation.team_input import/g'
```

- [ ] **Step 5: Assert no stale references remain**

Run: `grep -rnE "fantasy_football\.(optimisation|plan_report|team_input) import" fantasy_football tests main.py | grep -v "optimisation\.\(optimiser\|plan_report\|team_input\)"`
Expected: no output. (The `grep -v` filters out the correct new paths; anything left is a stale reference.)

- [ ] **Step 6: Fix import ordering and formatting**

Run: `uv run ruff check --fix . && uv run ruff format .`
Expected: fixes applied or all checks passed.

- [ ] **Step 7: Run the full check suite**

Run: `uv run pytest -q && uv run ruff check . && uv run ty check`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "refactor: move optimisation modules into fantasy_football/optimisation"
```

---

## Task 5: Update the README structure

**Files:**
- Modify: `README.md` (the "Current State" section, lines ~9-27, and the evaluation command, line ~105)

- [ ] **Step 1: Replace the flat "Current State" bullet list with a domain tree**

In `README.md`, replace the entire "## Current State" section (the intro line plus the flat `* fantasy_football/<file>.py — ...` bullet list) with a package-grouped view. Keep the short per-module descriptions, nested under their package:

````markdown
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

`tests/` mirrors this structure.
````

- [ ] **Step 2: Update the evaluation command path**

In `README.md`, find:

```bash
uv run python -m fantasy_football.evaluation
```

Replace with:

```bash
uv run python -m fantasy_football.modelling.evaluation
```

- [ ] **Step 3: Verify no other stale module paths remain in the README**

Run: `grep -nE "fantasy_football/[a-z_]+\.py|fantasy_football\.(data_extraction|data_transformation|fci_extraction|fpl|elo|fixtures|models|prediction|evaluation|metrics|optimisation|plan_report|team_input|seasons)\b" README.md`
Expected: only matches that are correct under the new layout (e.g. inside the tree block). Any old flat path like `fantasy_football/optimisation.py` outside the tree is stale — fix it.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: show domain package structure in README"
```

---

## Task 6: Final whole-repo verification

**Files:** none (verification only)

- [ ] **Step 1: Confirm the package tree matches the design**

Run: `find fantasy_football tests -name '*.py' | sort`
Expected: every module lives under `extraction/`, `features/`, `modelling/`, or `optimisation/`, except `constants.py`, `logging_config.py`, `fpl_types.py`, and `__init__.py` at the `fantasy_football/` root. `tests/` mirrors it.

- [ ] **Step 2: Confirm no old flat import paths survive anywhere**

Run: `grep -rnE "fantasy_football\.(data_extraction|data_transformation|fci_extraction|fpl|elo|fixtures|models|prediction|metrics|seasons|plan_report|team_input) import" fantasy_football tests main.py`
Expected: no output. (Note: `fantasy_football.fpl_types`, `.constants`, `.logging_config` are correct and excluded from this check.)

- [ ] **Step 3: Run the full check suite one final time**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check . && uv run ty check`
Expected: all pass.

- [ ] **Step 4: Smoke-test that `main.py` imports resolve**

Run: `uv run python -c "import main"`
Expected: no ImportError (it imports without running `main()`, which is guarded by `if __name__ == "__main__"`).
