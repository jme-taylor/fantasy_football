# Reorganise `fantasy_football/` into domain packages

**Date:** 2026-06-15
**Status:** Approved (design)

## Problem

`fantasy_football/` has grown to 17 flat modules. A flat layout no longer
communicates the project's architecture, and the crowded directory makes it
hard to see which modules belong together. The goal is to group modules into
domain sub-packages so the structure tells the story of the project, and to
mirror that structure in `tests/`.

## Target layout

```
fantasy_football/
├── __init__.py
├── constants.py
├── logging_config.py
├── fpl_types.py
├── extraction/
│   ├── __init__.py
│   ├── fpl.py            (was fpl.py)
│   ├── fci.py            (was fci_extraction.py)
│   ├── extractor.py      (was data_extraction.py)
│   └── seasons.py
├── features/
│   ├── __init__.py
│   ├── fixtures.py
│   ├── elo.py
│   └── transformation.py (was data_transformation.py)
├── modelling/
│   ├── __init__.py
│   ├── models.py
│   ├── prediction.py
│   ├── evaluation.py
│   └── metrics.py
└── optimisation/
    ├── __init__.py
    ├── optimiser.py      (was optimisation.py)
    ├── plan_report.py
    └── team_input.py
```

### Domain rationale

The boundaries follow the existing internal dependency graph:

| Domain | Modules | Purpose |
|---|---|---|
| **root (shared)** | `constants`, `logging_config`, `fpl_types` | Cross-cutting; depended on by everything, owned by nothing. |
| **extraction** | `fpl`, `fci`, `extractor`, `seasons` | Ingest raw data from FPL API / FCI / Vaastav sources. |
| **features** | `fixtures`, `elo`, `transformation` | Derive model-ready features (rolling points, fixture enrichment, team Elo). |
| **modelling** | `models`, `prediction`, `evaluation`, `metrics` | Train, predict, evaluate. |
| **optimisation** | `optimiser`, `plan_report`, `team_input` | Build and report the transfer/selection plan. |

`fpl_types.py` stays at root: it is used by `fpl` (extraction), `optimiser` and
`plan_report` (optimisation), so it crosses domains and belongs to neither.

`transformation.py` lives in `features/` (not `extraction/`): it produces a
derived feature (rolling points), nothing in `extraction/` imports it, and all
its consumers are in `modelling/`.

## Renames

Files are renamed to drop now-redundant prefixes while avoiding stutter against
the folder name (`extraction/extraction.py` would read as `extraction.extraction`
at the import site):

| From | To |
|---|---|
| `data_extraction.py` | `extraction/extractor.py` |
| `data_transformation.py` | `features/transformation.py` |
| `fci_extraction.py` | `extraction/fci.py` |
| `optimisation.py` | `optimisation/optimiser.py` |

All other modules move into their folder unchanged (their names do not stutter).

## Import strategy

Use full sub-package paths, continuing the existing module-path import style:

```python
from fantasy_football.features.transformation import create_rolling_points_data
from fantasy_football.optimisation.optimiser import optimise_plan
```

`__init__.py` files are not used for re-exports — explicit, greppable paths
only. Because ruff's `D` rules are enabled and `D104` (missing docstring in
public package) is **not** ignored, each `__init__.py` gets a one-line package
docstring, e.g. `"""Data extraction from FPL, FCI, and Vaastav sources."""`.

Every module that imports across a (new) domain boundary has its imports
updated, plus `main.py`.

## Tests

`tests/` mirrors the package layout, and test files follow the same renames:

```
tests/
├── extraction/
│   ├── test_fpl.py
│   ├── test_fci.py            (was test_fci_extraction.py)
│   ├── test_extractor.py      (was test_data_extraction.py)
│   └── test_seasons.py
├── features/
│   ├── test_fixtures.py
│   ├── test_elo.py
│   └── test_transformation.py (was test_data_transformation.py)
├── modelling/
│   ├── test_models.py
│   ├── test_prediction.py
│   ├── test_evaluation.py
│   └── test_metrics.py
└── optimisation/
    ├── test_optimiser.py       (was test_optimisation.py)
    ├── test_plan_report.py
    └── test_team_input.py
```

(`test_fpl_api.py` → `tests/extraction/test_fpl.py`.)

No `__init__.py` in test directories: pytest discovery works on unique file
basenames, which remain unique after the move. If a collection clash appears,
fall back to setting `[tool.pytest.ini_options] importmode = "importlib"` rather
than adding test packages.

## README

The README's **Current State** section is currently a flat per-file bullet
list (`fantasy_football/data_extraction.py — ...`). Rework it to show the new
domain structure as a directory tree, grouped by package, so the README
communicates the architecture at a glance:

```text
fantasy_football/
├── constants.py, logging_config.py, fpl_types.py   # shared
├── extraction/   — ingest raw data (FPL API, FCI, Vaastav)
├── features/     — derive model-ready features (rolling points, fixtures, Elo)
├── modelling/    — train, predict, evaluate
└── optimisation/ — build the squad / transfer plan
```

Keep a short per-module description, but nested under its package rather than
as a flat list. Also update every module path the README references, including
the evaluation command:

- `python -m fantasy_football.evaluation`
  → `python -m fantasy_football.modelling.evaluation`

## Out of scope / unchanged

- **`pyproject.toml`** — hatchling packages the whole `fantasy_football`
  directory (`[tool.hatch.build.targets.wheel] packages = ["fantasy_football"]`),
  so sub-packages are included automatically. No packaging change. There are no
  console entry-points.
- **`main.py`** stays at root as the orchestrator; only its imports change.

## Mechanics & sequence

- Use `git mv` for every move/rename so history follows each file.
- Move **one domain at a time**, in dependency order:
  `extraction → features → modelling → optimisation`. After each domain, run the
  full check suite so any breakage is localised rather than landing all moves in
  one untested heap.
- Verification after each step: `pytest`, `mypy`/`ty`, `ruff check`.

## Success criteria

- All 17 modules live under the layout above; `tests/` mirrors it.
- `pytest`, `ruff check`, and the type checker all pass.
- `main.py` runs end-to-end as before (imports resolve, no behaviour change).
- No module retains a stuttering or redundant-prefix name.
- The README's structure section and all module paths it references reflect the
  new layout.
