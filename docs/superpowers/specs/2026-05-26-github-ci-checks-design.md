# GitHub CI for lint, format, type, and test checks

## Goal

Add a GitHub Actions workflow that runs on every branch push and on PRs to
`main`, gating merges on:

- ruff lint
- ruff format (check-only)
- ty type check
- pytest test suite

## Non-goals

- Auto-formatting commits pushed back to the branch
- Coverage reporting or upload
- Multi-version Python matrix (project pins to 3.12)
- Caching strategies beyond what `astral-sh/setup-uv` provides out of the box

## Approach

Single workflow file, single job, sequential steps. Chosen over parallel
per-check jobs because the repo is small, runs are short, and a single readable
YAML file is worth more than seeing all failures simultaneously.

## Workflow file

Path: `.github/workflows/ci.yml`

### Triggers

- `push:` on all branches
- `pull_request:` targeting `main`

### Concurrency

Cancel in-progress runs on the same branch when a new commit is pushed, to save
minutes:

```yaml
concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true
```

### Job: `checks`

Runs on `ubuntu-latest`. Steps:

1. `actions/checkout@v4`
2. `astral-sh/setup-uv@v3` with caching enabled and Python 3.12 pinned
3. `uv sync --all-groups` — installs runtime + dev deps
4. `uv run ruff check .`
5. `uv run ruff format --check .`
6. `uv run ty check`
7. `uv run pytest`

All steps use default `continue-on-error: false`, so the first failure stops
the job. The user fixes locally and re-pushes.

## Companion change: drop mypy

Remove `mypy>=1.15.0` from `[dependency-groups].dev` in `pyproject.toml`. ty
replaces it; keeping both adds confusion. Run `uv lock` to refresh the
lockfile after editing.

## Success criteria

- Pushing a branch with a lint, format, type, or test failure produces a red
  check on the commit / PR.
- A clean push produces a green check.
- PRs to `main` show the same checks as branch status.
- mypy is no longer listed in `pyproject.toml` or `uv.lock`.

## Out of scope / future

- Adding coverage thresholds or codecov upload
- Running pytest on a matrix of OSes
- A separate `docs` or `build` job
