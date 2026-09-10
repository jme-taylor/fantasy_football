# Fantasy Football

A fully automated way of selecting your fantasy football team. At a high level it works in the following manner:

1. Collect all data from sources.
2. Train machine learning models for all points scoring components on all historic data.
3. Use trained machine learning models to make future predictions for upcoming gameweeks.
4. Run an optimisation algorithm over the future predictions, to make team selections and transfer decisions for the season.

There is further detail on all sections of this below.

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

To read your live squad rather than hand-maintaining a team file, set
`FPL_MANAGER_ID` (your FPL entry id — the number in the URL of your Points
page) in the same `.env`. No credentials are needed: the squad, bank and
transfer history all come from FPL's public `entry` endpoints.

A full example environment that you can copy is seen in `example.env`.

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
* `team_file` — path to a team JSON naming the squad carried into the
  upcoming gameweek. Players are declared by `name` or by `element`, each with
  a `purchase_price`. Passing this overrides the live squad, which is what
  makes it useful for asking "what if I owned this instead". With it unset the
  squad is read from the public entry endpoints using `FPL_MANAGER_ID`. One or
  the other is required to optimise once the season is under way; with neither,
  mid-season runs log and skip optimisation, leaving the ingest, training and
  prediction work of that run intact. At GW1 the optimiser free-builds and
  needs no squad at all.

  Every squad read from the API is recorded to `data/teams/{season}_gw{n}.json`
  in that same team-file format, so a run is always reproducible by handing the
  snapshot back as `team_file`. Snapshots are a record, never an automatic
  input: a stale one would silently plan transfers from a team you no longer
  own.

### Evaluating the models with MLflow

Each position (GK, DEF, MID, FWD) has its own points model, and each is
evaluated by its own expanding-window cross-validation during training,
logged to MLflow — one experiment per position (`gk-points-model`,
`def-points-model`, `mid-points-model`, `fwd-points-model`). The minutes
model logs to its own experiment (`minutes_played_classification`).

There is currently no harness that scores the four positions against each
other on one scale; the rolling-origin replay that used to do so measured
the deleted rolling-points formula rather than the models, and was removed
with it.

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

## Storage

The source of truth is a single-file DuckDB database at
`data/fantasy_football.duckdb` (gitignored). The storage layer is a
three-layer design: `storage/engine.py` holds the DuckDB primitives (the
only module that touches Arrow, `register`/`unregister`, and `.pl()`),
`storage/table.py` builds on it with the `Table` descriptor (column order
and dtypes from one `schema` dict, generated DDL, and the five operations
`coerce`/`load`/`seasons_present`/`write_immutable`/`upsert_current`), and
`storage/tables.py` declares the ten tables as `Table` specs gathered into
`TABLES`. `storage/database.py` reduces to `get_connection` and
`reset_database`, both driven by `TABLES`. Extractors hand the layer Polars
frames; downstream code reads frames back. The tables are:

* **`player_season`** — the `(season, element) -> player_code` identity
  dimension. `player_code` is FPL's stable global player identifier, unlike
  `element` (recycled per season) or player name (renamed, accented
  differently across sources, or shared by two players). It also carries
  slow-moving static attributes (`birth_date`, `team_join_date`, position)
  that are back- and forward-filled across a player's seasons from wherever
  they were first observed. This is what lets features span the Vaastav ->
  FPL Core Insights source seam and the close-season gap — see
  [Minutes-played model](#minutes-played-model).
* **`player_week`** — one row per `(season, gw, element)`, collapsed across
  double gameweeks. The substrate for the rolling-points features and the
  per-position points models.
* **`team_fixture`** — one row per `(season, gw, team, opposition)` with
  home/away and kickoff time.
* **`player_match`** — one row per `(season, gw, element, opponent)` (minutes
  and points only), **not** collapsed across double gameweeks. The training
  substrate for the minutes-played model.
* **`player_match_fpl`** — Vaastav's full per-fixture stat set, one row per
  `(season, gw, element, fixture)`, keeping every column the source
  publishes rather than the eight `player_match` narrows to.
* **`player_match_opta`** — FCI's Opta-grade per-fixture stat set from
  2024-25 onwards, one row per `(season, gw, element, match_id)`, covering
  every competition a player appeared in, not just the Premier League.
* **`player_availability`** — one row per `(season, gw, element)` capturing
  FPL's point-in-time `chance_of_playing_this_round`.
* **`minutes_prediction`** — one row per `(season, gw, element, opponent)`
  holding the production minutes model's 3-class probabilities (`p_zero`,
  `p_partial`, `p_sixty_plus`), the derived `expected_minutes`, and the
  `model_version` that produced them. Populated by the version-gated backfill
  (see [Minutes-played model](#minutes-played-model)).
* **`points_prediction`** — per-position predicted points at match grain,
  so a double gameweek is two rows summing to a gameweek total.
  `prediction_kind` is `backfill` (in-sample, for eyeballing a candidate
  model against actuals) or `forward` (the out-of-sample forecasts the
  optimiser consumes).

Completed (immutable) seasons are inserted once and skipped thereafter; the
current season is upserted (delete-then-insert) every run so late corrections,
new gameweeks, and injury news refresh cleanly. `main(rebuild=True)` drops and
reloads everything as a recovery escape hatch.

The optimiser reads `points_prediction` directly: forward-kind rows summed
from match to gameweek grain, joined onto the current roster for name, club,
position and price. Its own output (`optimisation_plan.jsonl` and
`optimisation_plan.md`) is written under `data/transformed/`, alongside the
`team_elo.csv` cache.

## Data sources

### Transfermarkt

[Transfermarkt](https://www.transfermarkt.co.uk/) is used as a source of data for player values (in €), granular positions and club history. To get this data, we use the [ScraperFc](https://scraperfc.readthedocs.io/en/latest/index.html) python package, however there were some bugs in there, so I've fixed these in my [own fork](https://github.com/jme-taylor/ScraperFC) of the package. This fork is what the uv environment installs.

Transfermarkt names do not fully align with FPL players, and there is no unique identifier that joins the two. So we join players to FPL data in the following manner:
* **`override`** — the hand-maintained CSV, which always wins
* **`dob_exact_name`** — date of birth plus an exact normalised name
* **`dob_fuzzy_name`** — date of birth plus a Jaro-Winkler name score ≥ 0.85
* **`dob_surname`** — date of birth plus a surname score ≥ 0.85
* **`club_exact_name`** — no date of birth either side: exact name plus a shared club
* **`exact_name`** — an exact name that is the only one of its kind on both sides
* **`exact_name_dob_conflict`** — as above, but the two dates of birth disagree

Each rung is tried in order, and an id claimed by one is never offered to a
later one, so the map stays one-to-one. Names are normalised first —
lowercased, accents stripped, punctuation collapsed. Ambiguity is never
guessed at: two people sharing a birthday both go to the human, whatever
their names score. Managers and players who never appeared are dropped.

Identity is resolved once, at person level, into `tm_player_map`; market
values and transfers join through it by date. The two hand-maintained files
in `fantasy_football/extraction/mappings/` are the only durable inputs, and
whatever stays unmatched lands in `data/tm_player_map_candidates.csv` for a
manual decision. Rebuild with `scripts/scrape_transfermarkt.py --map-only`.

## Other Data

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

### Match-level player stats

Two tables hold the comprehensive per-fixture record, alongside the thin
`player_match` table the prediction pipeline uses:

| Table | Source | Seasons | Key |
|---|---|---|---|
| `player_match_fpl` | Vaastav `gws/merged_gw.csv` | 2016-17 → 2025-26 | `(season, gw, element, fixture)` |
| `player_match_opta` | FCI `playermatchstats.csv` | 2024-25 → 2026-27 | `(season, gw, element, match_id)` |

Columns are ragged across seasons: Vaastav published a detailed stat set
from 2016-17 to 2018-19, dropped it for three seasons, added the
`expected_*` family in 2022-23 and `defensive_contribution` in 2025-26.
FCI added nine columns in 2025-26 and `corners` in 2026-27. A null
therefore may mean "not published that season" rather than zero —
`fantasy_football/storage/coverage.py` records which, and
`seasons_covering` returns the seasons in which a given set of columns is
all available.

`player_match_opta` keeps **every competition**, not just the Premier
League. Any consumer computing FPL stats must filter to
`competition = 'prem'`, or European and cup appearances will inflate the
totals.

`storage/validation.py` cross-checks the two sources over 2024-25 and
2025-26, where both publish.

**As ingested:** `player_match_fpl` holds all 10 seasons, 253,890 rows in
total, between 21,790 and 29,747 per season. `player_match_opta` holds
11,567 rows for 2024-25 (`prem` only, the sole competition FCI records that
season) and, for 2025-26: 12,754 `prem`, 1,047 `champions`, 835 `efl`, 463
`europa`, 241 `conference`. 2026-27 has 38 published gameweek files, all
empty — the season hasn't started yet, so nothing is stored there; this is
expected, not a fault.

Cross-checking the 23,986 player-gameweeks both sources cover in common
(2024-25 and 2025-26, Premier League only) shows tight agreement on the
counting stats and one well-understood gap in minutes:

| stat | disagreement rate |
|---|---|
| `minutes` | 12.45% |
| `assists` | 0.75% |
| `goals_scored` | 0.07% |
| `saves` | 0.04% |
| `penalties_missed` | 0.01% |

The `minutes` figure looks alarming on its own but is a benign convention
difference, not a bug: 2,859 of the 2,986 disagreements (95.7%) are off by
exactly one minute, almost always FCI reporting one more than Vaastav — a
substitution-rounding convention between the two sources. Only 127 of the
23,986 rows differ by more than a minute. The near-perfect agreement on the
other four stats is what confirms the join and aggregation logic are
correct. If a future run shows the disagreement rate climbing, or the
off-by-one pattern giving way to larger gaps, that would be worth
investigating — the one-minute rounding is the expected signature here.

The `player_availability` table's `chance_of_playing_this_round` is FPL's
0/25/50/75 percentage signal; `null` (no injury doubt) is stored as `100`. The
source is the [Randdalf/fplcache](https://github.com/Randdalf/fplcache)
bootstrap snapshot at each gameweek's deadline — the same source used to derive
per-gameweek `team_code`, because FCI only records a player's final club and so
gets mid-season transfers wrong for pre-transfer gameweeks. Coverage starts
from `2022-23` (fplcache's earliest snapshots, set by `FPLCACHE_FIRST_SEASON`).

> **Known gap: no 2018-19 / 2019-20 in `cleaned_merged_seasons.csv`.** That
> file only covers seasons from 2020-21 onwards. Earlier `merged_gw.csv` files
> lack the `position` and `team` columns those seasons would need, and 2018-19
> is additionally latin-1 encoded with no accompanying `teams.csv`. The two
> seasons were deliberately left out rather than backfilled, so
> `cleaned_merged_seasons.csv`-derived tables (e.g. `player_week`) have a gap
> there. `player_season`, which is built from a different source
> (`players_raw.csv`, present for every season), still has rows for both.

## Minutes-played model

`modelling/minutes.py` trains a single 3-class gradient-boosted classifier
(XGBoost) that predicts each
player's minutes bucket for an upcoming match: benched (`0_minutes`), partial
(`1_to_59_minutes`), or a full-ish shift (`60_minutes_plus`). It is scored on
the two decision boundaries the downstream points models care about —
probability of *any* appearance and probability of a *60+ minute* appearance —
cross-validated with an expanding window over seasons, then refit on all
seasons and logged to MLflow (experiment `minutes_played_classification`).

Numeric features pass through the pipeline unimputed and unscaled: XGBoost
learns a missing-direction per split, which says more than a median stand-in,
and trees are indifferent to scale. Categoricals are one-hot encoded with
nulls mapped to an explicit `unknown` category, and the label encoder lives
inside the estimator so `classes_` stays string-valued end to end.

Its features are assembled from the `player_match`, `player_week`,
`player_availability`, `player_season` and `tm_*` tables. Contemporaneous
features (knowable at the deadline): number of same-position teammates, FPL
chance-of-playing, positional availability (fit same-club, same-position
rivals ahead and at the same position), rolling 5-match minutes, and games
played so far this season. Cross-season features, joined through
`player_season.player_code` (see [Storage](#storage)): previous-season minutes
and start rate, seasons played in the Premier League, seasons since the player
was last in the Premier League, age, whether the player is a Premier League
newcomer, and whether their club was promoted.

Transfermarkt features (`features/transfermarkt.py`) run on two clocks.
Features derived from matches the player has played — minutes in each of the
last two games, days between them, season-to-date minutes and share of the
club's position-group minutes — **freeze** at the last played match, so a
forward gameweek inherits the state entering it and does not drift. Features
derived from the calendar — market value as of the fixture, and the count,
value and recency of rivals signed into the same position group within the
last 365 days — read the fixture's **own kickoff**, because a valuation or a
signing is a known fact at prediction time. Freezing those would hide a summer
signing from every pre-season prediction, which is the failure that motivated
the model: a keeper predicted as his club's number one after a replacement had
already been bought.

The `tm_*` tables are populated by hand (`scripts/scrape_transfermarkt.py`),
so `main()` calls `check_transfermarkt_freshness()` before predicting. It
fails loudly on valuations older than 45 days, and on fewer than 90% of the
season's actual appearers mapping to Transfermarkt — two different failures
that otherwise share one symptom, a model quietly reading nulls. Players with
no Transfermarkt match are never filtered out: their TM features are null and
the model reads the absence as signal. These
reach across the summer break, which is what lets the model say anything
useful about a player at GW1 of a new season, before any in-season evidence
exists — previously it had nothing but contemporaneous, in-season signal. The
model runs on every data refresh from `main`, and both `run_minutes_model()`
and `backfill_minutes()` are called bare — a modelling failure is **fatal**
and aborts `main()` before prediction and optimisation run, it is not logged
and swallowed. This is deliberate fail-fast behaviour, but it has a sharp
first-run consequence: `backfill_minutes()` calls `get_production_model()`,
which raises if the `production` alias has never been set, so a fresh clone
with no manually-promoted alias will abort `main()` on its very first run,
before the points models run. The first run after cloning must include a
manual promotion of a trained version to `production` in the MLflow UI (see
[Registry, promotion and backfill](#registry-promotion-and-backfill)) before
`main()` can complete.

**Measured impact (2026-07-30).** Training on the real database end to end,
7-fold expanding-window CV, comparing the run with the new cross-season
features against the most recent prior run (contemporaneous features only):

| metric | pre-branch baseline | with cross-season features | after dropping `days_since_team_join` | change vs baseline |
|---|---|---|---|---|
| `logloss_appear_mean` | 0.630 | 0.378 | 0.378 | lower is better — improved |
| `logloss_60_mean` | 0.637 | 0.349 | 0.349 | lower is better — improved |
| `auc_appear_mean` | 0.798 | 0.910 | 0.910 | higher is better — improved |
| `auc_60_mean` | 0.777 | 0.917 | 0.917 | higher is better — improved |
| `brier_appear_mean` | 0.212 | 0.118 | 0.118 | lower is better — improved |
| `e_min_mae_mean` | 30.93 | 19.57 | 19.56 | lower is better — improved |

The new features improved every CV metric over the pre-branch baseline, and
fold-to-fold variance (the `_std` companions to each metric above) also
dropped substantially, suggesting the gain is not a single lucky fold.
`days_since_team_join` was removed from the model's feature list (it is
100% null in 6 of 8 training seasons and 564/564 null in 2026-27, and its
non-nullness correlated with season membership well enough to risk acting as
a season proxy) — see the comment on `NUM_FEATURES` in `modelling/minutes.py`.
Re-measuring after the removal changed nothing beyond noise: every metric
above is within ±0.001 of the run that still had the column, confirming the
median-imputed, near-constant column was carrying no real signal for the
model to lose. `fit_rivals_ahead` is still all-null in at least one CV fold,
which the imputer skips with a warning rather than a failure; this remains a
candidate for follow-up but did not prevent training. This run was **not**
promoted — promotion to `production` remains a manual step in the MLflow UI.

### Registry, promotion and backfill

Every training run **registers** a new version of the model in the MLflow Model
Registry under `minutes_played_classifier`. It does **not** move any alias.
Promotion is manual: assign the `production` alias to a chosen version in the
MLflow UI (open the *Models* tab, pick a version, add the alias `production`).

On each `main` run, `run_minutes_backfill()` loads
`models:/minutes_played_classifier@production` and writes per-match predictions
to the `minutes_prediction` table. The backfill is **version-gated**: the
current season is re-scored every run (new gameweeks arrive continuously), while
historic seasons are re-scored **only** when the production version changes (or
rows/seasons are missing). Each stored row records the `model_version` that
produced it, which is what the gate compares against. Before the first manual
promotion — i.e. with no `production` alias set — the backfill is a logged
no-op.

> **Known leakage.** The single production ("champion") model scores its own
> training seasons, so the historic predictions are **in-sample**. This is fine
> for evaluation and the optimiser, but a points model that trains on these
> predictions as a feature learns from a mildly over-optimistic signal it won't
> have at real prediction time. Accepted for now; the leak-free fix is to store
> out-of-fold expanding-window predictions for historic seasons instead.

> **Planned: gated auto-promotion.** Manual promotion will eventually be
> replaced by automatically promoting a freshly trained version to `production`
> only when it beats the current champion on a CV metric (e.g.
> `logloss_appear`), removing the manual UI step.

### Defender points model

`modelling/defender.py` trains a random forest on match-grain defender
rows and writes its predictions to `points_prediction`. Every run of
`main.py` retrains and registers a new version under
`defender_points_regressor`; **which version is live is a manual alias
move in the MLflow UI.**

Because there is no formula fallback for defenders, `main.py` cannot
complete on a fresh database until a version has been promoted. The
first run trains and registers, then fails at prediction time with a
message naming the model and alias. Promote a version and every
subsequent run works. Failing here is deliberate: the optimiser needs
five defenders, so continuing would hand it an infeasible squad problem
far from the cause.

The same applies to every other position: each is served from its own
registered model's forward predictions, and the optimiser's input loader
fails when a whole position has none for a gameweek in the horizon.

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
  * [ ] Replace the removed rolling-origin replay with a harness that scores all four position models on one scale
* [~] Build a minutes-played model to feed appearance probabilities into the points models
  * [X] Per-fixture `player_match` table and point-in-time `player_availability` table
  * [X] Availability / valuation features
  * [X] 3-class classifier with expanding-window CV, scored on appearance and 60+ boundaries, logged to MLflow
  * [X] Register each trained model in the MLflow Model Registry (manual `production` promotion via the UI)
  * [X] Version-gated backfill of production predictions to the `minutes_prediction` table
  * [X] Cross-season player identity (`player_season`, keyed on FPL's stable `player_code`) feeding prior-season history and cold-start features into the model
  * [ ] Gated auto-promotion (promote a new version only when it beats the champion)
  * [ ] Feed its appearance probabilities into the per-position points models
* [ ] Dig into the worst-performing position and investigate its scoring errors
* [ ] Identify and incorporate additional features to improve the model

The rolling-points formula that used to serve as the baseline (scored per
position over all historical gameweeks) only narrowly beat predicting each
player's recent average — its value was in *ranking* players (Spearman ≈
0.6–0.8 by position) rather than predicting exact point totals. It has been
removed now that every position is served by its own model; beating that
recorded bar is the focus of the next milestones.

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
gameweeks, chips (wildcard, triple captain, etc.), managerial changes,
disciplinary suspensions, mid-season transfers, and backtesting how good the
optimiser actually is. Promoted teams and new players are handled by the
minutes model's `is_promoted_club` / `is_pl_newcomer` cold-start features
(measured to improve every CV metric, see
[Minutes-played model](#minutes-played-model)); it is no longer an open
question there.

## Resources

Historic data is sourced from the [Fantasy Premier League](https://github.com/vaastav/Fantasy-Premier-League)
repository by Vaastav Anand (through 2024-25). Current-season data (2025-26+)
is sourced from [FPL Core Insights](https://github.com/olbauday/FPL-Core-Insights),
and point-in-time team and availability snapshots from
[Randdalf/fplcache](https://github.com/Randdalf/fplcache). All three are
accessed via the GitHub API using `GITHUB_API_KEY`. Team Elo ratings are
scraped from ClubElo via [ScraperFC](https://github.com/oseymour/ScraperFC).
