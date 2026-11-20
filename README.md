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

## Points models

The points modelling is powered by various machine learning models, that map to the [FPL scoring system](https://www.premierleague.com/en/news/2174909/fpl-basics-scoring). These are then combined in `modelling/points.py` to get an overall 
expected points for the player (per match) that can then be rolled up to a gameweek level, and used by the optimiser for transfer and selection decisions. Below, we describe in more detail each point component's model.

### Minutes players model

`modelling/minutes.py` trains a single 3-class gradient-boosted classifier
(XGBoost) that predicts each
player's minutes bucket for an upcoming match: benched (`0_minutes`), partial
(`1_to_59_minutes`), or a full-ish shift (`60_minutes_plus`). These are then converted into
probabilities of appearing and probabilites of playing 60 minutes plus, which are the points scoring elements of
minutes for fantasy football that we care about. It also allows us to get an expected minutes component, that we can then combine with the outputs of other models (e.g goals scored, yellow cards rate) to get a player's expected points for a single match.

Scoring is measured in the `minutes_played_classification` experiment in MLflow. The following metrics are recorded:

| Metric | Description |
|--------|-------------|
|`logloss_appear`|Log loss of the probability of appearance|
|`logloss_60`|Log loss of the probability of playing 60 minutes plus|
|`brier_appear`||
|`brier_60`||
|`e_min_mae`||
|`auc_appear`||
|`auc_60`||

### Assists model

`modelling/assists.py` trains a regression model that outputs the per 90 expected assists for a given player. This only does per 90, rather than classification, as it is used in combination with the minutes model to give an expected points from assists for the single match that player is in.

Scoring is measured in the `assists-rate-model` experiment in MLflow. The following metrics are recorded:

| Metric | Description |
|--------|-------------|
|`poisson_deviance`||
|`brier`||
|`logloss`||
|`base_rate_brier`||
|`skill_score`||
|`assisting_rate`||
|`rate_mae`||
|`top_decile_ratio`||

### Conceding model

`modelling/conceding.py` trains a Poisson gradient-boosted regressor that outputs the expected goals conceded by a team in a given fixture. Unlike the other models, this is trained at the team level rather than the player level, and predicts per match rather than per 90, as goals conceded is a team quantity shared by every player on the pitch. The same prediction is then fanned out to every goalkeeper, defender and midfielder of that team, and converted into a probability of a clean sheet and an expected deduction for goals conceded (scaled by the player's expected minutes), giving the expected points from conceding for a single match.

Scoring is measured in the `team-conceding-model` experiment in MLflow. The following metrics are recorded:

| Metric | Description |
|--------|-------------|
|`poisson_deviance`||
|`mae`||
|`calibration`||
|`clean_sheet_brier`||
|`base_rate_brier`||
|`skill_score`||
|`clean_sheet_rate`||
|`deduction_mae`||

### Defensive contributions (DEFCON)

`modelling/defcon.py` has two defensive contribution regression models that both predict the rates of the stats that make up defensive contributions within FPL. The reasoning for having two of these is that defenders are measured on defensive contributions that include clearances, blocks, interceptions and tackles, whereas forwards and midfielders are measured on defensive contributions that include all of the defenders metric plus ball recoveries.

The defender defcon models is scored and measured in `def-defcon-rate-model` within MLflow. The midfielder and forwards defcon model is scored and measured in the `mid-fwd-cbirt-rate-model` within MLflow. They both use the following metrics:

| Metric | Description |
|--------|-------------|
|`brier`||
|`logloss`||
|`base_rate_brier`||
|`skill_score`||
|`hit_rate`||
|`rate_mae`||

### Goals

`modelling/goals.py` trains a regression model that predicts the per 90 rate of goals for a player. This is then combined with the expected minutes to create a player's points from goals. 

The mode is scored and measured in MLflow under the `goals-rate-model` experiment using the following metrics:

| Metric | Description |
|--------|-------------|
|`poisson_deviance`||
|`brier`||
|`logloss`||
|`base_rate_brier`||
|`skill_score`||
|`scoring_rate`||
|`rate_mae`||
|`top_decile_ratio`||

### Yellow Cards

`modelling/yellow_cards.py` trains a regression model that predicts the amount of yellow cards per 90 a player will get. This is combined with the expected minutes to create a player's points from yellow cards.

The mode is scored and measured in MLflow under the `yellow-cards-rate-model` experiment using the following metrics:

| Metric | Description |
|--------|-------------|
|`poisson_deviance`||
|`brier`||
|`base_rate_brier`||
|`skill_score`||
|`player_rate_skill`||
|`booking_rate`||
|`rate_mae`||
|`top_decile_ratio`||


### Saves

`modelling/saves.py` trains a regression model that predicts the amount of saves a goalkeeper will make per 90. This is combined wih their expected minutes to create a player's points from saves.

The mode is scored and measured in MLflow under the `goalkeeper-saves-model` experiment using the following metrics:

| Metric | Description |
|--------|-------------|
|`poisson_deviance`||
|`mae`||
|`calibration`||
|`points_mae`||
|`base_rate_points_mae`||
|`skill_score`||


## Resources

Historic data is sourced from the [Fantasy Premier League](https://github.com/vaastav/Fantasy-Premier-League)
repository by Vaastav Anand (through 2024-25). Current-season data (2025-26+)
is sourced from [FPL Core Insights](https://github.com/olbauday/FPL-Core-Insights),
and point-in-time team and availability snapshots from
[Randdalf/fplcache](https://github.com/Randdalf/fplcache). All three are
accessed via the GitHub API using `GITHUB_API_KEY`. Team Elo ratings are
scraped from ClubElo via [ScraperFC](https://github.com/oseymour/ScraperFC).
