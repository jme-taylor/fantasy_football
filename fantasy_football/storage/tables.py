"""The stored tables, declared as specs.

One ``Table`` per stored table. Adding a table means adding a spec here
and an entry in ``TABLES``; nothing else in the storage layer changes.
"""

import duckdb
import polars as pl

from fantasy_football.storage import engine
from fantasy_football.storage.table import Table

# How a stored prediction was produced. Backfilled rows are in-sample --
# the champion scored its own training seasons. Forward rows are genuine
# out-of-sample forecasts for fixtures that had not been played. Both
# minutes_prediction and points_prediction use these, which is why they
# live here rather than in either model's module.
BACKFILL_KIND = "backfill"
FORWARD_KIND = "forward"

# Attributes that are properties of the person, not of the season. One
# observation anywhere determines them everywhere, so they are filled
# across every season sharing a player_code on read. Everything else is
# season-varying and must never be propagated -- team_join_date in
# particular is the signal that a player is a recent signing.
PLAYER_SEASON_STATIC_COLUMNS: list[str] = [
    "first_name",
    "second_name",
    "birth_date",
    "region",
]


def gkp_to_gk(frame: pl.DataFrame) -> pl.DataFrame:
    """Collapse the legacy ``GKP`` position label to ``GK``.

    Parameters
    ----------
    frame : pl.DataFrame
        A frame carrying a ``position`` column.

    Returns
    -------
    pl.DataFrame
        The frame with ``GKP`` rewritten to ``GK``.
    """
    return frame.with_columns(
        pl.when(pl.col("position") == "GKP")
        .then(pl.lit("GK"))
        .otherwise(pl.col("position"))
        .alias("position")
    )


def propagate_static_columns(frame: pl.DataFrame) -> pl.DataFrame:
    """Fill person-level attributes across every season for each player.

    Within each ``player_code`` the first non-null value of each static
    column is used. Rows with a null ``player_code`` are left untouched,
    since there is no identity to propagate along. This matters because
    ``birth_date`` is only observable from 2022-23 onwards while the data
    reaches back to 2016-17.

    Parameters
    ----------
    frame : pl.DataFrame
        Player-season rows.

    Returns
    -------
    pl.DataFrame
        The frame with static columns filled.
    """
    return frame.with_columns(
        [
            pl.when(pl.col("player_code").is_null())
            .then(pl.col(column))
            .otherwise(pl.col(column).drop_nulls().first().over("player_code"))
            .alias(column)
            for column in PLAYER_SEASON_STATIC_COLUMNS
        ]
    )


PLAYER_WEEK = Table(
    name="player_week",
    schema={
        "season": pl.Utf8,
        "gw": pl.Int64,
        "element": pl.Int64,
        "name": pl.Utf8,
        "position": pl.Utf8,
        "team": pl.Utf8,
        "bonus": pl.Int64,
        "minutes": pl.Int64,
        "round": pl.Int64,
        "total_points": pl.Int64,
        "value": pl.Int64,
    },
    primary_key=("season", "gw", "element"),
    order_by=("season", "gw", "element"),
    normalise=gkp_to_gk,
)

TEAM_FIXTURE = Table(
    name="team_fixture",
    schema={
        "season": pl.Utf8,
        "gw": pl.Int64,
        "team": pl.Utf8,
        "is_home": pl.Boolean,
        "opposition": pl.Utf8,
        "kickoff_time": pl.Datetime("us"),
    },
    primary_key=("season", "gw", "team", "opposition"),
    order_by=("season", "gw", "team"),
)

PLAYER_MATCH = Table(
    name="player_match",
    schema={
        "season": pl.Utf8,
        "gw": pl.Int64,
        "element": pl.Int64,
        "opponent": pl.Int64,
        "is_home": pl.Boolean,
        "minutes": pl.Int64,
        "total_points": pl.Int64,
        "yellow_cards": pl.Int64,
        "red_cards": pl.Int64,
        "kickoff_time": pl.Datetime("us"),
    },
    primary_key=("season", "gw", "element", "opponent"),
    order_by=("season", "gw", "element", "opponent"),
)

# Vaastav's full per-fixture stat set, 2016-17 to 2025-26. Keyed on
# ``fixture`` rather than ``opponent`` because the FPL fixture id is
# published in every season and uniquely identifies each leg of a double
# gameweek. Columns are ragged across seasons -- see
# ``storage/coverage.py`` for which seasons publish which -- so a null
# here may mean "not published" rather than zero.
PLAYER_MATCH_FPL = Table(
    name="player_match_fpl",
    schema={
        "season": pl.Utf8,
        "gw": pl.Int64,
        "element": pl.Int64,
        "fixture": pl.Int64,
        "assists": pl.Int64,
        "attempted_passes": pl.Int64,
        "big_chances_created": pl.Int64,
        "big_chances_missed": pl.Int64,
        "bonus": pl.Int64,
        "bps": pl.Int64,
        "clean_sheets": pl.Int64,
        "clearances_blocks_interceptions": pl.Int64,
        "completed_passes": pl.Int64,
        "creativity": pl.Float64,
        "defensive_contribution": pl.Int64,
        "dribbles": pl.Int64,
        "errors_leading_to_goal": pl.Int64,
        "errors_leading_to_goal_attempt": pl.Int64,
        "expected_assists": pl.Float64,
        "expected_goal_involvements": pl.Float64,
        "expected_goals": pl.Float64,
        "expected_goals_conceded": pl.Float64,
        "fouls": pl.Int64,
        "goals_conceded": pl.Int64,
        "goals_scored": pl.Int64,
        "ict_index": pl.Float64,
        "influence": pl.Float64,
        "key_passes": pl.Int64,
        "kickoff_time": pl.Datetime("us"),
        "minutes": pl.Int64,
        "mng_clean_sheets": pl.Int64,
        "mng_draw": pl.Int64,
        "mng_goals_scored": pl.Int64,
        "mng_loss": pl.Int64,
        "mng_underdog_draw": pl.Int64,
        "mng_underdog_win": pl.Int64,
        "mng_win": pl.Int64,
        "name": pl.Utf8,
        "offside": pl.Int64,
        "open_play_crosses": pl.Int64,
        "opponent_team": pl.Int64,
        "own_goals": pl.Int64,
        "penalties_conceded": pl.Int64,
        "penalties_missed": pl.Int64,
        "penalties_saved": pl.Int64,
        "position": pl.Utf8,
        "recoveries": pl.Int64,
        "red_cards": pl.Int64,
        "round": pl.Int64,
        "saves": pl.Int64,
        "selected": pl.Int64,
        "starts": pl.Int64,
        "tackled": pl.Int64,
        "tackles": pl.Int64,
        "target_missed": pl.Int64,
        "team": pl.Utf8,
        "team_a_score": pl.Int64,
        "team_h_score": pl.Int64,
        "threat": pl.Float64,
        "total_points": pl.Int64,
        "transfers_balance": pl.Int64,
        "transfers_in": pl.Int64,
        "transfers_out": pl.Int64,
        "value": pl.Int64,
        "was_home": pl.Boolean,
        "winning_goals": pl.Int64,
        "xP": pl.Float64,
        "yellow_cards": pl.Int64,
    },
    primary_key=("season", "gw", "element", "fixture"),
    order_by=("season", "gw", "element", "fixture"),
    normalise=gkp_to_gk,
)

# FCI's Opta-grade per-fixture stat set, 2024-25 onwards. Every
# competition is kept, not just the Premier League: European and cup
# minutes are genuine rotation and fatigue signal. ``competition`` makes
# the FPL filter explicit -- any consumer computing FPL minutes must
# filter to ``competition == "prem"`` or it will overcount, which is the
# bug class that once gave a defender 180 minutes in one gameweek.
PLAYER_MATCH_OPTA = Table(
    name="player_match_opta",
    schema={
        "season": pl.Utf8,
        "gw": pl.Int64,
        "element": pl.Int64,
        "match_id": pl.Utf8,
        "competition": pl.Utf8,
        "accurate_crosses": pl.Int64,
        "accurate_crosses_percent": pl.Float64,
        "accurate_long_balls": pl.Int64,
        "accurate_long_balls_percent": pl.Float64,
        "accurate_passes": pl.Int64,
        "accurate_passes_percent": pl.Float64,
        "aerial_duels_won": pl.Int64,
        "aerial_duels_won_percent": pl.Float64,
        "assists": pl.Int64,
        "big_chances_missed": pl.Int64,
        "blocks": pl.Int64,
        "chances_created": pl.Int64,
        "clearances": pl.Int64,
        "corners": pl.Int64,
        "defensive_contributions": pl.Int64,
        "dispossessed": pl.Int64,
        "distance_covered": pl.Float64,
        "dribbled_past": pl.Int64,
        "duels_lost": pl.Int64,
        "duels_won": pl.Int64,
        "final_third_passes": pl.Int64,
        "finish_min": pl.Int64,
        "fouls_committed": pl.Int64,
        "gk_accurate_long_balls": pl.Int64,
        "gk_accurate_passes": pl.Int64,
        "goals": pl.Int64,
        "goals_conceded": pl.Int64,
        "goals_prevented": pl.Float64,
        "ground_duels_won": pl.Int64,
        "ground_duels_won_percent": pl.Float64,
        "headed_clearances": pl.Int64,
        "high_claim": pl.Int64,
        "interceptions": pl.Int64,
        "minutes_played": pl.Int64,
        "number_of_sprints": pl.Int64,
        "offsides": pl.Int64,
        "penalties_missed": pl.Int64,
        "penalties_scored": pl.Int64,
        "recoveries": pl.Int64,
        "running_distance": pl.Float64,
        "saves": pl.Int64,
        "saves_inside_box": pl.Int64,
        "shots_on_target": pl.Int64,
        "sprinting_distance": pl.Float64,
        "start_min": pl.Int64,
        "successful_dribbles": pl.Int64,
        "successful_dribbles_percent": pl.Float64,
        "sweeper_actions": pl.Int64,
        "tackles": pl.Int64,
        "tackles_won": pl.Int64,
        "tackles_won_percent": pl.Float64,
        "team_goals_conceded": pl.Int64,
        "top_speed": pl.Float64,
        "total_shots": pl.Int64,
        "touches": pl.Int64,
        "touches_opposition_box": pl.Int64,
        "walking_distance": pl.Float64,
        "was_fouled": pl.Int64,
        "xa": pl.Float64,
        "xg": pl.Float64,
        "xgot": pl.Float64,
        "xgot_faced": pl.Float64,
    },
    primary_key=("season", "gw", "element", "match_id"),
    order_by=("season", "gw", "element", "match_id"),
)

PLAYER_AVAILABILITY = Table(
    name="player_availability",
    schema={
        "season": pl.Utf8,
        "gw": pl.Int64,
        "element": pl.Int64,
        "chance_of_playing_this_round": pl.Int64,
    },
    primary_key=("season", "gw", "element"),
    order_by=("season", "gw", "element"),
)

# ``prediction_kind`` is part of the primary key, so the same
# (season, gw, element, opponent) can hold both a ``backfill`` row and a
# ``forward`` row. For the current season, where the two coexist, any
# consumer that joins this table without filtering on ``prediction_kind``
# fans its rows out 2x -- silently double-counting expected minutes.
# Always filter to one kind before joining.
MINUTES_PREDICTION = Table(
    name="minutes_prediction",
    schema={
        "season": pl.Utf8,
        "gw": pl.Int64,
        "element": pl.Int64,
        "opponent": pl.Int64,
        "p_zero": pl.Float64,
        "p_partial": pl.Float64,
        "p_sixty_plus": pl.Float64,
        "expected_minutes": pl.Float64,
        "model_version": pl.Utf8,
        "prediction_kind": pl.Utf8,
        "snapshot_captured_at": pl.Datetime("us"),
    },
    primary_key=(
        "season",
        "gw",
        "element",
        "opponent",
        "prediction_kind",
    ),
    order_by=("season", "gw", "element", "opponent", "prediction_kind"),
)

# Per-position points predictions at match grain, so a double gameweek
# is two rows that sum to a gameweek total. ``position`` is carried so
# GK/MID/FWD models can land here without a migration.
POINTS_PREDICTION = Table(
    name="points_prediction",
    schema={
        "season": pl.Utf8,
        "gw": pl.Int64,
        "element": pl.Int64,
        "opponent": pl.Int64,
        "position": pl.Utf8,
        "predicted_points": pl.Float64,
        "model_version": pl.Utf8,
        "prediction_kind": pl.Utf8,
    },
    primary_key=(
        "season",
        "gw",
        "element",
        "opponent",
        "prediction_kind",
    ),
    order_by=("season", "gw", "element", "opponent", "prediction_kind"),
)

# The scoring components a prediction is built from, one row per
# component per fixture leg. ``points_prediction`` is the sum of these,
# so a component is added by adding a Component value rather than a
# table. ``diagnostics`` is JSON for the same reason the evaluation
# tables carry features as JSON: each component wants to record different
# intermediates, and typed columns would mean a migration per component.
POINTS_COMPONENT = Table(
    name="points_component",
    schema={
        "season": pl.Utf8,
        "gw": pl.Int64,
        "element": pl.Int64,
        "opponent": pl.Int64,
        "position": pl.Utf8,
        "prediction_kind": pl.Utf8,
        "component": pl.Utf8,
        "points": pl.Float64,
        "model_version": pl.Utf8,
        "diagnostics": pl.Utf8,
    },
    primary_key=(
        "season",
        "gw",
        "element",
        "opponent",
        "prediction_kind",
        "component",
    ),
    order_by=(
        "season",
        "gw",
        "element",
        "opponent",
        "prediction_kind",
        "component",
    ),
)

# Evaluation predictions, one row per scored fold row. Deliberately not
# the serving tables: the optimiser reads points_prediction, and a join
# there that forgets prediction_kind already fans rows out. These stay
# where nothing but analysis reads them.
#
# ``run_id`` is the MLflow run, and it is in the primary key so runs
# accumulate side by side rather than overwriting each other. ``features``
# is JSON because each position feeds the model a different column list;
# typed columns would mean a migration per feature added. No error column:
# it is one expression away from the prediction and the actual, and a
# stored copy can drift from them.
TEST_POINTS_PREDICTION = Table(
    name="test_points_prediction",
    schema={
        "run_id": pl.Utf8,
        "season": pl.Utf8,
        "gw": pl.Int64,
        "element": pl.Int64,
        "opponent": pl.Int64,
        "position": pl.Utf8,
        "predicted_points": pl.Float64,
        "actual_points": pl.Float64,
        "features": pl.Utf8,
    },
    primary_key=("run_id", "season", "gw", "element", "opponent"),
    order_by=("run_id", "season", "gw", "element", "opponent"),
)

# Team grain, unlike every other evaluation table. The conceding head
# predicts one number per team-fixture; fanning its fold rows out to the
# eleven players who share it would store the same prediction eleven
# times and make the error analysis report a sample size the model never
# saw.
TEST_CONCEDING_PREDICTION = Table(
    name="test_conceding_prediction",
    schema={
        "run_id": pl.Utf8,
        "season": pl.Utf8,
        "gw": pl.Int64,
        "team": pl.Utf8,
        "opposition": pl.Utf8,
        "predicted_conceded": pl.Float64,
        "actual_conceded": pl.Float64,
        "features": pl.Utf8,
    },
    primary_key=("run_id", "season", "gw", "team", "opposition"),
    order_by=("run_id", "season", "gw", "team", "opposition"),
)

TEST_MINUTES_PREDICTION = Table(
    name="test_minutes_prediction",
    schema={
        "run_id": pl.Utf8,
        "season": pl.Utf8,
        "gw": pl.Int64,
        "element": pl.Int64,
        "opponent": pl.Int64,
        "p_zero": pl.Float64,
        "p_partial": pl.Float64,
        "p_sixty_plus": pl.Float64,
        "expected_minutes": pl.Float64,
        "actual_bucket": pl.Utf8,
        "actual_minutes": pl.Int64,
        "features": pl.Utf8,
    },
    primary_key=("run_id", "season", "gw", "element", "opponent"),
    order_by=("run_id", "season", "gw", "element", "opponent"),
)

PLAYER_SNAPSHOT = Table(
    name="player_snapshot",
    schema={
        "season": pl.Utf8,
        "captured_at": pl.Datetime("us"),
        "element": pl.Int64,
        "value": pl.Int64,
        "team": pl.Utf8,
        "position": pl.Utf8,
        "chance_of_playing_this_round": pl.Int64,
        "status": pl.Utf8,
    },
    primary_key=("season", "captured_at", "element"),
    order_by=("season", "captured_at", "element"),
    normalise=gkp_to_gk,
)

PLAYER_SEASON = Table(
    name="player_season",
    schema={
        "season": pl.Utf8,
        "element": pl.Int64,
        "player_code": pl.Int64,
        "web_name": pl.Utf8,
        "first_name": pl.Utf8,
        "second_name": pl.Utf8,
        "position": pl.Utf8,
        "team_code": pl.Int64,
        "birth_date": pl.Date,
        "region": pl.Int64,
        "team_join_date": pl.Date,
    },
    primary_key=("season", "element"),
    order_by=("season", "element"),
    enrich=propagate_static_columns,
)

# Raw Transfermarkt scrapes. Kept verbatim as VARCHAR with parsed
# siblings alongside, so a parser bug never destroys scraped data. These
# are not populated by main() -- see scripts/scrape_transfermarkt.py --
# so reset_database wipes them and a re-scrape is the only cure.
TM_PLAYER = Table(
    name="tm_player",
    schema={
        "tm_player_id": pl.Utf8,
        "name": pl.Utf8,
        "player_link": pl.Utf8,
        "dob": pl.Utf8,
        "dob_date": pl.Date,
        "height_m": pl.Float64,
        "nationality": pl.Utf8,
        "citizenship": pl.Utf8,
        "position": pl.Utf8,
        "team": pl.Utf8,
        "last_club": pl.Utf8,
        "since": pl.Utf8,
        "since_date": pl.Date,
        "joined": pl.Utf8,
        "joined_date": pl.Date,
        "contract_expiration": pl.Utf8,
        "contract_expiration_date": pl.Date,
        "value": pl.Utf8,
        "value_eur": pl.Int64,
        "value_last_updated": pl.Utf8,
        "value_last_updated_date": pl.Date,
        "scraped_at": pl.Datetime,
    },
    primary_key=("tm_player_id",),
    order_by=("tm_player_id",),
)

TM_PLAYER_SEASON = Table(
    name="tm_player_season",
    schema={
        "season": pl.Utf8,
        "tm_player_id": pl.Utf8,
        "player_link": pl.Utf8,
    },
    primary_key=("season", "tm_player_id"),
    order_by=("season", "tm_player_id"),
)

# transfer_seq is scrape order, and Transfermarkt lists newest first, so
# seq 0 is the most recent transfer. A synthetic key rather than the date
# because a player can have two transfers on one day (loan end plus new
# loan), and because a raw table must not reject a row for failing a parse.
TM_TRANSFER = Table(
    name="tm_transfer",
    schema={
        "tm_player_id": pl.Utf8,
        "transfer_seq": pl.Int64,
        "season": pl.Utf8,
        "transfer_date": pl.Utf8,
        "transfer_date_parsed": pl.Date,
        "left_club": pl.Utf8,
        "joined_club": pl.Utf8,
        "market_value": pl.Utf8,
        "market_value_eur": pl.Int64,
        "fee": pl.Utf8,
        "fee_eur": pl.Int64,
        "fee_type": pl.Utf8,
    },
    primary_key=("tm_player_id", "transfer_seq"),
    order_by=("tm_player_id", "transfer_seq"),
)

# value_eur arrives already typed from the market-value JSON endpoint, so
# this is the one table with nothing to parse but the date. That date is in
# the key, so a row whose date will not parse cannot be stored.
TM_MARKET_VALUE = Table(
    name="tm_market_value",
    schema={
        "tm_player_id": pl.Utf8,
        "value_date": pl.Date,
        "value_date_raw": pl.Utf8,
        "value_eur": pl.Int64,
    },
    primary_key=("tm_player_id", "value_date"),
    order_by=("tm_player_id", "value_date"),
)

# Person-level bridge from FPL to Transfermarkt, rebuilt wholesale by
# scripts/scrape_transfermarkt.py. UNIQUE on tm_player_id because the map
# must be one-to-one in both directions; a duplicate fans out the join.
TM_PLAYER_MAP = Table(
    name="tm_player_map",
    schema={
        "player_code": pl.Int64,
        "tm_player_id": pl.Utf8,
        "match_rule": pl.Utf8,
        "match_score": pl.Float64,
        "fpl_name": pl.Utf8,
        "tm_name": pl.Utf8,
    },
    primary_key=("player_code",),
    order_by=("player_code",),
    unique=(("tm_player_id",),),
)

# Every stored table. ``get_connection`` and ``reset_database`` loop over
# this, so a new table cannot be forgotten by either.
TABLES: tuple[Table, ...] = (
    PLAYER_WEEK,
    TEAM_FIXTURE,
    PLAYER_MATCH,
    PLAYER_MATCH_FPL,
    PLAYER_MATCH_OPTA,
    PLAYER_AVAILABILITY,
    MINUTES_PREDICTION,
    POINTS_COMPONENT,
    POINTS_PREDICTION,
    TEST_POINTS_PREDICTION,
    TEST_MINUTES_PREDICTION,
    TEST_CONCEDING_PREDICTION,
    PLAYER_SEASON,
    PLAYER_SNAPSHOT,
    TM_PLAYER,
    TM_PLAYER_SEASON,
    TM_TRANSFER,
    TM_MARKET_VALUE,
    TM_PLAYER_MAP,
)


def prediction_versions(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    seasons: list[str] | None = None,
    equals: dict[str, object] | None = None,
) -> set[str]:
    """Return the distinct ``model_version`` values stored in a table.

    Shared body for ``minutes_prediction_versions`` and
    ``points_prediction_versions``. ``table_name`` is always one of the
    two module-level literals those functions pass in, never external
    input, so it is safe to interpolate directly.

    ``equals`` restricts which rows count. ``points_prediction`` holds
    every position's predictions, so a model asking "what version am I
    stored at" must pass its own position or it reads another model's
    versions as its own and mis-gates the historic backfill.
    """
    clauses: list[str] = []
    params: list[object] = []
    if seasons is not None:
        if not seasons:
            # An empty ``IN ()`` clause is invalid SQL. An empty seasons
            # list means "no seasons requested", so the honest answer is
            # an empty set, without ever building the query.
            return set()
        placeholders = ", ".join("?" for _ in seasons)
        clauses.append(f"season IN ({placeholders})")
        params.extend(seasons)
    equality, equality_params = engine.equality_clauses(equals or {})
    clauses.extend(equality)
    params.extend(equality_params)
    query = f"SELECT DISTINCT model_version FROM {table_name}"
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    rows = connection.execute(query, params).fetchall()
    return {row[0] for row in rows if row[0] is not None}


def minutes_prediction_versions(
    connection: duckdb.DuckDBPyConnection,
    seasons: list[str] | None = None,
) -> set[str]:
    """Return the distinct ``model_version`` values stored.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        An open connection.
    seasons : list[str] | None, optional
        When given, restrict to these seasons (used to gate the historic
        backfill on the versions already stored for historic seasons).

    Returns
    -------
    set[str]
        Distinct non-null model versions.
    """
    return prediction_versions(connection, "minutes_prediction", seasons)


def points_prediction_versions(
    connection: duckdb.DuckDBPyConnection,
    seasons: list[str] | None = None,
) -> set[str]:
    """Return the distinct ``model_version`` values stored.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        An open connection.
    seasons : list[str] | None, optional
        When given, restrict to these seasons (used to gate the historic
        backfill on the versions already stored for historic seasons).

    Returns
    -------
    set[str]
        Distinct non-null model versions.
    """
    return prediction_versions(connection, "points_prediction", seasons)
