from datetime import datetime, timedelta

import duckdb
import polars as pl
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

from fantasy_football.modelling import defender
from fantasy_football.storage.tables import (
    PLAYER_MATCH,
    PLAYER_MATCH_OPTA,
    PLAYER_SEASON,
    PLAYER_WEEK,
    TABLES,
    TEAM_FIXTURE,
)

SEASON = "2025-26"
FIRST_KICKOFF = datetime(2025, 8, 16, 14, 0)

# Real FCI_SLUG_TO_FPL entries, reused from tests/unit/features/test_team_form.py
# and test_match_form.py so the club-name join actually resolves.
UNITED, ARSENAL, SPURS, CHELSEA = "Man Utd", "Arsenal", "Spurs", "Chelsea"
SLUGS = {
    UNITED: "manchester-united",
    ARSENAL: "arsenal",
    SPURS: "tottenham-hotspur",
    CHELSEA: "chelsea",
}
# fpl_team_id only maps a team name to an id when some player_match row
# has that team as its opponent, so every club needs at least one
# player_match row naming it as the opposition -- see the extra Spurs and
# Arsenal player_match rows below, which exist solely to mint the Man Utd
# and Chelsea mappings.
TEAM_IDS = {UNITED: 1, SPURS: 2, ARSENAL: 3, CHELSEA: 4}


@pytest.fixture
def connection():
    """Yield an in-memory DuckDB connection with every table created."""
    conn = duckdb.connect(":memory:")
    for table in TABLES:
        conn.execute(table.ddl)
    yield conn
    conn.close()


def _append(table, connection, rows: list[dict]) -> None:
    """Append dict rows, filling unstated columns with nulls."""
    table.append(connection, table.conform(pl.DataFrame(rows)))


def _opta_row(
    season: str,
    gw: int,
    element: int,
    match_id: str,
    xg: float,
    goals: int,
    conceded: int,
) -> dict:
    """Build a minimal player_match_opta row for one side of a fixture."""
    return {
        "season": season,
        "gw": gw,
        "element": element,
        "match_id": match_id,
        "competition": "prem",
        "minutes_played": 90,
        "xg": xg,
        "goals": goals,
        "team_goals_conceded": conceded,
    }


def _seed_defender_frame(connection: duckdb.DuckDBPyConnection) -> None:
    """Seed a minimal but real row set distinguishing own from opposition.

    Two prior-gameweek fixtures give Man Utd and Arsenal each their own,
    deliberately different, rolling defensive/attacking record before the
    gw2 fixture between them (our target row, element 1, a Man Utd
    defender). Man Utd's own xg_against/goals_against/clean_sheet in gw1
    (vs Spurs) and Arsenal's own xg_for/goals_for in gw1 (vs Chelsea) are
    picked to be unambiguously distinct, so a transposed own/opposition
    join is caught rather than coincidentally passing.
    """
    gw1_kickoff = FIRST_KICKOFF
    gw2_kickoff = FIRST_KICKOFF + timedelta(days=7)
    utd_v_spurs = f"25-26-prem-{SLUGS[UNITED]}-vs-{SLUGS[SPURS]}"
    arsenal_v_chelsea = f"25-26-prem-{SLUGS[ARSENAL]}-vs-{SLUGS[CHELSEA]}"
    utd_v_arsenal = f"25-26-prem-{SLUGS[UNITED]}-vs-{SLUGS[ARSENAL]}"

    _append(
        PLAYER_SEASON,
        connection,
        [{"season": SEASON, "element": 1, "position": "DEF"}],
    )
    _append(
        PLAYER_WEEK,
        connection,
        [
            {
                "season": SEASON,
                "gw": 1,
                "element": 1,
                "position": "DEF",
                "team": UNITED,
            },
            {
                "season": SEASON,
                "gw": 1,
                "element": 2,
                "position": "MID",
                "team": SPURS,
            },
            {
                "season": SEASON,
                "gw": 1,
                "element": 3,
                "position": "MID",
                "team": ARSENAL,
            },
            {
                "season": SEASON,
                "gw": 1,
                "element": 4,
                "position": "MID",
                "team": CHELSEA,
            },
            {
                "season": SEASON,
                "gw": 2,
                "element": 1,
                "position": "DEF",
                "team": UNITED,
            },
            {
                "season": SEASON,
                "gw": 2,
                "element": 3,
                "position": "MID",
                "team": ARSENAL,
            },
        ],
    )
    _append(
        PLAYER_MATCH,
        connection,
        [
            {
                "season": SEASON,
                "gw": 1,
                "element": 1,
                "opponent": TEAM_IDS[SPURS],
                "is_home": True,
                "minutes": 90,
                "kickoff_time": gw1_kickoff,
            },
            {
                "season": SEASON,
                "gw": 2,
                "element": 1,
                "opponent": TEAM_IDS[ARSENAL],
                "is_home": True,
                "minutes": 90,
                "kickoff_time": gw2_kickoff,
            },
            # These two exist only to mint the Man Utd and Chelsea
            # team-id mappings in fpl_team_id (see the TEAM_IDS comment
            # above) -- they play no other role in the test.
            {
                "season": SEASON,
                "gw": 1,
                "element": 2,
                "opponent": TEAM_IDS[UNITED],
                "is_home": False,
                "minutes": 90,
                "kickoff_time": gw1_kickoff,
            },
            {
                "season": SEASON,
                "gw": 1,
                "element": 3,
                "opponent": TEAM_IDS[CHELSEA],
                "is_home": True,
                "minutes": 90,
                "kickoff_time": gw1_kickoff,
            },
        ],
    )
    _append(
        TEAM_FIXTURE,
        connection,
        [
            {
                "season": SEASON,
                "gw": 1,
                "team": UNITED,
                "is_home": True,
                "opposition": SPURS,
                "kickoff_time": gw1_kickoff,
            },
            {
                "season": SEASON,
                "gw": 1,
                "team": SPURS,
                "is_home": False,
                "opposition": UNITED,
                "kickoff_time": gw1_kickoff,
            },
            {
                "season": SEASON,
                "gw": 1,
                "team": ARSENAL,
                "is_home": True,
                "opposition": CHELSEA,
                "kickoff_time": gw1_kickoff,
            },
            {
                "season": SEASON,
                "gw": 1,
                "team": CHELSEA,
                "is_home": False,
                "opposition": ARSENAL,
                "kickoff_time": gw1_kickoff,
            },
            {
                "season": SEASON,
                "gw": 2,
                "team": UNITED,
                "is_home": True,
                "opposition": ARSENAL,
                "kickoff_time": gw2_kickoff,
            },
            {
                "season": SEASON,
                "gw": 2,
                "team": ARSENAL,
                "is_home": False,
                "opposition": UNITED,
                "kickoff_time": gw2_kickoff,
            },
        ],
    )
    _append(
        PLAYER_MATCH_OPTA,
        connection,
        [
            # gw1: Man Utd 2-0 Spurs. Man Utd keep a clean sheet, facing
            # 0.3 xG against.
            _opta_row(SEASON, 1, 1, utd_v_spurs, 2.0, 2, 0),
            _opta_row(SEASON, 1, 2, utd_v_spurs, 0.3, 0, 2),
            # gw1: Arsenal 1-1 Chelsea. Arsenal's own attacking record
            # (1.0 xG for) is what must land in the *opposition* columns
            # of the gw2 fixture below, not the own-team columns.
            _opta_row(SEASON, 1, 3, arsenal_v_chelsea, 1.0, 1, 1),
            _opta_row(SEASON, 1, 4, arsenal_v_chelsea, 0.5, 1, 1),
            # gw2: Man Utd vs Arsenal, the target row. Values here are
            # irrelevant to the rolling columns -- the window excludes
            # the current match by construction.
            _opta_row(SEASON, 2, 1, utd_v_arsenal, 1.5, 1, 1),
            _opta_row(SEASON, 2, 3, utd_v_arsenal, 0.8, 1, 1),
        ],
    )


def test_features_exclude_keys_and_target() -> None:
    """Neither the target nor a key column leaks into FEATURES."""
    assert defender.TARGET not in defender.FEATURES
    for key in defender.KEY_COLUMNS:
        assert key not in defender.FEATURES


def test_model_frame_sql_filters_to_defenders_and_played_matches() -> None:
    """The SQL restricts to defenders and rows with recorded minutes."""
    sql = defender.model_frame_sql()
    assert "s.position = 'DEF'" in sql
    assert "m.minutes IS NOT NULL" in sql


def test_build_model_frame_returns_keys_target_and_features(
    connection,
) -> None:
    """The frame's columns are keys, then target, then features."""
    frame = defender.build_model_frame(connection)
    expected = defender.KEY_COLUMNS + [defender.TARGET] + defender.FEATURES
    assert frame.columns == expected


def test_pipeline_imputes_and_has_no_scaler() -> None:
    """The pipeline median-imputes into a random forest, with no scaler."""
    pipe = defender.make_pipeline()
    steps = dict(pipe.named_steps)
    assert isinstance(steps["impute"], SimpleImputer)
    assert steps["impute"].strategy == "median"
    assert isinstance(steps["model"], RandomForestRegressor)
    assert steps["model"].n_estimators == defender.N_ESTIMATORS
    assert not any("scale" in name for name in steps)


def test_pipeline_fits_and_predicts_with_nulls() -> None:
    """The pipeline fits and predicts on a frame containing null features."""
    pipe = defender.make_pipeline()
    x = pl.DataFrame(
        {name: [1.0, None, 3.0] for name in defender.FEATURES}
    ).to_pandas()
    y = [1.0, 2.0, 3.0]
    pipe.fit(x, y)
    assert len(pipe.predict(x)) == 3


def test_own_team_and_opposition_form_are_not_swapped(connection) -> None:
    """The gw2 row carries Man Utd's own record, not Arsenal's.

    A cartesian fan-out is caught by the row count; a transposed
    own/opposition join (``own.team = opp_id.team`` instead of
    ``own.team = pw.team``, and vice versa) is caught because Man Utd's
    and Arsenal's gw1 records are deliberately different -- if the two
    sides were swapped, this would fail rather than coincidentally pass.
    """
    _seed_defender_frame(connection)
    frame = defender.build_model_frame(connection)
    row = frame.filter(pl.col("gw") == 2)

    assert row.height == 1

    # Man Utd's own gw1 record (vs Spurs): 0.3 xG conceded, 0 goals
    # conceded, a clean sheet kept.
    assert row["xg_against_rolling_5"].item() == pytest.approx(0.3)
    assert row["goals_against_rolling_5"].item() == pytest.approx(0.0)
    assert row["clean_sheet_rolling_5"].item() == pytest.approx(1.0)

    # Arsenal's own gw1 record (vs Chelsea): 1.0 xG for, 1 goal for.
    assert row["xg_for_rolling_5"].item() == pytest.approx(1.0)
    assert row["goals_for_rolling_5"].item() == pytest.approx(1.0)
