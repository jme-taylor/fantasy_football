"""Tests for the rolling team-level form features."""

from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import polars as pl
import pytest

from fantasy_football.features import team_form
from fantasy_football.features.team_form import register_team_form
from fantasy_football.storage.database import get_connection
from fantasy_football.storage.lookups import register_lookups
from fantasy_football.storage.tables import (
    PLAYER_MATCH,
    PLAYER_MATCH_OPTA,
    PLAYER_WEEK,
    TEAM_FIXTURE,
)

SEASON = "2025-26"
FIRST_KICKOFF = datetime(2025, 8, 16, 14, 0)

UNITED, ARSENAL, SPURS = "Man Utd", "Arsenal", "Spurs"
TEAM_IDS = {UNITED: 14, ARSENAL: 1, SPURS: 18}
SLUGS = {
    UNITED: "manchester-united",
    ARSENAL: "arsenal",
    SPURS: "tottenham-hotspur",
}
# element -> club. One player per club keeps the fixtures readable; the
# aggregation is a sum either way.
ELEMENTS = {1: UNITED, 2: ARSENAL, 3: SPURS}


def _append(table, connection, rows: list[dict]) -> None:
    """Append dict rows, filling unstated columns with nulls."""
    table.append(connection, table.conform(pl.DataFrame(rows)))


def _build(tmp_path: Path, fixtures: list[dict]) -> duckdb.DuckDBPyConnection:
    """Return a connection holding the given fixtures.

    Each fixture is ``{"gw", "home", "away", "stats"}`` where ``stats``
    maps a club to its ``(xg, goals, team_goals_conceded)`` for that
    match. Kickoffs are a week apart, in gameweek order.
    """
    connection = get_connection(tmp_path / "test.duckdb")
    player_weeks, player_matches, team_fixtures, opta = [], [], [], []

    for fixture in fixtures:
        gw, home, away = fixture["gw"], fixture["home"], fixture["away"]
        kickoff = FIRST_KICKOFF + timedelta(days=7 * (gw - 1))
        match_id = f"25-26-prem-{SLUGS[home]}-vs-{SLUGS[away]}"

        for club, is_home in ((home, True), (away, False)):
            opponent = away if is_home else home
            element = next(e for e, c in ELEMENTS.items() if c == club)
            xg, goals, conceded = fixture["stats"][club]

            player_weeks.append(
                {
                    "season": SEASON,
                    "gw": gw,
                    "element": element,
                    "position": "MID",
                    "team": club,
                }
            )
            player_matches.append(
                {
                    "season": SEASON,
                    "gw": gw,
                    "element": element,
                    "opponent": TEAM_IDS[opponent],
                    "is_home": is_home,
                    "minutes": 90,
                    "kickoff_time": kickoff,
                }
            )
            team_fixtures.append(
                {
                    "season": SEASON,
                    "gw": gw,
                    "team": club,
                    "is_home": is_home,
                    "opposition": opponent,
                    "kickoff_time": kickoff,
                }
            )
            opta.append(
                {
                    "season": SEASON,
                    "gw": gw,
                    "element": element,
                    "match_id": match_id,
                    "competition": "prem",
                    "minutes_played": 90,
                    "xg": xg,
                    "goals": goals,
                    "team_goals_conceded": conceded,
                }
            )

    _append(PLAYER_WEEK, connection, player_weeks)
    _append(PLAYER_MATCH, connection, player_matches)
    _append(TEAM_FIXTURE, connection, team_fixtures)
    _append(PLAYER_MATCH_OPTA, connection, opta)
    register_lookups(connection)
    return connection


# United win 2-1, then win 3-0 away, then play a third match whose
# rolling columns describe only the first two.
_RUN = [
    {
        "gw": 1,
        "home": UNITED,
        "away": ARSENAL,
        "stats": {UNITED: (1.5, 2, 1), ARSENAL: (0.5, 1, 2)},
    },
    {
        "gw": 2,
        "home": SPURS,
        "away": UNITED,
        "stats": {SPURS: (0.2, 0, 3), UNITED: (2.0, 3, 0)},
    },
    {
        "gw": 3,
        "home": UNITED,
        "away": SPURS,
        "stats": {UNITED: (1.0, 1, 1), SPURS: (0.9, 1, 1)},
    },
]


@pytest.fixture
def connection(tmp_path: Path):
    """Yield a connection holding United's three-match run."""
    conn = _build(tmp_path, _RUN)
    try:
        yield conn
    finally:
        conn.close()


def test_feature_columns_names_every_measure() -> None:
    """One rolling column per measure, plus the context columns."""
    columns = team_form.feature_columns(5)
    assert columns[:5] == [
        "xg_for_rolling_5",
        "xg_against_rolling_5",
        "goals_for_rolling_5",
        "goals_against_rolling_5",
        "clean_sheet_rolling_5",
    ]
    assert columns[-1] == "days_since_last_match"


def test_clean_sheet_flags_a_goalless_defence(connection) -> None:
    """Conceding nothing is a clean sheet; conceding anything is not."""
    team_form.register_team_form(connection)
    result = connection.sql(
        "SELECT team, clean_sheet FROM team_match "
        "WHERE gw = 2 ORDER BY team"
    ).pl()
    # Spurs 0-3 United: United kept it, Spurs did not.
    assert dict(zip(result["team"], result["clean_sheet"])) == {
        UNITED: 1,
        SPURS: 0,
    }


def test_clean_sheet_rate_averages_prior_matches(connection) -> None:
    """The rolling column is a rate over the window, offset like the rest."""
    frame = team_form.load_team_form(connection)
    third = frame.filter((pl.col("team") == UNITED) & (pl.col("gw") == 3))
    # United conceded 1 in gw1 and 0 in gw2 -> one clean sheet in two.
    assert third["clean_sheet_rolling_5"].item() == pytest.approx(0.5)
    # And gw3's own clean sheet is excluded from that figure.
    assert third["clean_sheet"].item() == 0


def test_team_match_has_one_row_per_side(connection) -> None:
    """Each fixture reduces to exactly two rows."""
    team_form.register_team_form(connection)
    result = connection.sql(
        "SELECT count(*) FROM team_match WHERE gw = 1"
    ).fetchone()[0]
    assert result == 2


def test_for_and_against_are_symmetric(connection) -> None:
    """One side's xG for is the other's xG against."""
    team_form.register_team_form(connection)
    result = connection.sql(
        """
        SELECT a.xg_for, b.xg_against, a.goals_for, b.goals_against
        FROM team_match a JOIN team_match b
          ON a.match_id = b.match_id AND a.gw = b.gw AND a.team <> b.team
        WHERE a.gw = 1 AND a.team = 'Man Utd'
        """
    ).fetchone()
    assert result[0] == pytest.approx(result[1])
    assert result[2] == result[3]


def test_goals_come_from_declared_concessions(connection) -> None:
    """Goals for is the opponent's conceded count, not summed player goals."""
    team_form.register_team_form(connection)
    result = connection.sql(
        "SELECT goals_for, goals_against FROM team_match "
        "WHERE gw = 1 AND team = 'Man Utd'"
    ).fetchone()
    assert result == (2, 1)


def test_own_goal_counts_towards_goals_for(tmp_path: Path) -> None:
    """A goal no player is credited with still has to show up."""
    # Arsenal concede 1 while United's players are credited with none.
    connection = _build(
        tmp_path,
        [
            {
                "gw": 1,
                "home": UNITED,
                "away": ARSENAL,
                "stats": {UNITED: (0.3, 0, 0), ARSENAL: (0.4, 0, 1)},
            }
        ],
    )
    try:
        team_form.register_team_form(connection)
        result = connection.sql(
            "SELECT goals_for FROM team_match "
            "WHERE gw = 1 AND team = 'Man Utd'"
        ).fetchone()[0]
        assert result == 1
    finally:
        connection.close()


def test_first_match_has_no_prior_form(connection) -> None:
    """A window with nothing preceding it is null, not zero."""
    frame = team_form.load_team_form(connection)
    first = frame.filter((pl.col("team") == UNITED) & (pl.col("gw") == 1))
    assert first["form_matches"].item() == 0
    assert first["xg_for_rolling_5"].item() is None


def test_rolling_columns_exclude_the_current_match(connection) -> None:
    """The third match averages the first two only."""
    frame = team_form.load_team_form(connection)
    third = frame.filter((pl.col("team") == UNITED) & (pl.col("gw") == 3))
    assert third["form_matches"].item() == 2
    # xG for: (1.5 + 2.0) / 2. Against: (0.5 + 0.2) / 2.
    assert third["xg_for_rolling_5"].item() == pytest.approx(1.75)
    assert third["xg_against_rolling_5"].item() == pytest.approx(0.35)
    # Goals for: (2 + 3) / 2. Against: (1 + 0) / 2.
    assert third["goals_for_rolling_5"].item() == pytest.approx(2.5)
    assert third["goals_against_rolling_5"].item() == pytest.approx(0.5)
    # The current match's own 1-1 is absent from every figure above.
    assert third["goals_for"].item() == 1


def test_days_since_last_match_measures_the_gap(connection) -> None:
    """Fixture congestion is exposed, not just the averages."""
    frame = team_form.load_team_form(connection)
    third = frame.filter((pl.col("team") == UNITED) & (pl.col("gw") == 3))
    assert third["days_since_last_match"].item() == 7


def test_form_sql_inclusive_frame_includes_current_row():
    """The inclusive frame widens the window to the current row."""
    exclusive = team_form.form_sql(rolling_window=5, inclusive=False)
    inclusive = team_form.form_sql(rolling_window=5, inclusive=True)

    assert "ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING" in exclusive
    assert "ROWS BETWEEN 4 PRECEDING AND CURRENT ROW" in inclusive


def test_inclusive_view_registers_under_its_own_name(connection):
    """The inclusive view is a separate, additional view, not a swap."""
    register_lookups(connection)
    register_team_form(connection, inclusive=False)
    register_team_form(connection, inclusive=True)

    names = {
        row[0]
        for row in connection.execute(
            "SELECT view_name FROM duckdb_views()"
        ).fetchall()
    }
    assert "team_match_form" in names
    assert "team_match_form_inclusive" in names


def test_inclusive_view_rolling_measure_includes_the_current_match(
    connection,
) -> None:
    """The inclusive frame's rolling measure reaches into its own match."""
    register_team_form(connection, inclusive=True)
    inclusive = connection.sql(
        "SELECT form_matches, xg_for_rolling_5 "
        "FROM team_match_form_inclusive "
        "WHERE team = 'Man Utd' AND gw = 3"
    ).fetchone()
    # All three matches count, including gw3's own 1.0 xG for:
    # (1.5 + 2.0 + 1.0) / 3 = 1.5.
    assert inclusive[0] == 3
    assert inclusive[1] == pytest.approx(1.5)

    register_team_form(connection, inclusive=False)
    exclusive = connection.sql(
        "SELECT form_matches, xg_for_rolling_5 FROM team_match_form "
        "WHERE team = 'Man Utd' AND gw = 3"
    ).fetchone()
    # Only the two prior matches count, excluding gw3's own 1.0:
    # (1.5 + 2.0) / 2 = 1.75.
    assert exclusive[0] == 2
    assert exclusive[1] == pytest.approx(1.75)


def test_each_club_windows_over_its_own_matches(connection) -> None:
    """Opponents' form must not bleed into a team's window."""
    frame = team_form.load_team_form(connection)
    # Spurs played gw2 and gw3; their gw3 window holds only gw2.
    spurs = frame.filter((pl.col("team") == SPURS) & (pl.col("gw") == 3))
    assert spurs["form_matches"].item() == 1
    assert spurs["xg_for_rolling_5"].item() == pytest.approx(0.2)
    assert spurs["goals_against_rolling_5"].item() == pytest.approx(3.0)


def _exposure(connection, player: float, team: float, matches: float) -> float:
    """Evaluate the penalty exposure expression on literal counts."""
    return connection.sql(
        "SELECT "
        + team_form.penalty_exposure_sql(str(player), str(team), str(matches))
    ).fetchone()[0]


def test_penalty_exposure_rewards_the_sole_taker(connection) -> None:
    """A club's whole penalty duty concentrates on one player."""
    sole = _exposure(connection, player=6, team=6, matches=10)
    shared = _exposure(connection, player=3, team=6, matches=10)
    assert sole > shared > 0.0


def test_penalty_exposure_is_worthless_without_penalties(connection) -> None:
    """A designated taker at a club that wins none is worth nothing."""
    assert _exposure(connection, player=0, team=0, matches=10) == 0.0


def test_penalty_exposure_survives_a_club_with_no_matches(
    connection,
) -> None:
    """Zero matches is a division by zero, and must stay finite."""
    assert _exposure(connection, player=0, team=0, matches=0) == 0.0


def test_penalty_exposure_shrinks_a_single_attempt(connection) -> None:
    """One penalty does not make a player the designated taker."""
    once = _exposure(connection, player=1, team=1, matches=10)
    # An unshrunk share would be 1.0, so the exposure would be the
    # club's whole rate of 0.1.
    assert once < 0.1


def test_penalty_share_never_exceeds_one(connection) -> None:
    """A player cannot take more than his club's penalties."""
    for attempts in (1, 5, 50):
        exposure = _exposure(
            connection, player=attempts, team=attempts, matches=10
        )
        assert exposure <= attempts / 10


def test_penalty_share_is_clamped_for_a_mid_season_signing(
    connection,
) -> None:
    """His attempts follow him across clubs; the denominator does not.

    A player who took three penalties before a January move joins a
    club that has won one. The unclamped share exceeds 1 and would rate
    him above his new club's entire penalty output.
    """
    exposure = _exposure(connection, player=3, team=1, matches=10)
    club_rate = 1 / 10
    assert exposure == pytest.approx(club_rate)
