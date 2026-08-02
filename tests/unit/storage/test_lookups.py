"""Tests for the FCI-to-FPL fixture bridge views."""

from datetime import datetime
from pathlib import Path

import duckdb
import polars as pl
import pytest

from fantasy_football.storage import lookups
from fantasy_football.storage.database import get_connection
from fantasy_football.storage.tables import (
    PLAYER_MATCH,
    PLAYER_MATCH_OPTA,
    PLAYER_WEEK,
    TEAM_FIXTURE,
)

KICKOFF_ONE = datetime(2025, 8, 16, 14, 0)
KICKOFF_TWO = datetime(2025, 8, 19, 19, 30)


def _append(table, connection, data: dict) -> None:
    """Append a partial row set, filling unstated columns with nulls.

    ``Table.append`` coerces rather than conforms, so it needs every
    declared column present. These fixtures state only the columns the
    bridge actually reads.
    """
    table.append(connection, table.conform(pl.DataFrame(data)))


def _connection(tmp_path: Path) -> duckdb.DuckDBPyConnection:
    """Return a connection holding one double gameweek for element 1.

    Element 1 plays for Man Utd and faces Arsenal at home then Spurs away
    in the same gameweek -- the case gameweek-grain aggregation collapses.
    """
    connection = get_connection(tmp_path / "test.duckdb")
    _append(
        PLAYER_WEEK,
        connection,
        {
            "season": ["2025-26"],
            "gw": [1],
            "element": [1],
            "position": ["DEF"],
            "team": ["Man Utd"],
        },
    )
    _append(
        PLAYER_MATCH,
        connection,
        {
            "season": ["2025-26", "2025-26"],
            "gw": [1, 1],
            "element": [1, 1],
            "opponent": [1, 18],
            "is_home": [True, False],
            "minutes": [90, 45],
            "kickoff_time": [KICKOFF_ONE, KICKOFF_TWO],
        },
    )
    _append(
        TEAM_FIXTURE,
        connection,
        {
            "season": ["2025-26", "2025-26"],
            "gw": [1, 1],
            "team": ["Man Utd", "Man Utd"],
            "is_home": [True, False],
            "opposition": ["Arsenal", "Spurs"],
            "kickoff_time": [KICKOFF_ONE, KICKOFF_TWO],
        },
    )
    _append(
        PLAYER_MATCH_OPTA,
        connection,
        {
            "season": ["2025-26", "2025-26"],
            "gw": [1, 1],
            "element": [1, 1],
            "match_id": [
                "25-26-prem-manchester-united-vs-arsenal",
                "25-26-prem-tottenham-hotspur-vs-manchester-united",
            ],
            "competition": ["prem", "prem"],
            "minutes_played": [90, 45],
            "tackles": [3, 1],
        },
    )
    return connection


@pytest.fixture
def connection(tmp_path: Path):
    """Yield a populated connection with the lookup views registered."""
    conn = _connection(tmp_path)
    lookups.register_lookups(conn)
    try:
        yield conn
    finally:
        conn.close()


def test_slug_frame_has_a_row_per_known_slug() -> None:
    """The constant is exposed as a two-column relation."""
    frame = lookups.slug_frame()
    assert frame.columns == ["slug", "team"]
    assert frame.height == len(lookups.FCI_SLUG_TO_FPL)
    row = frame.filter(pl.col("slug") == "manchester-united")
    assert row["team"].item() == "Man Utd"


def test_brighton_respellings_resolve_to_one_club() -> None:
    """FCI changed Brighton's slug between seasons; both must map."""
    frame = lookups.slug_frame()
    teams = frame.filter(
        pl.col("slug").is_in(
            ["brighton-&-hove-albion", "brighton-hove-albion"]
        )
    )["team"].to_list()
    assert teams == ["Brighton", "Brighton"]


def test_fpl_team_id_derives_ids_from_the_data(connection) -> None:
    """Team ids come from the fixtures, not a hardcoded list."""
    result = connection.sql(
        "SELECT team_id, team FROM fpl_team_id ORDER BY team_id"
    ).pl()
    assert result["team_id"].to_list() == [1, 18]
    assert result["team"].to_list() == ["Arsenal", "Spurs"]


def test_opta_match_resolves_each_leg_of_a_double_gameweek(
    connection,
) -> None:
    """Both legs keep their own opponent instead of being summed."""
    result = connection.sql(
        "SELECT opponent, is_home, minutes_played, tackles "
        "FROM opta_match ORDER BY opponent"
    ).pl()
    assert result["opponent"].to_list() == [1, 18]
    assert result["is_home"].to_list() == [True, False]
    assert result["tackles"].to_list() == [3, 1]


def test_opta_match_joins_to_player_match_without_fanning_out(
    connection,
) -> None:
    """One Opta row per player_match leg, matched on the primary key."""
    result = connection.sql(
        """
        SELECT m.opponent, m.minutes, o.tackles
        FROM player_match AS m
        LEFT JOIN opta_match AS o USING (season, gw, element, opponent)
        ORDER BY m.opponent
        """
    ).pl()
    assert result.height == 2
    assert result["tackles"].to_list() == [3, 1]


def test_opta_match_excludes_non_prem_competitions(connection) -> None:
    """European minutes must not leak into an FPL-shaped join."""
    _append(
        PLAYER_MATCH_OPTA,
        connection,
        {
            "season": ["2025-26"],
            "gw": [1],
            "element": [1],
            "match_id": ["25-26-champions-manchester-united-vs-arsenal"],
            "competition": ["champions"],
            "minutes_played": [90],
            "tackles": [5],
        },
    )
    total = connection.sql(
        "SELECT sum(minutes_played) FROM opta_match"
    ).fetchone()[0]
    assert total == 135


def test_unknown_slugs_flags_an_unmapped_club(connection) -> None:
    """A promoted club FCI publishes early is reported, not swallowed."""
    assert lookups.unknown_slugs(connection) == []
    _append(
        PLAYER_MATCH_OPTA,
        connection,
        {
            "season": ["2025-26"],
            "gw": [2],
            "element": [1],
            "match_id": ["25-26-prem-luton-town-vs-arsenal"],
            "competition": ["prem"],
            "minutes_played": [90],
        },
    )
    assert lookups.unknown_slugs(connection) == ["luton-town"]
