"""Tests for the rolling per-90 match-form features."""

from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import polars as pl
import pytest

from fantasy_football.features import match_form
from fantasy_football.features.match_form import register_match_form
from fantasy_football.storage.database import get_connection
from fantasy_football.storage.lookups import register_lookups
from fantasy_football.storage.tables import (
    PLAYER_MATCH,
    PLAYER_MATCH_FPL,
    PLAYER_MATCH_OPTA,
    PLAYER_SEASON,
    PLAYER_WEEK,
    TEAM_FIXTURE,
)

SEASON = "2025-26"
FIRST_KICKOFF = datetime(2025, 8, 16, 14, 0)
# Opponent ids follow the alphabetical FPL ordering the fixtures below imply.
ARSENAL, SPURS = 1, 2


def _append(table, connection, data: dict) -> None:
    """Append a partial row set, filling unstated columns with nulls."""
    table.append(connection, table.conform(pl.DataFrame(data)))


def _build(
    tmp_path: Path,
    minutes: list[int],
    tackles: list[int | None],
    xg: list[float | None],
    yellow_cards: list[int] | None = None,
) -> duckdb.DuckDBPyConnection:
    """Return a connection holding one player's run of matches.

    Match ``i`` is in gameweek ``i + 1``, a week apart, alternating home
    and away against Arsenal and Spurs so each leg has a distinct
    opponent. Cards go to ``player_match_fpl``, the only source that
    publishes them.
    """
    connection = get_connection(tmp_path / "test.duckdb")
    count = len(minutes)
    gws = list(range(1, count + 1))
    kickoffs = [FIRST_KICKOFF + timedelta(days=7 * i) for i in range(count)]
    home = [i % 2 == 0 for i in range(count)]
    opponents = [ARSENAL if is_home else SPURS for is_home in home]

    _append(
        PLAYER_SEASON,
        connection,
        {"season": [SEASON], "element": [1], "player_code": [999]},
    )
    _append(
        PLAYER_WEEK,
        connection,
        {
            "season": [SEASON] * count,
            "gw": gws,
            "element": [1] * count,
            "position": ["DEF"] * count,
            "team": ["Man Utd"] * count,
        },
    )
    _append(
        PLAYER_MATCH,
        connection,
        {
            "season": [SEASON] * count,
            "gw": gws,
            "element": [1] * count,
            "opponent": opponents,
            "is_home": home,
            "minutes": minutes,
            "kickoff_time": kickoffs,
        },
    )
    _append(
        TEAM_FIXTURE,
        connection,
        {
            "season": [SEASON] * count,
            "gw": gws,
            "team": ["Man Utd"] * count,
            "is_home": home,
            "opposition": [
                "Arsenal" if is_home else "Spurs" for is_home in home
            ],
            "kickoff_time": kickoffs,
        },
    )
    _append(
        PLAYER_MATCH_OPTA,
        connection,
        {
            "season": [SEASON] * count,
            "gw": gws,
            "element": [1] * count,
            "match_id": [
                "25-26-prem-manchester-united-vs-arsenal"
                if is_home
                else "25-26-prem-tottenham-hotspur-vs-manchester-united"
                for is_home in home
            ],
            "competition": ["prem"] * count,
            "minutes_played": minutes,
            "tackles": tackles,
            "xg": xg,
        },
    )
    if yellow_cards is not None:
        _append(
            PLAYER_MATCH_FPL,
            connection,
            {
                "season": [SEASON] * count,
                "gw": gws,
                "element": [1] * count,
                "fixture": gws,
                "opponent_team": opponents,
                "minutes": minutes,
                "yellow_cards": yellow_cards,
                "red_cards": [0] * count,
            },
        )
    register_lookups(connection)
    return connection


@pytest.fixture
def connection(tmp_path: Path):
    """Yield a connection with three 90-minute matches, 3 tackles each."""
    conn = _build(
        tmp_path,
        minutes=[90, 90, 90],
        tackles=[3, 3, 3],
        xg=[0.2, 0.2, 0.2],
    )
    try:
        yield conn
    finally:
        conn.close()


def test_per90_column_name_matches_existing_convention() -> None:
    """Names line up with the repo's ``*_rolling_n`` feature columns."""
    assert match_form.per90_column_name("xg", 5) == "xg_per90_rolling_5"


def test_feature_columns_covers_every_stat_and_context_column() -> None:
    """Every declared stat produces one rate column."""
    columns = match_form.feature_columns(5)
    assert len(columns) == (
        len(match_form.PER90_STATS)
        + len(match_form.FPL_PER90_STATS)
        + len(match_form.CUMULATIVE_STATS)
        + len(match_form.FORM_CONTEXT_COLUMNS)
    )
    # No column may be emitted twice, whatever the stat lists hold.
    assert len(columns) == len(set(columns))
    assert columns[0] == "xg_per90_rolling_5"
    assert columns[-1] == "days_since_last_appearance"


def test_default_stats_are_all_valid() -> None:
    """The shipped stat list must itself pass the guard."""
    match_form.validate_stats(match_form.PER90_STATS)


def test_misspelled_stat_names_the_spelling_fix() -> None:
    """A typo fails before it reaches DuckDB's binder."""
    with pytest.raises(ValueError, match="Not player_match_opta columns"):
        match_form.validate_stats(("tackles_won_pct",))


def test_percentage_stat_is_rejected() -> None:
    """A percentage per 90 is a plausible-looking meaningless number."""
    with pytest.raises(ValueError, match="Per-90 is meaningless"):
        match_form.validate_stats(("tackles_won_percent",))


def test_opta_minutes_cannot_be_used_as_a_stat() -> None:
    """FPL minutes are the denominator; FCI's are not a numerator."""
    with pytest.raises(ValueError, match="Per-90 is meaningless"):
        match_form.validate_stats(("minutes_played",))


def test_never_populated_stat_is_rejected() -> None:
    """A column FCI declares but never fills would be null throughout."""
    with pytest.raises(ValueError, match="never populates"):
        match_form.validate_stats(("sprinting_distance",))


def test_covered_seasons_reports_the_usable_range() -> None:
    """A stat added mid-history narrows the range for the whole set."""
    assert "2024-25" in match_form.covered_seasons(["xg", "tackles"])
    # defensive_contributions only exists from 2025-26, so including it
    # pushes the first fully-covered season forward.
    assert "2024-25" not in match_form.covered_seasons(
        ["xg", "defensive_contributions"]
    )


def test_duplicate_stat_names_are_rejected() -> None:
    """A repeated stat would emit two columns of the same name."""
    with pytest.raises(ValueError, match="Listed more than once"):
        match_form.validate_stats(("xg", "xg"))


def test_card_stats_are_validated_against_the_fpl_table() -> None:
    """Cards live in player_match_fpl, not the Opta table."""
    match_form.validate_stats(match_form.FPL_PER90_STATS, source="fpl")
    # The same names are not Opta columns, and must not silently pass.
    with pytest.raises(ValueError, match="Not player_match_opta columns"):
        match_form.validate_stats(("yellow_cards",))


def test_first_match_has_no_prior_form(connection) -> None:
    """A window with nothing preceding it is null, not zero."""
    frame = match_form.load_match_form(connection)
    first = frame.filter(pl.col("gw") == 1)
    assert first["form_matches"].item() == 0
    assert first["tackles_per90_rolling_5"].item() is None


def test_rate_uses_prior_matches_only(connection) -> None:
    """The current match never contributes to its own rate."""
    frame = match_form.load_match_form(connection)
    second = frame.filter(pl.col("gw") == 2)
    # One prior match: 3 tackles in 90 minutes -> exactly 3.0 per 90.
    assert second["form_matches"].item() == 1
    assert second["tackles_per90_rolling_5"].item() == pytest.approx(3.0)
    assert second["form_minutes"].item() == 90


def test_rate_uses_player_match_minutes_not_opta_minutes(
    tmp_path: Path,
) -> None:
    """FPL minutes settle the denominator when the sources disagree."""
    connection = _build(
        tmp_path, minutes=[60, 90], tackles=[2, 0], xg=[0.0, 0.0]
    )
    # FCI claims 90 minutes for a match FPL records as 60.
    connection.execute(
        "UPDATE player_match_opta SET minutes_played = 90 WHERE gw = 1"
    )
    try:
        frame = match_form.load_match_form(connection)
        second = frame.filter(pl.col("gw") == 2)
        # 2 tackles over FPL's 60 minutes = 3.0, not 2.0 over FCI's 90.
        assert second["tackles_per90_rolling_5"].item() == pytest.approx(3.0)
    finally:
        connection.close()


def test_window_spans_appearances_not_fixtures(tmp_path: Path) -> None:
    """Benched fixtures do not consume window slots."""
    connection = _build(
        tmp_path,
        minutes=[90, 0, 0, 0, 0, 0, 90],
        tackles=[4, None, None, None, None, None, 0],
        xg=[0.0] * 7,
    )
    try:
        frame = match_form.load_match_form(connection)
        last = frame.filter(pl.col("gw") == 7)
        # The five 0-minute rows are excluded, so the GW1 appearance is
        # still the immediately preceding one.
        assert last["form_matches"].item() == 1
        assert last["tackles_per90_rolling_5"].item() == pytest.approx(4.0)
        assert last["days_since_last_appearance"].item() == 42
    finally:
        connection.close()


def test_stat_absent_from_a_match_is_left_out_of_its_denominator(
    tmp_path: Path,
) -> None:
    """A null stat means "not published", so it must not dilute the rate."""
    connection = _build(
        tmp_path, minutes=[90, 90, 90], tackles=[None, 4, 0], xg=[0.0] * 3
    )
    try:
        frame = match_form.load_match_form(connection)
        third = frame.filter(pl.col("gw") == 3)
        # GW1 published no tackles, so only GW2's 90 minutes count:
        # 4 tackles / 90 minutes = 4.0, not 4 / 180 = 2.0.
        assert third["tackles_per90_rolling_5"].item() == pytest.approx(4.0)
        # Both matches still count as form, and both supply minutes.
        assert third["form_matches"].item() == 2
        assert third["form_minutes"].item() == 180
    finally:
        connection.close()


def test_stat_never_published_yields_null_not_zero(tmp_path: Path) -> None:
    """A wholly-unpublished stat has an empty denominator."""
    connection = _build(
        tmp_path, minutes=[90, 90], tackles=[1, 1], xg=[0.1, 0.1]
    )
    try:
        frame = match_form.load_match_form(connection)
        second = frame.filter(pl.col("gw") == 2)
        # defensive_contributions is null throughout, as in 2024-25.
        assert second["defensive_contributions_per90_rolling_5"].item() is None
    finally:
        connection.close()


def test_null_player_codes_do_not_share_a_window(tmp_path: Path) -> None:
    """Two identity-less players must not average into each other."""
    connection = _build(
        tmp_path, minutes=[90, 90], tackles=[10, 10], xg=[0.0, 0.0]
    )
    try:
        # Strip the identity and add a second player with none either.
        connection.execute("DELETE FROM player_season")
        _append(
            PLAYER_WEEK,
            connection,
            {
                "season": [SEASON],
                "gw": [1],
                "element": [2],
                "position": ["DEF"],
                "team": ["Man Utd"],
            },
        )
        _append(
            PLAYER_MATCH,
            connection,
            {
                "season": [SEASON],
                "gw": [2],
                "element": [2],
                "opponent": [SPURS],
                "is_home": [False],
                "minutes": [90],
                "kickoff_time": [FIRST_KICKOFF + timedelta(days=7)],
            },
        )
        frame = match_form.load_match_form(connection)
        newcomer = frame.filter(pl.col("element") == 2)
        # Element 2's first ever match: no prior form of its own, and it
        # must not inherit element 1's 10-tackle history.
        assert newcomer["form_matches"].item() == 0
        assert newcomer["tackles_per90_rolling_5"].item() is None
    finally:
        connection.close()


def test_card_rate_and_running_total_use_prior_matches(
    tmp_path: Path,
) -> None:
    """Cards window exactly like every other stat: offset by one."""
    connection = _build(
        tmp_path,
        minutes=[90, 90, 90],
        tackles=[0, 0, 0],
        xg=[0.0] * 3,
        yellow_cards=[1, 0, 1],
    )
    try:
        frame = match_form.load_match_form(connection)
        third = frame.filter(pl.col("gw") == 3)
        # One booking across the two prior matches, 180 minutes.
        assert third["yellow_cards_per90_rolling_5"].item() == pytest.approx(
            0.5
        )
        assert third["yellow_cards_season_to_date"].item() == 1
        # The current match's own booking is in neither figure.
        assert third["red_cards_season_to_date"].item() == 0
    finally:
        connection.close()


def test_form_sql_inclusive_frame_includes_current_row():
    """The inclusive frame widens the window to the current row."""
    exclusive = match_form.form_sql(rolling_window=5, inclusive=False)
    inclusive = match_form.form_sql(rolling_window=5, inclusive=True)

    assert "ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING" in exclusive
    assert "ROWS BETWEEN 4 PRECEDING AND CURRENT ROW" in inclusive
    assert "AND 1 PRECEDING" not in inclusive


def test_inclusive_view_registers_under_its_own_name(connection):
    """The inclusive view is a separate, additional view, not a swap."""
    register_lookups(connection)
    register_match_form(connection, inclusive=False)
    register_match_form(connection, inclusive=True)

    names = {
        row[0]
        for row in connection.execute(
            "SELECT view_name FROM duckdb_views()"
        ).fetchall()
    }
    assert "player_match_form" in names
    assert "player_match_form_inclusive" in names


def test_running_total_starts_at_zero_not_null(tmp_path: Path) -> None:
    """A season's first appearance has genuinely accumulated nothing."""
    connection = _build(
        tmp_path,
        minutes=[90],
        tackles=[0],
        xg=[0.0],
        yellow_cards=[1],
    )
    try:
        frame = match_form.load_match_form(connection)
        first = frame.filter(pl.col("gw") == 1)
        # Zero, not null -- unlike the rolling rate, which has no evidence.
        assert first["yellow_cards_season_to_date"].item() == 0
        assert first["yellow_cards_per90_rolling_5"].item() is None
    finally:
        connection.close()
