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
    goals_prevented: list[float | None] | None = None,
    saves: list[int | None] | None = None,
    recoveries: list[int | None] | None = None,
) -> duckdb.DuckDBPyConnection:
    """Return a connection holding one player's run of matches.

    Match ``i`` is in gameweek ``i + 1``, a week apart, alternating home
    and away against Arsenal and Spurs so each leg has a distinct
    opponent. Cards go to ``player_match``, the spine that carries them
    for every season; passing ``None`` leaves them unfiled, which is what
    a season no source covers looks like.
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
            **(
                {
                    "yellow_cards": yellow_cards,
                    "red_cards": [0] * count,
                }
                if yellow_cards is not None
                else {}
            ),
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
            **(
                {"goals_prevented": goals_prevented}
                if goals_prevented is not None
                else {}
            ),
            **({"recoveries": recoveries} if recoveries is not None else {}),
        },
    )
    if saves is not None:
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
                "saves": saves,
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
        + len(match_form.CREATION_PER90_STATS)
        + len(match_form.DISCIPLINE_PER90_STATS)
        + len(match_form.FPL_PER90_STATS)
        + len(match_form.MATCH_PER90_STATS)
        + len(match_form.GK_FPL_PER90_STATS)
        + len(match_form.CUMULATIVE_STATS)
        + sum(
            len(variant.form_stats) for variant in match_form.DEFCON_VARIANTS
        )
        # Penalty exposure and the no-form flag, neither of which is a
        # windowed stat.
        + 2
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


def test_goalkeeper_stats_are_valid_against_their_own_source() -> None:
    """The keeper list must pass the guard for the table it comes from."""
    match_form.validate_stats(match_form.GK_FPL_PER90_STATS, source="fpl")


def test_goalkeeper_stats_stay_out_of_the_shared_lists() -> None:
    """Keeper stats are declared apart, so they cannot narrow the rest.

    ``covered_seasons`` over the shared list drives the other three
    positions' fold test seasons. A keeper stat folded in there could
    shrink their validation for a measure they never read.
    """
    shared = set(match_form.PER90_STATS) | set(match_form.FPL_PER90_STATS)
    assert not shared & set(match_form.GK_FPL_PER90_STATS)
    # The default is computed over PER90_STATS alone, so adding a keeper
    # stat cannot move it.
    assert match_form.covered_seasons() == match_form.covered_seasons(
        match_form.PER90_STATS
    )


def test_creation_stats_are_valid_against_their_source() -> None:
    """The creation list must pass the guard for FCI's table."""
    match_form.validate_stats(match_form.CREATION_PER90_STATS)


def test_creation_stats_stay_out_of_the_shared_lists() -> None:
    """Declared apart for the same reason the keeper stats are.

    Only the assists head reads them, and ``covered_seasons`` over the
    shared list drives three other models' fold test seasons. Today both
    creation stats have FCI's own coverage and would narrow nothing, but
    that is a property of these two columns rather than of the
    arrangement.
    """
    shared = set(match_form.PER90_STATS) | set(match_form.FPL_PER90_STATS)
    assert not shared & set(match_form.CREATION_PER90_STATS)
    # The comparison has to put the creation stats *in* to mean
    # anything: reading the default back is true by construction, since
    # covered_seasons defaults its argument to PER90_STATS.
    with_creation = match_form.covered_seasons(
        match_form.PER90_STATS + match_form.CREATION_PER90_STATS
    )
    assert set(match_form.covered_seasons()) >= set(with_creation)


def test_discipline_stats_are_valid_against_their_source() -> None:
    """The discipline list must pass the guard for FCI's table."""
    match_form.validate_stats(match_form.DISCIPLINE_PER90_STATS)


def test_discipline_stats_stay_out_of_the_shared_lists() -> None:
    """Declared apart for the reason the creation stats are.

    Only the yellow-cards head reads fouls, and ``covered_seasons`` over
    the shared list drives three other models' fold test seasons.
    """
    shared = set(match_form.PER90_STATS) | set(match_form.FPL_PER90_STATS)
    assert not shared & set(match_form.DISCIPLINE_PER90_STATS)
    with_discipline = match_form.covered_seasons(
        match_form.PER90_STATS + match_form.DISCIPLINE_PER90_STATS
    )
    assert set(match_form.covered_seasons()) >= set(with_discipline)


def test_cards_come_from_the_spine_not_a_provider_table() -> None:
    """The spine is the one relation filled for every season.

    Vaastav stops at 2025-26 and FCI has never published cards, so a
    provider-sourced card column is null for the live season.
    """
    assert set(match_form.MATCH_PER90_STATS) == {"yellow_cards", "red_cards"}
    assert not set(match_form.FPL_PER90_STATS) & {"yellow_cards", "red_cards"}
    match_form.validate_stats(
        match_form.MATCH_PER90_STATS, source="player_match"
    )
    match_form.validate_stats(
        match_form.CUMULATIVE_STATS, source="player_match"
    )


def test_the_assist_rate_comes_from_the_same_source_as_the_target() -> None:
    """FPL settles the assist, so FPL's count is what the rate windows.

    FCI counts a stricter event -- it logs no assist for a penalty won or
    a rebound -- so an FCI-sourced rate would describe a player by one
    definition and score him by another.
    """
    assert "assists" in match_form.FPL_PER90_STATS
    assert "assists" not in match_form.PER90_STATS
    match_form.validate_stats(match_form.FPL_PER90_STATS, source="fpl")


def test_goalkeeper_rate_uses_prior_matches_only(tmp_path: Path) -> None:
    """Keeper rates window the same way every other per-90 rate does."""
    connection = _build(
        tmp_path,
        minutes=[90, 90, 90],
        tackles=[0, 0, 0],
        xg=[0.0, 0.0, 0.0],
        saves=[3, 3, 3],
    )
    try:
        frame = match_form.load_match_form(connection).sort("gw")
    finally:
        connection.close()
    assert frame["saves_per90_rolling_5"].to_list()[0] is None
    assert frame["saves_per90_rolling_5"].to_list()[2] == 3.0


def test_card_stats_are_validated_against_the_fpl_table() -> None:
    """Cards live in player_match_fpl, not the Opta table."""
    match_form.validate_stats(match_form.FPL_PER90_STATS, source="fpl")
    # The same names are not Opta columns, and must not silently pass.
    with pytest.raises(ValueError, match="Not player_match_opta columns"):
        match_form.validate_stats(("yellow_cards",))


def test_view_exposes_the_rolling_identity_it_partitions_on(
    connection,
) -> None:
    """rolling_identity is a column, not just a window key.

    The forward-scoring path as-of joins an unplayed fixture back to the
    player's most recent appearance and has to match on the same key the
    window partitions by; if the view only computed it internally, that
    join would fall back to ``element`` and a player with no appearance
    yet this season would match nothing.
    """
    frame = match_form.load_match_form(connection)
    assert "rolling_identity" in frame.columns
    assert frame["rolling_identity"].unique().to_list() == ["999"]


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


def test_unplayed_fixture_carries_the_last_appearance_forward(
    tmp_path: Path,
) -> None:
    """A 0-minute row gets form, not nulls.

    Null form used to line up exactly with ``minutes = 0``, so the
    absence of a row told the model the player had not played -- an
    outcome it cannot know at prediction time.
    """
    connection = _build(
        tmp_path,
        minutes=[90, 90, 0],
        tackles=[4, 4, None],
        xg=[0.0] * 3,
    )
    try:
        frame = match_form.load_match_form(connection)
        benched = frame.filter(pl.col("gw") == 3)
        assert benched.height == 1
        assert benched["tackles_per90_rolling_5"].item() == pytest.approx(4.0)
        assert benched["form_matches"].item() == 2
        assert benched["days_since_last_appearance"].item() == 7
    finally:
        connection.close()


def test_every_fixture_gets_a_row(tmp_path: Path) -> None:
    """The view is one row per fixture, not one per appearance."""
    connection = _build(
        tmp_path, minutes=[90, 0, 0], tackles=[1, None, None], xg=[0.0] * 3
    )
    try:
        frame = match_form.load_match_form(connection)
        assert frame.height == 3
        assert sorted(frame["gw"].to_list()) == [1, 2, 3]
    finally:
        connection.close()


def test_carried_form_goes_stale_rather_than_disappearing(
    tmp_path: Path,
) -> None:
    """Staleness is reported, so a carried value is not read as fresh."""
    connection = _build(
        tmp_path,
        minutes=[90, 0, 0, 0, 0, 0, 0],
        tackles=[6, *([None] * 6)],
        xg=[0.0] * 7,
    )
    try:
        frame = match_form.load_match_form(connection).sort("gw")
        assert (
            frame["tackles_per90_rolling_5"].to_list()[1:]
            == [pytest.approx(6.0)] * 6
        )
        assert frame["days_since_last_appearance"].to_list() == [
            None,
            7,
            14,
            21,
            28,
            35,
            42,
        ]
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


def test_card_total_is_null_when_no_card_data_was_published(
    tmp_path: Path,
) -> None:
    """A source that filed no cards reads null, not a clean disciplinary record.

    Vaastav stopped publishing after 2025-26 and FCI never published
    cards at all, so for a season neither covers every card join misses.
    Coalescing that to zero made "we have no data" indistinguishable
    from "he has never been booked", and the four positional models read
    the column either way.
    """
    connection = _build(
        tmp_path,
        minutes=[90, 90],
        tackles=[0, 0],
        xg=[0.0] * 2,
        yellow_cards=None,
    )
    try:
        frame = match_form.load_match_form(connection)
        second = frame.filter(pl.col("gw") == 2)
        assert second["yellow_cards_season_to_date"].item() is None
        assert second["yellow_cards_per90_rolling_5"].item() is None
    finally:
        connection.close()


def test_form_sql_inclusive_frame_includes_current_row():
    """The inclusive frame widens the window to the current row.

    The exclusive view reaches the same window by as-of joining the
    previous appearance's inclusive figures, so it carries no
    ``1 PRECEDING`` frame of its own.
    """
    exclusive = match_form.form_sql(rolling_window=5, inclusive=False)
    inclusive = match_form.form_sql(rolling_window=5, inclusive=True)

    assert "ROWS BETWEEN 4 PRECEDING AND CURRENT ROW" in exclusive
    assert "ASOF LEFT JOIN" in exclusive
    assert "ROWS BETWEEN 4 PRECEDING AND CURRENT ROW" in inclusive
    assert "AND 1 PRECEDING" not in inclusive
    assert "ASOF" not in inclusive


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


def test_inclusive_view_rolling_rate_includes_the_current_appearance(
    tmp_path: Path,
) -> None:
    """The inclusive frame's per-90 rate reaches into its own row."""
    connection = _build(
        tmp_path, minutes=[90, 90, 90], tackles=[2, 4, 9], xg=[0.0] * 3
    )
    try:
        register_match_form(connection, inclusive=True)
        inclusive = connection.sql(
            "SELECT form_matches, tackles_per90_rolling_5 "
            "FROM player_match_form_inclusive WHERE gw = 3"
        ).fetchone()
        # All three appearances count, including gw3's own 9 tackles:
        # (2 + 4 + 9) tackles / 270 minutes = 5.0 per 90.
        assert inclusive[0] == 3
        assert inclusive[1] == pytest.approx(5.0)

        register_match_form(connection, inclusive=False)
        exclusive = connection.sql(
            "SELECT form_matches, tackles_per90_rolling_5 "
            "FROM player_match_form WHERE gw = 3"
        ).fetchone()
        # Only the two prior appearances count, excluding gw3's own 9:
        # (2 + 4) tackles / 180 minutes = 3.0 per 90.
        assert exclusive[0] == 2
        assert exclusive[1] == pytest.approx(3.0)
    finally:
        connection.close()


def test_inclusive_view_season_to_date_includes_the_current_match(
    tmp_path: Path,
) -> None:
    """The season-to-date total also reaches into the current match."""
    connection = _build(
        tmp_path,
        minutes=[90, 90, 90],
        tackles=[0, 0, 0],
        xg=[0.0] * 3,
        yellow_cards=[0, 0, 1],
    )
    try:
        register_match_form(connection, inclusive=True)
        inclusive_total = connection.sql(
            "SELECT yellow_cards_season_to_date "
            "FROM player_match_form_inclusive WHERE gw = 3"
        ).fetchone()[0]
        assert inclusive_total == 1

        register_match_form(connection, inclusive=False)
        exclusive_total = connection.sql(
            "SELECT yellow_cards_season_to_date "
            "FROM player_match_form WHERE gw = 3"
        ).fetchone()[0]
        assert exclusive_total == 0
    finally:
        connection.close()


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


# --- Defensive-contribution form --------------------------------------


def test_cbit_rate_sums_the_defender_counters(connection) -> None:
    """CBIT is the sum of the four counters FPL pays defenders for."""
    frame = match_form.load_match_form(connection)
    third = frame.filter(pl.col("gw") == 3)

    # The seeded run has 3 tackles per 90 and no other counter, so CBIT
    # is 3 per 90 over the two prior appearances.
    assert third["cbit_per90_rolling_5"].item() == pytest.approx(3.0)


def test_cbit_hit_rate_counts_appearances_clearing_the_threshold(
    tmp_path: Path,
) -> None:
    """The hit rate is the share of the window that reached ten."""
    conn = _build(
        tmp_path,
        minutes=[90, 90, 90],
        tackles=[12, 4, 3],
        xg=[0.0, 0.0, 0.0],
    )
    try:
        frame = match_form.load_match_form(conn)
        third = frame.filter(pl.col("gw") == 3)

        # One of the two prior appearances cleared ten.
        assert third["cbit_ten_plus_rate_rolling_5"].item() == pytest.approx(
            0.5
        )
    finally:
        conn.close()


def test_cbit_spread_separates_steady_from_streaky(tmp_path: Path) -> None:
    """Two defenders on the same mean differ in their spread.

    This is the threshold-specific signal a points regressor never
    needed: the same average CBIT can sit either side of the cliff edge
    depending on how much it varies.
    """
    steady = _build(
        tmp_path / "steady",
        minutes=[90, 90, 90],
        tackles=[8, 8, 0],
        xg=[0.0, 0.0, 0.0],
    )
    streaky = _build(
        tmp_path / "streaky",
        minutes=[90, 90, 90],
        tackles=[2, 14, 0],
        xg=[0.0, 0.0, 0.0],
    )
    try:
        steady_std = (
            match_form.load_match_form(steady)
            .filter(pl.col("gw") == 3)["cbit_std_rolling_5"]
            .item()
        )
        streaky_std = (
            match_form.load_match_form(streaky)
            .filter(pl.col("gw") == 3)["cbit_std_rolling_5"]
            .item()
        )

        assert steady_std == pytest.approx(0.0)
        assert streaky_std > steady_std
    finally:
        steady.close()
        streaky.close()


def test_cbit_is_null_when_no_counter_was_published(tmp_path: Path) -> None:
    """No FCI data is unknown CBIT, not zero CBIT.

    Zero would drag the rolling rate down and read as a quiet defender
    rather than as an unobserved one.
    """
    conn = _build(
        tmp_path,
        minutes=[90, 90],
        tackles=[None, None],
        xg=[0.0, 0.0],
    )
    try:
        frame = match_form.load_match_form(conn)
        second = frame.filter(pl.col("gw") == 2)

        assert second["cbit_per90_rolling_5"].item() is None
    finally:
        conn.close()


def test_defcon_form_columns_follow_the_window(tmp_path: Path) -> None:
    """The defcon columns are named for the window they cover.

    They were the only rates whose names were literals, so a changed
    ROLLING_WINDOW would have left them claiming five appearances while
    computing over another number.
    """
    assert match_form.CBIT_VARIANT.form_columns(5) == (
        "cbit_per90_rolling_5",
        "cbit_ten_plus_rate_rolling_5",
        "cbit_std_rolling_5",
    )
    assert match_form.CBIT_VARIANT.form_columns(3) == (
        "cbit_per90_rolling_3",
        "cbit_ten_plus_rate_rolling_3",
        "cbit_std_rolling_3",
    )


def test_cbirt_columns_are_named_for_their_own_threshold() -> None:
    """The midfield trio names its count and its threshold, not DEF's."""
    assert match_form.CBIRT_VARIANT.form_columns(5) == (
        "cbirt_per90_rolling_5",
        "cbirt_twelve_plus_rate_rolling_5",
        "cbirt_std_rolling_5",
    )


def test_cbirt_counts_recoveries_and_cbit_does_not(tmp_path: Path) -> None:
    """The two counts differ by exactly the recoveries.

    Recoveries are what separate the midfield and forward threshold of
    12 from the defender threshold of 10, so a variant that summed the
    same four counters twice would be two names for one feature.
    """
    conn = _build(
        tmp_path,
        minutes=[90, 90],
        tackles=[4, 4],
        xg=[0.0, 0.0],
        recoveries=[6, 6],
    )
    try:
        frame = match_form.load_match_form(conn)
        second = frame.filter(pl.col("gw") == 2)

        assert second["cbit_per90_rolling_5"].item() == pytest.approx(4.0)
        assert second["cbirt_per90_rolling_5"].item() == pytest.approx(10.0)
    finally:
        conn.close()


def test_cbirt_hit_rate_uses_the_twelve_threshold(tmp_path: Path) -> None:
    """Eleven does not clear twelve, though it clears ten."""
    conn = _build(
        tmp_path,
        minutes=[90, 90, 90],
        tackles=[5, 7, 0],
        xg=[0.0, 0.0, 0.0],
        recoveries=[6, 6, 0],
    )
    try:
        frame = match_form.load_match_form(conn)
        third = frame.filter(pl.col("gw") == 3)

        # Eleven then thirteen: one of the two cleared twelve, but both
        # cleared ten.
        assert third["cbirt_twelve_plus_rate_rolling_5"].item() == (
            pytest.approx(0.5)
        )
        assert third["cbit_ten_plus_rate_rolling_5"].item() == (
            pytest.approx(0.0)
        )
    finally:
        conn.close()


def test_cbirt_is_null_when_no_counter_was_published(tmp_path: Path) -> None:
    """No FCI data is unknown CBIRT, not zero CBIRT."""
    conn = _build(
        tmp_path,
        minutes=[90, 90],
        tackles=[None, None],
        xg=[0.0, 0.0],
        recoveries=[None, None],
    )
    try:
        frame = match_form.load_match_form(conn)
        second = frame.filter(pl.col("gw") == 2)

        assert second["cbirt_per90_rolling_5"].item() is None
    finally:
        conn.close()


def test_a_different_window_emits_the_renamed_defcon_columns(
    tmp_path: Path,
) -> None:
    """The view emits what feature_columns says it emits, at any window."""
    conn = _build(
        tmp_path,
        minutes=[90, 90, 90],
        tackles=[3, 3, 3],
        xg=[0.0, 0.0, 0.0],
    )
    try:
        register_match_form(conn, rolling_window=3)
        columns = (
            conn.sql("SELECT * FROM player_match_form LIMIT 0").pl().columns
        )

        for variant in match_form.DEFCON_VARIANTS:
            for name in variant.form_columns(3):
                assert name in columns
    finally:
        conn.close()


def _build_club(
    tmp_path: Path,
    pens: dict[int, list[int]],
    goals: dict[int, list[int]] | None = None,
) -> duckdb.DuckDBPyConnection:
    """Return a connection holding one club's two players over 3 matches.

    ``pens`` maps element to its penalties scored per gameweek. Both
    players turn out for Man Utd every week, which is what makes the
    club-level denominator behind the penalty share non-trivial.
    """
    connection = get_connection(tmp_path / "club.duckdb")
    gws = [1, 2, 3]
    kickoffs = [FIRST_KICKOFF + timedelta(days=7 * i) for i in range(3)]
    home = [True, False, True]
    opponents = [ARSENAL if is_home else SPURS for is_home in home]
    elements = sorted(pens)

    _append(
        PLAYER_SEASON,
        connection,
        {
            "season": [SEASON] * len(elements),
            "element": elements,
            "player_code": [900 + e for e in elements],
        },
    )
    _append(
        TEAM_FIXTURE,
        connection,
        {
            "season": [SEASON] * 3,
            "gw": gws,
            "team": ["Man Utd"] * 3,
            "is_home": home,
            "opposition": [
                "Arsenal" if is_home else "Spurs" for is_home in home
            ],
            "kickoff_time": kickoffs,
        },
    )
    for element in elements:
        _append(
            PLAYER_WEEK,
            connection,
            {
                "season": [SEASON] * 3,
                "gw": gws,
                "element": [element] * 3,
                "position": ["MID"] * 3,
                "team": ["Man Utd"] * 3,
            },
        )
        _append(
            PLAYER_MATCH,
            connection,
            {
                "season": [SEASON] * 3,
                "gw": gws,
                "element": [element] * 3,
                "opponent": opponents,
                "is_home": home,
                "minutes": [90] * 3,
                "kickoff_time": kickoffs,
            },
        )
        _append(
            PLAYER_MATCH_OPTA,
            connection,
            {
                "season": [SEASON] * 3,
                "gw": gws,
                "element": [element] * 3,
                "match_id": [
                    "25-26-prem-manchester-united-vs-arsenal"
                    if is_home
                    else "25-26-prem-tottenham-hotspur-vs-manchester-united"
                    for is_home in home
                ],
                "competition": ["prem"] * 3,
                "minutes_played": [90] * 3,
                "xg": [0.2] * 3,
                "penalties_scored": pens[element],
                "penalties_missed": [0] * 3,
            },
        )
        _append(
            PLAYER_MATCH_FPL,
            connection,
            {
                "season": [SEASON] * 3,
                "gw": gws,
                "element": [element] * 3,
                "fixture": gws,
                "opponent_team": opponents,
                "minutes": [90] * 3,
                "yellow_cards": [0] * 3,
                "red_cards": [0] * 3,
                "goals_scored": (goals or {}).get(element, [0] * 3),
            },
        )
    register_lookups(connection)
    return connection


def test_penalty_exposure_separates_the_taker_from_his_team_mate(
    tmp_path: Path,
) -> None:
    """The club's penalties concentrate on whoever actually takes them."""
    conn = _build_club(tmp_path, pens={1: [1, 1, 0], 2: [0, 0, 0]})
    try:
        frame = match_form.load_match_form(conn)
        third = frame.filter(pl.col("gw") == 3)
        taker = third.filter(pl.col("element") == 1)
        other = third.filter(pl.col("element") == 2)
        column = match_form.PENALTY_EXPOSURE_COLUMN
        assert taker[column].item() > other[column].item() > 0.0
    finally:
        conn.close()


def test_penalty_exposure_excludes_the_match_it_is_attached_to(
    tmp_path: Path,
) -> None:
    """A penalty won in gw3 cannot inform the gw3 feature."""
    conn = _build_club(tmp_path, pens={1: [0, 0, 5], 2: [0, 0, 0]})
    try:
        frame = match_form.load_match_form(conn)
        third = frame.filter((pl.col("gw") == 3) & (pl.col("element") == 1))
        assert third[match_form.PENALTY_EXPOSURE_COLUMN].item() == 0.0
    finally:
        conn.close()


def test_no_form_flag_marks_a_first_appearance(tmp_path: Path) -> None:
    """The flag is on when the window held nothing, and off after."""
    conn = _build_club(tmp_path, pens={1: [0, 0, 0]})
    try:
        frame = match_form.load_match_form(conn)
        flags = dict(
            zip(frame["gw"], frame[match_form.NO_FORM_COLUMN], strict=True)
        )
        assert flags[1] == 1.0
        assert flags[2] == 0.0
        assert flags[3] == 0.0
    finally:
        conn.close()


def test_goals_are_windowed_into_a_trailing_rate(tmp_path: Path) -> None:
    """The trailing goal rate reads FPL's count, offset by one match."""
    conn = _build_club(tmp_path, pens={1: [0, 0, 0]}, goals={1: [1, 2, 0]})
    try:
        frame = match_form.load_match_form(conn)
        third = frame.filter(pl.col("gw") == 3)
        # Three goals across two 90-minute matches.
        assert third["goals_scored_per90_rolling_5"].item() == pytest.approx(
            1.5
        )
    finally:
        conn.close()
