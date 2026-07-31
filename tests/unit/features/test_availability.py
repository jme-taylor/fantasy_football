from datetime import datetime, timedelta

import polars as pl
import pytest

from fantasy_football.features.availability import (
    add_chance_of_playing,
    add_games_played_this_season,
    add_positional_availability,
    add_rolling_minutes,
)


def _codes(pairs: list[tuple[str, int, int]]) -> pl.DataFrame:
    """Build a minimal player_season identity frame.

    Parameters
    ----------
    pairs : list[tuple[str, int, int]]
        ``(season, element, player_code)`` triples.

    Returns
    -------
    pl.DataFrame
        Identity rows with ``season``, ``element`` and ``player_code``.
    """
    return pl.DataFrame(
        {
            "season": [p[0] for p in pairs],
            "element": [p[1] for p in pairs],
            "player_code": [p[2] for p in pairs],
        }
    )


@pytest.fixture
def sample_minutes_data() -> pl.DataFrame:
    """Sample player-week minutes spanning one player across five gameweeks.

    Returns
    -------
    pl.DataFrame
        Player-week rows with ``season``, ``gw``, ``element`` and ``minutes``.
    """
    return pl.DataFrame(
        {
            "season": ["2020-21"] * 5,
            "gw": [1, 2, 3, 4, 5],
            "element": [1, 1, 1, 1, 1],
            "minutes": [90, 60, 0, 45, 90],
        }
    )


@pytest.fixture
def sample_match_stream(sample_minutes_data: pl.DataFrame) -> pl.DataFrame:
    """Match-grain view of ``sample_minutes_data`` with kickoff times added.

    Returns
    -------
    pl.DataFrame
        ``sample_minutes_data`` with a ``kickoff_time`` column, one week
        apart and in the same order as ``gw``.
    """
    n = sample_minutes_data.height
    return sample_minutes_data.with_columns(
        kickoff_time=pl.Series(
            [datetime(2020, 8, 1) + timedelta(weeks=i) for i in range(n)]
        )
    )


@pytest.fixture
def sample_player_season() -> pl.DataFrame:
    """Identity frame mapping element 1 to player_code 1 in 2020-21.

    Returns
    -------
    pl.DataFrame
        A single ``(season, element, player_code)`` row.
    """
    return _codes([("2020-21", 1, 1)])


def test_add_rolling_minutes_averages_prior_games(
    sample_minutes_data: pl.DataFrame,
    sample_match_stream: pl.DataFrame,
    sample_player_season: pl.DataFrame,
) -> None:
    """The window averages prior gameweeks and excludes the current one."""
    result = add_rolling_minutes(
        sample_minutes_data,
        sample_match_stream,
        sample_player_season,
        rolling_window=3,
    )

    by_gw = {
        row["gw"]: row["avg_minutes_rolling_3"]
        for row in result.iter_rows(named=True)
    }
    # GW4 averages GW1-3 minutes: (90 + 60 + 0) / 3 = 50, GW4's own 45 excluded.
    assert by_gw[4] == pytest.approx(50.0)
    # GW5 averages GW2-4: (60 + 0 + 45) / 3 = 35.
    assert by_gw[5] == pytest.approx(35.0)


def test_add_rolling_minutes_null_for_first_game(
    sample_minutes_data: pl.DataFrame,
    sample_match_stream: pl.DataFrame,
    sample_player_season: pl.DataFrame,
) -> None:
    """A player's first game of a season has no prior data, so it is null."""
    result = add_rolling_minutes(
        sample_minutes_data,
        sample_match_stream,
        sample_player_season,
        rolling_window=3,
    )

    gw1 = result.filter(pl.col("gw") == 1)
    assert gw1.get_column("avg_minutes_rolling_3").item() is None


def test_add_rolling_minutes_partial_window(
    sample_minutes_data: pl.DataFrame,
    sample_match_stream: pl.DataFrame,
    sample_player_season: pl.DataFrame,
) -> None:
    """Before the window fills, the average uses whatever prior games exist."""
    result = add_rolling_minutes(
        sample_minutes_data,
        sample_match_stream,
        sample_player_season,
        rolling_window=3,
    )

    by_gw = {
        row["gw"]: row["avg_minutes_rolling_3"]
        for row in result.iter_rows(named=True)
    }
    # GW2 has only GW1 prior: average is just 90.
    assert by_gw[2] == pytest.approx(90.0)
    # GW3 has GW1-2 prior: (90 + 60) / 2 = 75.
    assert by_gw[3] == pytest.approx(75.0)


def test_rolling_minutes_reaches_across_the_season_boundary() -> None:
    """GW1 of a new season uses last season's final matches, not null."""
    stream = pl.DataFrame(
        {
            "season": ["2025-26"] * 3 + ["2026-27"],
            "gw": [36, 37, 38, 1],
            "element": [10, 10, 10, 55],
            "kickoff_time": [
                datetime(2026, 5, 3, 14, 0),
                datetime(2026, 5, 10, 14, 0),
                datetime(2026, 5, 17, 14, 0),
                datetime(2026, 8, 21, 19, 0),
            ],
            "minutes": [90, 60, 30, None],
        },
        schema_overrides={"minutes": pl.Int64},
    )
    # Same person: element 10 in 2025-26 becomes element 55 in 2026-27.
    codes = _codes([("2025-26", 10, 999), ("2026-27", 55, 999)])
    weeks = pl.DataFrame({"season": ["2026-27"], "gw": [1], "element": [55]})

    result = add_rolling_minutes(weeks, stream, codes, rolling_window=5)

    # (90 + 60 + 30) / 3 = 60.0 — carried across the summer break.
    assert result["avg_minutes_rolling_5"][0] == pytest.approx(60.0)


def test_rolling_minutes_does_not_pool_reused_element_ids() -> None:
    """841 element ids map to 2+ players; they must never be merged."""
    stream = pl.DataFrame(
        {
            "season": ["2025-26", "2026-27"],
            "gw": [38, 1],
            "element": [10, 10],
            "kickoff_time": [
                datetime(2026, 5, 17, 14, 0),
                datetime(2026, 8, 21, 19, 0),
            ],
            "minutes": [90, None],
        },
        schema_overrides={"minutes": pl.Int64},
    )
    # Different people who happen to share element id 10.
    codes = _codes([("2025-26", 10, 111), ("2026-27", 10, 222)])
    weeks = pl.DataFrame({"season": ["2026-27"], "gw": [1], "element": [10]})

    result = add_rolling_minutes(weeks, stream, codes, rolling_window=5)

    assert result["avg_minutes_rolling_5"][0] is None


def test_rolling_minutes_orders_by_kickoff_not_gameweek() -> None:
    """A rescheduled fixture sorts by when it was played, not its gw."""
    stream = pl.DataFrame(
        {
            "season": ["2026-27"] * 3,
            "gw": [5, 3, 6],
            "element": [1] * 3,
            # GW3 was postponed and played after GW5.
            "kickoff_time": [
                datetime(2026, 9, 12, 14, 0),
                datetime(2026, 9, 20, 14, 0),
                datetime(2026, 9, 26, 14, 0),
            ],
            "minutes": [90, 0, None],
        },
        schema_overrides={"minutes": pl.Int64},
    )
    codes = _codes([("2026-27", 1, 777)])
    weeks = pl.DataFrame({"season": ["2026-27"], "gw": [6], "element": [1]})

    result = add_rolling_minutes(weeks, stream, codes, rolling_window=5)

    # Both prior matches count regardless of gw order: (90 + 0) / 2 = 45.0
    assert result["avg_minutes_rolling_5"][0] == pytest.approx(45.0)


def test_rolling_minutes_excludes_null_player_codes() -> None:
    """Null codes must be dropped, never pooled into one partition."""
    stream = pl.DataFrame(
        {
            "season": ["2026-27"] * 2,
            "gw": [1, 1],
            "element": [1, 2],
            "kickoff_time": [datetime(2026, 8, 21, 19, 0)] * 2,
            "minutes": [90, 45],
        },
        schema_overrides={"minutes": pl.Int64},
    )
    codes = pl.DataFrame(
        {
            "season": ["2026-27", "2026-27"],
            "element": [1, 2],
            "player_code": [None, None],
        },
        schema_overrides={"player_code": pl.Int64},
    )
    weeks = pl.DataFrame(
        {"season": ["2026-27"] * 2, "gw": [1, 1], "element": [1, 2]}
    )

    result = add_rolling_minutes(weeks, stream, codes, rolling_window=5)

    assert result["avg_minutes_rolling_5"].null_count() == 2


def test_rolling_minutes_is_constant_across_many_unplayed_fixtures() -> None:
    """Every unplayed fixture carries the same frozen played-match window.

    A shifted window over the raw stream feeds one null in per unplayed
    fixture, so the mean shrinks and then goes null from the sixth one on.
    In a pre-season run every gameweek is unplayed, which would strip the
    feature from GW6-GW38. All ten forward rows must equal 65.0 -- the mean
    of the player's three played matches.
    """
    played = [60, 90, 45]
    minutes: list[int | None] = [*played, *([None] * 10)]
    base = datetime(2026, 8, 1, 15, 0)
    stream = pl.DataFrame(
        {
            "season": ["2026-27"] * len(minutes),
            "gw": list(range(1, len(minutes) + 1)),
            "element": [7] * len(minutes),
            "kickoff_time": [
                base + timedelta(weeks=i) for i in range(len(minutes))
            ],
            "minutes": minutes,
        },
        schema_overrides={"minutes": pl.Int64},
    )
    codes = _codes([("2026-27", 7, 999)])
    weeks = stream.select("season", "gw", "element")

    result = add_rolling_minutes(weeks, stream, codes, rolling_window=5).sort(
        "gw"
    )

    forward = result.filter(pl.col("gw") > len(played))
    assert forward.height == 10
    values = forward["avg_minutes_rolling_5"].to_list()
    # Constant across every forward fixture, however far ahead...
    assert len(set(values)) == 1
    # ...and equal to the mean of the last 5 played matches: (60+90+45)/3.
    assert values[0] == pytest.approx(65.0)


def test_rolling_minutes_played_rows_keep_shifted_window() -> None:
    """A played row never sees its own minutes, even with forward rows after."""
    minutes: list[int | None] = [60, 90, 45, None, None]
    base = datetime(2026, 8, 1, 15, 0)
    stream = pl.DataFrame(
        {
            "season": ["2026-27"] * 5,
            "gw": [1, 2, 3, 4, 5],
            "element": [7] * 5,
            "kickoff_time": [base + timedelta(weeks=i) for i in range(5)],
            "minutes": minutes,
        },
        schema_overrides={"minutes": pl.Int64},
    )
    codes = _codes([("2026-27", 7, 999)])
    weeks = stream.select("season", "gw", "element")

    result = add_rolling_minutes(weeks, stream, codes, rolling_window=5).sort(
        "gw"
    )
    by_gw = {
        row["gw"]: row["avg_minutes_rolling_5"]
        for row in result.iter_rows(named=True)
    }

    assert by_gw[1] is None
    assert by_gw[2] == pytest.approx(60.0)
    assert by_gw[3] == pytest.approx(75.0)


def test_add_chance_of_playing_joins_known_value() -> None:
    """A covered (season, gw, element) keeps its availability percentage."""
    data = pl.DataFrame(
        {"season": ["2022-23"], "gw": [1], "element": [10], "minutes": [90]}
    )
    availability = pl.DataFrame(
        {
            "season": ["2022-23"],
            "gw": [1],
            "element": [10],
            "chance_of_playing_this_round": [25],
        }
    )

    result = add_chance_of_playing(data, availability)
    assert result["chance_of_playing_this_round"].to_list() == [25]


def test_add_chance_of_playing_defaults_uncovered_to_100() -> None:
    """An (season, gw, element) absent from availability defaults to 100."""
    data = pl.DataFrame(
        {"season": ["2022-23"], "gw": [2], "element": [99], "minutes": [0]}
    )
    availability = pl.DataFrame(
        {
            "season": ["2022-23"],
            "gw": [1],
            "element": [10],
            "chance_of_playing_this_round": [25],
        }
    )

    result = add_chance_of_playing(data, availability)
    assert result["chance_of_playing_this_round"].to_list() == [100]
    assert result.height == 1


def test_add_chance_of_playing_mixed_rows_no_fan_out_or_key_leak() -> None:
    """Two rows — one matched, one unmatched — produce exactly two output rows.

    Asserts:
    - The matched row keeps its availability value (25).
    - The unmatched row defaults to 100.
    - No rows are dropped or duplicated (height == 2).
    - No duplicate key columns (season_right, gw_right, element_right) appear.
    """
    data = pl.DataFrame(
        {
            "season": ["2022-23", "2022-23"],
            "gw": [1, 2],
            "element": [10, 99],
            "minutes": [90, 0],
        }
    )
    availability = pl.DataFrame(
        {
            "season": ["2022-23"],
            "gw": [1],
            "element": [10],
            "chance_of_playing_this_round": [25],
        }
    )

    result = add_chance_of_playing(data, availability)

    assert result.height == 2
    assert result["chance_of_playing_this_round"].to_list() == [25, 100]
    assert "season_right" not in result.columns
    assert "gw_right" not in result.columns
    assert "element_right" not in result.columns


def test_add_chance_of_playing_prefers_existing_non_null_value() -> None:
    """A non-null snapshot value on the incoming frame wins over the join.

    Forward rows carry the player snapshot's own
    ``chance_of_playing_this_round`` -- the only injury signal available
    for a season still being played, since ``player_availability``
    (fplcache) has no rows for it yet. That value must not be clobbered
    by a stale or absent availability-table join.
    """
    data = pl.DataFrame(
        {
            "season": ["2026-27"],
            "gw": [2],
            "element": [10],
            "chance_of_playing_this_round": [50],
        }
    )
    availability = pl.DataFrame(
        {
            "season": ["2026-27"],
            "gw": [2],
            "element": [10],
            "chance_of_playing_this_round": [100],
        }
    )

    result = add_chance_of_playing(data, availability)

    assert result["chance_of_playing_this_round"].to_list() == [50]


def test_add_chance_of_playing_falls_back_to_join_when_existing_is_null() -> (
    None
):
    """A null existing value still uses the availability join, not 100."""
    data = pl.DataFrame(
        {
            "season": ["2022-23"],
            "gw": [1],
            "element": [10],
            "chance_of_playing_this_round": [None],
        },
        schema_overrides={"chance_of_playing_this_round": pl.Int64},
    )
    availability = pl.DataFrame(
        {
            "season": ["2022-23"],
            "gw": [1],
            "element": [10],
            "chance_of_playing_this_round": [25],
        }
    )

    result = add_chance_of_playing(data, availability)

    assert result["chance_of_playing_this_round"].to_list() == [25]


def test_add_chance_of_playing_no_existing_column_behaves_as_before() -> None:
    """Absent the column entirely, behaviour is unchanged: join then default."""
    data = pl.DataFrame(
        {
            "season": ["2022-23", "2022-23"],
            "gw": [1, 2],
            "element": [10, 99],
        }
    )
    availability = pl.DataFrame(
        {
            "season": ["2022-23"],
            "gw": [1],
            "element": [10],
            "chance_of_playing_this_round": [25],
        }
    )

    result = add_chance_of_playing(data, availability)

    assert result["chance_of_playing_this_round"].to_list() == [25, 100]


@pytest.fixture
def sample_positional_data() -> pl.DataFrame:
    """One club+position group at a single gameweek, with mixed fitness.

    Arsenal DEF: element 1 (value 60, fit), element 2 (value 55, doubtful),
    element 3 (value 50, fit), element 4 (value 50, fit). Element 5 is an
    Arsenal MID and must never be counted as a DEF rival.
    """
    return pl.DataFrame(
        {
            "season": ["2024-25"] * 5,
            "gw": [1] * 5,
            "team": ["ARS"] * 5,
            "position": ["DEF", "DEF", "DEF", "DEF", "MID"],
            "element": [1, 2, 3, 4, 5],
            "value": [60, 55, 50, 50, 80],
            "chance_of_playing_this_round": [100, 50, 100, 100, 100],
        }
    )


def test_positional_availability_counts_fit_rivals_excluding_self(
    sample_positional_data: pl.DataFrame,
) -> None:
    """fit_rivals_same_pos counts fit same-position teammates, never self."""
    result = add_positional_availability(
        sample_positional_data, fit_threshold=75
    )
    by_element = {
        row["element"]: row["fit_rivals_same_pos"]
        for row in result.iter_rows(named=True)
    }
    # DEF group fit members: 1, 3, 4 (element 2 is doubtful at 50).
    # Element 1 is fit; its fit DEF rivals are 3 and 4 -> 2.
    assert by_element[1] == 2
    # Element 2 is doubtful; fit DEF rivals are 1, 3, 4 -> 3.
    assert by_element[2] == 3
    # Element 3 is fit; fit DEF rivals are 1 and 4 -> 2.
    assert by_element[3] == 2
    # Element 5 is the lone MID -> no same-position rivals.
    assert by_element[5] == 0


def test_positional_availability_ahead_uses_strict_value_and_fitness(
    sample_positional_data: pl.DataFrame,
) -> None:
    """fit_rivals_ahead counts only fit teammates with strictly higher value."""
    result = add_positional_availability(
        sample_positional_data, fit_threshold=75
    )
    by_element = {
        row["element"]: row["fit_rivals_ahead"]
        for row in result.iter_rows(named=True)
    }
    # Element 1 (value 60) is the most expensive DEF -> nobody ahead -> 0.
    assert by_element[1] == 0
    # Element 2 (value 55): only element 1 (60) is higher AND fit -> 1.
    assert by_element[2] == 1
    # Element 3 (value 50): higher-valued are 1 (fit) and 2 (doubtful) -> 1.
    assert by_element[3] == 1
    # Element 4 (value 50): equal-valued element 3 does NOT block (strict >);
    # higher-valued fit is only element 1 -> 1.
    assert by_element[4] == 1


def test_positional_availability_respects_threshold(
    sample_positional_data: pl.DataFrame,
) -> None:
    """Raising the threshold above a rival's chance drops them from the counts."""
    # At threshold 100, element 2 (chance 50) is already excluded; nothing changes
    # for the 100-chance players. Lower threshold to 50 to make element 2 count.
    result = add_positional_availability(
        sample_positional_data, fit_threshold=50
    )
    by_element = {
        row["element"]: row["fit_rivals_ahead"]
        for row in result.iter_rows(named=True)
    }
    # With element 2 (value 55) now fit, element 3 (value 50) has 1 (60) and
    # 2 (55) ahead -> 2.
    assert by_element[3] == 2


def test_positional_availability_fit_threshold_boundary() -> None:
    """chance_of_playing exactly equal to fit_threshold (75) counts as fit.

    Boundary conditions under the default threshold of 75:
    - A rival at exactly 75 IS counted (>= is inclusive).
    - A rival at 74 is NOT counted (strictly below threshold).
    """
    data = pl.DataFrame(
        {
            "season": ["2024-25", "2024-25", "2024-25"],
            "gw": [1, 1, 1],
            "team": ["ARS", "ARS", "ARS"],
            "position": ["MID", "MID", "MID"],
            "element": [1, 2, 3],
            "value": [70, 65, 60],
            "chance_of_playing_this_round": [100, 75, 74],
        }
    )
    result = add_positional_availability(data)
    by_element = {
        row["element"]: row["fit_rivals_same_pos"]
        for row in result.iter_rows(named=True)
    }
    # Elements 1 (100) and 2 (75) are fit; element 3 (74) is not.
    # Element 1: fit rivals are element 2 only -> 1.
    assert by_element[1] == 1
    # Element 2 (exactly 75, the boundary): fit rivals are element 1 only -> 1.
    assert by_element[2] == 1
    # Element 3 (74, just below threshold): fit rivals are elements 1 and 2 -> 2.
    assert by_element[3] == 2


def test_positional_availability_null_chance_counts_as_fit() -> None:
    """A null chance (upstream fills to 100) is treated as fit."""
    data = pl.DataFrame(
        {
            "season": ["2024-25", "2024-25"],
            "gw": [1, 1],
            "team": ["ARS", "ARS"],
            "position": ["DEF", "DEF"],
            "element": [1, 2],
            "value": [60, 50],
            "chance_of_playing_this_round": [None, 100],
        }
    )
    result = add_positional_availability(data, fit_threshold=75)
    by_element = {
        row["element"]: row["fit_rivals_same_pos"]
        for row in result.iter_rows(named=True)
    }
    # Element 2 sees element 1 as a fit rival because null is treated as fit.
    assert by_element[2] == 1


def test_add_games_played_excludes_the_current_gameweek() -> None:
    """A player's first gameweek has zero prior games, the second has one."""
    data = pl.DataFrame(
        {
            "season": ["2023-24"] * 3,
            "gw": [1, 2, 3],
            "element": [10, 10, 10],
            "minutes": [90, 60, 45],
        }
    )
    result = add_games_played_this_season(data).sort("gw")

    assert result["games_played_this_season"].to_list() == [0, 1, 2]


def test_add_games_played_resets_each_season() -> None:
    """The count restarts at zero in a new season."""
    data = pl.DataFrame(
        {
            "season": ["2023-24", "2023-24", "2024-25"],
            "gw": [1, 2, 1],
            "element": [10, 10, 55],
            "minutes": [90, 60, 45],
        }
    )
    result = add_games_played_this_season(data).sort(["season", "gw"])

    assert result["games_played_this_season"].to_list() == [0, 1, 0]


def test_games_played_freezes_across_unplayed_rows() -> None:
    """Rows with null minutes are future fixtures and must not count."""
    data = pl.DataFrame(
        {
            "season": ["2026-27"] * 6,
            "gw": [1, 2, 3, 4, 5, 6],
            "element": [1] * 6,
            "minutes": [90, 60, 45, None, None, None],
        },
        schema_overrides={"minutes": pl.Int64},
    )

    result = add_games_played_this_season(data)
    counts = {
        row["gw"]: row["games_played_this_season"]
        for row in result.iter_rows(named=True)
    }

    # Three played gameweeks, so every forward row sees exactly three.
    assert counts[1] == 0
    assert counts[4] == 3
    assert counts[5] == 3
    assert counts[6] == 3
