import polars as pl
import pytest

from fantasy_football.features.availability import (
    add_chance_of_playing,
    add_positional_availability,
    add_rolling_minutes,
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


def test_add_rolling_minutes_averages_prior_games(
    sample_minutes_data: pl.DataFrame,
) -> None:
    """The window averages prior gameweeks and excludes the current one."""
    result = add_rolling_minutes(sample_minutes_data, rolling_window=3)

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
) -> None:
    """A player's first game of a season has no prior data, so it is null."""
    result = add_rolling_minutes(sample_minutes_data, rolling_window=3)

    gw1 = result.filter(pl.col("gw") == 1)
    assert gw1.get_column("avg_minutes_rolling_3").item() is None


def test_add_rolling_minutes_partial_window(
    sample_minutes_data: pl.DataFrame,
) -> None:
    """Before the window fills, the average uses whatever prior games exist."""
    result = add_rolling_minutes(sample_minutes_data, rolling_window=3)

    by_gw = {
        row["gw"]: row["avg_minutes_rolling_3"]
        for row in result.iter_rows(named=True)
    }
    # GW2 has only GW1 prior: average is just 90.
    assert by_gw[2] == pytest.approx(90.0)
    # GW3 has GW1-2 prior: (90 + 60) / 2 = 75.
    assert by_gw[3] == pytest.approx(75.0)


def test_add_rolling_minutes_does_not_bleed_across_seasons() -> None:
    """The window resets each season and never mixes two players' minutes.

    The same ``element`` id refers to different players in different seasons,
    so the rolling window must partition by ``(season, element)``.
    """
    data = pl.DataFrame(
        {
            "season": ["2020-21", "2020-21", "2021-22", "2021-22"],
            "gw": [1, 2, 1, 2],
            "element": [1, 1, 1, 1],
            "minutes": [90, 90, 0, 30],
        }
    )

    result = add_rolling_minutes(data, rolling_window=3)

    by_key = {
        (row["season"], row["gw"]): row["avg_minutes_rolling_3"]
        for row in result.iter_rows(named=True)
    }
    # 2021-22 GW1 is a fresh season: no prior data despite 2020-21 rows.
    assert by_key[("2021-22", 1)] is None
    # 2021-22 GW2 averages only 2021-22 GW1 (0), not last season's 90s.
    assert by_key[("2021-22", 2)] == pytest.approx(0.0)


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
