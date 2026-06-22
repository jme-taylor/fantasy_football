import polars as pl
import pytest

from fantasy_football.features.availability import (
    add_chance_of_playing,
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
