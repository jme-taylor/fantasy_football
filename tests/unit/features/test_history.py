import polars as pl

from fantasy_football.features.history import add_history_features


def _player_season() -> pl.DataFrame:
    """Element ids deliberately differ per season, as FPL reassigns them."""
    return pl.DataFrame(
        {
            "season": ["2021-22", "2022-23", "2023-24", "2023-24"],
            "element": [10, 20, 30, 40],
            "player_code": [111, 111, 111, 999],
        }
    )


def _player_match() -> pl.DataFrame:
    """Player 111: two 2021-22 matches (one 60+), none in 2022-23."""
    return pl.DataFrame(
        {
            "season": ["2021-22", "2021-22", "2023-24", "2023-24"],
            "gw": [1, 2, 1, 1],
            "element": [10, 10, 30, 40],
            "opponent": [1, 2, 3, 3],
            "minutes": [90, 20, 90, 90],
            "total_points": [8, 1, 5, 6],
        }
    )


def _player_week() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": ["2021-22", "2022-23", "2023-24", "2023-24"],
            "gw": [1, 1, 1, 1],
            "element": [10, 20, 30, 40],
            "minutes": [90, 0, 90, 90],
            "total_points": [8, 0, 5, 6],
        }
    )


def _row(frame: pl.DataFrame, season: str, element: int) -> dict:
    return frame.filter(
        (pl.col("season") == season) & (pl.col("element") == element)
    ).to_dicts()[0]


def test_consecutive_seasons_carry_history() -> None:
    """A player who played last season gets seasons_since_last_pl == 0."""
    out = add_history_features(
        _player_week(), _player_match(), _player_season()
    )

    row = _row(out, "2022-23", 20)
    assert row["seasons_since_last_pl"] == 0
    assert row["prev_season_minutes"] == 110
    assert row["is_pl_newcomer"] is False


def test_start_rate_is_over_matches_not_gameweeks() -> None:
    """One 60+ appearance in two matches is a 0.5 start rate."""
    out = add_history_features(
        _player_week(), _player_match(), _player_season()
    )

    assert _row(out, "2022-23", 20)["prev_season_start_rate"] == 0.5


def test_points_per_start_averages_over_sixty_plus_matches_only() -> None:
    """The 20-minute, 1-point match is excluded from points_per_start."""
    out = add_history_features(
        _player_week(), _player_match(), _player_season()
    )

    assert _row(out, "2022-23", 20)["prev_season_points_per_start"] == 8.0


def test_gap_season_reports_distance_and_last_played_history() -> None:
    """After a blank 2022-23, 2023-24 sees a gap of 1 and 2021-22's history."""
    out = add_history_features(
        _player_week(), _player_match(), _player_season()
    )

    row = _row(out, "2023-24", 30)
    assert row["seasons_since_last_pl"] == 1
    assert row["prev_season_minutes"] == 110
    assert row["pl_seasons_played"] == 1


def test_first_ever_season_is_a_newcomer() -> None:
    """Player 999 debuts in 2023-24: newcomer, null history, no gap value."""
    out = add_history_features(
        _player_week(), _player_match(), _player_season()
    )

    row = _row(out, "2023-24", 40)
    assert row["is_pl_newcomer"] is True
    assert row["prev_season_minutes"] is None
    assert row["prev_season_start_rate"] is None
    assert row["seasons_since_last_pl"] is None
    assert row["pl_seasons_played"] == 0


def test_earliest_season_yields_nulls_without_raising() -> None:
    """The first season in the frame has no prior season; this is not an error."""
    out = add_history_features(
        _player_week(), _player_match(), _player_season()
    )

    row = _row(out, "2021-22", 10)
    assert row["prev_season_minutes"] is None
    assert row["is_pl_newcomer"] is True


def test_player_code_is_attached() -> None:
    """The output carries player_code so downstream windows can group on it."""
    out = add_history_features(
        _player_week(), _player_match(), _player_season()
    )

    assert _row(out, "2023-24", 30)["player_code"] == 111


def test_most_recent_of_multiple_prior_seasons_wins() -> None:
    """With two qualifying prior seasons, the more recent one's numbers win.

    Player 555 appears in 2019-20 and 2020-21, is blank in 2021-22 and
    2022-23, then reappears in 2023-24. If the "most recent" selection
    silently regressed to "any" prior season, this would pick up 2019-20's
    very different minutes total instead of 2020-21's.
    """
    player_season = _player_season().vstack(
        pl.DataFrame(
            {
                "season": ["2019-20", "2020-21", "2023-24"],
                "element": [1, 2, 5],
                "player_code": [555, 555, 555],
            }
        )
    )
    player_match = _player_match().vstack(
        pl.DataFrame(
            {
                "season": ["2019-20", "2020-21", "2020-21"],
                "gw": [1, 1, 2],
                "element": [1, 2, 2],
                "opponent": [9, 9, 8],
                "minutes": [90, 90, 45],
                "total_points": [3, 10, 2],
            }
        )
    )
    player_week = _player_week().vstack(
        pl.DataFrame(
            {
                "season": ["2023-24"],
                "gw": [1],
                "element": [5],
                "minutes": [90],
                "total_points": [7],
            }
        )
    )

    out = add_history_features(player_week, player_match, player_season)

    row = _row(out, "2023-24", 5)
    # 2020-21 (90 + 45 minutes, one of two matches a start), not 2019-20's
    # single 90-minute, 3-point match.
    assert row["prev_season_minutes"] == 135
    assert row["prev_season_start_rate"] == 0.5
    assert row["seasons_since_last_pl"] == 2


def test_points_per_start_is_null_without_any_sixty_plus_match() -> None:
    """A prior season with no 60+ minute match yields a null points-per-start."""
    player_season = _player_season().vstack(
        pl.DataFrame(
            {
                "season": ["2021-22", "2022-23"],
                "element": [50, 60],
                "player_code": [777, 777],
            }
        )
    )
    player_match = _player_match().vstack(
        pl.DataFrame(
            {
                "season": ["2021-22", "2021-22"],
                "gw": [1, 2],
                "element": [50, 50],
                "opponent": [4, 5],
                "minutes": [30, 45],
                "total_points": [1, 2],
            }
        )
    )
    player_week = _player_week().vstack(
        pl.DataFrame(
            {
                "season": ["2022-23"],
                "gw": [1],
                "element": [60],
                "minutes": [0],
                "total_points": [0],
            }
        )
    )

    out = add_history_features(player_week, player_match, player_season)

    row = _row(out, "2022-23", 60)
    assert row["prev_season_minutes"] == 75
    assert row["prev_season_start_rate"] == 0.0
    assert row["prev_season_points_per_start"] is None


def test_double_gameweek_does_not_inflate_start_rate() -> None:
    """Two matches in one gameweek count as two matches, not one."""
    matches = _player_match().vstack(
        pl.DataFrame(
            {
                "season": ["2021-22"],
                "gw": [2],
                "element": [10],
                "opponent": [7],
                "minutes": [90],
                "total_points": [6],
            }
        )
    )
    out = add_history_features(_player_week(), matches, _player_season())

    # Three matches, two of them 60+.
    assert _row(out, "2022-23", 20)["prev_season_start_rate"] == 2 / 3


def test_cold_start_features_compute_age_and_join_recency() -> None:
    """Age and days-since-joining come from the bio dates and the season start."""
    from datetime import date

    from fantasy_football.features.history import add_cold_start_features

    data = pl.DataFrame(
        {
            "season": ["2023-24"],
            "gw": [1],
            "element": [30],
            "team": ["Liverpool"],
            "birth_date": [date(1992, 6, 15)],
            "team_join_date": [date(2023, 7, 1)],
        },
        schema_overrides={"birth_date": pl.Date, "team_join_date": pl.Date},
    )
    fixtures = pl.DataFrame(
        {
            "season": ["2022-23", "2023-24"],
            "gw": [1, 1],
            "team": ["Liverpool", "Liverpool"],
            "opposition": ["Arsenal", "Arsenal"],
        }
    )
    out = add_cold_start_features(data, fixtures)

    # Season start is anchored at 1 August of the starting year: 2023-08-01.
    assert round(out["age_years"].item(), 1) == 31.1
    assert out["days_since_team_join"].item() == 31.0
    assert out["is_promoted_club"].item() is False


def test_cold_start_flags_a_promoted_club() -> None:
    """A team with no fixtures in the prior season is newly promoted."""
    from datetime import date

    from fantasy_football.features.history import add_cold_start_features

    data = pl.DataFrame(
        {
            "season": ["2023-24"],
            "gw": [1],
            "element": [30],
            "team": ["Luton"],
            "birth_date": [date(1999, 1, 1)],
            "team_join_date": [None],
        },
        schema_overrides={"birth_date": pl.Date, "team_join_date": pl.Date},
    )
    fixtures = pl.DataFrame(
        {
            "season": ["2022-23", "2023-24"],
            "gw": [1, 1],
            "team": ["Liverpool", "Luton"],
            "opposition": ["Arsenal", "Arsenal"],
        }
    )
    out = add_cold_start_features(data, fixtures)

    assert out["is_promoted_club"].item() is True
    assert out["days_since_team_join"].item() is None


def test_cold_start_earliest_season_is_not_flagged_promoted() -> None:
    """With no prior season on record, promotion is unknowable, not True."""
    from datetime import date

    from fantasy_football.features.history import add_cold_start_features

    data = pl.DataFrame(
        {
            "season": ["2022-23"],
            "gw": [1],
            "element": [30],
            "team": ["Liverpool"],
            "birth_date": [date(1999, 1, 1)],
            "team_join_date": [None],
        },
        schema_overrides={"birth_date": pl.Date, "team_join_date": pl.Date},
    )
    fixtures = pl.DataFrame(
        {
            "season": ["2022-23"],
            "gw": [1],
            "team": ["Liverpool"],
            "opposition": ["Arsenal"],
        }
    )
    out = add_cold_start_features(data, fixtures)

    assert out["is_promoted_club"].item() is False
