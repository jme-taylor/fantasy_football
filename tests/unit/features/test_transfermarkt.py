from datetime import date, datetime

import polars as pl

from fantasy_football.features.transfermarkt import (
    add_transfermarkt_features,
    gameweek_kickoff,
)

SEASON = "2024-25"
# Two Newcastle keepers: element 1 is the incumbent, element 2 the
# replacement signed mid-August. GW1 and GW2 are played, GW3 is forward.
KICKOFFS = {
    1: datetime(2024, 8, 17, 14, 0),
    2: datetime(2024, 8, 24, 14, 0),
    3: datetime(2024, 8, 31, 14, 0),
}


def _player_week() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [SEASON] * 6,
            "gw": [1, 1, 2, 2, 3, 3],
            "element": [1, 2, 1, 2, 1, 2],
            "team": ["Newcastle"] * 6,
            "position": ["GK"] * 6,
            "minutes": [90, 0, 90, 10, None, None],
        },
        schema_overrides={"minutes": pl.Int64},
    )


def _match_stream() -> pl.DataFrame:
    """Played GW1-2 plus an unplayed GW3, as build_feature_frame builds it."""
    return pl.DataFrame(
        {
            "season": [SEASON] * 6,
            "gw": [1, 1, 2, 2, 3, 3],
            "element": [1, 2, 1, 2, 1, 2],
            "kickoff_time": [
                KICKOFFS[1],
                KICKOFFS[1],
                KICKOFFS[2],
                KICKOFFS[2],
                KICKOFFS[3],
                KICKOFFS[3],
            ],
            "minutes": [90, 0, 90, 10, None, None],
        },
        schema_overrides={"minutes": pl.Int64},
    )


def _player_season() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [SEASON] * 2,
            "element": [1, 2],
            "player_code": [100, 200],
        }
    )


def _tm_player() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "tm_player_id": ["a", "b"],
            "position": ["Goalkeeper", "Goalkeeper"],
        }
    )


def _tm_player_map() -> pl.DataFrame:
    return pl.DataFrame(
        {"player_code": [100, 200], "tm_player_id": ["a", "b"]}
    )


def _tm_market_value() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "tm_player_id": ["a", "a", "b"],
            "value_date": [
                date(2024, 1, 1),
                date(2024, 8, 20),
                date(2024, 8, 1),
            ],
            "value_eur": [5_000_000, 4_000_000, 20_000_000],
        }
    )


def _tm_transfer() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "tm_player_id": ["a", "b"],
            "joined_club": ["Newcastle", "Newcastle"],
            "transfer_date_parsed": [date(2021, 7, 1), date(2024, 8, 5)],
        }
    )


def _features(**overrides: pl.DataFrame) -> pl.DataFrame:
    frames = {
        "data": _player_week(),
        "match_stream": _match_stream(),
        "player_season": _player_season(),
        "tm_player": _tm_player(),
        "tm_player_map": _tm_player_map(),
        "tm_market_value": _tm_market_value(),
        "tm_transfer": _tm_transfer(),
    }
    frames.update(overrides)
    return add_transfermarkt_features(**frames).sort("element", "gw")


def _by_gw(frame: pl.DataFrame, element: int, column: str) -> dict[int, float]:
    rows = frame.filter(pl.col("element") == element)
    return {row["gw"]: row[column] for row in rows.iter_rows(named=True)}


def test_gameweek_kickoff_takes_the_earliest_of_a_double() -> None:
    """Both legs of a double gameweek share the gameweek's first kickoff."""
    stream = pl.DataFrame(
        {
            "season": [SEASON] * 2,
            "gw": [1, 1],
            "element": [1, 1],
            "kickoff_time": [
                datetime(2024, 8, 20, 19, 45),
                datetime(2024, 8, 17, 14, 0),
            ],
        }
    )

    out = gameweek_kickoff(stream)

    assert out.height == 1
    assert out["as_of_date"].to_list() == [date(2024, 8, 17)]


def test_tm_value_reads_the_valuation_current_at_kickoff() -> None:
    """A revaluation between gameweeks moves the feature, forward included."""
    frame = _features()

    values = _by_gw(frame, 1, "value_tm")

    # Revalued down on 20 August, between GW1 and GW2.
    assert values[1] == 5_000_000
    assert values[2] == 4_000_000
    assert values[3] == 4_000_000


def test_prev_game_minutes_freezes_on_an_unplayed_fixture() -> None:
    """A forward gameweek inherits the last played match, and stays there."""
    frame = _features()

    prev = _by_gw(frame, 1, "prev_game_minutes")
    prev_2 = _by_gw(frame, 1, "prev_game_minutes_2")

    assert prev[1] is None  # no earlier match
    assert prev[2] == 90  # GW1
    assert prev[3] == 90  # frozen at GW2, not advanced past it
    assert prev_2[3] == 90  # GW1


def test_days_since_prev_game_freezes_rather_than_growing() -> None:
    """The gap is between the last two *played* matches, never to kickoff.

    GW3 is a week after GW2, so a value that tracked the fixture's own
    kickoff would read 7 as well. The played gap is what is asserted here:
    both played gameweeks are 7 days apart, so the frozen value is 7 and
    would be 14 if it advanced to GW3's kickoff.
    """
    frame = _features()

    days = _by_gw(frame, 1, "days_since_prev_game")

    assert days[1] is None
    assert days[2] == 7
    assert days[3] == 7


def test_minutes_to_date_excludes_the_current_gameweek() -> None:
    """Season-to-date minutes accumulate strictly before each gameweek."""
    frame = _features()

    minutes = _by_gw(frame, 1, "minutes_to_date")

    assert minutes[1] == 0
    assert minutes[2] == 90
    assert minutes[3] == 180  # frozen: GW3 itself is unplayed


def test_position_value_rank_normalises_within_the_club_group() -> None:
    """The cheaper of two keepers ranks 1.0, the dearer 0.0."""
    frame = _features()

    assert _by_gw(frame, 1, "tm_pos_value_rank_norm")[1] == 1.0
    assert _by_gw(frame, 2, "tm_pos_value_rank_norm")[1] == 0.0
    assert _by_gw(frame, 1, "players_same_tm_pos")[1] == 2


def test_arrivals_see_a_signing_made_after_the_last_played_match() -> None:
    """The incumbent's rival count advances into the forward gameweek.

    This is the shape the branch exists for: a keeper whose club has just
    signed a more valuable replacement. The arrival is calendar-driven, so
    freezing it at the last played match would hide a summer signing from
    every pre-season prediction.
    """
    frame = _features()

    rivals = _by_gw(frame, 1, "rivals_joined_same_pos")
    higher = _by_gw(frame, 1, "higher_value_rivals_joined_same_pos")
    max_value = _by_gw(frame, 1, "max_rival_value_eur")
    days = _by_gw(frame, 1, "days_since_rival_joined")

    assert rivals[3] == 1
    assert higher[3] == 1
    assert max_value[3] == 20_000_000
    # 5 August to each gameweek's kickoff: the clock keeps running.
    assert [days[1], days[2], days[3]] == [12, 19, 26]


def test_arrivals_ignore_a_signing_outside_the_lookback() -> None:
    """The 2021 arrival is not competition for the 2024 signing."""
    frame = _features()

    assert _by_gw(frame, 2, "rivals_joined_same_pos")[1] == 0
    assert _by_gw(frame, 2, "max_rival_value_eur")[1] is None


def test_unmapped_player_keeps_null_features_and_its_row() -> None:
    """A player absent from Transfermarkt is never dropped, only nulled."""
    frame = _features(
        tm_player_map=pl.DataFrame(
            {"player_code": [100], "tm_player_id": ["a"]}
        )
    )

    assert frame.height == 6
    unmapped = frame.filter(pl.col("element") == 2)
    assert unmapped["tm_player_id"].null_count() == unmapped.height
    assert unmapped["value_tm"].null_count() == unmapped.height
    assert unmapped["rivals_joined_same_pos"].null_count() == unmapped.height
    # The mapped player still gets real values alongside them.
    assert _by_gw(frame, 1, "value_tm")[1] == 5_000_000
