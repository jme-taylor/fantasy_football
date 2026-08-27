import json
import lzma
from datetime import datetime, timezone

import pytest
from pytest_mock import MockerFixture

from fantasy_football.extraction.extractor import GitHubAPIClient
from fantasy_football.extraction.fplcache import FplCacheExtractor


def _make_extractor() -> FplCacheExtractor:
    return FplCacheExtractor(
        api_client=GitHubAPIClient(
            api_key="k", owner="Randdalf", repo="fplcache", branch="main"
        )
    )


def test_read_snapshot_decodes_lzma_json(mocker: MockerFixture) -> None:
    """_read_snapshot downloads, LZMA-decompresses and JSON-parses a file."""
    payload = {"elements": [{"id": 1, "team_code": 43}]}
    compressed = lzma.compress(json.dumps(payload).encode())

    extractor = _make_extractor()
    response = mocker.Mock()
    response.content = compressed
    response.raise_for_status = mocker.Mock()
    mocker.patch(
        "fantasy_football.extraction.fplcache.requests.get",
        return_value=response,
    )

    result = extractor._read_snapshot("cache/2025/8/16/1250.json.xz")
    assert result == payload


def test_latest_snapshot_path_walks_descending(
    mocker: MockerFixture,
) -> None:
    """_latest_snapshot_path picks max year/month/day/time numerically."""
    extractor = _make_extractor()

    listings = {
        "cache": [{"name": "2024"}, {"name": "2025"}],
        "cache/2025": [{"name": "8"}, {"name": "12"}, {"name": "2"}],
        "cache/2025/12": [{"name": "9"}, {"name": "26"}],
        "cache/2025/12/26": [
            {"name": "0202.json.xz"},
            {"name": "1833.json.xz"},
            {"name": "1250.json.xz"},
        ],
    }
    mocker.patch.object(
        extractor.api_client,
        "get_file_details",
        side_effect=lambda path: listings[path],
    )

    assert extractor._latest_snapshot_path() == "cache/2025/12/26/1833.json.xz"


def test_snapshot_path_for_picks_first_at_or_after_deadline(
    mocker: MockerFixture,
) -> None:
    """The chosen snapshot is the earliest one at or after the deadline."""
    extractor = _make_extractor()
    mocker.patch.object(
        extractor,
        "_list_names",
        return_value=[
            "0202.json.xz",
            "0636.json.xz",
            "1250.json.xz",
            "1833.json.xz",
        ],
    )
    deadline = datetime(2025, 8, 16, 13, 0, tzinfo=timezone.utc)
    assert (
        extractor._snapshot_path_for(deadline)
        == "cache/2025/8/16/1833.json.xz"
    )


def test_snapshot_path_for_rolls_to_next_day(mocker: MockerFixture) -> None:
    """When no snapshot follows the deadline that day, roll to the next day."""
    extractor = _make_extractor()

    def fake_list(path: str) -> list[str]:
        if path == "cache/2025/8/16":
            return ["0202.json.xz", "0636.json.xz"]
        if path == "cache/2025/8/17":
            return ["0205.json.xz", "0640.json.xz"]
        return []

    mocker.patch.object(extractor, "_list_names", side_effect=fake_list)
    deadline = datetime(2025, 8, 16, 17, 30, tzinfo=timezone.utc)
    assert (
        extractor._snapshot_path_for(deadline)
        == "cache/2025/8/17/0205.json.xz"
    )


def test_season_probe_datetime_uses_october_of_start_year() -> None:
    """The probe lands inside the season so its snapshot has all deadlines."""
    extractor = _make_extractor()
    assert extractor._season_probe_datetime("2022-23") == datetime(
        2022, 10, 1, tzinfo=timezone.utc
    )


def test_season_event_deadlines_reads_in_season_snapshot(
    mocker: MockerFixture,
) -> None:
    """Deadlines come from a snapshot within the requested season."""
    extractor = _make_extractor()
    mocker.patch.object(
        extractor, "_snapshot_path_for", return_value="cache/in-season.json.xz"
    )
    mocker.patch.object(
        extractor,
        "_read_snapshot",
        return_value={
            "events": [
                {"id": 1, "deadline_time": "2022-08-05T17:30:00Z"},
                {"id": 2, "deadline_time": "2022-08-13T10:00:00Z"},
            ]
        },
    )

    deadlines = extractor.season_event_deadlines("2022-23")
    assert deadlines[1] == datetime(2022, 8, 5, 17, 30, tzinfo=timezone.utc)
    assert deadlines[2] == datetime(2022, 8, 13, 10, 0, tzinfo=timezone.utc)


def test_season_event_deadlines_falls_back_to_latest_snapshot(
    mocker: MockerFixture,
) -> None:
    """If no in-season probe snapshot exists, use the latest snapshot."""
    extractor = _make_extractor()
    mocker.patch.object(
        extractor, "_snapshot_path_for", side_effect=ValueError("none")
    )
    latest = mocker.patch.object(
        extractor, "_latest_snapshot_path", return_value="cache/latest.json.xz"
    )
    mocker.patch.object(
        extractor,
        "_read_snapshot",
        return_value={
            "events": [{"id": 1, "deadline_time": "2025-08-15T17:30:00Z"}]
        },
    )

    deadlines = extractor.season_event_deadlines("2025-26")
    assert deadlines[1] == datetime(2025, 8, 15, 17, 30, tzinfo=timezone.utc)
    assert deadlines[1].year == 2025
    latest.assert_called_once()


def test_build_player_chance_of_playing_coalesces_null_to_100(
    mocker: MockerFixture,
) -> None:
    """A null chance means no injury doubt and is stored as 100."""
    extractor = _make_extractor()
    mocker.patch.object(
        extractor,
        "season_event_deadlines",
        return_value={
            1: datetime(2022, 8, 5, 17, 30, tzinfo=timezone.utc),
            2: datetime(2022, 8, 13, 10, 0, tzinfo=timezone.utc),
        },
    )
    mocker.patch.object(
        extractor,
        "_snapshot_path_for",
        side_effect=lambda d: f"cache/{d.day}.json.xz",
    )
    snapshots = {
        "cache/5.json.xz": {
            "elements": [
                {"id": 10, "chance_of_playing_this_round": None},
                {"id": 11, "chance_of_playing_this_round": 25},
            ]
        },
        "cache/13.json.xz": {
            "elements": [
                {"id": 10, "chance_of_playing_this_round": 100},
                {"id": 11, "chance_of_playing_this_round": None},
            ]
        },
    }
    mocker.patch.object(
        extractor, "_read_snapshot", side_effect=lambda p: snapshots[p]
    )

    result = extractor.build_player_chance_of_playing("2022-23").sort(
        ["gw", "element"]
    )
    assert result.columns == [
        "season",
        "gw",
        "element",
        "chance_of_playing_this_round",
    ]
    assert result.to_dicts() == [
        {
            "season": "2022-23",
            "gw": 1,
            "element": 10,
            "chance_of_playing_this_round": 100,
        },
        {
            "season": "2022-23",
            "gw": 1,
            "element": 11,
            "chance_of_playing_this_round": 25,
        },
        {
            "season": "2022-23",
            "gw": 2,
            "element": 10,
            "chance_of_playing_this_round": 100,
        },
        {
            "season": "2022-23",
            "gw": 2,
            "element": 11,
            "chance_of_playing_this_round": 100,
        },
    ]


def test_build_player_chance_of_playing_skips_missing_snapshot(
    mocker: MockerFixture,
) -> None:
    """A gameweek with no snapshot yet is skipped, not fatal."""
    extractor = _make_extractor()
    mocker.patch.object(
        extractor,
        "season_event_deadlines",
        return_value={
            1: datetime(2025, 8, 15, 17, 30, tzinfo=timezone.utc),
            2: datetime(2025, 8, 22, 17, 30, tzinfo=timezone.utc),
        },
    )

    def fake_path(deadline: datetime) -> str:
        if deadline.day == 22:
            raise ValueError("no snapshot yet")
        return "cache/15.json.xz"

    mocker.patch.object(extractor, "_snapshot_path_for", side_effect=fake_path)
    mocker.patch.object(
        extractor,
        "_read_snapshot",
        return_value={
            "elements": [{"id": 10, "chance_of_playing_this_round": 75}]
        },
    )

    result = extractor.build_player_chance_of_playing("2025-26")
    assert result.to_dicts() == [
        {
            "season": "2025-26",
            "gw": 1,
            "element": 10,
            "chance_of_playing_this_round": 75,
        }
    ]


def test_build_player_gw_team_emits_per_gw_team_code(
    mocker: MockerFixture,
) -> None:
    """A player's team_code reflects the snapshot active each gameweek."""
    extractor = _make_extractor()
    mocker.patch.object(
        extractor,
        "season_event_deadlines",
        return_value={
            1: datetime(2025, 8, 15, 17, 30, tzinfo=timezone.utc),
            2: datetime(2025, 8, 22, 17, 30, tzinfo=timezone.utc),
        },
    )
    mocker.patch.object(
        extractor,
        "_snapshot_path_for",
        side_effect=lambda d: f"cache/{d.day}.json.xz",
    )
    snapshots = {
        "cache/15.json.xz": {"elements": [{"id": 82, "team_code": 91}]},
        "cache/22.json.xz": {"elements": [{"id": 82, "team_code": 43}]},
    }
    mocker.patch.object(
        extractor, "_read_snapshot", side_effect=lambda p: snapshots[p]
    )

    result = extractor.build_player_gw_team("2025-26", [1, 2]).sort("gw")
    assert result.columns == ["gw", "element", "team_code"]
    assert result.to_dicts() == [
        {"gw": 1, "element": 82, "team_code": 91},
        {"gw": 2, "element": 82, "team_code": 43},
    ]


def test_build_player_gw_team_uses_the_requested_seasons_deadlines(
    mocker: MockerFixture,
) -> None:
    """Deadlines come from the season being built, not the latest snapshot."""
    extractor = _make_extractor()
    season_deadlines = mocker.patch.object(
        extractor,
        "season_event_deadlines",
        return_value={1: datetime(2025, 8, 15, 17, 30, tzinfo=timezone.utc)},
    )
    mocker.patch.object(
        extractor, "_snapshot_path_for", return_value="cache/15.json.xz"
    )
    mocker.patch.object(
        extractor,
        "_read_snapshot",
        return_value={"elements": [{"id": 82, "team_code": 91}]},
    )

    extractor.build_player_gw_team("2025-26", [1])
    season_deadlines.assert_called_once_with("2025-26")


def test_build_player_gw_team_skips_missing_snapshot(
    mocker: MockerFixture,
) -> None:
    """A gameweek with no snapshot yet is skipped, not fatal."""
    extractor = _make_extractor()
    mocker.patch.object(
        extractor,
        "season_event_deadlines",
        return_value={
            1: datetime(2025, 8, 15, 17, 30, tzinfo=timezone.utc),
            2: datetime(2025, 8, 22, 17, 30, tzinfo=timezone.utc),
        },
    )

    def fake_path(deadline: datetime) -> str:
        if deadline.day == 22:
            raise ValueError("no snapshot yet")
        return "cache/15.json.xz"

    mocker.patch.object(extractor, "_snapshot_path_for", side_effect=fake_path)
    mocker.patch.object(
        extractor,
        "_read_snapshot",
        return_value={"elements": [{"id": 82, "team_code": 91}]},
    )

    result = extractor.build_player_gw_team("2025-26", [1, 2])
    assert result.to_dicts() == [{"gw": 1, "element": 82, "team_code": 91}]


def test_build_player_gw_team_raises_when_no_snapshots(
    mocker: MockerFixture,
) -> None:
    """No snapshot for any gameweek is fatal, not a silent empty mapping."""
    extractor = _make_extractor()
    mocker.patch.object(
        extractor,
        "season_event_deadlines",
        return_value={
            1: datetime(2026, 8, 21, 17, 30, tzinfo=timezone.utc),
            2: datetime(2026, 8, 28, 17, 30, tzinfo=timezone.utc),
        },
    )
    mocker.patch.object(
        extractor,
        "_snapshot_path_for",
        side_effect=ValueError("no snapshot yet"),
    )

    with pytest.raises(ValueError, match="2025-26"):
        extractor.build_player_gw_team("2025-26", [1, 2])


def test_season_event_deadlines_rejects_wrong_season(
    mocker: MockerFixture,
) -> None:
    """Reject deadline maps from a neighbouring season.

    A deadline map from a neighbouring season raises rather than silently
    passing off last season's dates as this season's.

    Parameters
    ----------
    mocker : MockerFixture
        Pytest fixture for mocking.
    """
    extractor = _make_extractor()
    mocker.patch.object(
        extractor, "_snapshot_path_for", side_effect=ValueError("none")
    )
    mocker.patch.object(
        extractor, "_latest_snapshot_path", return_value="cache/x"
    )
    # 2025-26 deadlines returned while 2026-27 was requested.
    mocker.patch.object(
        extractor,
        "_read_snapshot",
        return_value={
            "events": [
                {"id": 1, "deadline_time": "2025-08-15T17:30:00Z"},
                {"id": 2, "deadline_time": "2025-08-22T17:30:00Z"},
            ]
        },
    )
    with pytest.raises(ValueError, match="2026-27"):
        extractor.season_event_deadlines("2026-27")


def test_season_event_deadlines_accepts_correct_season(
    mocker: MockerFixture,
) -> None:
    """An August-to-May deadline map for the requested season is returned.

    Parameters
    ----------
    mocker : MockerFixture
        Pytest fixture for mocking.
    """
    extractor = _make_extractor()
    mocker.patch.object(
        extractor, "_snapshot_path_for", side_effect=ValueError("none")
    )
    mocker.patch.object(
        extractor, "_latest_snapshot_path", return_value="cache/x"
    )
    mocker.patch.object(
        extractor,
        "_read_snapshot",
        return_value={
            "events": [
                {"id": 1, "deadline_time": "2026-08-21T17:30:00Z"},
                {"id": 38, "deadline_time": "2027-05-30T13:30:00Z"},
            ]
        },
    )
    deadlines = extractor.season_event_deadlines("2026-27")
    assert deadlines[1] == datetime(2026, 8, 21, 17, 30, tzinfo=timezone.utc)
    assert deadlines[38] == datetime(2027, 5, 30, 13, 30, tzinfo=timezone.utc)


_DEADLINES = {
    1: datetime(2026, 8, 21, 17, 30, tzinfo=timezone.utc),
    2: datetime(2026, 8, 28, 17, 30, tzinfo=timezone.utc),
    3: datetime(2026, 9, 12, 10, 0, tzinfo=timezone.utc),
}


def test_played_gameweeks_empty_before_season_starts(
    mocker: MockerFixture,
) -> None:
    """Every deadline in the future yields no played gameweeks.

    Parameters
    ----------
    mocker : MockerFixture
        Pytest fixture for mocking.
    """
    extractor = _make_extractor()
    mocker.patch.object(
        extractor, "season_event_deadlines", return_value=_DEADLINES
    )
    now = datetime(2026, 7, 30, 12, 0, tzinfo=timezone.utc)
    assert extractor.played_gameweeks("2026-27", [1, 2, 3], now=now) == []


def test_played_gameweeks_includes_passed_deadlines_only(
    mocker: MockerFixture,
) -> None:
    """Mid-season, only gameweeks whose deadline has passed are returned.

    Parameters
    ----------
    mocker : MockerFixture
        Pytest fixture for mocking.
    """
    extractor = _make_extractor()
    mocker.patch.object(
        extractor, "season_event_deadlines", return_value=_DEADLINES
    )
    now = datetime(2026, 8, 29, 9, 0, tzinfo=timezone.utc)
    assert extractor.played_gameweeks("2026-27", [1, 2, 3], now=now) == [1, 2]


def test_played_gameweeks_includes_gameweek_at_its_deadline(
    mocker: MockerFixture,
) -> None:
    """A gameweek exactly at its deadline counts as played (boundary).

    Parameters
    ----------
    mocker : MockerFixture
        Pytest fixture for mocking.
    """
    extractor = _make_extractor()
    mocker.patch.object(
        extractor, "season_event_deadlines", return_value=_DEADLINES
    )
    assert extractor.played_gameweeks(
        "2026-27", [1, 2, 3], now=_DEADLINES[1]
    ) == [1]


def test_played_gameweeks_skips_gameweeks_with_no_deadline(
    mocker: MockerFixture,
) -> None:
    """A gameweek absent from the deadline map is excluded, not an error.

    FCI can list a gameweek folder that FPL's events array does not carry.

    Parameters
    ----------
    mocker : MockerFixture
        Pytest fixture for mocking.
    """
    extractor = _make_extractor()
    mocker.patch.object(
        extractor, "season_event_deadlines", return_value=_DEADLINES
    )
    now = datetime(2027, 6, 1, tzinfo=timezone.utc)
    assert extractor.played_gameweeks("2026-27", [1, 2, 3, 99], now=now) == [
        1,
        2,
        3,
    ]


def test_deadline_prices_reads_the_gameweeks_deadline_snapshot(
    mocker: MockerFixture,
) -> None:
    """Prices come from the snapshot active at that gameweek's deadline.

    Parameters
    ----------
    mocker : MockerFixture
        Pytest fixture for mocking.
    """
    extractor = _make_extractor()
    mocker.patch.object(
        extractor, "season_event_deadlines", return_value=_DEADLINES
    )
    mocker.patch.object(
        extractor,
        "_snapshot_path_for",
        side_effect=lambda d: f"cache/{d.day}.json.xz",
    )
    snapshots = {
        "cache/21.json.xz": {
            "elements": [
                {"id": 55, "now_cost": 80},
                {"id": 226, "now_cost": 55},
            ]
        },
        "cache/28.json.xz": {"elements": [{"id": 55, "now_cost": 79}]},
    }
    mocker.patch.object(
        extractor, "_read_snapshot", side_effect=lambda p: snapshots[p]
    )

    assert extractor.deadline_prices("2026-27", 1) == {55: 80, 226: 55}
    assert extractor.deadline_prices("2026-27", 2) == {55: 79}


def test_deadline_prices_rejects_a_gameweek_with_no_deadline(
    mocker: MockerFixture,
) -> None:
    """A gameweek absent from the season's events cannot be priced.

    Parameters
    ----------
    mocker : MockerFixture
        Pytest fixture for mocking.
    """
    extractor = _make_extractor()
    mocker.patch.object(
        extractor, "season_event_deadlines", return_value=_DEADLINES
    )

    with pytest.raises(ValueError, match="GW99"):
        extractor.deadline_prices("2026-27", 99)
