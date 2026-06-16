import json
import lzma
from datetime import datetime, timezone

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


def test_event_deadlines_parses_from_latest_snapshot(
    mocker: MockerFixture,
) -> None:
    """event_deadlines reads the events array from the latest snapshot."""
    extractor = _make_extractor()
    mocker.patch.object(
        extractor, "_latest_snapshot_path", return_value="cache/x.json.xz"
    )
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

    deadlines = extractor.event_deadlines()
    assert deadlines[1] == datetime(2025, 8, 15, 17, 30, tzinfo=timezone.utc)
    assert deadlines[2] == datetime(2025, 8, 22, 17, 30, tzinfo=timezone.utc)


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


def test_build_player_gw_team_emits_per_gw_team_code(
    mocker: MockerFixture,
) -> None:
    """A player's team_code reflects the snapshot active each gameweek."""
    extractor = _make_extractor()
    mocker.patch.object(
        extractor,
        "event_deadlines",
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

    result = extractor.build_player_gw_team([1, 2]).sort("gw")
    assert result.columns == ["gw", "element", "team_code"]
    assert result.to_dicts() == [
        {"gw": 1, "element": 82, "team_code": 91},
        {"gw": 2, "element": 82, "team_code": 43},
    ]
