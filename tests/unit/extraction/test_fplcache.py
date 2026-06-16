import json
import lzma

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
