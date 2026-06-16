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
