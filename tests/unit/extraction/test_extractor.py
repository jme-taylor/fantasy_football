from datetime import datetime
from pathlib import Path

import polars as pl
import pytest
import requests
from pytest_mock import MockerFixture

from fantasy_football.extraction.extractor import (
    DataExtractor,
    GitHubAPIClient,
    _build_player_match,
    _collapse_double_gameweeks,
)


@pytest.fixture(autouse=True)
def _isolate_github_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure no test inherits the developer's real GITHUB_API_KEY."""
    monkeypatch.delenv("GITHUB_API_KEY", raising=False)


@pytest.fixture
def mock_api_client() -> GitHubAPIClient:
    """Create a GitHubAPIClient with an explicit test key.

    Returns
    -------
    GitHubAPIClient
        A GitHubAPIClient instance with a fixed test key.
    """
    return GitHubAPIClient(api_key="test_key")


@pytest.fixture
def mock_data_extractor(mock_api_client: GitHubAPIClient) -> DataExtractor:
    """Create a DataExtractor with mocked API client.

    Parameters
    ----------
    mock_api_client : GitHubAPIClient
        The mocked API client.

    Returns
    -------
    DataExtractor
        A DataExtractor instance with mocked dependencies.
    """
    return DataExtractor(api_client=mock_api_client)


def test_github_api_client_init_with_key() -> None:
    """Test GitHubAPIClient initialization with provided API key."""
    client = GitHubAPIClient(api_key="test_key")
    assert client.api_key == "test_key"
    assert "Bearer test_key" in client.headers["Authorization"]


def test_github_api_client_init_no_key() -> None:
    """Test GitHubAPIClient initialization without API key raises ValueError."""
    with pytest.raises(ValueError, match="GitHub API key is required"):
        GitHubAPIClient()


def test_get_all_repo_files(
    mocker: MockerFixture, mock_api_client: GitHubAPIClient
) -> None:
    """Test getting all repo files from GitHub API.

    Parameters
    ----------
    mocker : MockerFixture
        Pytest fixture for mocking.
    mock_api_client : GitHubAPIClient
        Mocked API client.
    """
    mock_response = mocker.Mock()
    mock_response.json.return_value = {"tree": []}
    mock_response.raise_for_status.return_value = None
    mock_get = mocker.patch("requests.get", return_value=mock_response)

    result = mock_api_client.get_all_repo_files()

    assert result == {"tree": []}
    mock_get.assert_called_once_with(
        "https://api.github.com/repos/vaastav/Fantasy-Premier-League/git/trees/master?recursive=1",
        headers=mock_api_client.headers,
    )
    assert (
        mock_get.call_args.kwargs["headers"]["Authorization"]
        == "Bearer test_key"
    )
    mock_response.raise_for_status.assert_called_once()


def test_get_file_details(
    mocker: MockerFixture, mock_api_client: GitHubAPIClient
) -> None:
    """Test getting a single file from GitHub API.

    Parameters
    ----------
    mocker : MockerFixture
        Pytest fixture for mocking.
    mock_api_client : GitHubAPIClient
        Mocked API client.
    """
    mock_response = mocker.Mock()
    mock_response.json.return_value = {"content": "test"}
    mock_response.raise_for_status.return_value = None
    mock_get = mocker.patch("requests.get", return_value=mock_response)

    result = mock_api_client.get_file_details("data/2023-24/gws/gw1.csv")

    assert result == {"content": "test"}
    mock_get.assert_called_once_with(
        "https://api.github.com/repos/vaastav/Fantasy-Premier-League/contents/data/2023-24/gws/gw1.csv",
        headers=mock_api_client.headers,
    )
    assert (
        mock_get.call_args.kwargs["headers"]["Authorization"]
        == "Bearer test_key"
    )
    mock_response.raise_for_status.assert_called_once()


def test_get_all_repo_files_propagates_http_error(
    mocker: MockerFixture, mock_api_client: GitHubAPIClient
) -> None:
    """An HTTP error from raise_for_status should propagate to the caller.

    Parameters
    ----------
    mocker : MockerFixture
        Pytest fixture for mocking.
    mock_api_client : GitHubAPIClient
        Mocked API client.
    """
    mock_response = mocker.Mock()
    mock_response.raise_for_status.side_effect = requests.HTTPError("500")
    mocker.patch("requests.get", return_value=mock_response)

    with pytest.raises(requests.HTTPError):
        mock_api_client.get_all_repo_files()


def test_get_file_details_propagates_http_error(
    mocker: MockerFixture, mock_api_client: GitHubAPIClient
) -> None:
    """An HTTP error from raise_for_status should propagate to the caller.

    Parameters
    ----------
    mocker : MockerFixture
        Pytest fixture for mocking.
    mock_api_client : GitHubAPIClient
        Mocked API client.
    """
    mock_response = mocker.Mock()
    mock_response.raise_for_status.side_effect = requests.HTTPError("404")
    mocker.patch("requests.get", return_value=mock_response)

    with pytest.raises(requests.HTTPError):
        mock_api_client.get_file_details("data/missing.csv")


def test_get_raw_file_url(mock_api_client: GitHubAPIClient) -> None:
    """Test generating the raw file URL.

    Parameters
    ----------
    mock_api_client : GitHubAPIClient
        Mocked API client.
    """
    expected_url = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2023-24/gws/gw1.csv"
    assert (
        mock_api_client.get_raw_file_url("data/2023-24/gws/gw1.csv")
        == expected_url
    )


def test_save_file(
    mocker: MockerFixture, mock_data_extractor: DataExtractor, tmp_path: Path
) -> None:
    """Test saving a data file locally.

    Parameters
    ----------
    mocker : MockerFixture
        Pytest fixture for mocking.
    mock_data_extractor : DataExtractor
        Mocked data extractor.
    tmp_path : Path
        Pytest fixture providing a temporary directory.
    """
    mock_response = mocker.Mock()
    mock_response.content = b"test content"
    mock_response.raise_for_status.return_value = None
    mocker.patch("requests.get", return_value=mock_response)

    mock_data_extractor.raw_data_folder = tmp_path

    file_info = {"path": "data/2023-24/gws/gw1.csv"}
    mock_data_extractor.save_file(file_info)

    expected_file = tmp_path / "2023-24" / "gws" / "gw1.csv"
    assert expected_file.exists()
    assert expected_file.read_bytes() == b"test content"


def test_save_file_propagates_http_error_and_writes_no_file(
    mocker: MockerFixture, mock_data_extractor: DataExtractor, tmp_path: Path
) -> None:
    """A failed download should propagate and leave no partial file on disk.

    Parameters
    ----------
    mocker : MockerFixture
        Pytest fixture for mocking.
    mock_data_extractor : DataExtractor
        Mocked data extractor.
    tmp_path : Path
        Pytest fixture providing a temporary directory.
    """
    mock_response = mocker.Mock()
    mock_response.raise_for_status.side_effect = requests.HTTPError("500")
    mocker.patch("requests.get", return_value=mock_response)
    mock_data_extractor.raw_data_folder = tmp_path

    with pytest.raises(requests.HTTPError):
        mock_data_extractor.save_file({"path": "data/2023-24/gws/gw1.csv"})

    expected_file = tmp_path / "2023-24" / "gws" / "gw1.csv"
    assert not expected_file.exists()


def test_historic_loaded_detects_seasons_beyond_current_and_bridge() -> None:
    """A season outside current/bridge means the historic aggregate is loaded."""
    from fantasy_football.extraction.extractor import _historic_loaded

    assert not _historic_loaded(set(), "2025-26", ["2024-25"])
    assert not _historic_loaded({"2025-26", "2024-25"}, "2025-26", ["2024-25"])
    assert _historic_loaded({"2020-21"}, "2025-26", ["2024-25"])


def test_load_immutable_seasons_loads_aggregate_and_bridge(
    mocker: MockerFixture, mock_data_extractor: DataExtractor, tmp_path
) -> None:
    """Aggregate (split by season) and bridge seasons are inserted once."""
    from fantasy_football.storage.database import get_connection
    from fantasy_football.storage.tables import PLAYER_WEEK

    aggregate = pl.DataFrame(
        {
            "season_x": ["2019-20", "2020-21"],
            "name": ["A", "B"],
            "position": ["GKP", "DEF"],
            "team_x": ["Arsenal", "Chelsea"],
            "bonus": [1, 2],
            "element": [1, 2],
            "minutes": [90, 90],
            "round": [1, 1],
            "total_points": [6, 8],
            "value": [50, 55],
            "GW": [1, 1],
        }
    )
    bridge = pl.DataFrame(
        {
            "name": ["C"],
            "position": ["MID"],
            "team": ["Leeds"],
            "bonus": [3],
            "element": [3],
            "minutes": [90],
            "round": [5],
            "total_points": [9],
            "value": [60],
            "GW": [5],
        }
    )

    def fake_read_csv(path: str) -> pl.DataFrame:
        return bridge if "2024-25" in path else aggregate

    mocker.patch.object(
        mock_data_extractor, "_read_csv", side_effect=fake_read_csv
    )

    connection = get_connection(tmp_path / "t.duckdb")
    try:
        mock_data_extractor.load_immutable_seasons(
            connection, "2025-26", bridge_seasons=["2024-25"]
        )
        present = PLAYER_WEEK.seasons_present(connection)
    finally:
        connection.close()

    assert present == {"2019-20", "2020-21", "2024-25"}


def test_load_immutable_seasons_skips_when_already_present(
    mocker: MockerFixture, mock_data_extractor: DataExtractor, tmp_path
) -> None:
    """With historic + bridge already stored, no downloads happen."""
    from fantasy_football.storage.database import get_connection
    from fantasy_football.storage.tables import PLAYER_WEEK

    seeded = pl.DataFrame(
        {
            "season": ["2020-21", "2024-25"],
            "gw": [1, 1],
            "element": [1, 2],
            "name": ["A", "C"],
            "position": ["DEF", "MID"],
            "team": ["Arsenal", "Leeds"],
            "bonus": [1, 2],
            "minutes": [90, 90],
            "round": [1, 1],
            "total_points": [6, 8],
            "value": [50, 60],
        }
    )
    read_csv = mocker.patch.object(mock_data_extractor, "_read_csv")

    connection = get_connection(tmp_path / "t.duckdb")
    try:
        PLAYER_WEEK.write_immutable(
            connection, seeded.filter(pl.col("season") == "2020-21"), "2020-21"
        )
        PLAYER_WEEK.write_immutable(
            connection, seeded.filter(pl.col("season") == "2024-25"), "2024-25"
        )
        mock_data_extractor.load_immutable_seasons(
            connection, "2025-26", bridge_seasons=["2024-25"]
        )
    finally:
        connection.close()

    read_csv.assert_not_called()


def test_data_extractor_with_custom_client() -> None:
    """Test DataExtractor initialization with custom API client."""
    custom_client = GitHubAPIClient(api_key="custom_key")
    extractor = DataExtractor(api_client=custom_client)

    assert extractor.api_client is custom_client
    assert extractor.api_client.api_key == "custom_key"


def test_data_extractor_default_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test DataExtractor initialization with default client reads key from env."""
    monkeypatch.setenv("GITHUB_API_KEY", "env_key")

    extractor = DataExtractor()
    assert extractor.api_client.api_key == "env_key"


def test_github_api_client_custom_repo_urls() -> None:
    """Client builds API and raw URLs from provided repo coordinates."""
    client = GitHubAPIClient(
        api_key="test_key",
        owner="olbauday",
        repo="FPL-Core-Insights",
        branch="main",
    )
    assert client.base_url == (
        "https://api.github.com/repos/olbauday/FPL-Core-Insights"
    )
    assert client.get_raw_file_url("data/x.csv") == (
        "https://raw.githubusercontent.com/olbauday/FPL-Core-Insights/main/data/x.csv"
    )


def test_github_api_client_defaults_to_vaastav() -> None:
    """With no repo coords the client targets the Vaastav master branch."""
    client = GitHubAPIClient(api_key="test_key")
    assert client.base_url == (
        "https://api.github.com/repos/vaastav/Fantasy-Premier-League"
    )
    assert client.get_raw_file_url("data/x.csv") == (
        "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/x.csv"
    )


def test_load_immutable_seasons_collapses_double_gameweeks(
    mocker: MockerFixture, mock_data_extractor: DataExtractor, tmp_path
) -> None:
    """A player's two fixture-rows in a double gameweek collapse to one row."""
    from fantasy_football.storage.database import get_connection
    from fantasy_football.storage.tables import PLAYER_WEEK

    # Bridge season merged_gw with a double gameweek: element 5 plays twice in
    # GW24 (two fixture rows, same season/gw/element).
    bridge = pl.DataFrame(
        {
            "name": ["DGW Player", "DGW Player"],
            "position": ["MID", "MID"],
            "team": ["Arsenal", "Arsenal"],
            "bonus": [1, 2],
            "element": [5, 5],
            "minutes": [90, 75],
            "round": [24, 24],
            "total_points": [6, 8],
            "value": [80, 80],
            "GW": [24, 24],
        }
    )
    mocker.patch.object(mock_data_extractor, "_read_csv", return_value=bridge)

    connection = get_connection(tmp_path / "t.duckdb")
    try:
        # Seed a historic season so _historic_loaded() short-circuits the
        # aggregate download path, isolating the bridge-season collapse.
        seed = pl.DataFrame(
            {
                "season": ["2019-20"],
                "gw": [1],
                "element": [1],
                "name": ["Seed"],
                "position": ["DEF"],
                "team": ["Chelsea"],
                "bonus": [0],
                "minutes": [90],
                "round": [1],
                "total_points": [2],
                "value": [40],
            }
        )
        PLAYER_WEEK.write_immutable(connection, seed, "2019-20")
        mock_data_extractor.load_immutable_seasons(
            connection, "2025-26", bridge_seasons=["2024-25"]
        )
        result = PLAYER_WEEK.load(connection)
    finally:
        connection.close()

    dgw = result.filter(
        (pl.col("season") == "2024-25") & (pl.col("element") == 5)
    )
    assert dgw.height == 1
    row = dgw.row(0, named=True)
    assert row["total_points"] == 14  # 6 + 8
    assert row["minutes"] == 165  # 90 + 75
    assert row["bonus"] == 3  # 1 + 2
    assert row["name"] == "DGW Player"


def test_build_player_match_keeps_both_fixtures_of_a_dgw() -> None:
    """A double-gameweek player yields two rows, one per opponent, uncollapsed."""
    frame = pl.DataFrame(
        {
            "season": ["2023-24", "2023-24"],
            "gw": [1, 1],
            "element": [5, 5],
            "opponent_team": [12, 7],
            "was_home": [True, False],
            "minutes": [90, 70],
            "total_points": [6, 2],
            "yellow_cards": [1, 0],
            "red_cards": [0, 0],
            "name": ["A", "A"],
            "kickoff_time": [
                "2023-08-11T19:00:00Z",
                "2023-08-15T19:00:00Z",
            ],
        }
    )
    result = _build_player_match(frame)
    assert result.columns == [
        "season",
        "gw",
        "element",
        "opponent",
        "is_home",
        "minutes",
        "total_points",
        "yellow_cards",
        "red_cards",
        "kickoff_time",
    ]
    assert result.height == 2
    assert sorted(result["opponent"].to_list()) == [7, 12]
    assert sorted(result["minutes"].to_list()) == [70, 90]


def test_build_player_match_carries_kickoff_time() -> None:
    """Per-fixture rows keep the kickoff instant for chronological sorting."""
    frame = pl.DataFrame(
        {
            "season": ["2023-24"],
            "gw": [1],
            "element": [7],
            "opponent_team": [3],
            "was_home": [True],
            "minutes": [90],
            "total_points": [8],
            "yellow_cards": [0],
            "red_cards": [0],
            "kickoff_time": ["2023-08-11T19:00:00Z"],
        }
    )

    result = _build_player_match(frame)

    assert "kickoff_time" in result.columns
    assert result["kickoff_time"][0] == datetime(2023, 8, 11, 19, 0)


def test_build_player_match_tolerates_blank_kickoff_time() -> None:
    """A blank or null kickoff becomes a null, not an aborted load.

    Vaastav emits blank ``kickoff_time`` for a handful of rows; a strict
    parse would raise and take the whole ~150k-row historic load with it.
    """
    frame = pl.DataFrame(
        {
            "season": ["2023-24"] * 3,
            "gw": [1, 2, 3],
            "element": [7, 7, 7],
            "opponent_team": [3, 4, 5],
            "was_home": [True, False, True],
            "minutes": [90, 45, 0],
            "total_points": [8, 2, 0],
            "yellow_cards": [0, 0, 0],
            "red_cards": [0, 0, 0],
            "kickoff_time": ["2023-08-11T19:00:00Z", "", None],
        },
        schema_overrides={"kickoff_time": pl.Utf8},
    )

    result = _build_player_match(frame)

    assert result.height == 3
    assert result["kickoff_time"].to_list() == [
        datetime(2023, 8, 11, 19, 0),
        None,
        None,
    ]


def test_build_player_match_drops_verbatim_duplicate_rows() -> None:
    """Vaastav repeats some fixture rows byte-for-byte; keep only one.

    Regression: 2025-26's ``merged_gw.csv`` carries ten such rows, which
    violated ``player_match``'s (season, gw, element, opponent) primary
    key and aborted the whole historic load.
    """
    row = {
        "season": "2025-26",
        "gw": 1,
        "element": 391,
        "opponent_team": 4,
        "was_home": False,
        "minutes": 0,
        "total_points": 0,
        "yellow_cards": 0,
        "red_cards": 0,
        "kickoff_time": "2025-08-15T19:00:00Z",
    }
    frame = pl.DataFrame([row, row])

    result = _build_player_match(frame)

    assert result.height == 1
    assert result.row(0, named=True)["opponent"] == 4


def test_build_player_match_keeps_dgw_legs_that_share_a_gameweek() -> None:
    """Two fixtures in one gameweek differ by opponent and must both stay."""
    frame = pl.DataFrame(
        {
            "season": ["2025-26", "2025-26"],
            "gw": [24, 24],
            "element": [5, 5],
            "opponent_team": [4, 9],
            "was_home": [True, False],
            "minutes": [90, 90],
            "total_points": [6, 6],
            "yellow_cards": [0, 1],
            "red_cards": [0, 0],
            "kickoff_time": [
                "2026-02-01T15:00:00Z",
                "2026-02-04T19:45:00Z",
            ],
        }
    )

    result = _build_player_match(frame)

    assert result.height == 2
    assert sorted(result["opponent"].to_list()) == [4, 9]


def test_collapse_double_gameweeks_ignores_verbatim_duplicates() -> None:
    """A repeated row must not double the summed minutes and points.

    This is the silent half of the same defect: ``player_match`` fails
    loudly on the duplicate key, but the player-week collapse simply
    sums it, producing impossible totals such as 148 minutes.
    """
    row = {
        "season": "2025-26",
        "gw": 8,
        "element": 100,
        "opponent_team": 8,
        "fixture": 71,
        "bonus": 0,
        "minutes": 74,
        "total_points": 12,
        "name": "Junior Kroupi",
        "position": "FWD",
        "team": "Bournemouth",
        "round": 8,
        "value": 45,
    }
    frame = pl.DataFrame([row, row])

    result = _collapse_double_gameweeks(frame)

    assert result.height == 1
    collapsed = result.row(0, named=True)
    assert collapsed["minutes"] == 74
    assert collapsed["total_points"] == 12


def test_collapse_double_gameweeks_still_sums_genuine_dgw_legs() -> None:
    """Legs that differ by fixture are a real double gameweek: sum them."""
    frame = pl.DataFrame(
        {
            "season": ["2025-26", "2025-26"],
            "gw": [24, 24],
            "element": [5, 5],
            "opponent_team": [4, 9],
            "fixture": [230, 241],
            "bonus": [1, 2],
            "minutes": [90, 75],
            "total_points": [6, 8],
            "name": ["DGW Player", "DGW Player"],
            "position": ["MID", "MID"],
            "team": ["Arsenal", "Arsenal"],
            "round": [24, 24],
            "value": [80, 80],
        }
    )

    result = _collapse_double_gameweeks(frame)

    assert result.height == 1
    collapsed = result.row(0, named=True)
    assert collapsed["minutes"] == 165
    assert collapsed["total_points"] == 14
    assert collapsed["bonus"] == 3


def test_read_csv_defaults_to_strict_utf8(
    mocker: MockerFixture, mock_data_extractor: DataExtractor
) -> None:
    """The default encoding is unchanged, so existing callers are unaffected."""
    response = mocker.Mock()
    response.content = b"a,b\n1,2\n"
    mocker.patch(
        "fantasy_football.extraction.extractor.requests.get",
        return_value=response,
    )
    read_csv = mocker.patch(
        "fantasy_football.extraction.extractor.pl.read_csv",
        return_value=pl.DataFrame({"a": [1], "b": [2]}),
    )
    mock_data_extractor._read_csv("data/x.csv")
    assert read_csv.call_args.kwargs["encoding"] == "utf8"


def test_read_csv_passes_lossy_encoding_through_to_polars(
    mocker: MockerFixture, mock_data_extractor: DataExtractor
) -> None:
    """The caller's encoding reaches Polars, so legacy files can be read."""
    response = mocker.Mock()
    response.content = "name\nAdlène\n".encode("latin-1")
    mocker.patch(
        "fantasy_football.extraction.extractor.requests.get",
        return_value=response,
    )
    frame = mock_data_extractor._read_csv(
        "data/2016-17/gws/merged_gw.csv", encoding="utf8-lossy"
    )
    assert frame.height == 1
    assert frame.columns == ["name"]


def test_read_csv_raises_on_legacy_bytes_under_the_default_encoding(
    mocker: MockerFixture, mock_data_extractor: DataExtractor
) -> None:
    """Without the lossy encoding the 2016-19 files genuinely fail to parse."""
    response = mocker.Mock()
    response.content = "name\nAdlène\n".encode("latin-1")
    mocker.patch(
        "fantasy_football.extraction.extractor.requests.get",
        return_value=response,
    )
    with pytest.raises(Exception):
        mock_data_extractor._read_csv("data/2016-17/gws/merged_gw.csv")
