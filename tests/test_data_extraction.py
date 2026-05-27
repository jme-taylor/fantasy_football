import logging
from pathlib import Path

import pytest
import requests
from pytest_mock import MockerFixture

from fantasy_football.data_extraction import (
    DataExtractor,
    GitHubAPIClient,
)


@pytest.fixture(autouse=True)
def _isolate_github_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure no test inherits the developer's real GITHUB_API_KEY."""
    monkeypatch.delenv("GITHUB_API_KEY", raising=False)


@pytest.fixture
def mock_github_response() -> dict:
    """Create a mock GitHub API response.

    Returns
    -------
    dict
        A mock response from the GitHub API.
    """
    return {
        "tree": [
            {
                "path": "data/2023-24/gws/gw1.csv",
                "type": "blob",
                "sha": "abc123",
            },
            {
                "path": "data/2023-24/gws/gw2.csv",
                "type": "blob",
                "sha": "def456",
            },
            {"path": "README.md", "type": "blob", "sha": "ghi789"},
            {"path": "database/players.csv", "type": "blob", "sha": "jkl012"},
            {
                "path": "data/2023-24/gws/gw1.json",
                "type": "blob",
                "sha": "mno345",
            },
            {"path": "notdata/x.csv", "type": "blob", "sha": "pqr678"},
        ]
    }


@pytest.fixture
def mock_github_file_response() -> dict:
    """Create a mock GitHub file response.

    Returns
    -------
    dict
        A mock response from the GitHub API for a single file.
    """
    return {
        "path": "data/2023-24/gws/gw1.csv",
        "type": "file",
        "sha": "abc123",
        "content": "player_id,player_name,position\n1,Test Player,FWD",
        "encoding": "base64",
    }


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


def test_get_all_data_files(
    mocker: MockerFixture,
    mock_data_extractor: DataExtractor,
    mock_github_response: dict,
) -> None:
    """Test getting all data files from the repo.

    Parameters
    ----------
    mocker : MockerFixture
        Pytest fixture for mocking.
    mock_data_extractor : DataExtractor
        Mocked data extractor.
    mock_github_response : dict
        Mock GitHub API response.
    """
    mocker.patch.object(
        mock_data_extractor.api_client,
        "get_all_repo_files",
        return_value=mock_github_response,
    )

    result = mock_data_extractor.get_all_data_files()
    assert [file["path"] for file in result] == [
        "data/2023-24/gws/gw1.csv",
        "data/2023-24/gws/gw2.csv",
    ]


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


def test_save_all_data_files_calls_save_file_per_data_file(
    mocker: MockerFixture,
    mock_data_extractor: DataExtractor,
    mock_github_response: dict,
    tmp_path: Path,
) -> None:
    """save_all_data_files should create the raw folder and call save_file once per CSV under data/.

    Parameters
    ----------
    mocker : MockerFixture
        Pytest fixture for mocking.
    mock_data_extractor : DataExtractor
        Mocked data extractor.
    mock_github_response : dict
        Mock GitHub API response.
    tmp_path : Path
        Pytest fixture providing a temporary directory.
    """
    mock_data_extractor.raw_data_folder = tmp_path / "raw"
    assert not mock_data_extractor.raw_data_folder.exists()

    mocker.patch.object(
        mock_data_extractor.api_client,
        "get_all_repo_files",
        return_value=mock_github_response,
    )
    mock_save_file = mocker.patch.object(mock_data_extractor, "save_file")

    mock_data_extractor.save_all_data_files()

    assert mock_data_extractor.raw_data_folder.is_dir()
    assert mock_save_file.call_count == 2
    saved_paths = [
        call.args[0]["path"] for call in mock_save_file.call_args_list
    ]
    assert saved_paths == [
        "data/2023-24/gws/gw1.csv",
        "data/2023-24/gws/gw2.csv",
    ]


def test_save_all_data_files_continues_after_failure(
    mocker: MockerFixture,
    mock_data_extractor: DataExtractor,
    mock_github_response: dict,
    tmp_path: Path,
) -> None:
    """A failure on one file should not stop subsequent files from being saved.

    Parameters
    ----------
    mocker : MockerFixture
        Pytest fixture for mocking.
    mock_data_extractor : DataExtractor
        Mocked data extractor.
    mock_github_response : dict
        Mock GitHub API response.
    tmp_path : Path
        Pytest fixture providing a temporary directory.
    """
    mock_data_extractor.raw_data_folder = tmp_path

    mocker.patch.object(
        mock_data_extractor.api_client,
        "get_all_repo_files",
        return_value=mock_github_response,
    )
    mock_save_file = mocker.patch.object(
        mock_data_extractor,
        "save_file",
        side_effect=[requests.HTTPError("boom"), None],
    )

    mock_data_extractor.save_all_data_files()

    assert mock_save_file.call_count == 2


def test_update_current_season_data(
    mocker: MockerFixture,
    mock_data_extractor: DataExtractor,
    mock_github_file_response: dict,
) -> None:
    """Test updating current season data.

    Parameters
    ----------
    mocker : MockerFixture
        Pytest fixture for mocking.
    mock_data_extractor : DataExtractor
        Mocked data extractor.
    mock_github_file_response : dict
        Mock GitHub file response.
    """
    mock_get_file_details = mocker.patch.object(
        mock_data_extractor.api_client,
        "get_file_details",
        return_value=mock_github_file_response,
    )
    mock_save_file = mocker.patch.object(mock_data_extractor, "save_file")

    mock_data_extractor.update_current_season_data("2023-24")

    mock_get_file_details.assert_called_once_with(
        "data/2023-24/gws/merged_gw.csv"
    )
    mock_save_file.assert_called_once_with(mock_github_file_response)


def test_update_current_season_data_swallows_errors(
    mocker: MockerFixture,
    mock_data_extractor: DataExtractor,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Errors in update_current_season_data should be swallowed and logged, not raised.

    Parameters
    ----------
    mocker : MockerFixture
        Pytest fixture for mocking.
    mock_data_extractor : DataExtractor
        Mocked data extractor.
    caplog : pytest.LogCaptureFixture
        Pytest fixture for capturing log records.
    """
    mocker.patch.object(
        mock_data_extractor.api_client,
        "get_file_details",
        side_effect=requests.HTTPError("404"),
    )

    with caplog.at_level(
        logging.ERROR, logger="fantasy_football.data_extraction"
    ):
        mock_data_extractor.update_current_season_data("2023-24")

    assert any(
        "Error updating season data for 2023-24" in record.getMessage()
        for record in caplog.records
    )


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
