from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from fantasy_football.data_extraction import (
    DataExtractor,
    GitHubAPIClient,
)


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
            {
                "path": "README.md",
                "type": "blob",
                "sha": "ghi789",
            },
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
def mock_api_client(mocker: MockerFixture) -> GitHubAPIClient:
    """Create a mock GitHubAPIClient for testing.
    
    Parameters
    ----------
    mocker : MockerFixture
        Pytest fixture for mocking.
        
    Returns
    -------
    GitHubAPIClient
        A mocked GitHubAPIClient instance.
    """
    # Mock the environment variable and requests to avoid actual API calls
    mocker.patch.dict("os.environ", {"GITHUB_API_KEY": "test_key"})
    return GitHubAPIClient()


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


def test_github_api_client_init_no_key(mocker: MockerFixture) -> None:
    """Test GitHubAPIClient initialization without API key raises ValueError."""
    mocker.patch.dict("os.environ", {}, clear=True)
    with pytest.raises(ValueError, match="GitHub API key is required"):
        GitHubAPIClient()


def test_get_all_repo_files(mocker: MockerFixture, mock_api_client: GitHubAPIClient) -> None:
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
    mocker.patch("requests.get", return_value=mock_response)

    result = mock_api_client.get_all_repo_files()
    assert result == {"tree": []}
    mock_response.json.assert_called_once()


def test_get_file_details(mocker: MockerFixture, mock_api_client: GitHubAPIClient) -> None:
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
    mocker.patch("requests.get", return_value=mock_response)

    result = mock_api_client.get_file_details("test/path")
    assert result == {"content": "test"}
    mock_response.json.assert_called_once()


def test_get_raw_file_url(mock_api_client: GitHubAPIClient) -> None:
    """Test generating the raw file URL.
    
    Parameters
    ----------
    mock_api_client : GitHubAPIClient
        Mocked API client.
    """
    expected_url = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2023-24/gws/gw1.csv"
    assert mock_api_client.get_raw_file_url("data/2023-24/gws/gw1.csv") == expected_url


def test_get_all_data_files(
    mocker: MockerFixture, 
    mock_data_extractor: DataExtractor, 
    mock_github_response: dict
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
    assert len(result) == 2
    assert all(file["path"].startswith("data") for file in result)
    assert all(file["path"].endswith(".csv") for file in result)


def test_create_local_path(mock_data_extractor: DataExtractor, tmp_path: Path) -> None:
    """Test creating local path structure.
    
    Parameters
    ----------
    mock_data_extractor : DataExtractor
        Mocked data extractor.
    tmp_path : Path
        Pytest fixture providing a temporary directory.
    """
    mock_data_extractor.raw_data_folder = tmp_path
    
    result = mock_data_extractor._create_local_path("data/2023-24/gws/gw1.csv")
    expected_path = tmp_path / "2023-24" / "gws" / "gw1.csv"
    
    assert result == expected_path
    assert result.parent.exists()  # Folder should be created


def test_save_file(
    mocker: MockerFixture, 
    mock_data_extractor: DataExtractor, 
    tmp_path: Path
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


def test_save_all_data_files(
    mocker: MockerFixture, 
    mock_data_extractor: DataExtractor, 
    mock_github_response: dict, 
    tmp_path: Path
) -> None:
    """Test saving all data files.

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
    mocker.patch.object(mock_data_extractor, "save_file")

    mock_data_extractor.save_all_data_files()
    assert tmp_path.exists()


def test_update_current_season_data(
    mocker: MockerFixture, 
    mock_data_extractor: DataExtractor,
    mock_github_file_response: dict
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

    mock_get_file_details.assert_called_once_with("data/2023-24/gws/merged_gw.csv")
    mock_save_file.assert_called_once_with(mock_github_file_response)


def test_data_extractor_with_custom_client() -> None:
    """Test DataExtractor initialization with custom API client."""
    custom_client = GitHubAPIClient(api_key="custom_key")
    extractor = DataExtractor(api_client=custom_client)
    
    assert extractor.api_client is custom_client
    assert extractor.api_client.api_key == "custom_key"


def test_data_extractor_default_client(mocker: MockerFixture) -> None:
    """Test DataExtractor initialization with default client."""
    mocker.patch.dict("os.environ", {"GITHUB_API_KEY": "env_key"})
    
    extractor = DataExtractor()
    assert extractor.api_client.api_key == "env_key"
