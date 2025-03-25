from pathlib import Path

import pytest
import requests
from pytest_mock import MockerFixture

from fantasy_football.data_extraction import (
    get_all_data_files,
    get_all_repo_files,
    get_data_file_url,
    get_github_file,
    save_all_data_files,
    save_data_file,
    update_current_season_data,
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


def test_get_all_repo_files(mocker: MockerFixture) -> None:
    """Test getting all repo files from GitHub API.

    Parameters
    ----------
    mocker : MockerFixture
        Pytest fixture for mocking.
    """
    mock_response = mocker.Mock()
    mock_response.json.return_value = {"tree": []}
    mock_response.raise_for_status.return_value = None
    mocker.patch("requests.get", return_value=mock_response)

    result = get_all_repo_files()
    assert result == {"tree": []}
    mock_response.json.assert_called_once()


def test_get_github_file(mocker: MockerFixture) -> None:
    """Test getting a single file from GitHub API.

    Parameters
    ----------
    mocker : MockerFixture
        Pytest fixture for mocking.
    """
    mock_response = mocker.Mock()
    mock_response.json.return_value = {"content": "test"}
    mock_response.raise_for_status.return_value = None
    mocker.patch("requests.get", return_value=mock_response)

    result = get_github_file("test/path")
    assert result == {"content": "test"}
    mock_response.json.assert_called_once()


def test_get_all_data_files(mocker: MockerFixture, mock_github_response: dict) -> None:
    """Test getting all data files from the repo.

    Parameters
    ----------
    mocker : MockerFixture
        Pytest fixture for mocking.
    mock_github_response : dict
        Mock GitHub API response.
    """
    mocker.patch("fantasy_football.data_extraction.get_all_repo_files", return_value=mock_github_response)

    result = get_all_data_files()
    assert len(result) == 2
    assert all(file["path"].startswith("data") for file in result)
    assert all(file["path"].endswith(".csv") for file in result)


def test_get_data_file_url() -> None:
    """Test generating the data file URL."""
    file_dict = {"path": "data/2023-24/gws/gw1.csv"}
    expected_url = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2023-24/gws/gw1.csv"
    assert get_data_file_url(file_dict) == expected_url


def test_save_data_file(mocker: MockerFixture, tmp_path: Path) -> None:
    """Test saving a data file locally.

    Parameters
    ----------
    mocker : MockerFixture
        Pytest fixture for mocking.
    tmp_path : Path
        Pytest fixture providing a temporary directory.
    """
    mock_response = mocker.Mock()
    mock_response.content = b"test content"
    mock_response.raise_for_status.return_value = None
    mocker.patch("requests.get", return_value=mock_response)
    mocker.patch("fantasy_football.data_extraction.RAW_DATA_FOLDER", tmp_path)

    file_dict = {"path": "data/2023-24/gws/gw1.csv"}
    save_data_file(file_dict)

    expected_file = tmp_path / "2023-24" / "gws" / "gw1.csv"
    assert expected_file.exists()
    assert expected_file.read_bytes() == b"test content"


def test_save_all_data_files(mocker: MockerFixture, mock_github_response: dict, tmp_path: Path) -> None:
    """Test saving all data files.

    Parameters
    ----------
    mocker : MockerFixture
        Pytest fixture for mocking.
    mock_github_response : dict
        Mock GitHub API response.
    tmp_path : Path
        Pytest fixture providing a temporary directory.
    """
    mocker.patch("fantasy_football.data_extraction.get_all_repo_files", return_value=mock_github_response)
    mocker.patch("fantasy_football.data_extraction.RAW_DATA_FOLDER", tmp_path)
    mocker.patch("fantasy_football.data_extraction.save_data_file")

    save_all_data_files()
    assert tmp_path.exists()


def test_update_current_season_data(mocker: MockerFixture, mock_github_file_response: dict) -> None:
    """Test updating current season data.

    Parameters
    ----------
    mocker : MockerFixture
        Pytest fixture for mocking.
    mock_github_file_response : dict
        Mock GitHub file response.
    """
    mock_get_github_file = mocker.patch("fantasy_football.data_extraction.get_github_file", return_value=mock_github_file_response)
    mock_save_data_file = mocker.patch("fantasy_football.data_extraction.save_data_file")

    update_current_season_data("2023-24")
    
    mock_get_github_file.assert_called_once_with("data/2023-24/gws/merged_gw.csv")
    mock_save_data_file.assert_called_once_with(mock_github_file_response) 