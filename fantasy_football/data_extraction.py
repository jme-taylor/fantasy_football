import logging
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

from fantasy_football.constants import DATA_FOLDER, VASTAAV_BRIDGE_SEASONS

load_dotenv()

logger = logging.getLogger(__name__)

RAW_DATA_FOLDER = DATA_FOLDER.joinpath("raw")


class GitHubAPIClient:
    """Client for interacting with the GitHub API for Fantasy Premier League data."""

    def __init__(
        self,
        api_key: str | None = None,
        owner: str = "vaastav",
        repo: str = "Fantasy-Premier-League",
        branch: str = "master",
    ) -> None:
        """Initialize the GitHub API client.

        Parameters
        ----------
        api_key : str | None, optional
            GitHub API key. If None, will try to get from GITHUB_API_KEY environment variable.
        owner : str, optional
            GitHub repository owner. Defaults to the Vaastav repo owner.
        repo : str, optional
            GitHub repository name. Defaults to the Vaastav FPL repo.
        branch : str, optional
            Branch to read from. Defaults to "master".

        Raises
        ------
        ValueError
            If no API key is provided and GITHUB_API_KEY environment variable is not set.
        """
        self.api_key = api_key or os.getenv("GITHUB_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GitHub API key is required. Set GITHUB_API_KEY environment variable or pass api_key parameter."
            )

        self.branch = branch
        self.base_url = f"https://api.github.com/repos/{owner}/{repo}"
        self.raw_base_url = (
            f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}"
        )
        self.headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.api_key}",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def get_all_repo_files(self) -> dict:
        """Get details about all files in the Fantasy Premier League repo tree.

        Returns
        -------
        Dict
            Dictionary containing the JSON response from the GitHub API.

        Raises
        ------
        requests.HTTPError
            If the API request fails.
        """
        response = requests.get(
            f"{self.base_url}/git/trees/{self.branch}?recursive=1",
            headers=self.headers,
        )
        response.raise_for_status()
        return response.json()

    def get_file_details(self, path: str) -> dict:
        """Get details about a specific file in the Fantasy Premier League repo.

        Parameters
        ----------
        path : str
            The path of the file in the repo.

        Returns
        -------
        dict
            Dictionary containing file details from the GitHub API.

        Raises
        ------
        requests.HTTPError
            If the API request fails.
        """
        response = requests.get(
            f"{self.base_url}/contents/{path}",
            headers=self.headers,
        )
        response.raise_for_status()
        return response.json()

    def get_raw_file_url(self, path: str) -> str:
        """Get the raw download URL for a file.

        Parameters
        ----------
        path : str
            The path of the file in the repo.

        Returns
        -------
        str
            The raw download URL for the file.
        """
        return f"{self.raw_base_url}/{path}"


class DataExtractor:
    """Extracts and manages Fantasy Premier League data files."""

    HISTORIC_FILE = "data/cleaned_merged_seasons.csv"

    def __init__(self, api_client: GitHubAPIClient | None = None) -> None:
        """Initialize the data extractor.

        Parameters
        ----------
        api_client : GitHubAPIClient | None, optional
            GitHub API client. If None, creates a new client instance.
        """
        self.api_client = api_client or GitHubAPIClient()
        self.raw_data_folder = RAW_DATA_FOLDER

    def _create_local_path(self, file_path: str) -> Path:
        """Create local path structure for a given file path.

        Parameters
        ----------
        file_path : str
            The file path from the repo.

        Returns
        -------
        Path
            The local path where the file should be saved.
        """
        filename = Path(file_path).name
        folder = Path(file_path).parent.relative_to("data")
        local_folder = self.raw_data_folder / folder
        local_folder.mkdir(parents=True, exist_ok=True)
        return local_folder / filename

    def save_file(self, file_info: dict) -> None:
        """Save a single data file to the local filesystem.

        Parameters
        ----------
        file_info : dict
            Dictionary containing file information from GitHub API.

        Raises
        ------
        requests.HTTPError
            If the file download fails.
        """
        local_path = self._create_local_path(file_info["path"])
        url = self.api_client.get_raw_file_url(file_info["path"])

        response = requests.get(url)
        response.raise_for_status()

        with open(local_path, "wb") as f:
            f.write(response.content)

    def save_all_data_files(self) -> None:
        """Download the frozen Vaastav historic dataset.

        Fetches ``cleaned_merged_seasons.csv`` (the aggregate of older seasons)
        plus each Vaastav "bridge" season's ``merged_gw.csv`` — the recent
        seasons not yet folded into that aggregate (see
        ``VASTAAV_BRIDGE_SEASONS``). Current-season data comes from FCI.
        """
        self.raw_data_folder.mkdir(parents=True, exist_ok=True)
        bridge_files = [
            f"data/{season}/gws/merged_gw.csv"
            for season in VASTAAV_BRIDGE_SEASONS
        ]
        for path in [self.HISTORIC_FILE, *bridge_files]:
            try:
                self.save_file({"path": path})
                logger.info("Successfully saved: %s", path)
            except Exception:
                logger.exception("Error saving %s", path)
