import os

import requests
from dotenv import load_dotenv

from fantasy_football.constants import DATA_FOLDER

load_dotenv()

RAW_DATA_FOLDER = DATA_FOLDER.joinpath("raw")


def get_all_repo_files() -> dict:
    """Get details about all the files in the fantasy premier league repo.

    This function uses the GitHub API to get all files in the tree of the 
    master branch of the Fantasy Premier League repo, and will return this as
    a dictionary. The function will look for the GITHUB_API_KEY environment 
    variable to use as the API key for the request.

    Returns
    -------
    dict
        A dictionary of the JSON response from the github API.

    """
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {os.getenv("GITHUB_API_KEY")}",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    response = requests.get(
        "https://api.github.com/repos/vaastav/Fantasy-Premier-League/git/trees/master?recursive=1",
        headers=headers,
    )
    response.raise_for_status()
    response_json = response.json()
    return response_json


def get_all_data_files() -> list:
    """Get details about all the CSV files in the data folder in the repo.

    This function will use the get_all_repo_files function to get all files in 
    the repo, and then filter this list to only include files that are in the 
    data folder and have a .csv extension.

    Returns
    -------
    list
        A list of dictionaries, each containing details about a CSV file in the 
        data folder of the Fantasy Premier League repo.

    """
    all_files = get_all_repo_files()
    data_files = [
        file
        for file in all_files["tree"]
        if file["path"].startswith("data") and file["path"].endswith(".csv")
    ]
    return data_files


def get_data_file_url(file: dict) -> str:
    """Get the raw URL for a data file.

    This function takes a dictionary representing a file in the Fantasy 
    Premier League repo and returns the raw URL for that file.

    Parameters
    ----------
    file : dict
        A dictionary representing a file in the Fantasy Premier League repo.

    Returns
    -------
    str
        The raw URL for the file.

    """
    return f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/{file['path']}"


def save_data_file(file: dict) -> None:
    """Save a data file to the local filesystem.

    Using a dictionary representing a file in the Fantasy Premier League repo, 
    this function will download the file and save it to the local filesystem. 
    The file will be saved in a folder structure that mirrors the structure of 
    the data folder.

    Parameters
    ----------
    file : dict
        A dictionary representing a file in the Fantasy Premier League repo.

    """
    filename = file["path"].split("/")[-1]
    folder = file["path"].split(filename)[0].split("data/")[1]
    local_folder = RAW_DATA_FOLDER.joinpath(folder)
    local_folder.mkdir(parents=True, exist_ok=True)
    local_filepath = local_folder.joinpath(filename)
    url = get_data_file_url(file)
    result = requests.get(url)
    result.raise_for_status()
    with open(local_filepath, "wb") as f:
        f.write(result.content)


def save_all_data_files() -> None:
    """Save all data files from the Fantasy Premier League repo to local."""
    RAW_DATA_FOLDER.mkdir(parents=True, exist_ok=True)
    data_files = get_all_data_files()
    for file in data_files:
        try:
            save_data_file(file)
        except Exception as e:
            print(f"Error saving {file['path']}: {e}")
