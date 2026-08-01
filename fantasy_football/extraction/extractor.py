import io
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
import requests
from dotenv import load_dotenv

from fantasy_football.constants import DATA_FOLDER, VASTAAV_BRIDGE_SEASONS
from fantasy_football.storage.tables import PLAYER_MATCH, PLAYER_WEEK

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

load_dotenv()

logger = logging.getLogger(__name__)

RAW_DATA_FOLDER = DATA_FOLDER.joinpath("raw")


def _drop_duplicate_rows(frame: pl.DataFrame, context: str) -> pl.DataFrame:
    """Drop rows that are repeated verbatim in the source data.

    Vaastav's exports occasionally carry a player's fixture row twice,
    byte for byte -- 2025-26 has ten such rows. They are not double
    gameweeks: a genuine second fixture differs by ``opponent_team`` and
    ``fixture``, so it survives this filter. Left in, a duplicate breaks
    ``player_match``'s primary key and silently doubles the summed
    minutes in :func:`_collapse_double_gameweeks`.

    Deduplication is on the whole row deliberately. Two rows sharing a
    key but disagreeing on any value are a genuine conflict and are left
    alone, to fail loudly downstream rather than be silently reconciled
    here.

    Parameters
    ----------
    frame : pl.DataFrame
        Rows as read from the source.
    context : str
        Description of the source, used in the warning log.

    Returns
    -------
    pl.DataFrame
        ``frame`` with exact duplicate rows removed, order preserved.
    """
    deduped = frame.unique(maintain_order=True)
    dropped = frame.height - deduped.height
    if dropped:
        logger.warning(
            "Dropped %d verbatim duplicate row(s) from %s", dropped, context
        )
    return deduped


def _collapse_double_gameweeks(frame: pl.DataFrame) -> pl.DataFrame:
    """Collapse multi-fixture rows to one row per ``(season, gw, element)``.

    Vaastav data has one row per fixture, so a player in a double gameweek
    appears twice for the same player-gameweek. Sum the additive stats and keep
    a representative value for the descriptive columns, matching the
    one-row-per-player-gameweek shape the FCI adapter produces.

    Verbatim duplicate rows are dropped first. Because this function sums
    the additive stats, a repeated row would otherwise inflate them --
    silently, and to impossible values such as 148 minutes in a match.

    Parameters
    ----------
    frame : pl.DataFrame
        A frame with at least ``season, gw, element, bonus, minutes,
        total_points, name, position, team, round, value`` columns.

    Returns
    -------
    pl.DataFrame
        One row per ``(season, gw, element)``.
    """
    frame = _drop_duplicate_rows(frame, "player-week source rows")
    return frame.group_by(["season", "gw", "element"]).agg(
        pl.col("bonus").sum(),
        pl.col("minutes").sum(),
        pl.col("total_points").sum(),
        pl.col("name").first(),
        pl.col("position").first(),
        pl.col("team").first(),
        pl.col("round").first(),
        pl.col("value").first(),
    )


def _build_player_match(frame: pl.DataFrame) -> pl.DataFrame:
    """Select per-fixture player-match rows from Vaastav data without collapsing.

    Vaastav data has one row per fixture, so a double gameweek already appears
    as two rows. Unlike ``_collapse_double_gameweeks`` this keeps both rows,
    renaming the source columns to the canonical player-match names.

    Rows repeated verbatim in the source are dropped, since they would
    otherwise violate ``player_match``'s ``(season, gw, element,
    opponent)`` primary key. Deduplication runs on the selected columns:
    the two legs of a real double gameweek differ by ``opponent`` and so
    both survive, while two rows that reach the same key with different
    stored values still collide and fail loudly.

    Parameters
    ----------
    frame : pl.DataFrame
        Per-fixture rows carrying ``season, gw, element, opponent_team,
        was_home, minutes, total_points, kickoff_time``.

    Returns
    -------
    pl.DataFrame
        One row per fixture with columns ``season, gw, element, opponent,
        is_home, minutes, total_points, kickoff_time``.
    """
    selected = frame.select(
        pl.col("season"),
        pl.col("gw"),
        pl.col("element"),
        pl.col("opponent_team").alias("opponent"),
        pl.col("was_home").alias("is_home"),
        pl.col("minutes"),
        pl.col("total_points"),
        # strict=False: Vaastav emits blank kickoff_time for a handful of
        # rows, and one bad value in a ~150k-row CSV would otherwise abort
        # the whole historic load. A bad value becomes a null instead.
        pl.col("kickoff_time")
        .str.replace("Z", "+00:00")
        .str.to_datetime(time_zone="UTC", strict=False)
        .dt.replace_time_zone(None)
        .alias("kickoff_time"),
    )
    return _drop_duplicate_rows(selected, "player-match source rows")


def _historic_loaded(
    present: set[str], current_season: str, bridge_seasons: list[str]
) -> bool:
    """Return True if the historic aggregate has already been loaded.

    The aggregate covers many older seasons and is loaded once. We infer it is
    loaded when any stored season falls outside the current season and the
    enumerable bridge seasons.

    Parameters
    ----------
    present : set[str]
        Seasons already stored in the DB.
    current_season : str
        The current season (sourced from FCI, not the aggregate).
    bridge_seasons : list[str]
        Vaastav bridge seasons handled by their own per-season check.

    Returns
    -------
    bool
        True if historic seasons are present, False otherwise.
    """
    return bool(set(present) - {current_season} - set(bridge_seasons))


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

    def _read_csv(
        self, file_path: str, encoding: str = "utf8"
    ) -> pl.DataFrame:
        """Download a Vaastav repo CSV into a Polars DataFrame in memory.

        Parameters
        ----------
        file_path : str
            Repo-relative path of the CSV.
        encoding : str, optional
            Encoding passed to Polars. Defaults to ``"utf8"``. The
            2016-17 to 2018-19 ``merged_gw.csv`` files contain latin-1
            bytes (e.g. ``Adlène_Guédioura``) that abort a strict read,
            so their loader passes ``"utf8-lossy"``.

        Returns
        -------
        pl.DataFrame
            The parsed CSV.
        """
        url = self.api_client.get_raw_file_url(file_path)
        response = requests.get(url)
        response.raise_for_status()
        return pl.read_csv(
            io.BytesIO(response.content),
            infer_schema_length=10000,
            encoding=encoding,
        )

    def read_players_raw(self, season: str) -> pl.DataFrame:
        """Download a season's ``players_raw.csv`` from the Vaastav repo.

        Parameters
        ----------
        season : str
            Short-form season string, e.g. ``"2023-24"``.

        Returns
        -------
        pl.DataFrame
            The parsed file, one row per registered player.
        """
        return self._read_csv(f"data/{season}/players_raw.csv")

    def load_immutable_seasons(
        self,
        connection: "DuckDBPyConnection",
        current_season: str,
        bridge_seasons: list[str] | None = None,
    ) -> None:
        """Load completed seasons (aggregate + bridge) into the DB if absent.

        Bridge seasons are checked individually. The historic aggregate is
        downloaded and split by season only when no historic seasons are yet
        stored, so a normal run touches the network only for already-missing
        data.

        Parameters
        ----------
        connection : duckdb.DuckDBPyConnection
            Open connection to the player-week database.
        current_season : str
            The current season, excluded from the historic-loaded check.
        bridge_seasons : list[str] | None, optional
            Vaastav bridge seasons. Defaults to ``VASTAAV_BRIDGE_SEASONS``.
        """
        seasons = (
            bridge_seasons
            if bridge_seasons is not None
            else VASTAAV_BRIDGE_SEASONS
        )
        present = PLAYER_WEEK.seasons_present(connection)

        for season in seasons:
            if season in present:
                continue
            bridge = self._read_csv(f"data/{season}/gws/merged_gw.csv")
            shaped = bridge.rename({"GW": "gw"}).with_columns(
                pl.lit(season).alias("season")
            )
            shaped = _collapse_double_gameweeks(shaped)
            PLAYER_WEEK.write_immutable(connection, shaped, season)

        if not _historic_loaded(present, current_season, seasons):
            aggregate = self._read_csv(self.HISTORIC_FILE).rename(
                {"season_x": "season", "team_x": "team", "GW": "gw"}
            )
            aggregate = _collapse_double_gameweeks(aggregate)
            for season in sorted(aggregate["season"].unique().to_list()):
                slice_ = aggregate.filter(pl.col("season") == season)
                PLAYER_WEEK.write_immutable(connection, slice_, season)

    def load_immutable_player_match_seasons(
        self,
        connection: "DuckDBPyConnection",
        current_season: str,
        bridge_seasons: list[str] | None = None,
    ) -> None:
        """Load completed seasons' per-fixture rows into ``player_match``.

        Reads the same Vaastav files as ``load_immutable_seasons`` but keeps
        one row per fixture (no double-gameweek collapse), writing them to the
        ``player_match`` table. Existing seasons are skipped.

        Parameters
        ----------
        connection : duckdb.DuckDBPyConnection
            Open connection to the database.
        current_season : str
            The current season, excluded from the historic-loaded check.
        bridge_seasons : list[str] | None, optional
            Vaastav bridge seasons. Defaults to ``VASTAAV_BRIDGE_SEASONS``.
        """
        seasons = (
            bridge_seasons
            if bridge_seasons is not None
            else VASTAAV_BRIDGE_SEASONS
        )
        present = PLAYER_MATCH.seasons_present(connection)

        for season in seasons:
            if season in present:
                continue
            bridge = self._read_csv(f"data/{season}/gws/merged_gw.csv")
            shaped = bridge.rename({"GW": "gw"}).with_columns(
                pl.lit(season).alias("season")
            )
            PLAYER_MATCH.write_immutable(
                connection, _build_player_match(shaped), season
            )

        if not _historic_loaded(present, current_season, seasons):
            aggregate = self._read_csv(self.HISTORIC_FILE).rename(
                {"season_x": "season", "team_x": "team", "GW": "gw"}
            )
            for season in sorted(aggregate["season"].unique().to_list()):
                slice_ = aggregate.filter(pl.col("season") == season)
                PLAYER_MATCH.write_immutable(
                    connection, _build_player_match(slice_), season
                )
