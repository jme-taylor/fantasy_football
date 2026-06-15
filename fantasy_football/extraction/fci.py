"""Extract current-season FPL data from FPL Core Insights (FCI).

FCI replaces Vaastav as the data source from 2025-26 onwards. Its per-gameweek
folders hold mostly season-to-date snapshots, so this module reconstructs
Vaastav-shaped ``merged_gw.csv`` rows (one row per player per gameweek) from
them. The pure ``build_merged_gw`` adapter does the reshaping; the IO and
orchestration live in ``FciExtractor`` (added separately).
"""

import io
import logging
import re

import polars as pl
import requests

from fantasy_football.constants import RAW_DATA_FOLDER
from fantasy_football.extraction.extractor import GitHubAPIClient
from fantasy_football.extraction.fpl import FplAPI
from fantasy_football.extraction.seasons import season_short_to_long

logger = logging.getLogger(__name__)

# FCI position labels -> Vaastav/FPL short codes.
FCI_POSITION_TO_VAASTAV: dict[str, str] = {
    "Goalkeeper": "GK",
    "Defender": "DEF",
    "Midfielder": "MID",
    "Forward": "FWD",
}

MERGED_GW_COLUMNS: list[str] = [
    "name",
    "position",
    "team",
    "bonus",
    "element",
    "minutes",
    "round",
    "total_points",
    "GW",
    "value",
]

# Columns the adapter needs from each per-gameweek FCI file, with the dtypes we
# normalise them to. We select and cast these before concatenating because FCI's
# files carry many unused columns whose inferred dtype varies across gameweeks
# (rank/per-90 stats are numeric early but empty/String later), and even needed
# columns drift: ``bonus`` infers as Int64 in some gameweeks and Float64 in
# others. Either case breaks the diagonal concat with a SchemaError, so we pin
# the schema explicitly rather than trusting per-file inference.
SNAPSHOT_SCHEMA: dict[str, pl.DataType] = {
    "id": pl.Int64,
    "first_name": pl.Utf8,
    "second_name": pl.Utf8,
    "now_cost": pl.Float64,
    "event_points": pl.Int64,
    "bonus": pl.Int64,
}
MATCHSTATS_SCHEMA: dict[str, pl.DataType] = {
    "player_id": pl.Int64,
    "minutes_played": pl.Int64,
}


def build_merged_gw(
    snapshots: pl.DataFrame,
    matchstats: pl.DataFrame,
    players: pl.DataFrame,
    team_code_to_name: dict[int, str],
) -> pl.DataFrame:
    """Reconstruct Vaastav-shaped per-gameweek rows from FCI frames.

    Parameters
    ----------
    snapshots : pl.DataFrame
        Concatenated ``player_gameweek_stats`` across gameweeks, with an added
        ``gw`` column. Must contain ``gw, id, first_name, second_name,
        now_cost, event_points, bonus`` (``bonus`` cumulative for the season).
    matchstats : pl.DataFrame
        Concatenated ``playermatchstats`` with a ``gw`` column. Must contain
        ``gw, player_id, minutes_played``.
    players : pl.DataFrame
        Season ``players`` file. Must contain ``player_id, position,
        team_code``.
    team_code_to_name : dict[int, str]
        Maps FCI ``team_code`` (== FPL team ``code``) to official team name.

    Returns
    -------
    pl.DataFrame
        One row per (player, gameweek) with columns ``MERGED_GW_COLUMNS``.
    """
    # Per-GW minutes: sum across matches so double gameweeks accumulate.
    minutes = matchstats.group_by(["gw", "player_id"]).agg(
        pl.col("minutes_played").sum().alias("minutes")
    )

    # Event-level bonus: difference of cumulative season bonus per player; the
    # first gameweek keeps its cumulative value.
    snap = snapshots.sort(["id", "gw"]).with_columns(
        (pl.col("bonus") - pl.col("bonus").shift(1).over("id"))
        .fill_null(pl.col("bonus"))
        .alias("event_bonus")
    )

    # Map team_code -> name via a small mapping frame join (this polars version
    # lacks ``replace_strict``, so we join rather than replace-with-default).
    team_map = pl.DataFrame(
        {
            "team_code": list(team_code_to_name.keys()),
            "team": list(team_code_to_name.values()),
        }
    )

    merged = (
        snap.join(
            players.select("player_id", "position", "team_code"),
            left_on="id",
            right_on="player_id",
            how="left",
            coalesce=True,
        )
        .join(team_map, on="team_code", how="left", coalesce=True)
        .join(
            minutes,
            left_on=["gw", "id"],
            right_on=["gw", "player_id"],
            how="left",
            coalesce=True,
        )
        .with_columns(
            (pl.col("first_name") + " " + pl.col("second_name")).alias("name"),
            pl.col("position").replace(FCI_POSITION_TO_VAASTAV),
            pl.col("id").alias("element"),
            pl.col("minutes").fill_null(0).cast(pl.Int64),
            pl.col("gw").alias("round"),
            pl.col("event_points").alias("total_points"),
            pl.col("gw").alias("GW"),
            (pl.col("now_cost") * 10).round(0).cast(pl.Int64).alias("value"),
            pl.col("event_bonus").cast(pl.Int64).alias("bonus"),
        )
    )
    return merged.select(MERGED_GW_COLUMNS)


_GW_PATH_RE = re.compile(r"/By Gameweek/GW(\d+)/")


class FciExtractor:
    """Download FCI data and build a Vaastav-shaped current-season merged_gw.csv."""

    def __init__(
        self,
        api_client: GitHubAPIClient | None = None,
        fpl_api: FplAPI | None = None,
    ) -> None:
        """Initialise the extractor.

        Parameters
        ----------
        api_client : GitHubAPIClient | None, optional
            Client pointed at the FCI repo. Defaults to a new client for
            ``olbauday/FPL-Core-Insights`` on ``main``.
        fpl_api : FplAPI | None, optional
            Live FPL API client, used to resolve team names. Defaults to a new
            ``FplAPI`` instance.
        """
        self.api_client = api_client or GitHubAPIClient(
            owner="olbauday", repo="FPL-Core-Insights", branch="main"
        )
        self.fpl_api = fpl_api or FplAPI()
        self.raw_data_folder = RAW_DATA_FOLDER

    def _gw_number_from_path(self, path: str) -> int | None:
        """Return the gameweek number embedded in a By-Gameweek path, or None."""
        match = _GW_PATH_RE.search(path)
        return int(match.group(1)) if match else None

    def list_gameweeks(self, long_season: str) -> list[int]:
        """Return the sorted unique gameweek numbers available for a season.

        Parameters
        ----------
        long_season : str
            Long-form season string, e.g. ``"2025-2026"``.

        Returns
        -------
        list[int]
            Sorted unique gameweek numbers present under the season's
            ``By Gameweek`` folder.
        """
        tree = self.api_client.get_all_repo_files()
        prefix = f"data/{long_season}/By Gameweek/"
        gws: set[int] = set()
        for entry in tree["tree"]:
            path = entry["path"]
            if not path.startswith(prefix):
                continue
            gw = self._gw_number_from_path(path)
            if gw is not None:
                gws.add(gw)
        return sorted(gws)

    def _read_csv(self, path: str) -> pl.DataFrame:
        """Download a repo CSV and read it into a Polars DataFrame.

        Parameters
        ----------
        path : str
            Repo-relative path of the CSV.

        Returns
        -------
        pl.DataFrame
            The parsed CSV.
        """
        url = self.api_client.get_raw_file_url(path)
        response = requests.get(url)
        response.raise_for_status()
        return pl.read_csv(io.BytesIO(response.content))

    def _team_code_to_name(self) -> dict[int, str]:
        """Map FCI team_code (== FPL team code) to official team name.

        Returns
        -------
        dict[int, str]
            Mapping from team code to team name.
        """
        return {team.code: team.name for team in self.fpl_api.get_teams()}

    def fetch_season_frames(
        self, long_season: str
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Download and concatenate the FCI frames needed to build merged_gw.

        Parameters
        ----------
        long_season : str
            Long-form season string, e.g. ``"2025-2026"``.

        Returns
        -------
        tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]
            ``(snapshots, matchstats, players)`` where snapshots and matchstats
            are narrowed to the columns the adapter needs and carry an added
            ``gw`` column.
        """
        gameweeks = self.list_gameweeks(long_season)
        if not gameweeks:
            raise ValueError(
                f"No gameweek data found for season {long_season}"
            )
        base = f"data/{long_season}/By Gameweek"
        snapshot_frames: list[pl.DataFrame] = []
        matchstat_frames: list[pl.DataFrame] = []
        for gw in gameweeks:
            gw_dir = f"{base}/GW{gw}"
            snapshot_frames.append(
                self._read_csv(f"{gw_dir}/player_gameweek_stats.csv")
                .select(list(SNAPSHOT_SCHEMA))
                .cast(SNAPSHOT_SCHEMA, strict=False)
                .with_columns(pl.lit(gw).alias("gw"))
            )
            matchstat_frames.append(
                self._read_csv(f"{gw_dir}/playermatchstats.csv")
                .select(list(MATCHSTATS_SCHEMA))
                .cast(MATCHSTATS_SCHEMA, strict=False)
                .with_columns(pl.lit(gw).alias("gw"))
            )
        players = self._read_csv(f"data/{long_season}/players.csv")
        snapshots = pl.concat(snapshot_frames, how="diagonal")
        matchstats = pl.concat(matchstat_frames, how="diagonal")
        return snapshots, matchstats, players

    def build_current_season_merged_gw(
        self, short_season: str
    ) -> pl.DataFrame:
        """Build and write ``data/raw/{short_season}/gws/merged_gw.csv``.

        Parameters
        ----------
        short_season : str
            Short-form season string, e.g. ``"2025-26"``.

        Returns
        -------
        pl.DataFrame
            The merged_gw frame that was written.
        """
        long_season = season_short_to_long(short_season)
        snapshots, matchstats, players = self.fetch_season_frames(long_season)
        merged = build_merged_gw(
            snapshots, matchstats, players, self._team_code_to_name()
        )
        out_dir = self.raw_data_folder / short_season / "gws"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "merged_gw.csv"
        merged.write_csv(out_path)
        logger.info(
            "Wrote %d FCI merged_gw rows for %s", merged.height, short_season
        )
        return merged
