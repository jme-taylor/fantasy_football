"""Read per-gameweek team from the Randdalf/fplcache bootstrap time-series.

FCI records only a player's final club, so mid-season transfers are wrong for
pre-transfer gameweeks. ``Randdalf/fplcache`` snapshots FPL's bootstrap-static
endpoint four times a day; the snapshot active at a gameweek's deadline gives
the player's actual club that week. ``FplCacheExtractor`` selects that snapshot
per gameweek and emits a ``(gw, element, team_code)`` table.
"""

import io
import json
import logging
import lzma
from datetime import datetime, timedelta, timezone

import polars as pl
import requests

from fantasy_football.extraction.extractor import GitHubAPIClient

logger = logging.getLogger(__name__)

# Schema of the per-(season, gw, element) availability frame emitted by
# build_player_chance_of_playing.
_CHANCE_OF_PLAYING_SCHEMA: dict[str, pl.DataType] = {
    "season": pl.Utf8,
    "gw": pl.Int64,
    "element": pl.Int64,
    "chance_of_playing_this_round": pl.Int64,
}


def _season_window(season: str) -> tuple[datetime, datetime]:
    """Return the earliest and latest plausible deadline for a season.

    A Premier League season's gameweek deadlines fall between June of its
    start year and the end of June in its end year. The window is
    deliberately wide: it exists to catch a whole-season mismatch, not to
    validate individual fixtures.

    Parameters
    ----------
    season : str
        Short-form season string, e.g. ``"2026-27"``.

    Returns
    -------
    tuple[datetime, datetime]
        Inclusive ``(start, end)`` bounds in UTC.
    """
    start_year = int(season[:4])
    return (
        datetime(start_year, 6, 1, tzinfo=timezone.utc),
        datetime(start_year + 1, 7, 1, tzinfo=timezone.utc),
    )


class FplCacheExtractor:
    """Resolve each gameweek's player->team_code map from fplcache snapshots."""

    def __init__(self, api_client: GitHubAPIClient | None = None) -> None:
        """Initialise the extractor.

        Parameters
        ----------
        api_client : GitHubAPIClient | None, optional
            Client pointed at the fplcache repo. Defaults to a new client for
            ``Randdalf/fplcache`` on ``main``.
        """
        self.api_client = api_client or GitHubAPIClient(
            owner="Randdalf", repo="fplcache", branch="main"
        )

    def _list_names(self, path: str) -> list[str]:
        """List the entry names directly under a repo directory.

        Parameters
        ----------
        path : str
            Repo-relative directory path.

        Returns
        -------
        list[str]
            The ``name`` of each entry in the directory.
        """
        return [
            entry["name"] for entry in self.api_client.get_file_details(path)
        ]

    def _latest_snapshot_path(self) -> str:
        """Return the path of the newest snapshot in the cache.

        Returns
        -------
        str
            Repo-relative path of the most recent ``.json.xz`` snapshot.
        """
        year = max(self._list_names("cache"), key=int)
        month = max(self._list_names(f"cache/{year}"), key=int)
        day = max(self._list_names(f"cache/{year}/{month}"), key=int)
        time = max(self._list_names(f"cache/{year}/{month}/{day}"))
        return f"cache/{year}/{month}/{day}/{time}"

    def _read_snapshot(self, path: str) -> dict:
        """Download an LZMA-compressed JSON snapshot and parse it.

        Parameters
        ----------
        path : str
            Repo-relative path of the ``.json.xz`` snapshot.

        Returns
        -------
        dict
            The parsed bootstrap-static object.
        """
        url = self.api_client.get_raw_file_url(path)
        response = requests.get(url)
        response.raise_for_status()
        with lzma.open(io.BytesIO(response.content)) as snapshot:
            raw = snapshot.read()
        return json.loads(raw)

    # How many days past a deadline to search for a snapshot before giving up.
    _MAX_LOOKAHEAD_DAYS = 7

    def _snapshot_path_for(self, deadline: datetime) -> str:
        """Return the path of the first snapshot at or after a deadline.

        Parameters
        ----------
        deadline : datetime
            Timezone-aware gameweek deadline.

        Returns
        -------
        str
            Repo-relative path of the chosen snapshot.

        Raises
        ------
        ValueError
            If no snapshot is found within the lookahead window.
        """
        for offset in range(self._MAX_LOOKAHEAD_DAYS):
            day = deadline.date() + timedelta(days=offset)
            directory = f"cache/{day.year}/{day.month}/{day.day}"
            try:
                names = sorted(self._list_names(directory))
            except Exception:
                continue
            for name in names:
                stamp = name.removesuffix(".json.xz")
                taken = datetime(
                    day.year,
                    day.month,
                    day.day,
                    int(stamp[:2]),
                    int(stamp[2:]),
                    tzinfo=timezone.utc,
                )
                if taken >= deadline:
                    return f"{directory}/{name}"
        raise ValueError(f"No fplcache snapshot found at or after {deadline}")

    def build_player_gw_team(
        self, season: str, gameweeks: list[int]
    ) -> pl.DataFrame:
        """Build the per-gameweek player team_code table.

        Gameweeks with no snapshot yet (future rounds of a live season) are
        skipped; ``build_merged_gw`` fills their team from the neighbouring
        gameweeks. No snapshot for *any* gameweek is fatal, since the resulting
        empty mapping would silently fall back to FCI's end-of-season clubs.

        Parameters
        ----------
        season : str
            Short-form season string, e.g. ``"2025-26"``. Deadlines are read
            from a snapshot taken inside this season, not the latest snapshot
            in the cache, which belongs to whichever season FPL is currently
            serving.
        gameweeks : list[int]
            Gameweek numbers to resolve.

        Returns
        -------
        pl.DataFrame
            One row per ``(gw, element)`` with columns ``gw, element,
            team_code`` (all ``Int64``), where ``element`` is the FPL element
            id (== FCI ``player_id``).

        Raises
        ------
        ValueError
            If no gameweek could be resolved to a snapshot.
        """
        deadlines = self.season_event_deadlines(season)
        frames: list[pl.DataFrame] = []
        for gw in gameweeks:
            try:
                path = self._snapshot_path_for(deadlines[gw])
            # TODO (JT): If no future, stop the loop.
            except ValueError:
                logger.warning(
                    "No fplcache snapshot for %s GW%d yet; skipping.",
                    season,
                    gw,
                )
                continue
            snapshot = self._read_snapshot(path)
            frames.append(
                pl.DataFrame(
                    {
                        "gw": gw,
                        "element": [e["id"] for e in snapshot["elements"]],
                        "team_code": [
                            e["team_code"] for e in snapshot["elements"]
                        ],
                    },
                    schema={
                        "gw": pl.Int64,
                        "element": pl.Int64,
                        "team_code": pl.Int64,
                    },
                )
            )
        if not frames:
            raise ValueError(
                f"No fplcache snapshot found for any of {season} gameweeks "
                f"{gameweeks}"
            )
        return pl.concat(frames, how="vertical")

    def deadline_prices(self, season: str, gw: int) -> dict[int, int]:
        """Return every player's price at a gameweek's deadline.

        FPL prices move nightly, so a price recorded against a gameweek's
        matches is not the price paid at its deadline. Purchase prices need
        the deadline itself, which is what the snapshot at or after it holds.

        Parameters
        ----------
        season : str
            Short-form season string, e.g. ``"2026-27"``.
        gw : int
            Gameweek whose deadline to price at.

        Returns
        -------
        dict[int, int]
            Element id to price in tenths of a million.

        Raises
        ------
        ValueError
            If the season has no such gameweek, or no snapshot covers its
            deadline.
        """
        deadlines = self.season_event_deadlines(season)
        if gw not in deadlines:
            raise ValueError(f"{season} has no GW{gw} to price at.")
        snapshot = self._read_snapshot(self._snapshot_path_for(deadlines[gw]))
        return {
            element["id"]: element["now_cost"]
            for element in snapshot["elements"]
        }

    def _season_probe_datetime(self, season: str) -> datetime:
        """Return a datetime reliably inside a season (1 October of its start year).

        Snapshots exist continuously, so the snapshot at or after this instant
        belongs to ``season`` and carries that season's full deadline list.

        Parameters
        ----------
        season : str
            Short-form season string, e.g. ``"2022-23"``.

        Returns
        -------
        datetime
            ``1 October`` of the season's start year, in UTC.
        """
        return datetime(int(season[:4]), 10, 1, tzinfo=timezone.utc)

    def season_event_deadlines(self, season: str) -> dict[int, datetime]:
        """Map gameweek id to deadline for a specific season.

        Reads the ``events`` array from a snapshot taken inside ``season``.
        Falls back to the latest snapshot when no in-season snapshot exists yet
        (a freshly started current season before the October probe date).

        Parameters
        ----------
        season : str
            Short-form season string, e.g. ``"2022-23"``.

        Returns
        -------
        dict[int, datetime]
            ``{event_id: deadline_time}`` for the season.

        Raises
        ------
        ValueError
            If the resolved deadlines fall outside ``season``'s own window,
            meaning the snapshot belongs to a different season.
        """
        try:
            path = self._snapshot_path_for(self._season_probe_datetime(season))
        except ValueError:
            path = self._latest_snapshot_path()
        snapshot = self._read_snapshot(path)
        deadlines = {
            event["id"]: datetime.fromisoformat(
                event["deadline_time"].replace("Z", "+00:00")
            )
            for event in snapshot["events"]
        }
        start, end = _season_window(season)
        outside = [
            gw
            for gw, deadline in deadlines.items()
            if not start <= deadline <= end
        ]
        if outside:
            raise ValueError(
                f"Snapshot {path} does not hold {season} deadlines: "
                f"gameweeks {sorted(outside)} fall outside "
                f"{start.date()}..{end.date()}. The fplcache fallback "
                f"probably returned a different season's events."
            )
        return deadlines

    def played_gameweeks(
        self,
        season: str,
        gameweeks: list[int],
        now: datetime | None = None,
    ) -> list[int]:
        """Return the gameweeks whose deadline has already passed.

        FCI creates all 38 gameweek folders before a season kicks off, so the
        folder list alone says nothing about what has been played. A passed
        deadline is the cheap first test: it needs only the deadline map, so
        callers can narrow the gameweek list before downloading any CSV.

        Gameweeks missing from the season's deadline map are excluded — FCI
        occasionally lists a folder FPL's ``events`` array does not carry.

        Parameters
        ----------
        season : str
            Short-form season string, e.g. ``"2026-27"``.
        gameweeks : list[int]
            Candidate gameweek numbers.
        now : datetime | None, optional
            Timezone-aware instant to compare deadlines against. Defaults to
            the current UTC time.

        Returns
        -------
        list[int]
            Ascending gameweek numbers whose deadline is at or before ``now``.
        """
        moment = now or datetime.now(timezone.utc)
        deadlines = self.season_event_deadlines(season)
        return sorted(
            gw
            for gw in gameweeks
            if gw in deadlines and deadlines[gw] <= moment
        )

    def build_player_chance_of_playing(
        self, season: str, gameweeks: list[int] | None = None
    ) -> pl.DataFrame:
        """Build the per-gameweek ``chance_of_playing_this_round`` table.

        For each gameweek, the snapshot at or after that gameweek's deadline
        gives FPL's point-in-time availability percentage for the upcoming
        round. ``null`` (no injury doubt) is coalesced to ``100``. Gameweeks
        with no snapshot yet (future rounds of a live season) are skipped.

        Parameters
        ----------
        season : str
            Short-form season string, e.g. ``"2022-23"``.
        gameweeks : list[int] | None, optional
            Gameweeks to resolve. Defaults to every gameweek in the season's
            deadline list.

        Returns
        -------
        pl.DataFrame
            One row per ``(gw, element)`` with columns ``season, gw, element,
            chance_of_playing_this_round``.
        """
        deadlines = self.season_event_deadlines(season)
        if gameweeks is None:
            gameweeks = sorted(deadlines)
        frames: list[pl.DataFrame] = []
        for gw in gameweeks:
            try:
                path = self._snapshot_path_for(deadlines[gw])
            except ValueError:
                logger.warning(
                    "No fplcache snapshot for %s GW%d yet; skipping.",
                    season,
                    gw,
                )
                continue
            snapshot = self._read_snapshot(path)
            frames.append(
                pl.DataFrame(
                    {
                        "season": season,
                        "gw": gw,
                        "element": [e["id"] for e in snapshot["elements"]],
                        "chance_of_playing_this_round": [
                            100
                            if e.get("chance_of_playing_this_round") is None
                            else e["chance_of_playing_this_round"]
                            for e in snapshot["elements"]
                        ],
                    },
                    schema=_CHANCE_OF_PLAYING_SCHEMA,
                )
            )
        if not frames:
            return pl.DataFrame(schema=_CHANCE_OF_PLAYING_SCHEMA)
        return pl.concat(frames, how="vertical")

    def build_player_bio(self, season: str) -> pl.DataFrame:
        """Read static player bio fields from the season's GW1 snapshot.

        These fields exist only in the fplcache bootstrap snapshots (2022-23
        onwards); neither Vaastav's earlier files nor FCI's ``players.csv``
        carry ``birth_date``. GW1 is used because every registered player is
        present at the season's start.

        Parameters
        ----------
        season : str
            Short-form season string, e.g. ``"2025-26"``.

        Returns
        -------
        pl.DataFrame
            One row per ``element`` with ``birth_date``, ``region`` and
            ``team_join_date``. Empty when the season has no GW1 snapshot yet.
        """
        empty = pl.DataFrame(
            schema={
                "element": pl.Int64,
                "birth_date": pl.Date,
                "region": pl.Int64,
                "team_join_date": pl.Date,
            }
        )
        deadlines = self.season_event_deadlines(season)
        try:
            path = self._snapshot_path_for(deadlines[1])
        except (KeyError, ValueError):
            logger.warning(
                "No fplcache snapshot for %s GW1; no bio data available.",
                season,
            )
            return empty

        elements = self._read_snapshot(path)["elements"]
        return pl.DataFrame(
            {
                "element": [e["id"] for e in elements],
                "birth_date": [e.get("birth_date") for e in elements],
                "region": [e.get("region") for e in elements],
                "team_join_date": [e.get("team_join_date") for e in elements],
            },
            schema={
                "element": pl.Int64,
                "birth_date": pl.Utf8,
                "region": pl.Int64,
                "team_join_date": pl.Utf8,
            },
        ).with_columns(
            pl.col("birth_date").str.to_date(format="%Y-%m-%d", strict=False),
            pl.col("team_join_date").str.to_date(
                format="%Y-%m-%d", strict=False
            ),
        )
