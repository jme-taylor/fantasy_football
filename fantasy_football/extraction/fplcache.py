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

import requests

from fantasy_football.extraction.extractor import GitHubAPIClient

logger = logging.getLogger(__name__)


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

    def event_deadlines(self) -> dict[int, datetime]:
        """Map gameweek id to its deadline datetime (UTC).

        Returns
        -------
        dict[int, datetime]
            ``{event_id: deadline_time}`` parsed from the latest snapshot.
        """
        snapshot = self._read_snapshot(self._latest_snapshot_path())
        return {
            event["id"]: datetime.fromisoformat(
                event["deadline_time"].replace("Z", "+00:00")
            )
            for event in snapshot["events"]
        }

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
