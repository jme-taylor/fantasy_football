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
        raw = lzma.open(io.BytesIO(response.content)).read()
        return json.loads(raw)
