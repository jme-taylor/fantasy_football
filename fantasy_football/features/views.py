"""One door onto the session-scoped feature views.

The three registrations below must happen in this order: both form views
read ``opta_match``, which ``register_lookups`` creates. The views are
``TEMP``, so they live and die with the connection -- which is why every
consumer registers them itself rather than assuming a previous caller
did.
"""

from typing import TYPE_CHECKING

from fantasy_football.constants import ROLLING_WINDOW
from fantasy_football.features.match_form import register_match_form
from fantasy_football.features.team_form import register_team_form
from fantasy_football.storage.lookups import register_lookups

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection


def register_feature_views(
    connection: "DuckDBPyConnection",
    rolling_window: int = ROLLING_WINDOW,
) -> None:
    """Register the lookup, player-form and team-form views.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        An open connection. The views last only as long as it does.
    rolling_window : int, optional
        Number of preceding appearances or matches in the form windows.
    """
    register_lookups(connection)
    register_match_form(connection, rolling_window)
    register_team_form(connection, rolling_window)
