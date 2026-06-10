"""Render an optimisation Plan into a human-readable markdown report."""

from collections.abc import Mapping, Sequence

from fantasy_football.fpl_types import PlayerGameweekExpectedPoints

_POSITION_ORDER = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3, "?": 4}


def _ordered_rows(
    players: Sequence[PlayerGameweekExpectedPoints],
    positions: Mapping[str, str],
) -> list[tuple[str, PlayerGameweekExpectedPoints]]:
    """Pair players with their position and order them for display.

    Players are grouped by position in GK, DEF, MID, FWD order; within a
    group they are sorted by expected points descending. A player whose name
    is absent from ``positions`` is given the position ``"?"`` and sorts last.

    Parameters
    ----------
    players : Sequence[PlayerGameweekExpectedPoints]
        The players to order (e.g. a starting XI or a bench).
    positions : Mapping[str, str]
        Mapping of player name to position string (GK/DEF/MID/FWD).

    Returns
    -------
    list[tuple[str, PlayerGameweekExpectedPoints]]
        ``(position, player)`` pairs in display order.
    """
    rows = [(positions.get(p.player_name, "?"), p) for p in players]
    return sorted(
        rows,
        key=lambda row: (
            _POSITION_ORDER.get(row[0], 4),
            -row[1].expected_points,
        ),
    )
