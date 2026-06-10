"""Render an optimisation Plan into a human-readable markdown report."""

from collections.abc import Mapping, Sequence
from pathlib import Path

from fantasy_football.fpl_types import (
    GameWeekPlan,
    PlayerGameweekExpectedPoints,
)

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


def _row(
    position: str, player: PlayerGameweekExpectedPoints, flag: str
) -> str:
    """Format one markdown table row for a player."""
    return f"| {position} | {player.player_name} | {player.expected_points:.1f} | {flag} |"


def _render_gameweek(plan: GameWeekPlan, positions: Mapping[str, str]) -> str:
    """Render a single gameweek plan as a markdown section.

    Parameters
    ----------
    plan : GameWeekPlan
        The gameweek to render.
    positions : Mapping[str, str]
        Mapping of player name to position string (GK/DEF/MID/FWD).

    Returns
    -------
    str
        The markdown for this gameweek: a summary header, the XI table with a
        bench divider and bench rows, and (when any transfers happened) a
        trailing ``Transfers —`` line.
    """
    captain_name = plan.captain.player_name
    header = (
        f"## GW{plan.gameweek} — xPts {plan.expected_points:.1f}"
        f" · FT {plan.free_transfers}"
    )
    if plan.hits:
        header += f" · hit −{plan.hits}"
    header += f" · C: {captain_name}"

    in_names = {p.player_name for p in plan.transfers_in}
    xi_names = {p.player_name for p in plan.starting_xi}
    bench = [p for p in plan.squad if p.player_name not in xi_names]

    def flag(player: PlayerGameweekExpectedPoints) -> str:
        marks = []
        if player.player_name == captain_name:
            marks.append("⭐ C")
        if player.player_name in in_names:
            marks.append("↑")
        return " ".join(marks)

    lines = [
        header,
        "",
        "| Pos | Player | xPts | |",
        "|-----|--------|-----:|--|",
    ]
    for pos, player in _ordered_rows(plan.starting_xi, positions):
        lines.append(_row(pos, player, flag(player)))
    lines.append("| --- bench --- | | | |")
    for pos, player in _ordered_rows(bench, positions):
        lines.append(_row(pos, player, flag(player)))

    if plan.transfers_in or plan.transfers_out:
        ins = ", ".join(p.player_name for p in plan.transfers_in) or "—"
        outs = ", ".join(p.player_name for p in plan.transfers_out) or "—"
        lines += ["", f"Transfers — IN: {ins} · OUT: {outs}"]

    return "\n".join(lines)


def render_plan_markdown(
    plans: Sequence[GameWeekPlan],
    positions: Mapping[str, str],
) -> str:
    """Render a multi-gameweek plan as a markdown report.

    Parameters
    ----------
    plans : Sequence[GameWeekPlan]
        The per-gameweek plans, in any order.
    positions : Mapping[str, str]
        Mapping of player name to position string (GK/DEF/MID/FWD).

    Returns
    -------
    str
        The full markdown document: a title followed by one section per
        gameweek in ascending gameweek order.
    """
    ordered = sorted(plans, key=lambda p: p.gameweek)
    sections = [_render_gameweek(p, positions) for p in ordered]
    return "# Optimisation plan\n\n" + "\n\n".join(sections) + "\n"


def write_plan_report(
    plans: Sequence[GameWeekPlan],
    positions: Mapping[str, str],
    path: Path,
) -> None:
    """Render the plan and write it to ``path``.

    Parameters
    ----------
    plans : Sequence[GameWeekPlan]
        The per-gameweek plans.
    positions : Mapping[str, str]
        Mapping of player name to position string (GK/DEF/MID/FWD).
    path : Path
        Destination markdown file; its parent must already exist.
    """
    path.write_text(render_plan_markdown(plans, positions))
