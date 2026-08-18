"""Render an optimisation Plan into a human-readable markdown report."""

from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from fantasy_football.fpl_types import (
    GameWeekPlan,
    PlayerBreakdown,
    PlayerGameweekExpectedPoints,
)
from fantasy_football.modelling.components import (
    POSITION_COMPONENTS,
    Component,
)

_POSITION_ORDER = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3, "?": 4}

# Short column labels for the breakdown table.
_COMPONENT_HEADERS: dict[Component, str] = {
    Component.APPEARANCE: "App",
    Component.DEFCON: "DefCon",
    Component.GOALS: "Goals",
    Component.ASSISTS: "Ast",
    Component.CONCEDING: "Conc",
    Component.YELLOW_CARDS: "YC",
    Component.SAVES: "Saves",
}

_BONUS_FOOTNOTE = (
    "No component carries bonus or red cards, so every number runs low "
    "by roughly a bonus expectation. A GK additionally carries no "
    "yellow cards, worth about 0.1 a match."
)


class PlayerPrices(BaseModel):
    """The three prices a squad player carries, in tenths of a million.

    They differ only for a carried-in player who has moved: a player bought
    during the horizon has all three equal, because they were bought at
    today's price and no profit has accrued to halve.

    Attributes
    ----------
    purchase : int
        What the player cost when bought.
    current : int
        What the player is worth now.
    selling : int
        What the player would actually fetch, per the FPL selling rule.
    """

    model_config = ConfigDict(extra="forbid")

    purchase: int
    current: int
    selling: int


def _money(tenths: int) -> str:
    """Render a price held in tenths of a million the way the game shows it."""
    return f"{tenths / 10:.1f}"


def _ordered_rows(
    players: Sequence[PlayerGameweekExpectedPoints],
    positions: Mapping[int, str],
) -> list[tuple[str, PlayerGameweekExpectedPoints]]:
    """Pair players with their position and order them for display.

    Players are grouped by position in GK, DEF, MID, FWD order; within a
    group they are sorted by expected points descending. A player whose id
    is absent from ``positions`` is given the position ``"?"`` and sorts last.

    Parameters
    ----------
    players : Sequence[PlayerGameweekExpectedPoints]
        The players to order (e.g. a starting XI or a bench).
    positions : Mapping[int, str]
        Mapping of element id to position string (GK/DEF/MID/FWD).

    Returns
    -------
    list[tuple[str, PlayerGameweekExpectedPoints]]
        ``(position, player)`` pairs in display order.
    """
    rows = [(positions.get(p.player_id, "?"), p) for p in players]
    return sorted(
        rows,
        key=lambda row: (
            _POSITION_ORDER.get(row[0], 4),
            -row[1].expected_points,
        ),
    )


def _row(
    position: str,
    player: PlayerGameweekExpectedPoints,
    flag: str,
    prices: Mapping[int, PlayerPrices],
) -> str:
    """Format one markdown table row for a player."""
    money = prices.get(player.player_id)
    cells = (
        ["—", "—", "—"]
        if money is None
        else [
            _money(money.purchase),
            _money(money.current),
            _money(money.selling),
        ]
    )
    return (
        f"| {position} | {player.player_name} "
        f"| {player.expected_points:.1f} "
        f"| {' | '.join(cells)} | {flag} |"
    )


def _value_line(plan: GameWeekPlan, prices: Mapping[int, PlayerPrices]) -> str:
    """Summarise the money behind a gameweek's squad.

    Squad value is what the squad would raise if liquidated -- selling prices
    plus the bank -- and is the quantity the next transfer is actually
    constrained by. Team value is the number the FPL site displays: current
    prices plus the bank. They are shown together and labelled distinctly
    because reading one as the other is how a plan's affordability stops
    making sense.
    """
    owned = [prices[p.player_id] for p in plan.squad if p.player_id in prices]
    squad_value = sum(price.selling for price in owned) + plan.bank
    team_value = sum(price.current for price in owned) + plan.bank
    return (
        f"Value — squad {_money(squad_value)} "
        f"· team {_money(team_value)} "
        f"· bank {_money(plan.bank)}"
    )


def _breakdown_table(
    position: str,
    players: Sequence[tuple[str, PlayerGameweekExpectedPoints]],
    xi_ids: set[int],
    breakdowns: Mapping[int, PlayerBreakdown],
) -> list[str]:
    """Render one position's component table.

    A component with no row renders as a dash rather than a zero: nothing
    was predicted, which is a different statement from a prediction that
    the player scores nothing there.
    """
    components = POSITION_COMPONENTS[position]
    headers = [_COMPONENT_HEADERS[component] for component in components]
    lines = [
        "",
        f"#### {position}",
        "",
        f"| Player | {' | '.join(headers)} | |",
        f"|--------|{'|'.join(['-----:'] * len(headers))}|--|",
    ]
    for _, player in players:
        breakdown = breakdowns.get(player.player_id)
        points = breakdown.components if breakdown else {}
        cells = [
            "—" if component not in points else f"{points[component]:.2f}"
            for component in components
        ]
        marks = []
        if player.player_id in xi_ids:
            marks.append("XI")
        if breakdown is not None and breakdown.fixtures > 1:
            marks.append(f"×{breakdown.fixtures}")
        lines.append(
            f"| {player.player_name} | {' | '.join(cells)} "
            f"| {' '.join(marks)} |"
        )
    return lines


def _render_breakdown(
    plan: GameWeekPlan,
    positions: Mapping[int, str],
    breakdowns: Mapping[int, PlayerBreakdown],
) -> list[str]:
    """Render the squad's component tables, one per position."""
    rows = _ordered_rows(plan.squad, positions)
    xi_ids = {p.player_id for p in plan.starting_xi}
    lines = ["", "### Breakdown"]
    ordered_positions = sorted(
        POSITION_COMPONENTS, key=lambda p: _POSITION_ORDER.get(p, 4)
    )
    for position in ordered_positions:
        players = [row for row in rows if row[0] == position]
        if players:
            lines += _breakdown_table(position, players, xi_ids, breakdowns)
    return lines


def _render_gameweek(
    plan: GameWeekPlan,
    positions: Mapping[int, str],
    prices: Mapping[int, PlayerPrices],
    breakdowns: Mapping[int, PlayerBreakdown],
) -> str:
    """Render a single gameweek plan as a markdown section.

    Parameters
    ----------
    plan : GameWeekPlan
        The gameweek to render.
    positions : Mapping[int, str]
        Mapping of element id to position string (GK/DEF/MID/FWD).
    prices : Mapping[int, PlayerPrices]
        Mapping of element id to purchase, current and selling price.
    breakdowns : Mapping[int, PlayerBreakdown]
        Mapping of element id to this gameweek's component points.

    Returns
    -------
    str
        The markdown for this gameweek: a summary header, the XI table with a
        bench divider and bench rows, (when any transfers happened) a
        ``Transfers —`` line, a ``Value —`` line, and the per-position
        component breakdown.
    """
    captain_id = plan.captain.player_id
    header = (
        f"## GW{plan.gameweek} — xPts {plan.expected_points:.1f}"
        f" · FT {plan.free_transfers}"
    )
    if plan.hits:
        header += f" · hit −{plan.hits}"
    header += f" · C: {plan.captain.player_name}"

    in_ids = {p.player_id for p in plan.transfers_in}
    xi_ids = {p.player_id for p in plan.starting_xi}
    bench = [p for p in plan.squad if p.player_id not in xi_ids]

    def flag(player: PlayerGameweekExpectedPoints) -> str:
        marks = []
        if player.player_id == captain_id:
            marks.append("⭐ C")
        if player.player_id in in_ids:
            marks.append("↑")
        return " ".join(marks)

    lines = [
        header,
        "",
        "| Pos | Player | xPts | Buy | Now | Sell | |",
        "|-----|--------|-----:|----:|----:|-----:|--|",
    ]
    for pos, player in _ordered_rows(plan.starting_xi, positions):
        lines.append(_row(pos, player, flag(player), prices))
    lines.append("| --- bench --- | | | | | | |")
    for pos, player in _ordered_rows(bench, positions):
        lines.append(_row(pos, player, flag(player), prices))

    if plan.transfers_in or plan.transfers_out:
        ins = ", ".join(p.player_name for p in plan.transfers_in) or "—"
        outs = ", ".join(p.player_name for p in plan.transfers_out) or "—"
        lines += ["", f"Transfers — IN: {ins} · OUT: {outs}"]

    lines += ["", _value_line(plan, prices)]
    lines += _render_breakdown(plan, positions, breakdowns)

    return "\n".join(lines)


def render_plan_markdown(
    plans: Sequence[GameWeekPlan],
    positions: Mapping[int, str],
    prices: Mapping[int, PlayerPrices],
    breakdowns: Mapping[tuple[int, int], PlayerBreakdown],
) -> str:
    """Render a multi-gameweek plan as a markdown report.

    Parameters
    ----------
    plans : Sequence[GameWeekPlan]
        The per-gameweek plans, in any order.
    positions : Mapping[int, str]
        Mapping of element id to position string (GK/DEF/MID/FWD).
    prices : Mapping[int, PlayerPrices]
        Mapping of element id to purchase, current and selling price.
    breakdowns : Mapping[tuple[int, int], PlayerBreakdown]
        Mapping of (element id, gameweek) to that gameweek's component
        points.

    Returns
    -------
    str
        The full markdown document: a title followed by one section per
        gameweek in ascending gameweek order.
    """
    ordered = sorted(plans, key=lambda p: p.gameweek)
    sections = [
        _render_gameweek(
            plan,
            positions,
            prices,
            {
                element: breakdown
                for (element, gw), breakdown in breakdowns.items()
                if gw == plan.gameweek
            },
        )
        for plan in ordered
    ]
    body = "\n\n".join(sections)
    return f"# Optimisation plan\n\n{body}\n\n{_BONUS_FOOTNOTE}\n"


def write_plan_report(
    plans: Sequence[GameWeekPlan],
    positions: Mapping[int, str],
    prices: Mapping[int, PlayerPrices],
    breakdowns: Mapping[tuple[int, int], PlayerBreakdown],
    path: Path,
) -> None:
    """Render the plan and write it to ``path``.

    Parameters
    ----------
    plans : Sequence[GameWeekPlan]
        The per-gameweek plans.
    positions : Mapping[int, str]
        Mapping of element id to position string (GK/DEF/MID/FWD).
    prices : Mapping[int, PlayerPrices]
        Mapping of element id to purchase, current and selling price.
    breakdowns : Mapping[tuple[int, int], PlayerBreakdown]
        Mapping of (element id, gameweek) to that gameweek's component
        points.
    path : Path
        Destination markdown file; its parent must already exist.
    """
    path.write_text(
        render_plan_markdown(plans, positions, prices, breakdowns),
        encoding="utf-8",
    )
