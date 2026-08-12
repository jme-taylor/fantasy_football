import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import polars as pl
import pulp
from pydantic import TypeAdapter

from fantasy_football.constants import TRANSFORMED_DATA_FOLDER
from fantasy_football.fpl_types import (
    GameWeekPlan,
    PlayerGameweekExpectedPoints,
)
from fantasy_football.optimisation.inputs import (
    forward_gameweeks,
    load_optimiser_inputs,
)
from fantasy_football.optimisation.plan_report import write_plan_report

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

logger = logging.getLogger(__name__)

BUDGET = 1000
SQUAD_SIZE = 15
XI_SIZE = 11
MAX_PER_CLUB = 3
SQUAD_BY_POSITION = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
FORMATION_BOUNDS = {"GK": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}
BENCH_WEIGHT = 0.1
HIT_COST = 4
MAX_FREE_TRANSFERS = 5
DEFAULT_HORIZON = 8


@dataclass
class GameweekPlan:
    """The optimiser's decisions for a single gameweek, keyed on element id."""

    gw: int
    squad: list[int]
    starting_xi: list[int]
    captain: int
    transfers_in: list[int]
    transfers_out: list[int]
    hits: int
    free_transfers: int
    expected_points: float


@dataclass
class Plan:
    """A full multi-gameweek optimisation plan."""

    start_gw: int
    horizon: int
    gameweeks: list[GameweekPlan] = field(default_factory=list)
    total_expected_points: float = 0.0

    def to_frame(self, names: Mapping[int, str]) -> pl.DataFrame:
        """Return one row per gameweek summarising the plan.

        Parameters
        ----------
        names : Mapping[int, str]
            Element id to display name. Ids are the plan's identity; names
            exist so the frame is readable.

        Returns
        -------
        pl.DataFrame
            A DataFrame with one row per gameweek in the plan, containing
            columns for the squad, starting XI, captain, transfers, hits,
            free transfers, and expected points.
        """

        def render(elements: list[int]) -> str:
            return ", ".join(names.get(e, str(e)) for e in elements)

        return pl.DataFrame(
            [
                {
                    "gw": g.gw,
                    "squad": render(g.squad),
                    "starting_xi": render(g.starting_xi),
                    "captain": names.get(g.captain, str(g.captain)),
                    "transfers_in": render(g.transfers_in),
                    "transfers_out": render(g.transfers_out),
                    "hits": g.hits,
                    "free_transfers": g.free_transfers,
                    "expected_points": g.expected_points,
                }
                for g in self.gameweeks
            ]
        )


def _by_element(predictions: pl.DataFrame, column: str) -> dict:
    """Return a column keyed on element id.

    Parameters
    ----------
    predictions : pl.DataFrame
        The optimiser's input rows, one per (element, gameweek).
    column : str
        The column to read.

    Returns
    -------
    dict
        Element id to that column's value. A player's price, name and
        position are constant across the horizon, so collapsing their
        gameweek rows loses nothing.
    """
    return dict(
        zip(
            predictions["element"].to_list(),
            predictions[column].to_list(),
            strict=True,
        )
    )


def _validate_initial_squad(
    initial_squad: list[int],
    predictions: pl.DataFrame,
    prices: dict[int, int],
    names: Mapping[int, str],
    budget: int = BUDGET,
) -> None:
    """Validate a carried-in squad, raising ValueError with named offenders.

    Checks, in order: data availability (price + prediction rows), duplicates,
    squad size and position split, the per-club cap, and the budget. Each
    failure raises a ValueError naming the offending players, clubs, or value.
    Offenders are reported by name -- the squad is keyed on element id, but an
    error message full of ids is not actionable.

    Parameters
    ----------
    initial_squad : list[int]
        The carried-in squad (element ids).
    predictions : pl.DataFrame
        Prediction rows for the horizon (filtered to the optimised weeks).
    prices : dict[int, int]
        Player price in tenths of a million, keyed on element id.
    names : Mapping[int, str]
        Element id to display name, for error messages.
    budget : int, optional
        The effective budget ceiling. Defaults to BUDGET (1000).
    """

    def label(element: int) -> str:
        return f"{names.get(element, '?')} ({element})"

    known = set(predictions["element"].to_list())
    missing_pred = sorted(p for p in initial_squad if p not in known)
    missing_price = sorted(p for p in initial_squad if p not in prices)
    if missing_pred or missing_price:
        raise ValueError(
            "initial_squad players missing data: no predictions for "
            f"{[label(p) for p in missing_pred]}, no price for "
            f"{[label(p) for p in missing_price]}"
        )

    duplicates = sorted(
        {p for p in initial_squad if initial_squad.count(p) > 1}
    )
    if duplicates:
        raise ValueError(
            f"initial_squad has duplicate players: "
            f"{[label(p) for p in duplicates]}"
        )

    if len(initial_squad) != SQUAD_SIZE:
        raise ValueError(
            f"initial_squad must have {SQUAD_SIZE} players, "
            f"got {len(initial_squad)}"
        )

    pos = dict(
        zip(predictions["element"], predictions["position"], strict=False)
    )
    club = dict(zip(predictions["element"], predictions["team"], strict=False))

    counts = {position: 0 for position in SQUAD_BY_POSITION}
    for p in initial_squad:
        counts[pos[p]] += 1
    if counts != SQUAD_BY_POSITION:
        raise ValueError(
            f"initial_squad has wrong position split {counts}, "
            f"expected {SQUAD_BY_POSITION}"
        )

    club_counts: dict[str, int] = {}
    for p in initial_squad:
        club_counts[club[p]] = club_counts.get(club[p], 0) + 1
    over = {c: n for c, n in club_counts.items() if n > MAX_PER_CLUB}
    if over:
        raise ValueError(
            f"initial_squad exceeds {MAX_PER_CLUB} players per club: {over}"
        )

    value = sum(prices[p] for p in initial_squad)
    if value > budget:
        raise ValueError(
            f"initial_squad value {value} exceeds budget {budget}"
        )


def _build_problem(
    predictions: pl.DataFrame,
    prices: dict[int, int],
    weeks: list[int],
    start_gw: int,
    initial_squad: list[int] | None = None,
    free_transfers: int = 1,
    bench_weight: float = BENCH_WEIGHT,
    budget: int = BUDGET,
) -> tuple[pulp.LpProblem, dict]:
    """Construct the multi-week FPL MILP.

    Decision variables, per player p and week t:
      own[p,t], start[p,t], cap[p,t], buy[p,t], sell[p,t]  (all binary)
      ft[t]   integer free transfers banked at the start of week t (0..5)
      paid[t] integer transfers paid for as hits in week t (>= 0)

    Free-transfer banking is enforced with upper bounds only:
      ft[t+1] <= 5  and  ft[t+1] <= ft[t] - transfers[t] + paid[t] + 1.
    The objective rewards banked transfers indirectly (they relax future
    `paid` constraints), so the solver drives ft to min(5, earned) without
    needing auxiliary binaries.

    Returns the problem plus a dict of variable containers for extraction.

    Parameters
    ----------
    predictions : pl.DataFrame
        Rows of (element, position, team, gw, predicted_points).
    prices : dict[int, int]
        Player price in tenths of a million, keyed on element id.
    weeks : list[int]
        Gameweek numbers to optimise over.
    start_gw : int
        The first gameweek of the horizon.
    initial_squad : list[int] or None, optional
        Players already owned before the horizon starts. When provided,
        start_gw uses the carried-in squad as the baseline and applies
        the transfer identity (own = initial + buy - sell). When None,
        a fresh free-build squad is constructed at start_gw.
    free_transfers : int, optional
        Number of free transfers available at start_gw when initial_squad
        is provided. Ignored for a free-build (initial_squad=None).
        Defaults to 1.
    bench_weight : float, optional
        Weight applied to bench players' predicted points.
    budget : int, optional
        The effective budget ceiling in tenths of a million. Defaults to
        BUDGET (1000).

    Returns
    -------
    tuple[pulp.LpProblem, dict]
        The PuLP problem and a dict of variable containers keyed by
        ``own``, ``start``, ``cap``, ``buy``, ``sell``, ``ft``,
        ``paid``, ``points``, and ``pos``.
    """
    players = predictions["element"].unique().to_list()
    pos = dict(
        zip(predictions["element"], predictions["position"], strict=False)
    )
    club = dict(zip(predictions["element"], predictions["team"], strict=False))
    points = {
        (row["element"], row["gw"]): row["predicted_points"]
        for row in predictions.iter_rows(named=True)
    }

    prob = pulp.LpProblem("fpl_optimisation", pulp.LpMaximize)

    own, start, cap, buy, sell = {}, {}, {}, {}, {}
    for t in weeks:
        for p in players:
            own[p, t] = pulp.LpVariable(f"own_{p}_{t}", cat="Binary")
            start[p, t] = pulp.LpVariable(f"start_{p}_{t}", cat="Binary")
            cap[p, t] = pulp.LpVariable(f"cap_{p}_{t}", cat="Binary")
            buy[p, t] = pulp.LpVariable(f"buy_{p}_{t}", cat="Binary")
            sell[p, t] = pulp.LpVariable(f"sell_{p}_{t}", cat="Binary")

    ft, paid, transfers = {}, {}, {}
    for t in weeks:
        ft[t] = pulp.LpVariable(
            f"ft_{t}",
            lowBound=0,
            upBound=MAX_FREE_TRANSFERS,
            cat="Integer",
        )
        paid[t] = pulp.LpVariable(f"paid_{t}", lowBound=0, cat="Integer")
        transfers[t] = pulp.lpSum(buy[p, t] for p in players)

    for t in weeks:
        prob += pulp.lpSum(own[p, t] for p in players) == SQUAD_SIZE
        for position, n in SQUAD_BY_POSITION.items():
            prob += (
                pulp.lpSum(own[p, t] for p in players if pos[p] == position)
                == n
            )
        prob += pulp.lpSum(prices[p] * own[p, t] for p in players) <= budget
        for c in set(club.values()):
            prob += (
                pulp.lpSum(own[p, t] for p in players if club[p] == c)
                <= MAX_PER_CLUB
            )
        prob += pulp.lpSum(start[p, t] for p in players) == XI_SIZE
        for p in players:
            prob += start[p, t] <= own[p, t]
            prob += cap[p, t] <= start[p, t]
        for position, (lo, hi) in FORMATION_BOUNDS.items():
            count = pulp.lpSum(
                start[p, t] for p in players if pos[p] == position
            )
            prob += count >= lo
            prob += count <= hi
        prob += pulp.lpSum(cap[p, t] for p in players) == 1

    free_build = initial_squad is None
    squad_set = set(initial_squad or [])
    initial = {p: int(p in squad_set) for p in players}

    ordered = sorted(weeks)
    for k, t in enumerate(ordered):
        if k == 0:
            # First gameweek: set the opening conditions.
            if free_build:
                prob += ft[t] == 1
                prob += paid[t] == 0
                for p in players:
                    prob += buy[p, t] == own[p, t]
                    prob += sell[p, t] == 0
            else:
                prob += ft[t] == free_transfers
                for p in players:
                    prob += own[p, t] == initial[p] + buy[p, t] - sell[p, t]
                    prob += buy[p, t] + sell[p, t] <= 1
                prob += paid[t] >= transfers[t] - ft[t]
            continue
        prev = ordered[k - 1]
        for p in players:
            prob += own[p, t] == own[p, prev] + buy[p, t] - sell[p, t]
        prob += paid[t] >= transfers[t] - ft[t]
        prob += ft[t] <= MAX_FREE_TRANSFERS
        first_is_free_build = prev == start_gw and free_build
        prev_transfers = 0 if first_is_free_build else transfers[prev]
        prev_paid = 0 if first_is_free_build else paid[prev]
        prob += ft[t] <= ft[prev] - prev_transfers + prev_paid + 1

    xi_points = pulp.lpSum(
        points.get((p, t), 0.0) * start[p, t] for t in weeks for p in players
    )
    captain_points = pulp.lpSum(
        points.get((p, t), 0.0) * cap[p, t] for t in weeks for p in players
    )
    bench_points = pulp.lpSum(
        points.get((p, t), 0.0) * (own[p, t] - start[p, t])
        for t in weeks
        for p in players
    )
    hit_cost = pulp.lpSum(HIT_COST * paid[t] for t in weeks)
    prob += xi_points + captain_points + bench_weight * bench_points - hit_cost

    variables = {
        "own": own,
        "start": start,
        "cap": cap,
        "buy": buy,
        "sell": sell,
        "ft": ft,
        "paid": paid,
        "points": points,
        "pos": pos,
    }
    return prob, variables


def _solve_problem(prob: pulp.LpProblem) -> str:
    """Solve the problem with CBC (silently) and return the status name.

    Parameters
    ----------
    prob : pulp.LpProblem
        A fully-constructed PuLP optimisation problem.

    Returns
    -------
    str
        The solver status string, e.g. ``"Optimal"``.
    """
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    return pulp.LpStatus[prob.status]


def _extract_plan(
    variables: dict,
    weeks: list[int],
    start_gw: int,
    initial_squad: list[int] | None = None,
) -> Plan:
    """Convert solved MILP variables into a Plan.

    Parameters
    ----------
    variables: dict
        The container returned by `_build_problem`.
    weeks: list[int]
        The gameweeks that were optimised, in any order.
    start_gw: int
        The first gameweek of the horizon.
    initial_squad: list[int] or None, optional
        The squad carried into the horizon. Controls whether start_gw transfers
        are reported: when None (free build), start_gw transfers are blanked
        (the opening squad is just "bought", not transferred into). When
        provided, real start_gw transfers are reported in the plan.

    Returns
    -------
    Plan
        The structured per-gameweek plan.
    """
    own, start, cap = variables["own"], variables["start"], variables["cap"]
    buy, sell = variables["buy"], variables["sell"]
    ft, paid = variables["ft"], variables["paid"]
    points = variables["points"]

    def chosen(container, t):
        return [
            element
            for (element, week), var in container.items()
            if week == t and round(var.value()) == 1
        ]

    free_build = initial_squad is None
    gameweeks: list[GameweekPlan] = []
    total = 0.0
    for t in sorted(weeks):
        squad = chosen(own, t)
        xi = chosen(start, t)
        captain = chosen(cap, t)[0]
        blank_start = t == start_gw and free_build
        ins = [] if blank_start else chosen(buy, t)
        outs = [] if blank_start else chosen(sell, t)
        hits = int(round(paid[t].value())) * HIT_COST
        xi_pts = sum(points.get((e, t), 0.0) for e in xi)
        captain_pts = points.get((captain, t), 0.0)
        bench_pts = sum(points.get((e, t), 0.0) for e in squad if e not in xi)
        expected = xi_pts + captain_pts + BENCH_WEIGHT * bench_pts - hits
        total += expected
        gameweeks.append(
            GameweekPlan(
                gw=t,
                squad=squad,
                starting_xi=xi,
                captain=captain,
                transfers_in=ins,
                transfers_out=outs,
                hits=hits,
                free_transfers=int(round(ft[t].value())),
                expected_points=expected,
            )
        )
    return Plan(
        start_gw=start_gw,
        horizon=len(weeks),
        gameweeks=gameweeks,
        total_expected_points=total,
    )


def _to_gameweek_plans(
    plan: Plan,
    names: Mapping[int, str],
    points: dict[tuple[int, int], float],
) -> list[GameWeekPlan]:
    """Convert an internal element-keyed Plan into typed GameWeekPlans.

    Parameters
    ----------
    plan : Plan
        The solved internal plan (element ids).
    names : Mapping[int, str]
        Mapping of element id to display name.
    points : dict[tuple[int, int], float]
        Mapping of (element, gw) to predicted points.

    Returns
    -------
    list[GameWeekPlan]
        One typed plan per gameweek, players carried as
        PlayerGameweekExpectedPoints.

    Raises
    ------
    KeyError
        If a planned player has no name in ``names``.
    """

    def to_player(element: int, gw: int) -> PlayerGameweekExpectedPoints:
        if element not in names:
            raise KeyError(f"No name for element {element}")
        return PlayerGameweekExpectedPoints(
            player_id=element,
            player_name=names[element],
            expected_points=points.get((element, gw), 0.0),
        )

    plans: list[GameWeekPlan] = []
    for g in plan.gameweeks:
        plans.append(
            GameWeekPlan(
                gameweek=g.gw,
                squad=[to_player(e, g.gw) for e in g.squad],
                starting_xi=[to_player(e, g.gw) for e in g.starting_xi],
                captain=to_player(g.captain, g.gw),
                transfers_in=[to_player(e, g.gw) for e in g.transfers_in],
                transfers_out=[to_player(e, g.gw) for e in g.transfers_out],
                hits=g.hits,
                free_transfers=g.free_transfers,
                expected_points=g.expected_points,
            )
        )
    return plans


def _resolve_weeks(
    season: str,
    start_gw: int,
    horizon: int,
    connection: "DuckDBPyConnection | None",
) -> list[int]:
    """Return the horizon's gameweeks, checking start_gw against the data.

    Forward predictions always begin at the gameweek after the last played
    one, so ``start_gw`` is not free: it has to be the first gameweek that
    has been predicted for. A team file naming any other gameweek is stale
    (or the pipeline has not been re-run), and silently optimising the wrong
    week is worse than refusing to.

    Parameters
    ----------
    season : str
        The season being optimised.
    start_gw : int
        The requested first gameweek.
    horizon : int
        Number of gameweeks to plan from start_gw inclusive.
    connection : duckdb.DuckDBPyConnection | None
        An open connection, or None to open one.

    Returns
    -------
    list[int]
        The gameweeks to optimise.

    Raises
    ------
    ValueError
        When nothing has been predicted, or start_gw is not the first
        predicted gameweek.
    """
    available = forward_gameweeks(season, connection)
    if not available:
        raise ValueError(
            f"No forward predictions stored for {season}. Run the points "
            f"models' predict_forward before optimising."
        )
    if start_gw != available[0]:
        raise ValueError(
            f"start_gw {start_gw} is not the first predicted gameweek "
            f"({available[0]}) for {season}. Forward predictions start after "
            f"the last played gameweek, so either the team file is for a "
            f"gameweek that has already been played, or the pipeline needs "
            f"re-running."
        )
    return [
        gw for gw in range(start_gw, start_gw + horizon) if gw in available
    ]


def optimise_plan(
    season: str,
    start_gw: int,
    horizon: int | None = None,
    initial_squad: list[int] | None = None,
    free_transfers: int = 1,
    bank: int = 0,
    connection: "DuckDBPyConnection | None" = None,
) -> list[GameWeekPlan]:
    """Optimise the squad/XI/captain/transfers over a future horizon.

    Loads predictions and prices from the database, builds and solves the
    MILP, then writes and returns the plans.

    Parameters
    ----------
    season: str
        The season to optimise (e.g. "2026-27").
    start_gw: int
        The gameweek treated as "now"; the first gameweek of the horizon. Must
        be the first gameweek with forward predictions.
    horizon: int | None
        Number of gameweeks to plan from start_gw inclusive. Defaults to
        DEFAULT_HORIZON, clamped to the gameweeks available in predictions.
    initial_squad: list[int] | None
        Required when start_gw > 1: the element ids carried into that
        gameweek. Ignored at GW1 (free build).
    free_transfers: int
        Free transfers available at start_gw (1..MAX_FREE_TRANSFERS).
        Ignored at start_gw == 1 (a free build always opens with one free transfer).
    bank: int
        Money in the bank (tenths of a million) added to the carried-in
        squad's value to form the budget. Ignored for a free build
        (start_gw == 1). Defaults to 0.
    connection: duckdb.DuckDBPyConnection | None
        An open connection. When None, one is opened per table read.

    Returns
    -------
    list[GameWeekPlan]
        One typed plan per gameweek. The same plans are written to
        ``optimisation_plan.jsonl`` (one GameWeekPlan per line).
    """
    if not 1 <= free_transfers <= MAX_FREE_TRANSFERS:
        raise ValueError(
            f"free_transfers must be in 1..{MAX_FREE_TRANSFERS}, "
            f"got {free_transfers}"
        )
    if start_gw == 1:
        if initial_squad is not None:
            logger.warning(
                "start_gw == 1 is a free build; ignoring initial_squad."
            )
        if bank:
            logger.warning("start_gw == 1 is a free build; ignoring bank.")
        initial_squad = None
    elif initial_squad is None:
        raise ValueError(
            f"start_gw {start_gw} > 1 requires an initial_squad "
            "(the team carried into that gameweek)."
        )

    if horizon is None:
        horizon = DEFAULT_HORIZON
    weeks = _resolve_weeks(season, start_gw, horizon, connection)
    predictions = load_optimiser_inputs(season, weeks, connection)

    # TODO(JT): prices are the current market value, so the optimiser assumes
    # a player can be sold for what they now cost. FPL sells at purchase price
    # plus half the profit, so any squad player who has risen is worth less
    # than this thinks and the budget is over-estimated. Fixing it needs
    # per-player purchase prices carried in the team file.
    prices = _by_element(predictions, "value")
    names = _by_element(predictions, "name")
    positions = _by_element(predictions, "position")

    if initial_squad is not None:
        # Any squad player missing a price is caught with a clear error in
        # _validate_initial_squad below; the guard just avoids a KeyError here.
        budget = sum(prices[p] for p in initial_squad if p in prices) + bank
        _validate_initial_squad(
            initial_squad, predictions, prices, names, budget
        )
    else:
        budget = BUDGET

    prob, variables = _build_problem(
        predictions,
        prices,
        weeks,
        start_gw,
        initial_squad,
        free_transfers,
        budget=budget,
    )
    status = _solve_problem(prob)
    if status != "Optimal":
        raise RuntimeError(f"Solver finished with status {status!r}")
    plan = _extract_plan(variables, weeks, start_gw, initial_squad)

    points = variables["points"]
    gameweek_plans = _to_gameweek_plans(plan, names, points)

    TRANSFORMED_DATA_FOLDER.mkdir(exist_ok=True, parents=True)
    adapter = TypeAdapter(GameWeekPlan)
    out_path = TRANSFORMED_DATA_FOLDER.joinpath("optimisation_plan.jsonl")
    with out_path.open("wb") as f:
        for gw_plan in gameweek_plans:
            f.write(adapter.dump_json(gw_plan))
            f.write(b"\n")
    write_plan_report(
        gameweek_plans,
        positions,
        TRANSFORMED_DATA_FOLDER.joinpath("optimisation_plan.md"),
    )
    logger.info(
        "Optimised gws %s, total expected points %.1f",
        weeks,
        plan.total_expected_points,
    )
    return gameweek_plans
