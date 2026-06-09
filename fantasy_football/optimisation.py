import logging
from dataclasses import dataclass, field

import polars as pl
import pulp
from pydantic import TypeAdapter

from fantasy_football.constants import RAW_DATA_FOLDER, TRANSFORMED_DATA_FOLDER
from fantasy_football.fpl_types import (
    GameWeekPlan,
    PlayerGameweekExpectedPoints,
)

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
    """The optimiser's decisions for a single gameweek."""

    gw: int
    squad: list[str]
    starting_xi: list[str]
    captain: str
    transfers_in: list[str]
    transfers_out: list[str]
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

    def to_frame(self) -> pl.DataFrame:
        """Return one row per gameweek summarising the plan.

        Returns
        -------
        pl.DataFrame
            A DataFrame with one row per gameweek in the plan, containing
            columns for the squad, starting XI, captain, transfers, hits,
            free transfers, and expected points.
        """
        return pl.DataFrame(
            [
                {
                    "gw": g.gw,
                    "squad": ", ".join(g.squad),
                    "starting_xi": ", ".join(g.starting_xi),
                    "captain": g.captain,
                    "transfers_in": ", ".join(g.transfers_in),
                    "transfers_out": ", ".join(g.transfers_out),
                    "hits": g.hits,
                    "free_transfers": g.free_transfers,
                    "expected_points": g.expected_points,
                }
                for g in self.gameweeks
            ]
        )


def _load_prices(season: str, start_gw: int) -> dict[str, int]:
    """Return each player's price (tenths) as of the latest GW <= start_gw.

    Parameters
    ----------
    season: str
        The season whose raw gameweek data to read.
    start_gw: int
        The pivot gameweek treated as "now".

    Returns
    -------
    dict[str, int]
        Mapping of player name to price in tenths of a million.
    """
    merged = pl.read_csv(
        RAW_DATA_FOLDER.joinpath(season, "gws", "merged_gw.csv")
    ).filter(pl.col("GW") <= start_gw)
    latest_gw = merged.group_by("name").agg(pl.col("GW").max().alias("GW"))
    latest = merged.join(latest_gw, on=["name", "GW"], how="inner")
    return dict(
        zip(
            latest["name"].to_list(),
            latest["value"].to_list(),
            strict=True,
        )
    )


def _prune_players(predictions: pl.DataFrame, k: int) -> pl.DataFrame:
    """Keep only the top-k players per position by mean predicted points.

    Parameters
    ----------
    predictions: pl.DataFrame
        Rows of (name, position, team, gw, predicted_points).
    k: int
        Number of players to keep per position.

    Returns
    -------
    pl.DataFrame
        The input filtered to the retained players (all their rows kept).
    """
    means = predictions.group_by("name", "position").agg(
        pl.col("predicted_points").mean().alias("mean_points")
    )
    ranked = means.with_columns(
        pl.col("mean_points")
        .rank("ordinal", descending=True)
        .over("position")
        .alias("rank")
    )
    keep = ranked.filter(pl.col("rank") <= k)["name"]
    return predictions.filter(pl.col("name").is_in(keep))


def _build_problem(
    predictions: pl.DataFrame,
    prices: dict[str, int],
    weeks: list[int],
    start_gw: int,
    initial_squad: list[str] | None = None,
    free_transfers: int = 1,
    bench_weight: float = BENCH_WEIGHT,
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
        Rows of (name, position, team, gw, predicted_points).
    prices : dict[str, int]
        Mapping of player name to price in tenths of a million.
    weeks : list[int]
        Gameweek numbers to optimise over.
    start_gw : int
        The first gameweek of the horizon.
    initial_squad : list[str] or None, optional
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

    Returns
    -------
    tuple[pulp.LpProblem, dict]
        The PuLP problem and a dict of variable containers keyed by
        ``own``, ``start``, ``cap``, ``buy``, ``sell``, ``ft``,
        ``paid``, ``points``, and ``pos``.
    """
    players = predictions["name"].unique().to_list()
    pos = dict(zip(predictions["name"], predictions["position"], strict=False))
    club = dict(zip(predictions["name"], predictions["team"], strict=False))
    points = {
        (row["name"], row["gw"]): row["predicted_points"]
        for row in predictions.iter_rows(named=True)
    }
    idx = {name: i for i, name in enumerate(players)}

    prob = pulp.LpProblem("fpl_optimisation", pulp.LpMaximize)

    own, start, cap, buy, sell = {}, {}, {}, {}, {}
    for t in weeks:
        for p in players:
            i = idx[p]
            own[p, t] = pulp.LpVariable(f"own_{i}_{t}", cat="Binary")
            start[p, t] = pulp.LpVariable(f"start_{i}_{t}", cat="Binary")
            cap[p, t] = pulp.LpVariable(f"cap_{i}_{t}", cat="Binary")
            buy[p, t] = pulp.LpVariable(f"buy_{i}_{t}", cat="Binary")
            sell[p, t] = pulp.LpVariable(f"sell_{i}_{t}", cat="Binary")

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
        prob += pulp.lpSum(prices[p] * own[p, t] for p in players) <= BUDGET
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
        prev_transfers = 0 if prev == start_gw else transfers[prev]
        prev_paid = 0 if prev == start_gw else paid[prev]
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


def _extract_plan(variables: dict, weeks: list[int], start_gw: int) -> Plan:
    """Convert solved MILP variables into a Plan.

    Parameters
    ----------
    variables: dict
        The container returned by `_build_problem`.
    weeks: list[int]
        The gameweeks that were optimised, in any order.
    start_gw: int
        The first (free-build) gameweek.

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
            name
            for (name, week), var in container.items()
            if week == t and round(var.value()) == 1
        ]

    gameweeks: list[GameweekPlan] = []
    total = 0.0
    for t in sorted(weeks):
        squad = chosen(own, t)
        xi = chosen(start, t)
        captain = chosen(cap, t)[0]
        ins = [] if t == start_gw else chosen(buy, t)
        outs = [] if t == start_gw else chosen(sell, t)
        hits = int(round(paid[t].value())) * HIT_COST
        xi_pts = sum(points.get((n, t), 0.0) for n in xi)
        captain_pts = points.get((captain, t), 0.0)
        bench_pts = sum(points.get((n, t), 0.0) for n in squad if n not in xi)
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
    player_id_map: dict[str, int],
    points: dict[tuple[str, int], float],
) -> list[GameWeekPlan]:
    """Convert an internal name-based Plan into typed GameWeekPlans.

    Parameters
    ----------
    plan : Plan
        The solved internal plan (player names).
    player_id_map : dict[str, int]
        Mapping of player name to FPL element id.
    points : dict[tuple[str, int], float]
        Mapping of (name, gw) to predicted points.

    Returns
    -------
    list[GameWeekPlan]
        One typed plan per gameweek, players carried as
        PlayerGameweekExpectedPoints.

    Raises
    ------
    KeyError
        If a planned player has no id in ``player_id_map``.
    """

    def to_player(name: str, gw: int) -> PlayerGameweekExpectedPoints:
        if name not in player_id_map:
            raise KeyError(f"No player_id for {name!r}")
        return PlayerGameweekExpectedPoints(
            player_id=player_id_map[name],
            player_name=name,
            expected_points=points.get((name, gw), 0.0),
        )

    plans: list[GameWeekPlan] = []
    for g in plan.gameweeks:
        plans.append(
            GameWeekPlan(
                gameweek=g.gw,
                squad=[to_player(n, g.gw) for n in g.squad],
                starting_xi=[to_player(n, g.gw) for n in g.starting_xi],
                captain=to_player(g.captain, g.gw),
                transfers_in=[to_player(n, g.gw) for n in g.transfers_in],
                transfers_out=[to_player(n, g.gw) for n in g.transfers_out],
                hits=g.hits,
                free_transfers=g.free_transfers,
                expected_points=g.expected_points,
            )
        )
    return plans


def optimise_plan(
    season: str,
    start_gw: int,
    horizon: int | None = None,
    initial_squad: list[str] | None = None,
    k: int = 30,
) -> list[GameWeekPlan]:
    """Optimise the squad/XI/captain/transfers over a future horizon.

    Loads predictions and prices, prunes to the top-k players per position,
    builds and solves the MILP, then writes and returns the plans.

    Parameters
    ----------
    season: str
        The season to optimise (e.g. "2025-26").
    start_gw: int
        The gameweek treated as "now"; the first free-build week.
    horizon: int | None
        Number of gameweeks to plan from start_gw inclusive. Defaults to
        DEFAULT_HORIZON, clamped to the gameweeks available in predictions.
    initial_squad: list[str] | None
        Reserved for future use (a pre-existing squad). None means a free
        build at start_gw.
    k: int
        Players retained per position before solving.

    Returns
    -------
    list[GameWeekPlan]
        One typed plan per gameweek. The same plans are written to
        ``optimisation_plan.jsonl`` (one GameWeekPlan per line).
    """
    predictions = pl.read_csv(
        TRANSFORMED_DATA_FOLDER.joinpath("predictions.csv")
    )
    if horizon is None:
        horizon = DEFAULT_HORIZON
    weeks = [
        gw
        for gw in range(start_gw, start_gw + horizon)
        if gw in predictions["gw"].to_list()
    ]
    if not weeks:
        raise ValueError(
            f"No prediction rows for gameweeks "
            f"{start_gw}..{start_gw + horizon - 1}"
        )
    predictions = predictions.filter(pl.col("gw").is_in(weeks))
    predictions = _prune_players(predictions, k)
    prices = _load_prices(season, start_gw)
    missing = set(predictions["name"].to_list()) - set(prices)
    if missing:
        logger.warning(
            "Dropping %d players with no price: %s",
            len(missing),
            sorted(missing)[:5],
        )
        predictions = predictions.filter(~pl.col("name").is_in(list(missing)))

    prob, variables = _build_problem(
        predictions, prices, weeks, start_gw, initial_squad
    )
    status = _solve_problem(prob)
    if status != "Optimal":
        raise RuntimeError(f"Solver finished with status {status!r}")
    plan = _extract_plan(variables, weeks, start_gw)

    player_id_map = dict(
        zip(
            predictions["name"].to_list(),
            predictions["player_id"].to_list(),
            strict=True,
        )
    )
    points = variables["points"]
    gameweek_plans = _to_gameweek_plans(plan, player_id_map, points)

    TRANSFORMED_DATA_FOLDER.mkdir(exist_ok=True, parents=True)
    adapter = TypeAdapter(GameWeekPlan)
    out_path = TRANSFORMED_DATA_FOLDER.joinpath("optimisation_plan.jsonl")
    with out_path.open("wb") as f:
        for gw_plan in gameweek_plans:
            f.write(adapter.dump_json(gw_plan))
            f.write(b"\n")
    logger.info(
        "Optimised gws %s, total expected points %.1f",
        weeks,
        plan.total_expected_points,
    )
    return gameweek_plans
