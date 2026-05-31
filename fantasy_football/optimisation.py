"""FPL squad optimisation via mixed-integer linear programming."""

import logging
from dataclasses import dataclass, field

import polars as pl

from fantasy_football.constants import DATA_FOLDER

logger = logging.getLogger(__name__)

TRANSFORMED_DATA_FOLDER = DATA_FOLDER.joinpath("transformed")
RAW_DATA_FOLDER = DATA_FOLDER.joinpath("raw")

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
