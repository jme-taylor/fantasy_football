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
