"""Tests for the FPL squad optimiser."""

import polars as pl

from fantasy_football.optimisation import GameweekPlan, Plan


def test_plan_to_frame_has_one_row_per_gameweek() -> None:
    """Plan.to_frame returns one row per GameweekPlan with correct values."""
    plan = Plan(
        start_gw=10,
        horizon=2,
        gameweeks=[
            GameweekPlan(
                gw=10,
                squad=[f"P{i}" for i in range(15)],
                starting_xi=[f"P{i}" for i in range(11)],
                captain="P0",
                transfers_in=[],
                transfers_out=[],
                hits=0,
                free_transfers=1,
                expected_points=50.0,
            ),
            GameweekPlan(
                gw=11,
                squad=[f"P{i}" for i in range(15)],
                starting_xi=[f"P{i}" for i in range(11)],
                captain="P1",
                transfers_in=["P15"],
                transfers_out=["P14"],
                hits=0,
                free_transfers=2,
                expected_points=48.0,
            ),
        ],
        total_expected_points=98.0,
    )
    frame = plan.to_frame()
    assert frame.height == 2
    assert frame["gw"].to_list() == [10, 11]
    assert frame.filter(pl.col("gw") == 11)["captain"].item() == "P1"
    assert frame.filter(pl.col("gw") == 11)["transfers_in"].item() == "P15"
