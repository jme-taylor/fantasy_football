"""Tests for the FPL squad optimiser."""

import polars as pl
import pytest

from fantasy_football import optimisation
from fantasy_football.optimisation import GameweekPlan, Plan, _load_prices


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


def _write_merged_gw(tmp_path, rows) -> None:
    """Write a minimal merged_gw.csv under tmp_path/raw/<season>/gws/."""
    raw = tmp_path / "raw" / "2025-26" / "gws"
    raw.mkdir(parents=True)
    pl.DataFrame(rows).write_csv(raw / "merged_gw.csv")


def test_load_prices_uses_latest_value_at_or_before_start_gw(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_load_prices returns the latest price at or before start_gw per player."""
    _write_merged_gw(
        tmp_path,
        {
            "name": ["P1", "P1", "P1", "P2"],
            "value": [50, 55, 60, 80],
            "GW": [9, 10, 11, 10],
        },
    )
    monkeypatch.setattr(optimisation, "RAW_DATA_FOLDER", tmp_path / "raw")
    prices = _load_prices("2025-26", start_gw=10)
    assert prices["P1"] == 55  # GW10 value, not the later GW11=60
    assert prices["P2"] == 80
