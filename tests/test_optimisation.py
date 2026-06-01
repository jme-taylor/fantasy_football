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


from fantasy_football.optimisation import _prune_players


def test_prune_players_keeps_top_k_per_position_by_mean_points() -> None:
    """_prune_players keeps the top-k players per position by mean predicted points."""
    predictions = pl.DataFrame(
        {
            "name": ["A", "A", "B", "C", "G1", "G2", "G3"],
            "position": ["MID", "MID", "MID", "MID", "GK", "GK", "GK"],
            "team": ["T"] * 7,
            "gw": [10, 11, 10, 10, 10, 10, 10],
            "predicted_points": [9.0, 9.0, 5.0, 1.0, 4.0, 3.0, 2.0],
        }
    )
    kept = _prune_players(predictions, k=2)
    names = set(kept["name"].to_list())
    # Top-2 MIDs by mean are A (9.0) and B (5.0); C is dropped.
    assert "A" in names and "B" in names and "C" not in names
    # Top-2 GKs are G1 and G2; G3 dropped.
    assert "G1" in names and "G2" in names and "G3" not in names


from fantasy_football.optimisation import _build_problem, _solve_problem


def _feasible_universe(gws):
    """Return (predictions_df, prices) with a legal 15-man squad available."""
    counts = {"GK": 3, "DEF": 7, "MID": 7, "FWD": 5}
    rows = {
        "name": [],
        "position": [],
        "team": [],
        "gw": [],
        "predicted_points": [],
    }
    prices = {}
    clubs = ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]
    idx = 0
    for pos, n in counts.items():
        for j in range(n):
            name = f"{pos}{j}"
            prices[name] = 50
            for gw in gws:
                rows["name"].append(name)
                rows["position"].append(pos)
                rows["team"].append(clubs[idx % len(clubs)])
                rows["gw"].append(gw)
                rows["predicted_points"].append(10.0 - j + gw * 0.0)
            idx += 1
    return pl.DataFrame(rows), prices


def test_single_week_squad_is_legal() -> None:
    """The optimiser returns a rules-legal squad, XI, and captain."""
    predictions, prices = _feasible_universe([10])
    prob, v = _build_problem(predictions, prices, weeks=[10], start_gw=10)
    status = _solve_problem(prob)
    assert status == "Optimal"
    own = [
        name
        for (name, t), var in v["own"].items()
        if t == 10 and round(var.value()) == 1
    ]
    start = [
        name
        for (name, t), var in v["start"].items()
        if t == 10 and round(var.value()) == 1
    ]
    cap = [
        name
        for (name, t), var in v["cap"].items()
        if t == 10 and round(var.value()) == 1
    ]
    pos = dict(zip(predictions["name"], predictions["position"], strict=False))
    assert len(own) == 15
    assert sorted(pos[n] for n in own) == (
        ["DEF"] * 5 + ["FWD"] * 3 + ["GK"] * 2 + ["MID"] * 5
    )
    assert len(start) == 11
    assert all(n in own for n in start)
    assert len(cap) == 1 and cap[0] in start
    start_pos = [pos[n] for n in start]
    assert start_pos.count("GK") == 1
    assert 3 <= start_pos.count("DEF") <= 5
    assert 2 <= start_pos.count("MID") <= 5
    assert 1 <= start_pos.count("FWD") <= 3


from fantasy_football.optimisation import _extract_plan, optimise_plan


def test_extract_plan_reads_solved_variables() -> None:
    """A solved single-week problem converts to a populated Plan."""
    predictions, prices = _feasible_universe([10])
    prob, v = _build_problem(predictions, prices, weeks=[10], start_gw=10)
    assert _solve_problem(prob) == "Optimal"
    plan = _extract_plan(v, weeks=[10], start_gw=10)
    assert plan.start_gw == 10
    gw = plan.gameweeks[0]
    assert len(gw.squad) == 15
    assert len(gw.starting_xi) == 11
    assert gw.captain in gw.starting_xi
    assert gw.hits == 0
    assert gw.expected_points > 0
    assert plan.total_expected_points == pytest.approx(gw.expected_points)


def _setup_artifacts(tmp_path, monkeypatch, predictions, prices, gws):
    """Write predictions/merged_gw to a temp tree and patch folder constants."""
    transformed = tmp_path / "transformed"
    transformed.mkdir()
    predictions.write_csv(transformed / "predictions.csv")
    raw = tmp_path / "raw" / "2025-26" / "gws"
    raw.mkdir(parents=True)
    rows = {"name": [], "value": [], "GW": []}
    for name, value in prices.items():
        for gw in gws:
            rows["name"].append(name)
            rows["value"].append(value)
            rows["GW"].append(gw)
    pl.DataFrame(rows).write_csv(raw / "merged_gw.csv")
    monkeypatch.setattr(optimisation, "TRANSFORMED_DATA_FOLDER", transformed)
    monkeypatch.setattr(optimisation, "RAW_DATA_FOLDER", tmp_path / "raw")
    return transformed


def test_optimise_plan_writes_csv_and_returns_plan(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: optimise_plan returns a plan and writes the CSV."""
    predictions, prices = _feasible_universe([10, 11])
    transformed = _setup_artifacts(
        tmp_path, monkeypatch, predictions, prices, gws=[9, 10, 11]
    )
    plan = optimise_plan(season="2025-26", start_gw=10, horizon=2, k=20)
    assert [g.gw for g in plan.gameweeks] == [10, 11]
    assert (transformed / "optimisation_plan.csv").exists()
    written = pl.read_csv(transformed / "optimisation_plan.csv")
    assert written.height == 2
