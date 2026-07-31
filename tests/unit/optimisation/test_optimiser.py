import polars as pl
import pytest

from fantasy_football.optimisation import optimiser as optimisation
from fantasy_football.optimisation.optimiser import (
    GameweekPlan,
    Plan,
    _load_prices,
)
from fantasy_football.storage import database
from fantasy_football.storage.database import get_connection
from fantasy_football.storage.tables import PLAYER_WEEK


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


def _seed_player_week(
    tmp_path, monkeypatch, *, name, element, gw, value=None, season="2025-26"
) -> None:
    """Seed the player_week DB with rows and point DATABASE_PATH at it."""
    n = len(name)
    frame = pl.DataFrame(
        {
            "season": [season] * n,
            "gw": gw,
            "element": element,
            "name": name,
            "position": ["MID"] * n,
            "team": ["T"] * n,
            "bonus": [0] * n,
            "minutes": [0] * n,
            "round": gw,
            "total_points": [0] * n,
            "value": value if value is not None else [0] * n,
        }
    )
    db_path = tmp_path / "t.duckdb"
    monkeypatch.setattr(database, "DATABASE_PATH", db_path)
    connection = get_connection(db_path)
    try:
        PLAYER_WEEK.upsert_current(connection, frame, season)
    finally:
        connection.close()


def test_load_prices_uses_latest_value_at_or_before_start_gw(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_load_prices returns the latest price at or before start_gw per player."""
    _seed_player_week(
        tmp_path,
        monkeypatch,
        name=["P1", "P1", "P1", "P2"],
        element=[1, 1, 1, 2],
        gw=[9, 10, 11, 10],
        value=[50, 55, 60, 80],
    )
    prices = _load_prices("2025-26", start_gw=10)
    assert prices["P1"] == 55  # GW10 value, not the later GW11=60
    assert prices["P2"] == 80


from fantasy_football.optimisation.optimiser import (
    _build_problem,
    _solve_problem,
)


def _feasible_universe(gws):
    """Return (predictions_df, prices) with a legal 15-man squad available."""
    counts = {"GK": 3, "DEF": 7, "MID": 7, "FWD": 5}
    rows = {
        "name": [],
        "player_id": [],
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
                rows["player_id"].append(idx + 1)
                rows["position"].append(pos)
                rows["team"].append(clubs[idx % len(clubs)])
                rows["gw"].append(gw)
                rows["predicted_points"].append(10.0 - j + gw * 0.0)
            idx += 1
    return pl.DataFrame(rows), prices


# A legal, point-optimal 15-man squad for _feasible_universe (lowest-index
# players score highest: predicted_points = 10 - j).
_LEGAL_SQUAD = (
    ["GK0", "GK1"]
    + ["DEF0", "DEF1", "DEF2", "DEF3", "DEF4"]
    + ["MID0", "MID1", "MID2", "MID3", "MID4"]
    + ["FWD0", "FWD1", "FWD2"]
)
# Same squad but holding FWD3 (7.0) instead of FWD2 (8.0) — a single
# beneficial upgrade is available.
_SQUAD_ONE_OFF = _LEGAL_SQUAD[:-1] + ["FWD3"]


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


def test_initial_squad_with_no_better_option_makes_no_transfers() -> None:
    """A carried-in squad that is already optimal triggers no transfers."""
    predictions, prices = _feasible_universe([10])
    prob, v = _build_problem(
        predictions,
        prices,
        weeks=[10],
        start_gw=10,
        initial_squad=_LEGAL_SQUAD,
        free_transfers=1,
    )
    assert _solve_problem(prob) == "Optimal"
    own = {
        name
        for (name, t), var in v["own"].items()
        if t == 10 and round(var.value()) == 1
    }
    assert own == set(_LEGAL_SQUAD)
    buys = sum(
        round(var.value()) for (name, t), var in v["buy"].items() if t == 10
    )
    assert buys == 0
    assert round(v["paid"][10].value()) == 0


def test_initial_squad_uses_one_free_transfer_to_upgrade() -> None:
    """One free transfer upgrades FWD3 (7.0) to FWD2 (8.0), no hit."""
    predictions, prices = _feasible_universe([10])
    prob, v = _build_problem(
        predictions,
        prices,
        weeks=[10],
        start_gw=10,
        initial_squad=_SQUAD_ONE_OFF,
        free_transfers=1,
    )
    assert _solve_problem(prob) == "Optimal"
    own = {
        name
        for (name, t), var in v["own"].items()
        if t == 10 and round(var.value()) == 1
    }
    assert own == set(_LEGAL_SQUAD)
    buys = sum(
        round(var.value()) for (name, t), var in v["buy"].items() if t == 10
    )
    assert buys == 1
    assert round(v["paid"][10].value()) == 0


def test_initial_squad_pays_a_hit_to_make_extra_transfer() -> None:
    """With one free transfer, two compelling upgrades force a paid hit."""
    rows: dict = {
        "name": [],
        "position": [],
        "team": [],
        "gw": [],
        "predicted_points": [],
    }
    prices: dict = {}
    counts = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
    clubs = ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]
    squad: list[str] = []
    idx = 0
    for pos, n in counts.items():
        for j in range(n):
            name = f"{pos}{j}"
            squad.append(name)
            prices[name] = 50
            rows["name"].append(name)
            rows["position"].append(pos)
            rows["team"].append(clubs[idx % len(clubs)])
            rows["gw"].append(10)
            rows["predicted_points"].append(2.0)
            idx += 1
    # Two MID stars NOT in the initial squad, each worth far more than -4.
    for s, club in enumerate(["C5", "C6"]):
        name = f"STAR{s}"
        prices[name] = 50
        rows["name"].append(name)
        rows["position"].append("MID")
        rows["team"].append(club)
        rows["gw"].append(10)
        rows["predicted_points"].append(100.0)
    predictions = pl.DataFrame(rows)

    prob, v = _build_problem(
        predictions,
        prices,
        weeks=[10],
        start_gw=10,
        initial_squad=squad,
        free_transfers=1,
    )
    assert _solve_problem(prob) == "Optimal"
    buys = sum(
        round(var.value()) for (name, t), var in v["buy"].items() if t == 10
    )
    assert buys == 2  # both stars brought in
    assert round(v["paid"][10].value()) == 1  # one free, one paid (-4)


from fantasy_football.optimisation.optimiser import (
    _extract_plan,
    optimise_plan,
)


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


def test_extract_plan_reports_start_gw_transfers_with_initial_squad() -> None:
    """With a carried-in squad, start_gw transfers appear in the plan."""
    predictions, prices = _feasible_universe([10])
    prob, v = _build_problem(
        predictions,
        prices,
        weeks=[10],
        start_gw=10,
        initial_squad=_SQUAD_ONE_OFF,
        free_transfers=1,
    )
    assert _solve_problem(prob) == "Optimal"
    plan = _extract_plan(
        v, weeks=[10], start_gw=10, initial_squad=_SQUAD_ONE_OFF
    )
    gw = plan.gameweeks[0]
    assert gw.transfers_in == ["FWD2"]
    assert gw.transfers_out == ["FWD3"]
    assert gw.hits == 0


def _setup_artifacts(tmp_path, monkeypatch, predictions, prices, gws):
    """Write predictions to a temp tree, seed prices into the DB, patch paths."""
    transformed = tmp_path / "transformed"
    transformed.mkdir()
    predictions.write_csv(transformed / "predictions.csv")
    monkeypatch.setattr(optimisation, "TRANSFORMED_DATA_FOLDER", transformed)

    name_col: list[str] = []
    element_col: list[int] = []
    gw_col: list[int] = []
    value_col: list[int] = []
    for index, (name, value) in enumerate(prices.items(), start=1):
        for gw in gws:
            name_col.append(name)
            element_col.append(index)
            gw_col.append(gw)
            value_col.append(value)
    _seed_player_week(
        tmp_path,
        monkeypatch,
        name=name_col,
        element=element_col,
        gw=gw_col,
        value=value_col,
    )
    return transformed


def test_optimise_plan_writes_jsonl_and_returns_gameweek_plans(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: GW1 free build returns typed plans and writes JSON Lines."""
    import json

    predictions, prices = _feasible_universe([1, 2])
    transformed = _setup_artifacts(
        tmp_path, monkeypatch, predictions, prices, gws=[1]
    )
    plans = optimise_plan(season="2025-26", start_gw=1, horizon=2)

    assert [p.gameweek for p in plans] == [1, 2]
    assert not (transformed / "optimisation_plan.csv").exists()
    out = transformed / "optimisation_plan.jsonl"
    assert out.exists()

    lines = out.read_text().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["gameweek"] == 1
    assert len(first["squad"]) == 15
    assert len(first["starting_xi"]) == 11
    assert all(p["player_id"] is not None for p in first["squad"])
    xi_names = {p["player_name"] for p in first["starting_xi"]}
    assert first["captain"]["player_name"] in xi_names


def test_optimise_plan_writes_markdown_report(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A GW1 free build also writes a human-readable markdown report."""
    predictions, prices = _feasible_universe([1, 2])
    transformed = _setup_artifacts(
        tmp_path, monkeypatch, predictions, prices, gws=[1]
    )

    optimise_plan(season="2025-26", start_gw=1, horizon=2)

    report = transformed / "optimisation_plan.md"
    assert report.exists()
    text = report.read_text()
    assert "## GW1" in text
    assert "--- bench ---" in text


def test_optimise_plan_requires_initial_squad_after_gw1() -> None:
    """start_gw > 1 with no initial_squad is rejected before any file I/O."""
    with pytest.raises(ValueError, match="initial_squad"):
        optimise_plan(season="2025-26", start_gw=10, horizon=2)


def test_optimise_plan_rejects_bad_free_transfers() -> None:
    """free_transfers must be within 1..MAX_FREE_TRANSFERS."""
    with pytest.raises(ValueError, match="free_transfers"):
        optimise_plan(
            season="2025-26",
            start_gw=1,
            horizon=1,
            free_transfers=6,
        )


def test_optimise_plan_with_initial_squad_reports_start_gw_transfer(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mid-season run honours the carried-in squad and reports its transfer."""
    predictions, prices = _feasible_universe([10, 11])
    _setup_artifacts(tmp_path, monkeypatch, predictions, prices, gws=[9, 10])
    plans = optimise_plan(
        season="2025-26",
        start_gw=10,
        horizon=2,
        initial_squad=_SQUAD_ONE_OFF,
        free_transfers=1,
    )
    gw10 = next(p for p in plans if p.gameweek == 10)
    ins = {p.player_name for p in gw10.transfers_in}
    outs = {p.player_name for p in gw10.transfers_out}
    assert ins == {"FWD2"}
    assert outs == {"FWD3"}
    assert gw10.hits == 0


def test_optimise_plan_ignores_initial_squad_at_gw1(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """At GW1 the build is free: a provided initial_squad is ignored, not required."""
    predictions, prices = _feasible_universe([1, 2])
    _setup_artifacts(tmp_path, monkeypatch, predictions, prices, gws=[1])
    # Passing a squad at GW1 must not raise; it free-builds (no start-gw transfers).
    plans = optimise_plan(
        season="2025-26",
        start_gw=1,
        horizon=2,
        initial_squad=_SQUAD_ONE_OFF,
    )
    gw1 = next(p for p in plans if p.gameweek == 1)
    assert gw1.transfers_in == []
    assert gw1.transfers_out == []


def test_first_week_is_a_free_build_with_no_hits() -> None:
    """The opening squad costs no transfer hit and records no transfers."""
    predictions, prices = _feasible_universe([10])
    prob, v = _build_problem(predictions, prices, weeks=[10], start_gw=10)
    assert _solve_problem(prob) == "Optimal"
    plan = _extract_plan(v, weeks=[10], start_gw=10)
    assert plan.gameweeks[0].hits == 0
    assert plan.gameweeks[0].transfers_in == []
    assert plan.gameweeks[0].transfers_out == []


def test_extra_transfers_incur_hits() -> None:
    """Free-transfer depletion in one week forces paid hits the next.

    Over a three-week horizon the solver:
    - GW10 (free build): ft=1, 0 transfers recorded.
    - GW11: banks 2 FTs, makes 2 transfers (both free) to acquire five
      GW11-specialists, depleting the banked FT allowance.
    - GW12: only ft=1 remains; all five MID slots need swapping for
      GW12-specialists, requiring 4 paid transfers — a hits penalty of 16.

    The scenario uses five GW11-specialist MIDs (score 0/101/0 across
    GW10/11/12) and five GW12-specialist MIDs (score 0/0/100), each pair
    sharing a club with existing base players so the club-ownership cap
    (3 per club) limits how many can be carried in the free build.
    """
    gw11_rows: dict = {
        "name": [],
        "position": [],
        "team": [],
        "gw": [],
        "predicted_points": [],
    }
    prices: dict = {}
    counts = {"GK": 3, "DEF": 7, "MID": 7, "FWD": 5}
    clubs = ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]
    idx = 0
    for pos, n in counts.items():
        for j in range(n):
            name = f"{pos}{j}"
            prices[name] = 50
            for gw in [10, 11, 12]:
                gw11_rows["name"].append(name)
                gw11_rows["position"].append(pos)
                gw11_rows["team"].append(clubs[idx % len(clubs)])
                gw11_rows["gw"].append(gw)
                gw11_rows["predicted_points"].append(10.0 - j)
            idx += 1

    # Five GW11-specialist MIDs (one per club C0-C4).
    for i, club in enumerate(["C0", "C1", "C2", "C3", "C4"]):
        name = f"GW11S{i}"
        prices[name] = 50
        for gw, pts in [(10, 0.0), (11, 101.0), (12, 0.0)]:
            gw11_rows["name"].append(name)
            gw11_rows["position"].append("MID")
            gw11_rows["team"].append(club)
            gw11_rows["gw"].append(gw)
            gw11_rows["predicted_points"].append(pts)

    # Five GW12-specialist MIDs (one per club C0-C4).
    for i, club in enumerate(["C0", "C1", "C2", "C3", "C4"]):
        name = f"GW12S{i}"
        prices[name] = 50
        for gw, pts in [(10, 0.0), (11, 0.0), (12, 100.0)]:
            gw11_rows["name"].append(name)
            gw11_rows["position"].append("MID")
            gw11_rows["team"].append(club)
            gw11_rows["gw"].append(gw)
            gw11_rows["predicted_points"].append(pts)

    predictions = pl.DataFrame(gw11_rows)
    prob, v = _build_problem(
        predictions, prices, weeks=[10, 11, 12], start_gw=10
    )
    assert _solve_problem(prob) == "Optimal"
    plan = _extract_plan(v, weeks=[10, 11, 12], start_gw=10)

    gw10, gw11, gw12 = plan.gameweeks
    # GW10 is a free build: no transfers, no hits.
    assert gw10.hits == 0
    assert gw10.transfers_in == []
    # The optimal objective forces 16 points of hits across the horizon to
    # load the GW12 specialists, even though the solver may split which week
    # the paid transfers land in (the per-week split is a degenerate optimum).
    assert gw10.hits + gw11.hits + gw12.hits == 16
    # By GW12 the squad must hold all five GW12-specialist MIDs.
    assert {f"GW12S{i}" for i in range(5)} <= set(gw12.squad)


def test_start_gw_transfers_reduce_banked_free_transfers() -> None:
    """A start-gw transfer consumes the FT, banking fewer into gw11.

    A free build would bank 2 FTs into gw11 (1 - 0 + 1); honouring the
    carried-in squad's start-gw transfer banks only 1 (1 - 1 + 1). With two
    compelling upgrades waiting at gw11, that missing FT forces a paid hit.
    """
    rows: dict = {
        "name": [],
        "position": [],
        "team": [],
        "gw": [],
        "predicted_points": [],
    }
    prices: dict = {}
    counts = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
    clubs = ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]
    squad: list[str] = []
    idx = 0
    for pos, n in counts.items():
        for j in range(n):
            name = f"{pos}{j}"
            squad.append(name)
            prices[name] = 50
            for gw in (10, 11):
                rows["name"].append(name)
                rows["position"].append(pos)
                rows["team"].append(clubs[idx % len(clubs)])
                rows["gw"].append(gw)
                rows["predicted_points"].append(2.0)
            idx += 1
    # A start-gw upgrade (DEF), strong in both weeks -> bought at gw10.
    # New club C7 (not used by the base squad) to avoid the club cap.
    prices["UP10"] = 50
    for gw in (10, 11):
        rows["name"].append("UP10")
        rows["position"].append("DEF")
        rows["team"].append("C7")
        rows["gw"].append(gw)
        rows["predicted_points"].append(100.0)
    # Two gw11-only MID stars (worthless at gw10) in fresh clubs -> bought at gw11.
    for s, club in enumerate(["C8", "C9"]):
        name = f"STAR{s}"
        prices[name] = 50
        for gw, pts in ((10, 0.0), (11, 100.0)):
            rows["name"].append(name)
            rows["position"].append("MID")
            rows["team"].append(club)
            rows["gw"].append(gw)
            rows["predicted_points"].append(pts)
    predictions = pl.DataFrame(rows)

    prob, v = _build_problem(
        predictions,
        prices,
        weeks=[10, 11],
        start_gw=10,
        initial_squad=squad,
        free_transfers=1,
    )
    assert _solve_problem(prob) == "Optimal"
    # gw10 buys UP10 (1 transfer), so only 1 FT banks into gw11 (1 - 1 + 1).
    assert round(v["ft"][11].value()) == 1
    # gw11: two star upgrades, one free + one paid -> a single hit.
    g11_buys = sum(
        round(var.value()) for (name, t), var in v["buy"].items() if t == 11
    )
    assert g11_buys == 2
    assert round(v["paid"][11].value()) == 1


from fantasy_football.optimisation.optimiser import _validate_initial_squad


def test_validate_initial_squad_accepts_a_legal_squad() -> None:
    """A legal carried-in squad validates without raising."""
    predictions, prices = _feasible_universe([10])
    # Returns None (no raise) for a legal squad.
    assert _validate_initial_squad(_LEGAL_SQUAD, predictions, prices) is None


def test_validate_initial_squad_rejects_missing_data() -> None:
    """A squad member with no price/prediction is rejected by name."""
    predictions, prices = _feasible_universe([10])
    squad = _LEGAL_SQUAD[:-1] + ["GHOST"]  # GHOST has no price/prediction
    with pytest.raises(ValueError, match="GHOST"):
        _validate_initial_squad(squad, predictions, prices)


def test_validate_initial_squad_rejects_wrong_composition() -> None:
    """A 15-man squad with an illegal position split is rejected."""
    predictions, prices = _feasible_universe([10])
    # 3 GK / 5 DEF / 5 MID / 2 FWD = 15 players but illegal split.
    squad = (
        ["GK0", "GK1", "GK2"]
        + ["DEF0", "DEF1", "DEF2", "DEF3", "DEF4"]
        + ["MID0", "MID1", "MID2", "MID3", "MID4"]
        + ["FWD0", "FWD1"]
    )
    with pytest.raises(ValueError, match="position split"):
        _validate_initial_squad(squad, predictions, prices)


def test_validate_initial_squad_rejects_club_cap_breach() -> None:
    """A squad with more than the per-club cap is rejected."""
    predictions, prices = _feasible_universe([10])
    # GK0, DEF4, MID4, FWD4 all share club C0 (idx % 7) -> 4 from one club.
    squad = (
        ["GK0", "GK1"]
        + ["DEF0", "DEF1", "DEF2", "DEF3", "DEF4"]
        + ["MID0", "MID1", "MID2", "MID3", "MID4"]
        + ["FWD0", "FWD1", "FWD4"]
    )
    with pytest.raises(ValueError, match="per club"):
        _validate_initial_squad(squad, predictions, prices)


def test_validate_initial_squad_rejects_over_budget() -> None:
    """A squad valued above the budget is rejected."""
    predictions, prices = _feasible_universe([10])
    dear = {name: 100 for name in prices}  # 15 * 100 = 1500 > BUDGET (1000)
    with pytest.raises(ValueError, match="budget"):
        _validate_initial_squad(_LEGAL_SQUAD, predictions, dear)


def test_validate_initial_squad_rejects_duplicates() -> None:
    """A squad containing a duplicated player is rejected."""
    predictions, prices = _feasible_universe([10])
    # GK0 appears twice (only 14 distinct players).
    squad = ["GK0", "GK0"] + _LEGAL_SQUAD[2:]
    with pytest.raises(ValueError, match="duplicate"):
        _validate_initial_squad(squad, predictions, prices)


def test_optimise_plan_derives_budget_from_squad_value_over_1000(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A carried-in squad worth > 1000 is admitted via its derived budget."""
    predictions, prices = _feasible_universe([10, 11])
    dear = {name: 80 for name in prices}  # 15 * 80 = 1200 > BUDGET (1000)
    _setup_artifacts(tmp_path, monkeypatch, predictions, dear, gws=[9, 10])
    plans = optimise_plan(
        season="2025-26",
        start_gw=10,
        horizon=2,
        initial_squad=_LEGAL_SQUAD,
        free_transfers=1,
        bank=0,
    )
    gw10 = next(p for p in plans if p.gameweek == 10)
    assert {p.player_name for p in gw10.squad} == set(_LEGAL_SQUAD)


def test_optimise_plan_bank_raises_effective_budget(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A larger bank admits an upgrade that a zero bank cannot afford.

    The squad is fully priced at the budget ceiling, so an upgrade to a
    pricier, far better player is only affordable when the bank covers the
    price difference.
    """
    predictions, prices = _feasible_universe([10])
    prices = {name: 66 for name in prices}  # 15 * 66 = 990 squad value
    # A standout upgrade at MID priced 16 above the player it replaces, in a
    # fresh club (CX) so the per-club cap cannot interfere with the assertion.
    extra = pl.DataFrame(
        {
            "name": ["UPGRADE"],
            "player_id": [9999],
            "position": ["MID"],
            "team": ["CX"],
            "gw": [10],
            "predicted_points": [100.0],
        }
    )
    predictions = pl.concat([predictions, extra])
    prices["UPGRADE"] = 82  # 66 + 16; needs 16 of bank to afford the swap

    # bank=0 -> cannot afford the swap; no UPGRADE bought.
    _setup_artifacts(tmp_path, monkeypatch, predictions, prices, gws=[9, 10])
    poor = optimise_plan(
        season="2025-26",
        start_gw=10,
        horizon=1,
        initial_squad=_LEGAL_SQUAD,
        free_transfers=1,
        bank=0,
    )
    # 15*66=990 budget; buying UPGRADE (82) for any 66 player spends 1006 > 990.
    assert "UPGRADE" not in {p.player_name for p in poor[0].squad}

    # bank=16 -> can afford the swap; UPGRADE bought.
    rich = optimise_plan(
        season="2025-26",
        start_gw=10,
        horizon=1,
        initial_squad=_LEGAL_SQUAD,
        free_transfers=1,
        bank=16,
    )
    assert "UPGRADE" in {p.player_name for p in rich[0].squad}


from fantasy_football.fpl_types import GameWeekPlan
from fantasy_football.optimisation.optimiser import _to_gameweek_plans


def test_to_gameweek_plans_maps_ids_and_points() -> None:
    """_to_gameweek_plans builds typed GameWeekPlans with ids and points."""
    plan = Plan(
        start_gw=10,
        horizon=1,
        gameweeks=[
            GameweekPlan(
                gw=10,
                squad=["A", "B"],
                starting_xi=["A"],
                captain="A",
                transfers_in=[],
                transfers_out=[],
                hits=0,
                free_transfers=1,
                expected_points=12.0,
            )
        ],
        total_expected_points=12.0,
    )
    player_id_map = {"A": 1, "B": 2}
    points = {("A", 10): 7.0, ("B", 10): 3.0}

    result = _to_gameweek_plans(plan, player_id_map, points)

    assert len(result) == 1
    gw = result[0]
    assert isinstance(gw, GameWeekPlan)
    assert gw.gameweek == 10
    assert {p.player_id for p in gw.squad} == {1, 2}
    a = next(p for p in gw.squad if p.player_name == "A")
    assert a.player_id == 1
    assert a.expected_points == 7.0
    assert gw.captain.player_id == 1
    assert gw.captain.player_name == "A"
    assert gw.hits == 0
    assert gw.free_transfers == 1
    assert gw.expected_points == 12.0
