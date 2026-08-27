"""Tests for the squad optimiser: the MILP, and the end-to-end plan."""

import json
from datetime import datetime

import polars as pl
import pytest

from fantasy_football.fpl_types import GameWeekPlan
from fantasy_football.optimisation import optimiser as optimisation
from fantasy_football.optimisation.inputs import (
    MissingPositionPredictionsError,
)
from fantasy_football.optimisation.optimiser import (
    GameweekPlan,
    Plan,
    _build_problem,
    _extract_plan,
    _solve_problem,
    _to_gameweek_plans,
    _validate_initial_squad,
    optimise_plan,
)
from fantasy_football.optimisation.team_input import OwnedPlayer
from fantasy_football.storage import database
from fantasy_football.storage.database import get_connection
from fantasy_football.storage.tables import (
    FORWARD_KIND,
    PLAYER_SEASON,
    PLAYER_SNAPSHOT,
    POINTS_PREDICTION,
)

SEASON = "2026-27"

# The universe every test draws on: enough of each position to build a legal
# squad with spares. Element ids are assigned in this order from 1.
_COUNTS = {"GK": 3, "DEF": 7, "MID": 7, "FWD": 5}
_CLUBS = ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]

# Label to element id, so tests can talk in "MID3" and the code in ids.
E: dict[str, int] = {}
_labels: list[str] = []
for _pos, _n in _COUNTS.items():
    for _j in range(_n):
        _labels.append(f"{_pos}{_j}")
for _index, _label in enumerate(_labels, start=1):
    E[_label] = _index


def _display(label: str) -> str:
    """Return the full name the roster builds for a test label."""
    return f"Player {label}"


def _feasible_universe(
    gws: list[int], price: int = 50
) -> tuple[pl.DataFrame, dict[tuple[int, int], int]]:
    """Return (predictions, prices) with a legal 15-man squad available.

    Prices are keyed on (element, gameweek), the grain the optimiser works
    at, even though this universe holds them flat across the horizon.
    """
    rows: list[dict] = []
    prices: dict[tuple[int, int], int] = {}
    for index, label in enumerate(_labels):
        element = E[label]
        position = label.rstrip("0123456789")
        rank = int(label[len(position) :])
        for gw in gws:
            prices[element, gw] = price
        for gw in gws:
            rows.append(
                {
                    "element": element,
                    "gw": gw,
                    "name": _display(label),
                    "position": position,
                    "team": _CLUBS[index % len(_CLUBS)],
                    "value": price,
                    "predicted_points": 10.0 - rank,
                }
            )
    return pl.DataFrame(rows), prices


# A legal, point-optimal 15-man squad for _feasible_universe (lowest-index
# players score highest: predicted_points = 10 - j).
_LEGAL_SQUAD = [
    E[label]
    for label in (
        ["GK0", "GK1"]
        + ["DEF0", "DEF1", "DEF2", "DEF3", "DEF4"]
        + ["MID0", "MID1", "MID2", "MID3", "MID4"]
        + ["FWD0", "FWD1", "FWD2"]
    )
]
# Same squad but holding FWD3 (7.0) instead of FWD2 (8.0) — a single
# beneficial upgrade is available.
_SQUAD_ONE_OFF = _LEGAL_SQUAD[:-1] + [E["FWD3"]]


def _owned(elements: list[int], purchase: int = 50) -> list[OwnedPlayer]:
    """Wrap element ids as a carried-in squad bought at a flat price."""
    return [
        OwnedPlayer(element=element, purchase_price=purchase)
        for element in elements
    ]


def _now(prices: dict[tuple[int, int], int], gw: int) -> dict[int, int]:
    """Collapse per-gameweek prices to the price in one gameweek."""
    return {
        element: price
        for (element, week), price in prices.items()
        if week == gw
    }


def _names_of(predictions: pl.DataFrame) -> dict[int, str]:
    """Element id to display name, as optimise_plan builds it."""
    return dict(
        zip(
            predictions["element"].to_list(),
            predictions["name"].to_list(),
            strict=True,
        )
    )


def _seed_db(
    tmp_path,
    monkeypatch,
    predictions: pl.DataFrame,
    season: str = SEASON,
) -> None:
    """Seed roster and forward-prediction rows from a predictions frame."""
    roster = predictions.unique(subset=["element"]).sort("element")
    captured = datetime(2026, 8, 1, 12, 0)
    snapshot = pl.DataFrame(
        [
            {
                "season": season,
                "captured_at": captured,
                "element": row["element"],
                "value": row["value"],
                "team": row["team"],
                "position": row["position"],
                "chance_of_playing_this_round": 100,
            }
            for row in roster.iter_rows(named=True)
        ]
    )
    identity = pl.DataFrame(
        [
            {
                "season": season,
                "element": row["element"],
                "player_code": 10_000 + row["element"],
                "web_name": row["name"],
                "first_name": row["name"].split(" ")[0],
                "second_name": row["name"].split(" ", 1)[1],
                "position": row["position"],
                "team_code": row["element"],
                "birth_date": None,
                "region": None,
                "team_join_date": None,
            }
            for row in roster.iter_rows(named=True)
        ],
        schema_overrides={
            "birth_date": pl.Date,
            "region": pl.Int64,
            "team_join_date": pl.Date,
        },
    )
    stored = pl.DataFrame(
        [
            {
                "season": season,
                "gw": row["gw"],
                "element": row["element"],
                "opponent": 99,
                "position": row["position"],
                "predicted_points": row["predicted_points"],
                "model_version": "1",
                "prediction_kind": FORWARD_KIND,
            }
            for row in predictions.iter_rows(named=True)
        ]
    )
    db_path = tmp_path / "optimiser.duckdb"
    monkeypatch.setattr(database, "DATABASE_PATH", db_path)
    connection = get_connection(db_path)
    try:
        PLAYER_SNAPSHOT.upsert_current(connection, snapshot, season)
        PLAYER_SEASON.upsert_current(connection, identity, season)
        POINTS_PREDICTION.upsert_current(connection, stored, season)
    finally:
        connection.close()


def _setup(tmp_path, monkeypatch, predictions: pl.DataFrame):
    """Seed the database and redirect the output folder to a temp tree."""
    transformed = tmp_path / "transformed"
    transformed.mkdir()
    monkeypatch.setattr(optimisation, "TRANSFORMED_DATA_FOLDER", transformed)
    _seed_db(tmp_path, monkeypatch, predictions)
    return transformed


def test_plan_to_frame_has_one_row_per_gameweek() -> None:
    """Plan.to_frame returns one row per GameweekPlan with correct values."""
    plan = Plan(
        start_gw=10,
        horizon=2,
        gameweeks=[
            GameweekPlan(
                gw=10,
                squad=list(range(15)),
                starting_xi=list(range(11)),
                captain=0,
                transfers_in=[],
                transfers_out=[],
                hits=0,
                free_transfers=1,
                expected_points=50.0,
                bank=5,
            ),
            GameweekPlan(
                gw=11,
                squad=list(range(15)),
                starting_xi=list(range(11)),
                captain=1,
                transfers_in=[15],
                transfers_out=[14],
                hits=0,
                free_transfers=2,
                expected_points=48.0,
                bank=3,
            ),
        ],
        total_expected_points=98.0,
    )
    names = {i: f"P{i}" for i in range(16)}

    frame = plan.to_frame(names)

    assert frame.height == 2
    assert frame["gw"].to_list() == [10, 11]
    assert frame.filter(pl.col("gw") == 11)["captain"].item() == "P1"
    assert frame.filter(pl.col("gw") == 11)["transfers_in"].item() == "P15"


def test_single_week_squad_is_legal() -> None:
    """The optimiser returns a rules-legal squad, XI, and captain."""
    predictions, prices = _feasible_universe([10])
    prob, v = _build_problem(predictions, prices, weeks=[10], start_gw=10)
    assert _solve_problem(prob) == "Optimal"

    owned = [p for (p, t), var in v["own"].items() if round(var.value()) == 1]
    started = [
        p for (p, t), var in v["start"].items() if round(var.value()) == 1
    ]
    captains = [
        p for (p, t), var in v["cap"].items() if round(var.value()) == 1
    ]
    assert len(owned) == 15
    assert len(started) == 11
    assert len(captains) == 1
    assert captains[0] in started
    positions = [v["pos"][p] for p in owned]
    assert {p: positions.count(p) for p in set(positions)} == {
        "GK": 2,
        "DEF": 5,
        "MID": 5,
        "FWD": 3,
    }


def test_initial_squad_with_no_better_option_makes_no_transfers() -> None:
    """The optimal carried-in squad is held rather than churned."""
    predictions, prices = _feasible_universe([10])
    prob, v = _build_problem(
        predictions,
        prices,
        weeks=[10],
        start_gw=10,
        initial_squad=_owned(_LEGAL_SQUAD),
        free_transfers=1,
    )
    assert _solve_problem(prob) == "Optimal"
    buys = sum(round(var.value()) for var in v["buy"].values())
    assert buys == 0


def test_initial_squad_uses_one_free_transfer_to_upgrade() -> None:
    """A single beneficial upgrade is taken with the free transfer."""
    predictions, prices = _feasible_universe([10])
    prob, v = _build_problem(
        predictions,
        prices,
        weeks=[10],
        start_gw=10,
        initial_squad=_owned(_SQUAD_ONE_OFF),
        free_transfers=1,
    )
    assert _solve_problem(prob) == "Optimal"
    buys = [p for (p, t), var in v["buy"].items() if round(var.value()) == 1]
    assert buys == [E["FWD2"]]
    assert round(v["paid"][10].value()) == 0


def test_initial_squad_pays_a_hit_to_make_extra_transfer() -> None:
    """Two compelling upgrades on one free transfer means one paid hit."""
    predictions, prices = _feasible_universe([10])
    stars = pl.DataFrame(
        [
            {
                "element": 900 + i,
                "gw": 10,
                "name": _display(f"STAR{i}"),
                "position": "FWD",
                "team": f"X{i}",
                "value": 50,
                "predicted_points": 100.0,
            }
            for i in range(2)
        ]
    )
    predictions = pl.concat([predictions, stars])
    prices[900, 10] = 50
    prices[901, 10] = 50

    prob, v = _build_problem(
        predictions,
        prices,
        weeks=[10],
        start_gw=10,
        initial_squad=_owned(_LEGAL_SQUAD),
        free_transfers=1,
    )
    assert _solve_problem(prob) == "Optimal"
    buys = sum(
        round(var.value()) for (p, t), var in v["buy"].items() if t == 10
    )
    assert buys == 2  # both stars brought in
    assert round(v["paid"][10].value()) == 1  # one free, one paid (-4)


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
        initial_squad=_owned(_SQUAD_ONE_OFF),
        free_transfers=1,
    )
    assert _solve_problem(prob) == "Optimal"
    plan = _extract_plan(
        v, weeks=[10], start_gw=10, initial_squad=_owned(_SQUAD_ONE_OFF)
    )
    gw = plan.gameweeks[0]
    assert gw.transfers_in == [E["FWD2"]]
    assert gw.transfers_out == [E["FWD3"]]
    assert gw.hits == 0


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
    predictions, prices = _feasible_universe([10, 11, 12])
    rows: list[dict] = []
    for i, club in enumerate(["C0", "C1", "C2", "C3", "C4"]):
        for tag, element, points in (
            ("GW11S", 900 + i, [(10, 0.0), (11, 101.0), (12, 0.0)]),
            ("GW12S", 950 + i, [(10, 0.0), (11, 0.0), (12, 100.0)]),
        ):
            for gw, pts in points:
                prices[element, gw] = 50
                rows.append(
                    {
                        "element": element,
                        "gw": gw,
                        "name": _display(f"{tag}{i}"),
                        "position": "MID",
                        "team": club,
                        "value": 50,
                        "predicted_points": pts,
                    }
                )
    predictions = pl.concat([predictions, pl.DataFrame(rows)])

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
    assert {950 + i for i in range(5)} <= set(gw12.squad)


def test_start_gw_transfers_reduce_banked_free_transfers() -> None:
    """A start-gw transfer consumes the FT, banking fewer into gw11.

    A free build would bank 2 FTs into gw11 (1 - 0 + 1); honouring the
    carried-in squad's start-gw transfer banks only 1 (1 - 1 + 1). With two
    compelling upgrades waiting at gw11, that missing FT forces a paid hit.
    """
    rows: list[dict] = []
    prices: dict[tuple[int, int], int] = {}
    counts = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
    squad: list[int] = []
    element = 0
    for pos, n in counts.items():
        for j in range(n):
            element += 1
            squad.append(element)
            for gw in (10, 11):
                prices[element, gw] = 50
                rows.append(
                    {
                        "element": element,
                        "gw": gw,
                        "name": _display(f"{pos}{j}"),
                        "position": pos,
                        "team": _CLUBS[(element - 1) % len(_CLUBS)],
                        "value": 50,
                        "predicted_points": 2.0,
                    }
                )
    # A start-gw upgrade (DEF), strong in both weeks -> bought at gw10.
    # New club C7 (not used by the base squad) to avoid the club cap.
    for gw in (10, 11):
        prices[900, gw] = 50
        rows.append(
            {
                "element": 900,
                "gw": gw,
                "name": _display("UP10"),
                "position": "DEF",
                "team": "C7",
                "value": 50,
                "predicted_points": 100.0,
            }
        )
    # Two gw11-only MID stars (worthless at gw10) in fresh clubs.
    for s, club in enumerate(["C8", "C9"]):
        for gw, pts in ((10, 0.0), (11, 100.0)):
            prices[910 + s, gw] = 50
            rows.append(
                {
                    "element": 910 + s,
                    "gw": gw,
                    "name": _display(f"STAR{s}"),
                    "position": "MID",
                    "team": club,
                    "value": 50,
                    "predicted_points": pts,
                }
            )
    predictions = pl.DataFrame(rows)

    prob, v = _build_problem(
        predictions,
        prices,
        weeks=[10, 11],
        start_gw=10,
        initial_squad=_owned(squad),
        free_transfers=1,
    )
    assert _solve_problem(prob) == "Optimal"
    # gw10 buys UP10 (1 transfer), so only 1 FT banks into gw11 (1 - 1 + 1).
    assert round(v["ft"][11].value()) == 1
    # gw11: two star upgrades, one free + one paid -> a single hit.
    g11_buys = sum(
        round(var.value()) for (p, t), var in v["buy"].items() if t == 11
    )
    assert g11_buys == 2
    assert round(v["paid"][11].value()) == 1


def test_validate_initial_squad_accepts_a_legal_squad() -> None:
    """A legal carried-in squad validates without raising."""
    predictions, prices = _feasible_universe([10])
    assert (
        _validate_initial_squad(
            _owned(_LEGAL_SQUAD),
            predictions,
            _now(prices, 10),
            _names_of(predictions),
        )
        is None
    )


def test_validate_initial_squad_rejects_missing_data() -> None:
    """A squad member with no price/prediction is rejected by name."""
    predictions, prices = _feasible_universe([10])
    squad = _owned(_LEGAL_SQUAD[:-1] + [777])  # 777 has no price
    with pytest.raises(ValueError, match="777"):
        _validate_initial_squad(
            squad, predictions, _now(prices, 10), _names_of(predictions)
        )


def test_validate_initial_squad_names_offenders() -> None:
    """Errors carry the display name, not just the element id."""
    predictions, prices = _feasible_universe([10])
    squad = _owned([_LEGAL_SQUAD[0]] + _LEGAL_SQUAD)  # 16, one duplicated
    with pytest.raises(ValueError, match=_display("GK0")):
        _validate_initial_squad(
            squad, predictions, _now(prices, 10), _names_of(predictions)
        )


def test_validate_initial_squad_rejects_wrong_composition() -> None:
    """A 15-man squad with an illegal position split is rejected."""
    predictions, prices = _feasible_universe([10])
    # 3 GK / 5 DEF / 5 MID / 2 FWD = 15 players but illegal split.
    squad = _owned(
        [
            E[label]
            for label in (
                ["GK0", "GK1", "GK2"]
                + ["DEF0", "DEF1", "DEF2", "DEF3", "DEF4"]
                + ["MID0", "MID1", "MID2", "MID3", "MID4"]
                + ["FWD0", "FWD1"]
            )
        ]
    )
    with pytest.raises(ValueError, match="position split"):
        _validate_initial_squad(
            squad, predictions, _now(prices, 10), _names_of(predictions)
        )


def test_validate_initial_squad_rejects_club_cap_breach() -> None:
    """A squad with more than the per-club cap is rejected."""
    predictions, prices = _feasible_universe([10])
    # GK0, DEF4, MID4, FWD4 all share club C0 (idx % 7) -> 4 from one club.
    squad = _owned(
        [
            E[label]
            for label in (
                ["GK0", "GK1"]
                + ["DEF0", "DEF1", "DEF2", "DEF3", "DEF4"]
                + ["MID0", "MID1", "MID2", "MID3", "MID4"]
                + ["FWD0", "FWD1", "FWD4"]
            )
        ]
    )
    with pytest.raises(ValueError, match="per club"):
        _validate_initial_squad(
            squad, predictions, _now(prices, 10), _names_of(predictions)
        )


def test_validate_initial_squad_rejects_duplicates() -> None:
    """A squad containing a duplicated player is rejected."""
    predictions, prices = _feasible_universe([10])
    squad = _owned([E["GK0"], E["GK0"]] + _LEGAL_SQUAD[2:])
    with pytest.raises(ValueError, match="duplicate"):
        _validate_initial_squad(
            squad, predictions, _now(prices, 10), _names_of(predictions)
        )


def test_validate_initial_squad_rejects_a_non_positive_purchase_price() -> (
    None
):
    """A purchase price of zero or less is an entry error, not a squad."""
    predictions, prices = _feasible_universe([10])
    squad = _owned(_LEGAL_SQUAD)
    squad[0] = OwnedPlayer(element=_LEGAL_SQUAD[0], purchase_price=0)
    with pytest.raises(ValueError, match="purchase price"):
        _validate_initial_squad(
            squad, predictions, _now(prices, 10), _names_of(predictions)
        )


def test_validate_initial_squad_accepts_a_squad_valued_over_the_budget(
    caplog,
) -> None:
    """Squad value is no longer a constraint -- the bank is.

    A carried-in squad cost whatever it cost; affordability is expressed by
    the bank never going negative, so an expensive squad is not an error.
    """
    predictions, _ = _feasible_universe([10], price=100)
    dear = {element: 100 for element in E.values()}  # 15 * 100 = 1500
    assert (
        _validate_initial_squad(
            _owned(_LEGAL_SQUAD, purchase=100),
            predictions,
            dear,
            _names_of(predictions),
        )
        is None
    )


def test_validate_initial_squad_warns_on_an_implausible_purchase_price(
    caplog,
) -> None:
    """A purchase price far from the current price is flagged, not fatal."""
    predictions, prices = _feasible_universe([10])
    squad = _owned(_LEGAL_SQUAD)
    # Bought at 5.0, "now" 12.0 -- a typo or a stale team file.
    squad[0] = OwnedPlayer(element=_LEGAL_SQUAD[0], purchase_price=120)
    with caplog.at_level("WARNING"):
        _validate_initial_squad(
            squad, predictions, _now(prices, 10), _names_of(predictions)
        )
    assert _display("GK0") in caplog.text


def test_validate_initial_squad_accepts_a_large_but_legal_rise(
    caplog,
) -> None:
    """A genuine riser inside the tolerance draws no warning."""
    predictions, prices = _feasible_universe([10])
    squad = _owned(_LEGAL_SQUAD)
    squad[0] = OwnedPlayer(element=_LEGAL_SQUAD[0], purchase_price=40)
    with caplog.at_level("WARNING"):
        _validate_initial_squad(
            squad, predictions, _now(prices, 10), _names_of(predictions)
        )
    assert caplog.text == ""


def test_optimise_plan_writes_jsonl_and_returns_gameweek_plans(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: GW1 free build returns typed plans and writes JSON Lines."""
    predictions, _ = _feasible_universe([1, 2])
    transformed = _setup(tmp_path, monkeypatch, predictions)

    plans = optimise_plan(season=SEASON, start_gw=1, horizon=2)

    assert [p.gameweek for p in plans] == [1, 2]
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
    predictions, _ = _feasible_universe([1, 2])
    transformed = _setup(tmp_path, monkeypatch, predictions)

    optimise_plan(season=SEASON, start_gw=1, horizon=2)

    report = transformed / "optimisation_plan.md"
    assert report.exists()
    text = report.read_text()
    assert "## GW1" in text
    assert "--- bench ---" in text
    # Positions resolve, so no player falls through to the "?" bucket.
    assert "| ? |" not in text


def test_free_build_cannot_exceed_the_opening_budget() -> None:
    """A universe priced beyond 100.0m for 15 players has no legal squad."""
    predictions, prices = _feasible_universe([10], price=70)  # 15*70 = 1050
    prob, _ = _build_problem(predictions, prices, weeks=[10], start_gw=10)
    assert _solve_problem(prob) != "Optimal"


def test_free_build_bank_is_the_budget_less_the_squad_cost() -> None:
    """The opening bank is what the free build did not spend."""
    predictions, prices = _feasible_universe([10])  # 15 * 50 = 750
    prob, v = _build_problem(predictions, prices, weeks=[10], start_gw=10)
    assert _solve_problem(prob) == "Optimal"
    assert round(v["bank"][10].value()) == 250


def test_a_risen_player_sells_for_less_than_their_current_price() -> None:
    """Selling a riser banks purchase plus half the profit, not the price.

    The carried-in FWD2 was bought at 5.0 and is now worth 7.0, so he sells
    for 6.0 -- not 7.0. Replacing him with a 5.0 star therefore leaves 1.0
    in the bank; a model believing the current price would leave 2.0.
    """
    predictions, prices = _feasible_universe([10])
    riser = E["FWD2"]
    prices[riser, 10] = 70
    predictions = predictions.with_columns(
        pl.when(pl.col("element") == riser)
        .then(70)
        .otherwise(pl.col("value"))
        .alias("value")
    )
    star = pl.DataFrame(
        [
            {
                "element": 900,
                "gw": 10,
                "name": _display("STAR"),
                "position": "FWD",
                "team": "X0",
                "value": 50,
                "predicted_points": 100.0,
            }
        ]
    )
    predictions = pl.concat([predictions, star])
    prices[900, 10] = 50

    squad = _owned(_LEGAL_SQUAD)
    squad[-1] = OwnedPlayer(element=riser, purchase_price=50)
    prob, v = _build_problem(
        predictions,
        prices,
        weeks=[10],
        start_gw=10,
        initial_squad=squad,
        free_transfers=1,
        bank=0,
    )
    assert _solve_problem(prob) == "Optimal"
    sold = [p for (p, t), var in v["sell"].items() if round(var.value()) == 1]
    assert sold == [riser]
    assert round(v["bank"][10].value()) == 10


def _swap_universe() -> (
    tuple[pl.DataFrame, dict[tuple[int, int], int], list[int]]
):
    """Build a squad plus two FWDs that must be swapped every week.

    The carried-in RISER (bought 5.0, now 7.0, so sells for 6.0) scores only
    in gw11; the SWAP target (7.8) scores only in gw10 and gw12. With 2.0 in
    the bank, SWAP is affordable only by selling RISER, which is what forces
    the sell/rebuy/sell dance the spread accounting has to survive.
    """
    rows: list[dict] = []
    prices: dict[tuple[int, int], int] = {}
    weeks = [10, 11, 12]
    counts = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 2}
    squad: list[int] = []
    element = 0
    for pos, n in counts.items():
        for j in range(n):
            element += 1
            squad.append(element)
            for gw in weeks:
                prices[element, gw] = 50
                rows.append(
                    {
                        "element": element,
                        "gw": gw,
                        "name": _display(f"{pos}{j}"),
                        "position": pos,
                        "team": _CLUBS[(element - 1) % len(_CLUBS)],
                        "value": 50,
                        "predicted_points": 2.0,
                    }
                )
    for tag, ident, price, club, points in (
        ("RISER", 900, 70, "X0", [0.0, 1000.0, 0.0]),
        ("SWAP", 901, 78, "X1", [1000.0, 0.0, 1000.0]),
    ):
        for gw, pts in zip(weeks, points, strict=True):
            prices[ident, gw] = price
            rows.append(
                {
                    "element": ident,
                    "gw": gw,
                    "name": _display(tag),
                    "position": "FWD",
                    "team": club,
                    "value": price,
                    "predicted_points": pts,
                }
            )
    squad.append(900)
    return pl.DataFrame(rows), prices, squad


def test_a_rebought_player_sells_at_their_new_purchase_price() -> None:
    """The profit spread is consumed by the first sale and never again.

    The plan sells RISER in gw10, buys him back in gw11 and sells him again
    in gw12. The first sale banks the spread price (6.0); the second banks
    the price actually paid to get him back (7.0), because the repurchase
    reset his purchase price. Both assertions matter -- either one alone
    passes against a broken implementation.
    """
    predictions, prices, squad = _swap_universe()
    riser, swap = 900, 901
    prob, v = _build_problem(
        predictions,
        prices,
        weeks=[10, 11, 12],
        start_gw=10,
        initial_squad=_owned(squad),
        free_transfers=1,
        bank=20,
    )
    assert _solve_problem(prob) == "Optimal"

    def moved(kind: str, gw: int) -> list[int]:
        return sorted(
            p
            for (p, t), var in v[kind].items()
            if t == gw and round(var.value()) == 1
        )

    assert (moved("sell", 10), moved("buy", 10)) == ([riser], [swap])
    assert (moved("sell", 11), moved("buy", 11)) == ([swap], [riser])
    assert (moved("sell", 12), moved("buy", 12)) == ([riser], [swap])

    # gw10: 2.0 + 6.0 (spread) - 7.8 = 0.2
    assert round(v["bank"][10].value()) == 2
    # gw11: 0.2 + 7.8 - 7.0 = 1.0
    assert round(v["bank"][11].value()) == 10
    # gw12: 1.0 + 7.0 (NOT the 6.0 spread again) - 7.8 = 0.2
    assert round(v["bank"][12].value()) == 2


def test_the_bank_never_goes_negative() -> None:
    """An upgrade costing more than the squad can raise is not planned."""
    predictions, prices = _feasible_universe([10])
    star = pl.DataFrame(
        [
            {
                "element": 900,
                "gw": 10,
                "name": _display("STAR"),
                "position": "FWD",
                "team": "X0",
                "value": 90,
                "predicted_points": 100.0,
            }
        ]
    )
    predictions = pl.concat([predictions, star])
    prices[900, 10] = 90

    prob, v = _build_problem(
        predictions,
        prices,
        weeks=[10],
        start_gw=10,
        initial_squad=_owned(_LEGAL_SQUAD),
        free_transfers=1,
        bank=0,
    )
    assert _solve_problem(prob) == "Optimal"
    # Selling a 5.0 player raises 5.0 against a 9.0 price tag, so no deal.
    buys = [p for (p, t), var in v["buy"].items() if round(var.value()) == 1]
    assert buys == []
    assert round(v["bank"][10].value()) == 0


def test_extract_plan_reports_the_bank_per_gameweek() -> None:
    """The extracted plan carries the solved bank for each gameweek."""
    predictions, prices = _feasible_universe([10])
    prob, v = _build_problem(predictions, prices, weeks=[10], start_gw=10)
    assert _solve_problem(prob) == "Optimal"
    plan = _extract_plan(v, weeks=[10], start_gw=10)
    assert plan.gameweeks[0].bank == 250


def test_optimise_plan_sums_double_gameweek_fixtures(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second fixture in a gameweek adds points rather than replacing them."""
    predictions, _ = _feasible_universe([1])
    _setup(tmp_path, monkeypatch, predictions)
    # MID0 plays twice in GW1: 10.0 already stored, plus another 10.0.
    extra = pl.DataFrame(
        [
            {
                "season": SEASON,
                "gw": 1,
                "element": E["MID0"],
                "opponent": 42,
                "position": "MID",
                "predicted_points": 10.0,
                "model_version": "1",
                "prediction_kind": FORWARD_KIND,
            }
        ]
    )
    connection = get_connection(tmp_path / "optimiser.duckdb")
    try:
        POINTS_PREDICTION.append(connection, extra)
    finally:
        connection.close()

    plans = optimise_plan(season=SEASON, start_gw=1, horizon=1)

    doubled = next(p for p in plans[0].squad if p.player_id == E["MID0"])
    assert doubled.expected_points == pytest.approx(20.0)
    assert plans[0].captain.player_id == E["MID0"]


def test_optimise_plan_requires_initial_squad_after_gw1() -> None:
    """start_gw > 1 with no initial_squad is rejected before any file I/O."""
    with pytest.raises(ValueError, match="initial_squad"):
        optimise_plan(season=SEASON, start_gw=10, horizon=2)


def test_zero_free_transfers_pays_a_hit_for_every_transfer() -> None:
    """Having already used the week's transfers is a real state.

    ``my-team`` reports free transfers as an allowance less what has been
    made, so a mid-week run legitimately arrives with none left; every
    transfer from there costs four points.
    """
    predictions, prices = _feasible_universe([10])
    star = pl.DataFrame(
        [
            {
                "element": 900,
                "gw": 10,
                "name": _display("STAR0"),
                "position": "FWD",
                "team": "X0",
                "value": 50,
                "predicted_points": 100.0,
            }
        ]
    )
    predictions = pl.concat([predictions, star])
    prices[900, 10] = 50

    prob, v = _build_problem(
        predictions,
        prices,
        weeks=[10],
        start_gw=10,
        initial_squad=_owned(_LEGAL_SQUAD),
        free_transfers=0,
    )
    assert _solve_problem(prob) == "Optimal"
    buys = sum(
        round(var.value()) for (p, t), var in v["buy"].items() if t == 10
    )
    assert buys == 1
    assert round(v["paid"][10].value()) == 1


def test_optimise_plan_accepts_zero_free_transfers() -> None:
    """Zero passes validation; it is a state, not a mistake."""
    predictions, _ = _feasible_universe([10])
    with pytest.raises(ValueError, match="initial_squad"):
        optimise_plan(
            season=SEASON,
            start_gw=10,
            horizon=1,
            free_transfers=0,
        )


def test_optimise_plan_rejects_bad_free_transfers() -> None:
    """free_transfers must be within 0..MAX_FREE_TRANSFERS."""
    with pytest.raises(ValueError, match="free_transfers"):
        optimise_plan(
            season=SEASON,
            start_gw=1,
            horizon=1,
            free_transfers=6,
        )
    with pytest.raises(ValueError, match="free_transfers"):
        optimise_plan(
            season=SEASON,
            start_gw=1,
            horizon=1,
            free_transfers=-1,
        )


def test_optimise_plan_rejects_a_start_gw_that_is_not_first_predicted(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale team-file gameweek is refused rather than silently shifted."""
    predictions, _ = _feasible_universe([10, 11])
    _setup(tmp_path, monkeypatch, predictions)

    with pytest.raises(ValueError, match="first predicted gameweek"):
        optimise_plan(
            season=SEASON,
            start_gw=11,
            horizon=1,
            initial_squad=_owned(_LEGAL_SQUAD),
        )


def test_optimise_plan_without_forward_predictions_raises(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing predicted means nothing to optimise, said plainly."""
    predictions, _ = _feasible_universe([10])
    _setup(tmp_path, monkeypatch, predictions)

    with pytest.raises(ValueError, match="No forward predictions"):
        optimise_plan(
            season="2019-20",
            start_gw=10,
            horizon=1,
            initial_squad=_owned(_LEGAL_SQUAD),
        )


def test_optimise_plan_fails_when_a_position_has_no_predictions(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unscored position aborts rather than being fielded on zeroes."""
    predictions, _ = _feasible_universe([10])
    _setup(tmp_path, monkeypatch, predictions)
    connection = get_connection(tmp_path / "optimiser.duckdb")
    try:
        connection.execute(
            "DELETE FROM points_prediction WHERE position = 'GK'"
        )
    finally:
        connection.close()

    with pytest.raises(MissingPositionPredictionsError, match="GK"):
        optimise_plan(
            season=SEASON,
            start_gw=10,
            horizon=1,
            initial_squad=_owned(_LEGAL_SQUAD),
        )


def test_optimise_plan_with_initial_squad_reports_start_gw_transfer(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mid-season run honours the carried-in squad and reports its transfer."""
    predictions, _ = _feasible_universe([10, 11])
    _setup(tmp_path, monkeypatch, predictions)

    plans = optimise_plan(
        season=SEASON,
        start_gw=10,
        horizon=2,
        initial_squad=_owned(_SQUAD_ONE_OFF),
        free_transfers=1,
    )
    gw10 = next(p for p in plans if p.gameweek == 10)
    assert {p.player_id for p in gw10.transfers_in} == {E["FWD2"]}
    assert {p.player_id for p in gw10.transfers_out} == {E["FWD3"]}
    assert gw10.hits == 0


def test_optimise_plan_ignores_initial_squad_at_gw1(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """At GW1 the build is free: a squad is ignored, not required."""
    predictions, _ = _feasible_universe([1, 2])
    _setup(tmp_path, monkeypatch, predictions)

    plans = optimise_plan(
        season=SEASON,
        start_gw=1,
        horizon=2,
        initial_squad=_owned(_SQUAD_ONE_OFF),
    )
    gw1 = next(p for p in plans if p.gameweek == 1)
    assert gw1.transfers_in == []
    assert gw1.transfers_out == []


def test_optimise_plan_derives_budget_from_squad_value_over_1000(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A carried-in squad worth > 1000 is admitted via its derived budget."""
    predictions, _ = _feasible_universe([10, 11], price=80)  # 15 * 80 = 1200
    _setup(tmp_path, monkeypatch, predictions)

    plans = optimise_plan(
        season=SEASON,
        start_gw=10,
        horizon=2,
        initial_squad=_owned(_LEGAL_SQUAD),
        free_transfers=1,
        bank=0,
    )
    gw10 = next(p for p in plans if p.gameweek == 10)
    assert {p.player_id for p in gw10.squad} == set(_LEGAL_SQUAD)


def test_optimise_plan_bank_raises_effective_budget(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A larger bank admits an upgrade that a zero bank cannot afford.

    The squad is fully priced at the budget ceiling, so an upgrade to a
    pricier, far better player is only affordable when the bank covers the
    price difference. Every player is carried in at what they are worth now,
    so no profit spread muddies the arithmetic -- that is tested separately.
    """
    predictions, _ = _feasible_universe([10], price=66)  # 15 * 66 = 990
    # A standout upgrade at MID priced 16 above the player it replaces, in a
    # fresh club (CX) so the per-club cap cannot interfere with the assertion.
    upgrade = pl.DataFrame(
        [
            {
                "element": 9999,
                "gw": 10,
                "name": _display("UPGRADE"),
                "position": "MID",
                "team": "CX",
                "value": 82,  # 66 + 16; needs 16 of bank to afford the swap
                "predicted_points": 100.0,
            }
        ]
    )
    _setup(tmp_path, monkeypatch, pl.concat([predictions, upgrade]))

    poor = optimise_plan(
        season=SEASON,
        start_gw=10,
        horizon=1,
        initial_squad=_owned(_LEGAL_SQUAD, purchase=66),
        free_transfers=1,
        bank=0,
    )
    assert 9999 not in {p.player_id for p in poor[0].squad}

    rich = optimise_plan(
        season=SEASON,
        start_gw=10,
        horizon=1,
        initial_squad=_owned(_LEGAL_SQUAD, purchase=66),
        free_transfers=1,
        bank=16,
    )
    assert 9999 in {p.player_id for p in rich[0].squad}


def test_to_gameweek_plans_maps_ids_and_points() -> None:
    """_to_gameweek_plans builds typed GameWeekPlans with ids and points."""
    plan = Plan(
        start_gw=10,
        horizon=1,
        gameweeks=[
            GameweekPlan(
                gw=10,
                squad=[1, 2],
                starting_xi=[1],
                captain=1,
                transfers_in=[],
                transfers_out=[],
                hits=0,
                free_transfers=1,
                expected_points=12.0,
                bank=9,
            )
        ],
        total_expected_points=12.0,
    )
    names = {1: "A", 2: "B"}
    points = {(1, 10): 7.0, (2, 10): 3.0}

    result = _to_gameweek_plans(plan, names, points)

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
