from pathlib import Path

from fantasy_football.fpl_types import (
    GameWeekPlan,
    PlayerGameweekExpectedPoints,
)
from fantasy_football.optimisation.plan_report import (
    PlayerPrices,
    _ordered_rows,
    _render_gameweek,
    render_plan_markdown,
    write_plan_report,
)

# Stable element id per name, assigned on first use. The report keys on
# ids, so the tests need a name-to-id map they can build positions from.
_IDS: dict[str, int] = {}


def _id(name: str) -> int:
    return _IDS.setdefault(name, len(_IDS) + 1)


def _p(name: str, pts: float) -> PlayerGameweekExpectedPoints:
    return PlayerGameweekExpectedPoints(
        player_id=_id(name), player_name=name, expected_points=pts
    )


def _prices(
    holdings: dict[str, tuple[int, int, int]],
) -> dict[int, PlayerPrices]:
    """Rekey a name-to-(purchase, current, selling) map onto element ids."""
    return {
        _id(name): PlayerPrices(
            purchase=purchase, current=current, selling=selling
        )
        for name, (purchase, current, selling) in holdings.items()
    }


def _by_id(positions: dict[str, str]) -> dict[int, str]:
    """Rekey a name-to-position map onto element ids."""
    return {_id(name): position for name, position in positions.items()}


def test_ordered_rows_groups_by_position_then_points_desc() -> None:
    """Rows come out GK, DEF, MID, FWD; within a group, highest xPts first."""
    players = [
        _p("Striker", 5.0),
        _p("Keeper", 4.0),
        _p("Mid1", 9.0),
        _p("Mid2", 2.0),
    ]
    positions = _by_id(
        {
            "Striker": "FWD",
            "Keeper": "GK",
            "Mid1": "MID",
            "Mid2": "MID",
        }
    )

    rows = _ordered_rows(players, positions)

    assert [(pos, p.player_name) for pos, p in rows] == [
        ("GK", "Keeper"),
        ("MID", "Mid1"),
        ("MID", "Mid2"),
        ("FWD", "Striker"),
    ]


def test_ordered_rows_missing_position_renders_as_question_mark() -> None:
    """An id absent from the positions map gets '?' and sorts last."""
    players = [_p("Known", 5.0), _p("Unknown", 9.0)]
    positions = _by_id({"Known": "MID"})

    rows = _ordered_rows(players, positions)

    assert [(pos, p.player_name) for pos, p in rows] == [
        ("MID", "Known"),
        ("?", "Unknown"),
    ]


def _gw_plan() -> tuple[GameWeekPlan, dict[int, str]]:
    salah = _p("Salah", 8.2)
    vicario = _p("Vicario", 4.1)
    vvd = _p("Van Dijk", 5.4)
    mateta = _p("Mateta", 5.6)
    bench_gk = _p("King", 0.2)
    bench_def = _p("Lewis", 1.5)
    bench_mid = _p("Smith Rowe", 1.1)
    bench_fwd = _p("Pedro", 3.0)
    out = _p("Milenkovic", 1.4)
    # 11 starters: build a legal-ish XI for rendering (counts not enforced here).
    xi = [
        vicario,
        vvd,
        _p("Munoz", 4.8),
        _p("Timber", 7.3),
        salah,
        _p("Kudus", 6.0),
        _p("Semenyo", 5.0),
        _p("Gibbs-White", 4.5),
        mateta,
        _p("Gyokeres", 6.5),
        _p("Enzo", 5.8),
    ]
    squad = xi + [bench_gk, bench_def, bench_mid, bench_fwd]
    positions = _by_id(
        {
            "Vicario": "GK",
            "Van Dijk": "DEF",
            "Munoz": "DEF",
            "Timber": "DEF",
            "Salah": "MID",
            "Kudus": "MID",
            "Semenyo": "MID",
            "Gibbs-White": "MID",
            "Mateta": "FWD",
            "Gyokeres": "FWD",
            "Enzo": "MID",
            "King": "GK",
            "Lewis": "DEF",
            "Smith Rowe": "MID",
            "Pedro": "FWD",
            "Milenkovic": "DEF",
        }
    )
    plan = GameWeekPlan(
        gameweek=5,
        squad=squad,
        starting_xi=xi,
        captain=salah,
        transfers_in=[vvd],
        transfers_out=[out],
        hits=4,
        free_transfers=1,
        expected_points=62.3,
        bank=7,
    )
    return plan, positions


def test_render_gameweek_has_header_captain_transfers_and_bench() -> None:
    """A GW section shows summary, captain/in markers, bench divider, transfers."""
    plan, positions = _gw_plan()

    md = _render_gameweek(plan, positions, {})

    assert "## GW5 — xPts 62.3 · FT 1 · hit −4 · C: Salah" in md
    assert any("Salah" in line and "⭐ C" in line for line in md.splitlines())
    assert any("Van Dijk" in line and "↑" in line for line in md.splitlines())
    assert "--- bench ---" in md
    assert "Transfers — IN: Van Dijk · OUT: Milenkovic" in md


def test_render_gameweek_freebuild_omits_transfers_line() -> None:
    """A GW with no transfers in or out omits the Transfers line."""
    plan, positions = _gw_plan()
    free_build = GameWeekPlan(
        gameweek=1,
        squad=plan.squad,
        starting_xi=plan.starting_xi,
        captain=plan.captain,
        transfers_in=[],
        transfers_out=[],
        hits=0,
        free_transfers=1,
        expected_points=70.0,
        bank=7,
    )

    md = _render_gameweek(free_build, positions, {})

    assert "Transfers —" not in md
    assert "hit −0" not in md


def test_render_plan_markdown_orders_gameweeks_ascending() -> None:
    """Sections appear in ascending gameweek order regardless of input order."""
    plan_a, positions = _gw_plan()  # gameweek 5
    plan_b = GameWeekPlan(
        gameweek=6,
        squad=plan_a.squad,
        starting_xi=plan_a.starting_xi,
        captain=plan_a.captain,
        transfers_in=[],
        transfers_out=[],
        hits=0,
        free_transfers=2,
        expected_points=64.0,
        bank=7,
    )

    md = render_plan_markdown([plan_b, plan_a], positions, {})

    assert md.index("## GW5") < md.index("## GW6")


def test_write_plan_report_writes_file(tmp_path: Path) -> None:
    """write_plan_report writes a non-empty markdown file at the given path."""
    plan, positions = _gw_plan()
    out = tmp_path / "optimisation_plan.md"

    write_plan_report([plan], positions, {}, out)

    text = out.read_text()
    assert "## GW5" in text
    assert text.endswith("\n")


def test_render_gameweek_shows_purchase_current_and_selling_prices() -> None:
    """Each player row carries what they cost, are worth, and would fetch."""
    plan, positions = _gw_plan()
    prices = _prices({"Salah": (125, 145, 135)})

    md = _render_gameweek(plan, positions, prices)

    row = next(
        line
        for line in md.splitlines()
        if line.startswith("|") and "Salah" in line
    )
    assert "| 12.5 | 14.5 | 13.5 |" in row


def test_render_gameweek_dashes_prices_for_an_unpriced_player() -> None:
    """A player with no price entry renders dashes rather than failing."""
    plan, positions = _gw_plan()

    md = _render_gameweek(plan, positions, {})

    row = next(
        line
        for line in md.splitlines()
        if line.startswith("|") and "Salah" in line
    )
    assert "| — | — | — |" in row


def test_value_line_separates_squad_value_from_team_value() -> None:
    """Squad value uses selling prices; team value uses current prices.

    They are different numbers whenever the squad holds a riser, and the
    constraint the optimiser actually obeys is the squad value.
    """
    plan, positions = _gw_plan()
    # Every squad player flat at 5.0 except Salah, bought at 12.5 and now
    # 14.5, so he sells for 13.5. 14 flat players + Salah + 0.7 bank:
    #   squad 14*5.0 + 13.5 + 0.7 = 84.2
    #   team  14*5.0 + 14.5 + 0.7 = 85.2
    holdings = {
        player.player_name: (50, 50, 50)
        for player in plan.squad
        if player.player_name != "Salah"
    }
    holdings["Salah"] = (125, 145, 135)

    md = _render_gameweek(plan, positions, _prices(holdings))

    assert "Value — squad 84.2 · team 85.2 · bank 0.7" in md


def test_value_line_is_present_even_without_transfers() -> None:
    """The money line is unconditional; the transfers line is not."""
    plan, positions = _gw_plan()
    free_build = GameWeekPlan(
        gameweek=1,
        squad=plan.squad,
        starting_xi=plan.starting_xi,
        captain=plan.captain,
        transfers_in=[],
        transfers_out=[],
        hits=0,
        free_transfers=1,
        expected_points=70.0,
        bank=250,
    )

    md = _render_gameweek(free_build, positions, {})

    assert "Transfers —" not in md
    assert "· bank 25.0" in md
