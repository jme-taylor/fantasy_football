"""Rebuilding a carried-in squad from FPL's public endpoints."""

import pytest
import requests

from fantasy_football.fpl_types import Entry, EntryPicks, EntryTransfer
from fantasy_football.optimisation.team_input import (
    ChipActiveError,
    OpeningBudgetError,
    OwnedPlayer,
    Squad,
    SquadUnavailableError,
    check_opening_budget,
    load_public_squad,
    load_team_file,
    purchase_prices,
    resolve_squad,
    save_squad_snapshot,
)

# Fifteen prices totalling 960, which with a bank of 40 is the
# 100.0m every manager starts on.
OPENING = {100 + i: 64 for i in range(15)}


def _transfer(out_element: int, in_element: int, cost: int, event: int):
    return EntryTransfer(
        element_in=in_element,
        element_in_cost=cost,
        element_out=out_element,
        event=event,
    )


def test_purchase_prices_are_the_opening_prices_before_any_transfer() -> None:
    """An untouched squad was bought at its opening-deadline prices."""
    assert purchase_prices(OPENING, []) == OPENING


def test_purchase_prices_uses_what_was_paid_for_a_transfer_in() -> None:
    """``element_in_cost`` is the price paid, not today's price.

    That is the whole reason the public path can price a squad at all.
    """
    prices = purchase_prices(OPENING, [_transfer(100, 900, 77, 2)])

    assert 100 not in prices
    assert prices[900] == 77


def test_purchase_prices_replays_transfers_in_gameweek_order() -> None:
    """A player bought then sold then re-bought keeps the latest price."""
    prices = purchase_prices(
        OPENING,
        [
            _transfer(100, 900, 77, 2),
            _transfer(900, 901, 80, 3),
            _transfer(901, 900, 85, 4),
        ],
    )

    assert prices[900] == 85
    assert 901 not in prices


def test_purchase_prices_keeps_the_squad_at_fifteen() -> None:
    """Every transfer is one out and one in."""
    prices = purchase_prices(
        OPENING, [_transfer(100, 900, 77, 2), _transfer(101, 901, 60, 3)]
    )

    assert len(prices) == 15


def test_purchase_prices_rejects_selling_a_player_not_owned() -> None:
    """A replay that drifts from reality must not price a squad."""
    with pytest.raises(ValueError, match="999"):
        purchase_prices(OPENING, [_transfer(999, 900, 77, 2)])


def test_opening_budget_checksum_passes_on_a_real_squad() -> None:
    """Every manager starts on exactly 100.0m, spent or banked.

    Verified against a live account: fifteen GW1 prices summing to 960
    against a GW1 bank of 40.
    """
    check_opening_budget({i: 64 for i in range(15)}, bank=40)


def test_opening_budget_checksum_fails_when_it_does_not_add_up() -> None:
    """A mismatch means the opening prices are wrong, so selling is too."""
    with pytest.raises(OpeningBudgetError):
        check_opening_budget({i: 64 for i in range(15)}, bank=0)


class _FakeApi:
    """An FplAPI stand-in that never touches the network."""

    def __init__(
        self,
        next_gw=2,
        started=1,
        current=1,
        picks=None,
        transfers=(),
        errors=None,
    ):
        self._next_gw = next_gw
        self._entry = Entry(started_event=started, current_event=current)
        self._picks = picks or {}
        self._transfers = list(transfers)
        self._errors = errors or {}

    def _maybe_raise(self, key):
        if key in self._errors:
            raise self._errors[key]

    def next_gameweek(self):
        self._maybe_raise("next_gameweek")
        return self._next_gw

    def get_entry(self, manager_id):
        self._maybe_raise("entry")
        return self._entry

    def get_entry_picks(self, manager_id, event):
        self._maybe_raise("picks")
        return self._picks[event]

    def get_entry_transfers(self, manager_id):
        self._maybe_raise("transfers")
        return self._transfers


def _picks(elements, bank=40, chip=None) -> EntryPicks:
    return EntryPicks(elements=list(elements), bank=bank, active_chip=chip)


def _load(api, tmp_path, **kwargs):
    return load_public_squad(
        "2230710",
        season="2026-27",
        expected_gameweek=kwargs.pop("expected_gameweek", 2),
        free_transfers=kwargs.pop("free_transfers", 1),
        prices_at_gameweek=kwargs.pop(
            "prices_at_gameweek", lambda gw: OPENING
        ),
        api=api,
        snapshot_folder=tmp_path,
        **kwargs,
    )


def test_load_public_squad_prices_an_untouched_squad(tmp_path) -> None:
    """No transfers means every player sits at their opening price."""
    api = _FakeApi(picks={1: _picks(OPENING)})

    squad = _load(api, tmp_path)

    assert squad.gameweek == 2
    assert squad.free_transfers == 1
    assert squad.bank == 40
    assert {p.element: p.purchase_price for p in squad.players} == OPENING


def test_load_public_squad_writes_a_snapshot(tmp_path) -> None:
    """The squad is recorded before anything validates it."""
    api = _FakeApi(picks={1: _picks(OPENING)})

    _load(api, tmp_path)

    assert (tmp_path / "2026-27_gw2.json").exists()


def test_load_public_squad_reads_the_latest_settled_gameweek(
    tmp_path,
) -> None:
    """Picks for the upcoming gameweek do not exist until it kicks off."""
    later = dict(OPENING)
    api = _FakeApi(
        next_gw=6,
        current=5,
        picks={1: _picks(OPENING), 5: _picks(later, bank=7)},
    )

    squad = _load(api, tmp_path, expected_gameweek=6)

    assert squad.gameweek == 6
    assert squad.bank == 7


def test_load_public_squad_prices_a_transferred_in_player(tmp_path) -> None:
    """What was paid comes from the transfer, not the opening squad."""
    after = {e: p for e, p in OPENING.items() if e != 100}
    after[900] = 0
    api = _FakeApi(
        next_gw=4,
        current=3,
        picks={1: _picks(OPENING), 3: _picks(after)},
        transfers=[_transfer(100, 900, 77, 3)],
    )

    squad = _load(api, tmp_path, expected_gameweek=4)

    assert {p.element: p.purchase_price for p in squad.players}[900] == 77


def test_load_public_squad_fails_when_the_budget_does_not_add_up(
    tmp_path,
) -> None:
    """Wrong opening prices would make every selling price wrong."""
    api = _FakeApi(picks={1: _picks(OPENING, bank=0)})

    with pytest.raises(OpeningBudgetError):
        _load(api, tmp_path)


def test_load_public_squad_fails_when_fpl_disagrees_on_the_gameweek(
    tmp_path,
) -> None:
    """Stale forward predictions must not be planned on."""
    api = _FakeApi(next_gw=3, picks={1: _picks(OPENING)})

    with pytest.raises(ValueError, match="3"):
        _load(api, tmp_path, expected_gameweek=2)


def test_load_public_squad_rejects_a_free_hit_squad(tmp_path) -> None:
    """A free hit squad reverts, so it is not what is carried in."""
    api = _FakeApi(picks={1: _picks(OPENING, chip="freehit")})

    with pytest.raises(ChipActiveError):
        _load(api, tmp_path)


def test_load_public_squad_allows_a_past_wildcard(tmp_path) -> None:
    """A wildcard squad persists, so it is carried in like any other."""
    api = _FakeApi(picks={1: _picks(OPENING, chip="wildcard")})

    assert len(_load(api, tmp_path).players) == 15


def test_load_public_squad_survives_a_network_failure(tmp_path) -> None:
    """Being offline must not throw away the run's training output."""
    for stage in ("next_gameweek", "entry", "picks", "transfers"):
        api = _FakeApi(
            picks={1: _picks(OPENING)},
            errors={stage: requests.ConnectionError("no route")},
        )
        with pytest.raises(SquadUnavailableError):
            _load(api, tmp_path)


def test_load_public_squad_reports_a_manager_who_has_not_started(
    tmp_path,
) -> None:
    """A brand-new entry has no settled gameweek to read."""
    api = _FakeApi(current=None, picks={1: _picks(OPENING)})

    with pytest.raises(SquadUnavailableError):
        _load(api, tmp_path)


def _squad() -> Squad:
    return Squad(
        gameweek=7,
        free_transfers=2,
        bank=12,
        players=[
            OwnedPlayer(element=element, purchase_price=price)
            for element, price in OPENING.items()
        ],
    )


def test_save_squad_snapshot_round_trips_through_the_file_reader(
    tmp_path,
) -> None:
    """What is written is a team file the next run could be given.

    The snapshot is element-keyed, so it round-trips without inverting
    ids back through the roster.
    """
    squad = _squad()

    path = save_squad_snapshot(squad, "2026-27", folder=tmp_path)

    team = load_team_file(path)
    assert team.gameweek == 7
    assert team.free_transfers == 2
    assert team.bank == 12
    assert resolve_squad(team.players, "2026-27") == squad.players


def test_save_squad_snapshot_names_the_file_by_season_and_gameweek(
    tmp_path,
) -> None:
    """One file per gameweek, so re-running a week overwrites in place."""
    path = save_squad_snapshot(_squad(), "2026-27", folder=tmp_path)

    assert path.name == "2026-27_gw7.json"
    assert path.parent == tmp_path
