"""Turning an authenticated ``my-team`` response into a carried-in squad."""

import json
import logging
from pathlib import Path

import pytest
import requests

from fantasy_football.fpl_types import MyTeam
from fantasy_football.optimisation.team_input import (
    ChipActiveError,
    OwnedPlayer,
    Squad,
    SquadUnavailableError,
    check_selling_prices,
    load_api_squad,
    load_team_file,
    resolve_squad,
    save_squad_snapshot,
    squad_from_my_team,
)

FIXTURE = Path(__file__).parents[2] / "fixtures" / "my_team.json"


def _my_team(**overrides) -> MyTeam:
    payload = json.loads(FIXTURE.read_text())
    payload["transfers"].update(overrides)
    return MyTeam(**payload)


def test_squad_from_my_team_carries_purchase_prices_and_elements() -> None:
    """The optimiser keys on elements and prices from purchase price."""
    squad = squad_from_my_team(_my_team(), gameweek=7)

    assert isinstance(squad, Squad)
    assert squad.gameweek == 7
    assert len(squad.players) == 15
    assert squad.players[0] == OwnedPlayer(element=100, purchase_price=45)


def test_squad_from_my_team_reads_transfers_and_bank() -> None:
    """Free transfers and bank come from the transfers block."""
    squad = squad_from_my_team(_my_team(), gameweek=7)

    assert squad.free_transfers == 2
    assert squad.bank == 12


def test_squad_from_my_team_discounts_transfers_already_made() -> None:
    """``limit`` is the week's allowance, not what is left of it.

    Reading it whole would plan a transfer that has already been spent,
    and the -4 it really costs would never appear in the plan.
    """
    squad = squad_from_my_team(_my_team(limit=2, made=1), gameweek=7)

    assert squad.free_transfers == 1


def test_squad_from_my_team_floors_free_transfers_at_zero() -> None:
    """Transfers taken as hits push ``made`` past the allowance."""
    squad = squad_from_my_team(_my_team(limit=1, made=3), gameweek=7)

    assert squad.free_transfers == 0


def test_squad_from_my_team_rejects_an_active_chip() -> None:
    """A null transfer limit means a wildcard or free hit is in play.

    There is no representation for unlimited transfers downstream, and a
    free hit squad reverts next gameweek, so planning off it would be a
    plan for a team that will not exist.
    """
    with pytest.raises(ChipActiveError):
        squad_from_my_team(_my_team(limit=None), gameweek=7)


def test_save_squad_snapshot_round_trips_through_the_file_reader(
    tmp_path,
) -> None:
    """What is written is a team file the next run could be given.

    The snapshot is element-keyed, so it round-trips without inverting
    ids back through the roster.
    """
    squad = squad_from_my_team(_my_team(), gameweek=7)

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
    squad = squad_from_my_team(_my_team(), gameweek=7)

    path = save_squad_snapshot(squad, "2026-27", folder=tmp_path)

    assert path.name == "2026-27_gw7.json"
    assert path.parent == tmp_path


def test_check_selling_prices_is_quiet_when_they_agree(caplog) -> None:
    """FPL's selling price should match the rule we implement by hand."""
    my_team = _my_team()
    prices = {
        pick.element: pick.purchase_price + (4 if i % 3 == 0 else 0)
        for i, pick in enumerate(my_team.picks)
    }

    with caplog.at_level(logging.WARNING):
        check_selling_prices(my_team, prices)

    assert caplog.records == []


def test_check_selling_prices_warns_on_a_disagreement(caplog) -> None:
    """A mismatch is a warning, never a failure.

    Price changes land at 02:00, so the stored price and FPL's can
    legitimately disagree by timing, and losing a whole plan to a 0.1m
    discrepancy is the worse trade.
    """
    my_team = _my_team()
    prices = {
        pick.element: pick.purchase_price + (4 if i % 3 == 0 else 0)
        for i, pick in enumerate(my_team.picks)
    }
    prices[my_team.picks[1].element] = 999

    with caplog.at_level(logging.WARNING):
        check_selling_prices(my_team, prices)

    assert len(caplog.records) == 1
    assert "101" in caplog.records[0].getMessage()


def test_check_selling_prices_skips_players_with_no_stored_price(
    caplog,
) -> None:
    """An element the roster has no price for is not a mismatch."""
    with caplog.at_level(logging.WARNING):
        check_selling_prices(_my_team(), {})

    assert caplog.records == []


class _FakeApi:
    """An FplAPI stand-in that never touches the network."""

    def __init__(self, next_gw=7, my_team=None, error=None, gw_error=None):
        self._next_gw = next_gw
        self._my_team = my_team
        self._error = error
        self._gw_error = gw_error
        self.cookie = None

    def next_gameweek(self):
        if self._gw_error is not None:
            raise self._gw_error
        return self._next_gw

    def get_my_team(self, manager_id, cookie):
        self.cookie = cookie
        if self._error is not None:
            raise self._error
        return self._my_team


def test_load_api_squad_returns_the_squad_and_writes_a_snapshot(
    tmp_path,
) -> None:
    """The happy path: one fetch, one snapshot, one squad."""
    api = _FakeApi(my_team=_my_team())

    squad = load_api_squad(
        "7515957",
        "cookie",
        season="2026-27",
        expected_gameweek=7,
        api=api,
        prices={},
        snapshot_folder=tmp_path,
    )

    assert squad.gameweek == 7
    assert len(squad.players) == 15
    assert (tmp_path / "2026-27_gw7.json").exists()


def test_load_api_squad_fails_when_fpl_and_the_predictions_disagree(
    tmp_path,
) -> None:
    """A disagreement means the forward predictions are stale.

    Planning off that silently is how a confidently wrong transfer gets
    made, so it stops the run.
    """
    api = _FakeApi(next_gw=8, my_team=_my_team())

    with pytest.raises(ValueError, match="8"):
        load_api_squad(
            "7515957",
            "cookie",
            season="2026-27",
            expected_gameweek=7,
            api=api,
            prices={},
            snapshot_folder=tmp_path,
        )


def test_load_api_squad_fails_when_the_season_is_over(tmp_path) -> None:
    """No event is flagged next, so there is no gameweek to plan."""
    api = _FakeApi(next_gw=None, my_team=_my_team())

    with pytest.raises(ValueError):
        load_api_squad(
            "7515957",
            "cookie",
            season="2026-27",
            expected_gameweek=7,
            api=api,
            prices={},
            snapshot_folder=tmp_path,
        )


def test_load_api_squad_turns_an_expired_cookie_into_a_skip(tmp_path) -> None:
    """A 403 is a paste-a-cookie problem, not a crash."""
    api = _FakeApi(my_team=None, error=requests.HTTPError("403 Forbidden"))

    with pytest.raises(SquadUnavailableError):
        load_api_squad(
            "7515957",
            "cookie",
            season="2026-27",
            expected_gameweek=7,
            api=api,
            prices={},
            snapshot_folder=tmp_path,
        )


def test_load_api_squad_turns_a_network_failure_into_a_skip(tmp_path) -> None:
    """Being offline must not throw away the run's training output."""
    api = _FakeApi(my_team=None, error=requests.ConnectionError("no route"))

    with pytest.raises(SquadUnavailableError):
        load_api_squad(
            "7515957",
            "cookie",
            season="2026-27",
            expected_gameweek=7,
            api=api,
            prices={},
            snapshot_folder=tmp_path,
        )


def test_load_api_squad_never_puts_the_cookie_in_the_error(tmp_path) -> None:
    """The cookie is a credential; it belongs in no message."""
    secret = "pl_profile=SECRETVALUE"
    api = _FakeApi(my_team=None, error=requests.HTTPError("403 Forbidden"))

    with pytest.raises(SquadUnavailableError) as exc:
        load_api_squad(
            "7515957",
            secret,
            season="2026-27",
            expected_gameweek=7,
            api=api,
            prices={},
            snapshot_folder=tmp_path,
        )

    assert "SECRETVALUE" not in str(exc.value)


def test_load_api_squad_survives_a_failure_reading_the_next_gameweek(
    tmp_path,
) -> None:
    """The bootstrap call is on the network too, and unguarded by FPL.

    ``get_bootstrap_data`` never raises for status, so a 5xx serving an
    HTML error page fails on the JSON decode rather than the request.
    """
    api = _FakeApi(gw_error=requests.ConnectionError("no route"))

    with pytest.raises(SquadUnavailableError):
        load_api_squad(
            "7515957",
            "cookie",
            season="2026-27",
            expected_gameweek=7,
            api=api,
            prices={},
            snapshot_folder=tmp_path,
        )
