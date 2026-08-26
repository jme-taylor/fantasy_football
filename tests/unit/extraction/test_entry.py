"""The public entry endpoints the optimiser reads a squad from."""

import json
from pathlib import Path

import pytest
import requests
from pytest_mock import MockerFixture

from fantasy_football.extraction.fpl import FplAPI

FIXTURES = Path(__file__).parents[2] / "fixtures"
MANAGER = "2230710"


def _payload(name: str):
    return json.loads((FIXTURES / f"{name}.json").read_text())


def _mock_get(mocker: MockerFixture, payload, status: int = 200):
    response = mocker.Mock()
    response.json.return_value = payload
    if status >= 400:
        response.raise_for_status.side_effect = requests.HTTPError(str(status))
    else:
        response.raise_for_status.return_value = None
    return mocker.patch("requests.get", return_value=response)


def test_get_entry_reads_the_opening_and_settled_gameweeks(
    mocker: MockerFixture,
) -> None:
    """Both are needed: one prices the squad, one reads it."""
    _mock_get(mocker, _payload("entry"))

    entry = FplAPI().get_entry(MANAGER)

    assert entry.started_event == 1
    assert entry.current_event == 1


def test_get_entry_picks_reads_elements_bank_and_chip(
    mocker: MockerFixture,
) -> None:
    """The slice of the picks payload a carried-in squad needs."""
    _mock_get(mocker, _payload("entry_picks"))

    picks = FplAPI().get_entry_picks(MANAGER, 1)

    assert len(picks.elements) == 15
    assert picks.elements[0] == 100
    assert picks.bank == 40
    assert picks.active_chip is None


def test_get_entry_picks_addresses_the_requested_gameweek(
    mocker: MockerFixture,
) -> None:
    """Picks are per gameweek, and only exist once it has kicked off."""
    get = _mock_get(mocker, _payload("entry_picks"))

    FplAPI().get_entry_picks(MANAGER, 7)

    assert get.call_args[0][0].endswith(f"entry/{MANAGER}/event/7/picks/")


def test_get_entry_transfers_returns_oldest_first(
    mocker: MockerFixture,
) -> None:
    """Replaying transfers out of order would price the wrong squad."""
    _mock_get(mocker, _payload("entry_transfers"))

    transfers = FplAPI().get_entry_transfers(MANAGER)

    assert [t.event for t in transfers] == [2, 3]
    assert transfers[0].element_in == 901
    assert transfers[0].element_in_cost == 55


def test_get_entry_transfers_is_empty_before_any_are_made(
    mocker: MockerFixture,
) -> None:
    """A manager who has not transferred returns an empty list."""
    _mock_get(mocker, [])

    assert FplAPI().get_entry_transfers(MANAGER) == []


def test_entry_endpoints_send_a_browser_user_agent(
    mocker: MockerFixture,
) -> None:
    """FPL serves a challenge page to obviously scripted clients."""
    get = _mock_get(mocker, _payload("entry"))

    FplAPI().get_entry(MANAGER)

    assert "Mozilla" in get.call_args[1]["headers"]["User-Agent"]


def test_entry_endpoints_time_out(mocker: MockerFixture) -> None:
    """A hung connection would defeat the fail-soft squad read."""
    get = _mock_get(mocker, _payload("entry"))

    FplAPI().get_entry(MANAGER)

    assert get.call_args[1]["timeout"] > 0


def test_entry_endpoints_raise_on_an_error_status(
    mocker: MockerFixture,
) -> None:
    """A 404 must not be parsed into an empty squad."""
    _mock_get(mocker, {}, status=404)

    with pytest.raises(requests.HTTPError):
        FplAPI().get_entry(MANAGER)


def test_entry_endpoints_ignore_fields_they_do_not_read(
    mocker: MockerFixture,
) -> None:
    """FPL adding a key must not break the parse."""
    payload = _payload("entry")
    payload["brand_new"] = {"nested": True}
    _mock_get(mocker, payload)

    assert FplAPI().get_entry(MANAGER).started_event == 1
