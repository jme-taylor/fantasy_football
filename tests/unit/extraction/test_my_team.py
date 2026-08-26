"""The authenticated ``my-team`` endpoint and the next-gameweek lookup."""

import json
from pathlib import Path

import pytest
import requests
from pytest_mock import MockerFixture

from fantasy_football.extraction.fpl import FplAPI

FIXTURE = Path(__file__).parents[2] / "fixtures" / "my_team.json"
COOKIE = "pl_profile=abc; sessionid=def"


def _payload() -> dict:
    return json.loads(FIXTURE.read_text())


def _mock_get(mocker: MockerFixture, payload: dict, status: int = 200):
    response = mocker.Mock()
    response.json.return_value = payload
    if status >= 400:
        response.raise_for_status.side_effect = requests.HTTPError(
            f"{status} Client Error"
        )
    else:
        response.raise_for_status.return_value = None
    return mocker.patch("requests.get", return_value=response)


def test_get_my_team_parses_picks_and_transfers(
    mocker: MockerFixture,
) -> None:
    """The slice of the response the optimiser needs is parsed out."""
    _mock_get(mocker, _payload())

    my_team = FplAPI().get_my_team("7515957", COOKIE)

    assert len(my_team.picks) == 15
    assert my_team.picks[0].element == 100
    assert my_team.picks[0].purchase_price == 45
    assert my_team.picks[0].selling_price == 47
    assert my_team.transfers.limit == 2
    assert my_team.transfers.bank == 12


def test_get_my_team_sends_the_cookie(mocker: MockerFixture) -> None:
    """Without the cookie the endpoint 403s, so it has to be sent."""
    get = _mock_get(mocker, _payload())

    FplAPI().get_my_team("7515957", COOKIE)

    _, kwargs = get.call_args
    assert kwargs["headers"]["Cookie"] == COOKIE


def test_get_my_team_requests_the_managers_own_team(
    mocker: MockerFixture,
) -> None:
    """The manager id is the only thing addressing the endpoint."""
    get = _mock_get(mocker, _payload())

    FplAPI().get_my_team("7515957", COOKIE)

    url = get.call_args[0][0]
    assert url.endswith("my-team/7515957/")


def test_get_my_team_raises_on_an_expired_cookie(
    mocker: MockerFixture,
) -> None:
    """A 403 is surfaced rather than parsed into an empty squad."""
    _mock_get(mocker, {}, status=403)

    with pytest.raises(requests.HTTPError):
        FplAPI().get_my_team("7515957", COOKIE)


def test_get_my_team_ignores_fields_it_does_not_read(
    mocker: MockerFixture,
) -> None:
    """FPL adding a key must not break the parse."""
    payload = _payload()
    payload["something_new"] = {"nested": True}
    payload["picks"][0]["brand_new"] = 1
    _mock_get(mocker, payload)

    assert len(FplAPI().get_my_team("7515957", COOKIE).picks) == 15


def _bootstrap(next_event: int | None) -> dict:
    return {
        "events": [
            {"id": gw, "is_next": gw == next_event} for gw in range(1, 39)
        ]
    }


def test_next_gameweek_reads_the_event_flagged_next(
    mocker: MockerFixture,
) -> None:
    """``is_next`` is FPL's own answer to which deadline is upcoming."""
    api = FplAPI()
    mocker.patch.object(api, "get_bootstrap_data", return_value=_bootstrap(7))

    assert api.next_gameweek() == 7


def test_next_gameweek_is_none_when_the_season_is_over(
    mocker: MockerFixture,
) -> None:
    """After GW38 no event is flagged next."""
    api = FplAPI()
    mocker.patch.object(
        api, "get_bootstrap_data", return_value=_bootstrap(None)
    )

    assert api.next_gameweek() is None
