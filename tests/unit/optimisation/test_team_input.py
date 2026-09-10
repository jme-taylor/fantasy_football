import json
from datetime import datetime

import polars as pl
import pytest

from fantasy_football.optimisation.team_input import (
    OwnedPlayer,
    TeamFile,
    load_team_file,
    resolve_squad,
    selling_price,
    squad_from_team_file,
)
from fantasy_football.storage import database
from fantasy_football.storage.database import get_connection
from fantasy_football.storage.tables import PLAYER_SEASON, PLAYER_SNAPSHOT

SEASON = "2026-27"


def _write_team(tmp_path, payload) -> str:
    path = tmp_path / "team.json"
    path.write_text(json.dumps(payload))
    return str(path)


def _squad_payload(count: int = 15) -> list[dict]:
    """Return a team file's ``players`` block with distinct names."""
    return [{"name": f"P{i}", "purchase_price": 50 + i} for i in range(count)]


def test_selling_price_returns_purchase_price_when_price_is_unchanged() -> (
    None
):
    """A player who has not moved sells for exactly what they cost."""
    assert selling_price(purchase=50, current=50) == 50


def test_selling_price_halves_an_even_profit_exactly() -> None:
    """A profit of four tenths banks two of them."""
    assert selling_price(purchase=50, current=54) == 52


def test_selling_price_rounds_an_odd_profit_down() -> None:
    """A profit of three tenths banks one, not one and a half."""
    assert selling_price(purchase=50, current=53) == 51


def test_selling_price_banks_nothing_on_a_one_tenth_rise() -> None:
    """Half of a single tenth rounds down to nothing."""
    assert selling_price(purchase=50, current=51) == 50


def test_selling_price_takes_the_full_loss_on_a_faller() -> None:
    """A player below their purchase price sells at the current price."""
    assert selling_price(purchase=50, current=47) == 47


def test_selling_price_takes_the_full_loss_however_far_the_fall() -> None:
    """The loss is never halved, however large it is."""
    assert selling_price(purchase=120, current=95) == 95


def test_load_team_file_reads_fields_and_defaults_bank(tmp_path) -> None:
    """load_team_file parses all fields correctly and defaults bank to 0."""
    path = _write_team(
        tmp_path,
        {
            "gameweek": 5,
            "free_transfers": 2,
            "players": _squad_payload(),
        },
    )
    team = load_team_file(path)
    assert isinstance(team, TeamFile)
    assert team.gameweek == 5
    assert team.free_transfers == 2
    assert team.bank == 0
    assert [p.name for p in team.players] == [f"P{i}" for i in range(15)]
    assert [p.purchase_price for p in team.players] == [
        50 + i for i in range(15)
    ]


def test_load_team_file_rejects_bare_player_names(tmp_path) -> None:
    """The pre-purchase-price format is rejected rather than defaulted.

    Silently defaulting a missing purchase price to the current price would
    reinstate the exact over-valuation this format change exists to fix.
    """
    path = _write_team(
        tmp_path,
        {
            "gameweek": 5,
            "free_transfers": 1,
            "players": [f"P{i}" for i in range(15)],
        },
    )
    with pytest.raises(ValueError):
        load_team_file(path)


def test_load_team_file_rejects_a_player_without_a_purchase_price(
    tmp_path,
) -> None:
    """A player object missing its purchase price is rejected."""
    path = _write_team(
        tmp_path,
        {
            "gameweek": 5,
            "free_transfers": 1,
            "players": [{"name": "P0"}],
        },
    )
    with pytest.raises(ValueError):
        load_team_file(path)


def test_load_team_file_rejects_extra_keys(tmp_path) -> None:
    """load_team_file raises ValueError when the JSON contains extra keys."""
    path = _write_team(
        tmp_path,
        {
            "gameweek": 5,
            "free_transfers": 1,
            "players": [],
            "unexpected": True,
        },
    )
    with pytest.raises(ValueError):
        load_team_file(path)


def test_load_team_file_rejects_extra_keys_on_a_player(tmp_path) -> None:
    """An unexpected key inside a player object is rejected too."""
    path = _write_team(
        tmp_path,
        {
            "gameweek": 5,
            "free_transfers": 1,
            "players": [
                {"name": "P0", "purchase_price": 50, "unexpected": True}
            ],
        },
    )
    with pytest.raises(ValueError):
        load_team_file(path)


def test_load_team_file_rejects_missing_field(tmp_path) -> None:
    """load_team_file raises ValueError when a required field is absent."""
    path = _write_team(tmp_path, {"gameweek": 5, "players": []})
    with pytest.raises(ValueError):
        load_team_file(path)


def _seed_roster(
    tmp_path, monkeypatch, *, names, elements, season=SEASON
) -> None:
    """Seed the snapshot and identity rows the roster is built from."""
    captured = datetime(2026, 8, 1, 12, 0)
    snapshot = pl.DataFrame(
        [
            {
                "season": season,
                "captured_at": captured,
                "element": element,
                "value": 50,
                "team": "T",
                "position": "MID",
                "chance_of_playing_this_round": 100,
                "status": "a",
            }
            for element in dict.fromkeys(elements)
        ]
    )
    identity = pl.DataFrame(
        [
            {
                "season": season,
                "element": element,
                "player_code": 10_000 + element,
                "web_name": name.split(" ", 1)[1],
                "first_name": name.split(" ")[0],
                "second_name": name.split(" ", 1)[1],
                "position": "MID",
                "team_code": element,
                "birth_date": None,
                "region": None,
                "team_join_date": None,
            }
            for name, element in zip(names, elements, strict=True)
        ],
        schema_overrides={
            "birth_date": pl.Date,
            "region": pl.Int64,
            "team_join_date": pl.Date,
        },
    )
    db_path = tmp_path / "t.duckdb"
    monkeypatch.setattr(database, "DATABASE_PATH", db_path)
    connection = get_connection(db_path)
    try:
        PLAYER_SNAPSHOT.upsert_current(connection, snapshot, season)
        PLAYER_SEASON.upsert_current(connection, identity, season)
    finally:
        connection.close()


def _declared(name: str, purchase_price: int) -> dict:
    return {"name": name, "purchase_price": purchase_price}


def _team(tmp_path, players: list[dict]) -> TeamFile:
    path = _write_team(
        tmp_path,
        {"gameweek": 5, "free_transfers": 1, "players": players},
    )
    return load_team_file(path)


def test_resolve_squad_maps_names_and_carries_purchase_prices(
    tmp_path, monkeypatch
) -> None:
    """resolve_squad returns OwnedPlayers in order, prices intact."""
    _seed_roster(
        tmp_path,
        monkeypatch,
        names=["Mohamed Salah", "Erling Haaland"],
        elements=[328, 351],
    )
    team = _team(
        tmp_path,
        [
            _declared("Erling Haaland", 140),
            _declared("Mohamed Salah", 125),
        ],
    )

    owned = resolve_squad(team.players, SEASON)

    assert owned == [
        OwnedPlayer(element=351, purchase_price=140),
        OwnedPlayer(element=328, purchase_price=125),
    ]


def test_resolve_squad_reports_unmatched(tmp_path, monkeypatch) -> None:
    """An unmatched name is named in the raised ValueError."""
    _seed_roster(
        tmp_path, monkeypatch, names=["Mohamed Salah"], elements=[328]
    )
    team = _team(
        tmp_path,
        [_declared("Mohamed Salah", 125), _declared("Ghost Player", 50)],
    )
    with pytest.raises(ValueError, match="Ghost Player"):
        resolve_squad(team.players, SEASON)


def test_resolve_squad_reports_ambiguous(tmp_path, monkeypatch) -> None:
    """A name mapping to multiple elements raises an 'ambiguous' ValueError."""
    _seed_roster(
        tmp_path,
        monkeypatch,
        names=["Danny Ward", "Danny Ward"],
        elements=[11, 22],
    )
    team = _team(tmp_path, [_declared("Danny Ward", 45)])
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_squad(team.players, SEASON)


def test_resolve_squad_reports_unmatched_and_ambiguous_together(
    tmp_path, monkeypatch
) -> None:
    """Both unmatched and ambiguous offenders are named in one error."""
    _seed_roster(
        tmp_path,
        monkeypatch,
        names=["Danny Ward", "Danny Ward"],
        elements=[11, 22],
    )
    team = _team(
        tmp_path,
        [_declared("Danny Ward", 45), _declared("Ghost Player", 50)],
    )
    with pytest.raises(ValueError) as exc:
        resolve_squad(team.players, SEASON)
    assert "Danny Ward" in str(exc.value)
    assert "Ghost Player" in str(exc.value)


def test_resolve_squad_finds_a_player_who_has_not_played(
    tmp_path, monkeypatch
) -> None:
    """A summer signing is on the roster, so their name resolves."""
    _seed_roster(tmp_path, monkeypatch, names=["New Signing"], elements=[500])
    team = _team(tmp_path, [_declared("New Signing", 55)])
    assert resolve_squad(team.players, SEASON) == [
        OwnedPlayer(element=500, purchase_price=55)
    ]


def test_load_team_file_accepts_an_element_keyed_player(tmp_path) -> None:
    """A player may be declared by element id instead of by name.

    This is the form the API snapshot is written in, so it has to load
    back through the same reader a hand-authored file uses.
    """
    path = _write_team(
        tmp_path,
        {
            "gameweek": 5,
            "free_transfers": 1,
            "players": [{"element": 328, "purchase_price": 125}],
        },
    )
    team = load_team_file(path)
    assert team.players[0].element == 328
    assert team.players[0].name is None


def test_load_team_file_rejects_a_player_with_both_name_and_element(
    tmp_path,
) -> None:
    """Declaring both leaves no answer for which one wins."""
    path = _write_team(
        tmp_path,
        {
            "gameweek": 5,
            "free_transfers": 1,
            "players": [{"name": "P0", "element": 328, "purchase_price": 125}],
        },
    )
    with pytest.raises(ValueError):
        load_team_file(path)


def test_load_team_file_rejects_a_player_with_neither(tmp_path) -> None:
    """A player identified by nothing at all is rejected."""
    path = _write_team(
        tmp_path,
        {
            "gameweek": 5,
            "free_transfers": 1,
            "players": [{"purchase_price": 125}],
        },
    )
    with pytest.raises(ValueError):
        load_team_file(path)


def test_resolve_squad_passes_element_keyed_players_through(
    tmp_path, monkeypatch
) -> None:
    """An element needs no roster lookup, so it is carried straight over.

    The optimiser rejects an element with no predictions anyway, which is
    a stricter check than roster membership.
    """
    _seed_roster(
        tmp_path, monkeypatch, names=["Mohamed Salah"], elements=[328]
    )
    path = _write_team(
        tmp_path,
        {
            "gameweek": 5,
            "free_transfers": 1,
            "players": [{"element": 999, "purchase_price": 60}],
        },
    )
    team = load_team_file(path)

    assert resolve_squad(team.players, SEASON) == [
        OwnedPlayer(element=999, purchase_price=60)
    ]


def test_resolve_squad_mixes_named_and_element_keyed_players(
    tmp_path, monkeypatch
) -> None:
    """Both forms can appear in one file, in order."""
    _seed_roster(
        tmp_path, monkeypatch, names=["Mohamed Salah"], elements=[328]
    )
    path = _write_team(
        tmp_path,
        {
            "gameweek": 5,
            "free_transfers": 1,
            "players": [
                {"element": 999, "purchase_price": 60},
                {"name": "Mohamed Salah", "purchase_price": 125},
            ],
        },
    )
    team = load_team_file(path)

    assert resolve_squad(team.players, SEASON) == [
        OwnedPlayer(element=999, purchase_price=60),
        OwnedPlayer(element=328, purchase_price=125),
    ]


def test_squad_from_team_file_resolves_names_and_carries_the_rest(
    tmp_path, monkeypatch
) -> None:
    """The file path reaches the same Squad the API path produces."""
    _seed_roster(
        tmp_path, monkeypatch, names=["Mohamed Salah"], elements=[328]
    )
    path = _write_team(
        tmp_path,
        {
            "gameweek": 5,
            "free_transfers": 2,
            "bank": 8,
            "players": [_declared("Mohamed Salah", 125)],
        },
    )

    squad = squad_from_team_file(load_team_file(path), SEASON)

    assert squad.gameweek == 5
    assert squad.free_transfers == 2
    assert squad.bank == 8
    assert squad.players == [OwnedPlayer(element=328, purchase_price=125)]
