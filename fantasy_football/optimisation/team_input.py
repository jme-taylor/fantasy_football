import json
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, TypeAdapter

from fantasy_football.features.roster import current_roster

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

config = ConfigDict(extra="forbid")


def selling_price(purchase: int, current: int) -> int:
    """Return what a player would sell for, in tenths of a million.

    FPL pays out the purchase price plus half of any profit, rounded down to
    the nearest 0.1m: a player bought at 12.5m and now worth 13.5m sells for
    13.0m, not 13.5m. A player who has *fallen* is a separate rule rather
    than a consequence of the same one -- there is no halving, you take the
    whole loss and sell at the current price.

    Parameters
    ----------
    purchase : int
        What the player cost, in tenths of a million.
    current : int
        What the player is worth now, in tenths of a million.

    Returns
    -------
    int
        The selling price in tenths of a million.
    """
    if current <= purchase:
        return current
    return purchase + (current - purchase) // 2


class SquadPlayer(BaseModel):
    """A player in a hand-authored team file.

    Attributes
    ----------
    name : str
        Full player name exactly as it appears in the data's ``name`` column.
    purchase_price : int
        What the player cost when bought, in tenths of a million. Hand-entered
        for now; sourcing it from the authenticated FPL ``my-team`` endpoint
        is a future improvement.
    """

    model_config = config

    name: str
    purchase_price: int


class OwnedPlayer(BaseModel):
    """A carried-in squad player, keyed the way the optimiser keys players.

    The team file speaks names because that is what a human can write down;
    the optimiser speaks element ids. ``resolve_squad`` is the one step
    between the two vocabularies, and this is what it produces.

    Attributes
    ----------
    element : int
        The player's FPL element id.
    purchase_price : int
        What the player cost when bought, in tenths of a million.
    """

    model_config = config

    element: int
    purchase_price: int


class TeamFile(BaseModel):
    """A human-authored FPL team for a single gameweek.

    Attributes
    ----------
    gameweek : int
        The gameweek the team is for; the optimiser's start_gw.
    free_transfers : int
        Free transfers available at that gameweek.
    players : list[SquadPlayer]
        The squad, each player carrying the price they were bought at.
    bank : int
        Money in the bank in tenths of a million. Defaults to 0.
    """

    model_config = config

    gameweek: int
    free_transfers: int
    players: list[SquadPlayer]
    bank: int = 0


def load_team_file(path: "Path | str") -> TeamFile:
    """Read and validate a team JSON file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the JSON file (str or pathlib.Path).

    Returns
    -------
    TeamFile
        The validated team.
    """
    data = json.loads(Path(path).read_text())
    return TypeAdapter(TeamFile).validate_python(data)


def resolve_squad(
    players: list[SquadPlayer],
    season: str,
    connection: "DuckDBPyConnection | None" = None,
) -> list[OwnedPlayer]:
    """Map a declared squad onto element ids, carrying purchase prices.

    Names resolve against the current roster -- the same set of players the
    optimiser can buy -- so a name that resolves is always a player the plan
    can actually hold. Resolving against played gameweeks instead would
    accept a name the optimiser has no price for, and reject a summer signing
    who has not played yet.

    Parameters
    ----------
    players : list[SquadPlayer]
        The declared squad, as read from a team file.
    season : str
        Season to read, e.g. ``"2026-27"``.
    connection : duckdb.DuckDBPyConnection | None, optional
        An open connection. When None, one is opened per table read.

    Returns
    -------
    list[OwnedPlayer]
        One entry per input player, in order.

    Raises
    ------
    ValueError
        If any name has no row, or maps to more than one distinct element.
    """
    roster = current_roster(season, connection)
    name_to_ids: dict[str, set[int]] = {}
    for name, element in zip(
        roster["name"].to_list(), roster["element"].to_list(), strict=True
    ):
        name_to_ids.setdefault(name, set()).add(element)

    names = [player.name for player in players]
    unmatched = sorted(n for n in names if n not in name_to_ids)
    ambiguous = sorted(
        n for n in names if n in name_to_ids and len(name_to_ids[n]) > 1
    )
    if unmatched or ambiguous:
        raise ValueError(
            f"could not resolve names to ids: unmatched={unmatched}, "
            f"ambiguous={ambiguous}"
        )
    return [
        OwnedPlayer(
            element=next(iter(name_to_ids[player.name])),
            purchase_price=player.purchase_price,
        )
        for player in players
    ]
