import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import requests
from pydantic import BaseModel, ConfigDict, TypeAdapter, model_validator

from fantasy_football.constants import DATA_FOLDER
from fantasy_football.features.roster import current_roster
from fantasy_football.fpl_types import MyTeam

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

logger = logging.getLogger(__name__)

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
    """A player in a team file, identified by name or by element id.

    A hand-authored file names players because that is what a human can
    write down. A file written from the ``my-team`` endpoint has element
    ids and no names, and inverting them back through the roster would
    make the write fail on the ambiguity the read already handles. So the
    schema takes either, and exactly one.

    Attributes
    ----------
    name : str or None
        Full player name exactly as it appears in the data's ``name``
        column. None when the player is declared by element.
    element : int or None
        The player's FPL element id. None when declared by name.
    purchase_price : int
        What the player cost when bought, in tenths of a million.
    """

    model_config = config

    name: str | None = None
    element: int | None = None
    purchase_price: int

    @model_validator(mode="after")
    def _exactly_one_identifier(self) -> "SquadPlayer":
        if (self.name is None) == (self.element is None):
            raise ValueError(
                "a squad player needs exactly one of name or element, "
                f"got name={self.name!r} element={self.element!r}"
            )
        return self


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


class Squad(BaseModel):
    """A carried-in squad, however it was sourced.

    What the optimiser is given. A team file and the ``my-team`` endpoint
    are two ways of arriving here and neither is the contract: the file
    speaks names and needs resolving, the endpoint speaks elements and
    does not.

    Attributes
    ----------
    gameweek : int
        The gameweek the squad is carried into; the optimiser's start_gw.
    free_transfers : int
        Free transfers available at that gameweek.
    bank : int
        Money in the bank in tenths of a million.
    players : list[OwnedPlayer]
        The squad, each player carrying the price they were bought at.
    """

    model_config = config

    gameweek: int
    free_transfers: int
    bank: int
    players: list[OwnedPlayer]


class ChipActiveError(ValueError):
    """Raised when a wildcard or free hit makes the squad unplannable."""


class SquadUnavailableError(RuntimeError):
    """Raised when the endpoint cannot be reached or refuses the cookie.

    Distinct from a bad squad: nothing is wrong with the plan, the
    credential or the network is. Callers skip optimisation on this and
    keep everything the run has already produced.
    """


def squad_from_my_team(my_team: MyTeam, gameweek: int) -> Squad:
    """Adapt an authenticated ``my-team`` response into a Squad.

    Parameters
    ----------
    my_team : MyTeam
        The parsed endpoint response.
    gameweek : int
        The gameweek the response is for. The endpoint does not say, so
        it is passed in.

    Returns
    -------
    Squad
        The carried-in squad. ``limit`` is the gameweek's allowance
        rather than what is left of it, so what has already been made is
        taken off. Transfers taken as hits can push that below zero,
        which is floored: no fewer than none are free.

    Raises
    ------
    ChipActiveError
        If the transfer limit is null, which is how the response reports
        an active wildcard or free hit.
    """
    if my_team.transfers.limit is None:
        raise ChipActiveError(
            f"GW{gameweek} has a wildcard or free hit active, so there is "
            "no free transfer count and the squad may not survive the "
            "gameweek; plan this one by hand."
        )
    return Squad(
        gameweek=gameweek,
        free_transfers=max(
            my_team.transfers.limit - my_team.transfers.made, 0
        ),
        bank=my_team.transfers.bank,
        players=[
            OwnedPlayer(
                element=pick.element, purchase_price=pick.purchase_price
            )
            for pick in my_team.picks
        ],
    )


def check_selling_prices(my_team: MyTeam, prices: Mapping[int, int]) -> None:
    """Warn where FPL's selling price disagrees with the one we derive.

    A free oracle on a rounding rule implemented by hand. Prices come
    from the latest stored snapshot, which can predate the 02:00 price
    changes FPL has already applied, so a warning is as likely to be that
    timing as a bug in the parse or in ``selling_price``. Either way it is
    not something the manager can act on, which is why it never fails.

    Parameters
    ----------
    my_team : MyTeam
        The parsed endpoint response, carrying FPL's own selling prices.
    prices : Mapping[int, int]
        Current price by element, in tenths of a million. Elements absent
        from the mapping are skipped.
    """
    for pick in my_team.picks:
        current = prices.get(pick.element)
        if current is None:
            continue
        derived = selling_price(pick.purchase_price, current)
        if derived != pick.selling_price:
            logger.warning(
                "Element %d: FPL sells at %d but purchase %d against "
                "current %d derives %d.",
                pick.element,
                pick.selling_price,
                pick.purchase_price,
                current,
                derived,
            )


def squad_from_team_file(
    team: TeamFile,
    season: str,
    connection: "DuckDBPyConnection | None" = None,
) -> Squad:
    """Adapt a team file into a Squad, resolving any names it declares.

    Parameters
    ----------
    team : TeamFile
        The validated file.
    season : str
        Season to resolve names against, e.g. ``"2026-27"``.
    connection : duckdb.DuckDBPyConnection | None, optional
        An open connection. When None, one is opened per table read.

    Returns
    -------
    Squad
        The carried-in squad.
    """
    return Squad(
        gameweek=team.gameweek,
        free_transfers=team.free_transfers,
        bank=team.bank,
        players=resolve_squad(team.players, season, connection),
    )


#: Where API-sourced squads are recorded. Gitignored with the rest of
#: data/, so these are a local record and a fallback to hand back to
#: ``--team-file``, never an automatic input.
SNAPSHOT_FOLDER = DATA_FOLDER.joinpath("teams")


def save_squad_snapshot(
    squad: Squad, season: str, folder: "Path | None" = None
) -> Path:
    """Record a squad as a team file, one per gameweek.

    Written as soon as the squad is known and before anything validates
    it, because a squad that fails validation is the one most worth
    having on disk.

    Parameters
    ----------
    squad : Squad
        The squad to record.
    season : str
        Season the squad belongs to, e.g. ``"2026-27"``.
    folder : pathlib.Path or None, optional
        Where to write. Defaults to ``SNAPSHOT_FOLDER``.

    Returns
    -------
    pathlib.Path
        The file written.
    """
    folder = SNAPSHOT_FOLDER if folder is None else folder
    folder.mkdir(parents=True, exist_ok=True)
    path = folder.joinpath(f"{season}_gw{squad.gameweek}.json")
    team = TeamFile(
        gameweek=squad.gameweek,
        free_transfers=squad.free_transfers,
        bank=squad.bank,
        players=[
            SquadPlayer(
                element=player.element, purchase_price=player.purchase_price
            )
            for player in squad.players
        ],
    )
    path.write_text(
        json.dumps(team.model_dump(exclude_none=True), indent=2) + "\n"
    )
    return path


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


def _owned(
    player: SquadPlayer, name_to_ids: dict[str, set[int]]
) -> OwnedPlayer:
    """Key one declared player by element, resolving a name if that is all."""
    element = player.element
    if element is None:
        # SquadPlayer guarantees one identifier, so a missing element
        # means a name is present.
        element = next(iter(name_to_ids[player.name or ""]))
    return OwnedPlayer(element=element, purchase_price=player.purchase_price)


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

    Element-keyed players are carried straight through. The optimiser
    already rejects an element it holds no predictions for, which is
    strictly stronger than roster membership, so checking here would only
    give the same condition a second error message.

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
    names = [p.name for p in players if p.name is not None]
    if not names:
        return [_owned(player, {}) for player in players]

    roster = current_roster(season, connection)
    name_to_ids: dict[str, set[int]] = {}
    for name, element in zip(
        roster["name"].to_list(), roster["element"].to_list(), strict=True
    ):
        name_to_ids.setdefault(name, set()).add(element)

    unmatched = sorted(n for n in names if n not in name_to_ids)
    ambiguous = sorted(
        n for n in names if n in name_to_ids and len(name_to_ids[n]) > 1
    )
    if unmatched or ambiguous:
        raise ValueError(
            f"could not resolve names to ids: unmatched={unmatched}, "
            f"ambiguous={ambiguous}"
        )
    return [_owned(player, name_to_ids) for player in players]


class MyTeamSource(Protocol):
    """What ``load_api_squad`` needs of an FPL client.

    FplAPI satisfies this; so does a stub that never opens a socket,
    which is the point.
    """

    def next_gameweek(self) -> int | None:
        """Return the gameweek whose deadline is next, if there is one."""
        ...

    def get_my_team(self, manager_id: str, cookie: str) -> MyTeam:
        """Return the authenticated squad pending the next deadline."""
        ...


def load_api_squad(
    manager_id: str,
    cookie: str,
    *,
    season: str,
    expected_gameweek: int,
    connection: "DuckDBPyConnection | None" = None,
    api: MyTeamSource | None = None,
    prices: Mapping[int, int] | None = None,
    snapshot_folder: "Path | None" = None,
) -> Squad:
    """Fetch the live squad, record it, and hand it to the optimiser.

    The gameweek is checked before the cookie is sent: if FPL and the
    stored forward predictions disagree about which week is next there is
    nothing worth planning, and asking anyway only spends a credential.

    Parameters
    ----------
    manager_id : str
        The manager's FPL entry id.
    cookie : str
        The ``Cookie`` header from a logged-in browser session.
    season : str
        Season the squad belongs to, e.g. ``"2026-27"``.
    expected_gameweek : int
        The gameweek the forward predictions start at.
    connection : duckdb.DuckDBPyConnection | None, optional
        An open connection, used to price the selling-price check.
    api : MyTeamSource | None, optional
        The client to fetch through. Defaults to a fresh FplAPI.
    prices : Mapping[int, int] | None, optional
        Current price by element. Read from the roster when None.
    snapshot_folder : pathlib.Path | None, optional
        Where the squad is recorded. Defaults to ``SNAPSHOT_FOLDER``.

    Returns
    -------
    Squad
        The carried-in squad.

    Raises
    ------
    ValueError
        If FPL's next gameweek is absent or is not ``expected_gameweek``.
    ChipActiveError
        If a wildcard or free hit is active.
    SquadUnavailableError
        If the endpoint cannot be reached or rejects the cookie.
    """
    from fantasy_football.extraction.fpl import FplAPI

    api = FplAPI() if api is None else api
    try:
        next_gameweek = api.next_gameweek()
    except requests.RequestException as error:
        raise SquadUnavailableError(
            f"Could not read the next gameweek from FPL "
            f"({type(error).__name__})."
        ) from None

    if next_gameweek != expected_gameweek:
        raise ValueError(
            f"FPL says gameweek {next_gameweek} is next but the forward "
            f"predictions start at {expected_gameweek}; the predictions "
            "are stale, so re-run before planning."
        )

    try:
        my_team = api.get_my_team(manager_id, cookie)
    except requests.RequestException as error:
        raise SquadUnavailableError(
            f"Could not read manager {manager_id}'s team from FPL "
            f"({type(error).__name__}). A 403 means the cookie has "
            "expired -- paste a fresh one into FPL_COOKIE."
        ) from None

    squad = squad_from_my_team(my_team, next_gameweek)
    save_squad_snapshot(squad, season, folder=snapshot_folder)
    if prices is None:
        roster = current_roster(season, connection)
        prices = dict(
            zip(
                roster["element"].to_list(),
                roster["value"].to_list(),
                strict=True,
            )
        )
    check_selling_prices(my_team, prices)
    return squad
