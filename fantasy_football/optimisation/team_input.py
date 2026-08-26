import json
import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import requests
from pydantic import BaseModel, ConfigDict, TypeAdapter, model_validator

from fantasy_football.constants import BUDGET, DATA_FOLDER
from fantasy_football.features.roster import current_roster
from fantasy_football.fpl_types import Entry, EntryPicks, EntryTransfer

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
    """Raised when a free hit makes the squad read back a temporary one."""


#: The chip whose squad is temporary: a free hit team reverts at the
#: next deadline, so it is never what is carried in.
FREE_HIT = "freehit"


class OpeningBudgetError(ValueError):
    """Raised when reconstructed opening prices do not total the budget.

    Every manager starts on exactly 100.0m, spent or banked, so this is a
    hard check on the reconstruction rather than a judgement about the
    squad. Failing it means every selling price would be wrong.
    """


class SquadUnavailableError(RuntimeError):
    """Raised when the endpoint cannot be reached or refuses the cookie.

    Distinct from a bad squad: nothing is wrong with the plan, the
    credential or the network is. Callers skip optimisation on this and
    keep everything the run has already produced.
    """


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


class EntrySource(Protocol):
    """What the public squad read needs of an FPL client.

    FplAPI satisfies this; so does a stub that never opens a socket,
    which is the point.
    """

    def next_gameweek(self) -> int | None:
        """Return the gameweek whose deadline is next, if there is one."""
        ...

    def get_entry(self, manager_id: str) -> Entry:
        """Return the manager's opening and latest settled gameweeks."""
        ...

    def get_entry_picks(self, manager_id: str, event: int) -> EntryPicks:
        """Return the settled squad for one gameweek."""
        ...

    def get_entry_transfers(self, manager_id: str) -> list[EntryTransfer]:
        """Return every completed transfer, oldest first."""
        ...


def purchase_prices(
    opening: Mapping[int, int], transfers: Sequence[EntryTransfer]
) -> dict[int, int]:
    """Replay transfers over an opening squad to price what is owned now.

    A player still holding their opening place is priced at the opening
    deadline; anyone transferred in since is priced at what was actually
    paid for them, which FPL publishes.

    Parameters
    ----------
    opening : Mapping[int, int]
        Element to price at the manager's opening deadline.
    transfers : Sequence[EntryTransfer]
        Completed transfers. Replayed in gameweek order.

    Returns
    -------
    dict[int, int]
        Element to purchase price for the squad as it now stands.

    Raises
    ------
    ValueError
        If a transfer sells a player the replay does not hold, which
        means the reconstruction has drifted from reality.
    """
    prices = dict(opening)
    for transfer in sorted(transfers, key=lambda t: t.event):
        if transfer.element_out not in prices:
            raise ValueError(
                f"GW{transfer.event} sells element {transfer.element_out}, "
                "which the reconstructed squad does not hold; the opening "
                "squad or the transfer history is incomplete."
            )
        del prices[transfer.element_out]
        prices[transfer.element_in] = transfer.element_in_cost
    return prices


def check_opening_budget(opening: Mapping[int, int], bank: int) -> None:
    """Assert an opening squad and its bank total the starting budget.

    Parameters
    ----------
    opening : Mapping[int, int]
        Element to price at the manager's opening deadline.
    bank : int
        Money banked at that deadline, in tenths of a million.

    Raises
    ------
    OpeningBudgetError
        If the two do not total ``BUDGET``.
    """
    total = sum(opening.values()) + bank
    if total != BUDGET:
        raise OpeningBudgetError(
            f"Opening squad of {sum(opening.values())} plus bank {bank} "
            f"is {total}, not the {BUDGET} every manager starts with; the "
            "stored opening-gameweek prices do not match what was paid."
        )


def load_public_squad(
    manager_id: str,
    *,
    season: str,
    expected_gameweek: int,
    free_transfers: int,
    prices_at_gameweek: Callable[[int], Mapping[int, int]],
    api: EntrySource | None = None,
    snapshot_folder: "Path | None" = None,
) -> Squad:
    """Rebuild the carried-in squad from FPL's public endpoints.

    No credentials are involved. The squad and bank come from the last
    settled gameweek, and purchase prices from the opening squad's prices
    replayed through every completed transfer. The reconstruction is
    checked against the starting budget before it is used, because a
    wrong purchase price is a wrong selling price in every gameweek of
    the plan.

    Free transfers are the one thing no public endpoint exposes, so they
    are passed in rather than derived.

    Parameters
    ----------
    manager_id : str
        The manager's FPL entry id.
    season : str
        Season the squad belongs to, e.g. ``"2026-27"``.
    expected_gameweek : int
        The gameweek the forward predictions start at.
    free_transfers : int
        Free transfers available at that gameweek.
    prices_at_gameweek : Callable[[int], Mapping[int, int]]
        Given a gameweek, the price of every player in it. Called for
        the manager's opening gameweek, which the entry reports rather
        than the caller knowing it up front.
    api : EntrySource | None, optional
        The client to read through. Defaults to a fresh FplAPI.
    snapshot_folder : pathlib.Path | None, optional
        Where the squad is recorded. Defaults to ``SNAPSHOT_FOLDER``.

    Returns
    -------
    Squad
        The carried-in squad.

    Raises
    ------
    ValueError
        If FPL's next gameweek is not ``expected_gameweek``.
    ChipActiveError
        If the settled squad was a free hit, which reverts.
    OpeningBudgetError
        If the reconstructed opening squad does not total the budget.
    SquadUnavailableError
        If FPL cannot be reached, or the manager has no settled gameweek.
    """
    from fantasy_football.extraction.fpl import FplAPI

    api = FplAPI() if api is None else api
    try:
        next_gameweek = api.next_gameweek()
        entry = api.get_entry(manager_id)
        if entry.current_event is None:
            raise SquadUnavailableError(
                f"Manager {manager_id} has no settled gameweek yet, so "
                "there is no squad to carry in."
            )
        settled = api.get_entry_picks(manager_id, entry.current_event)
        opening = (
            settled
            if entry.current_event == entry.started_event
            else api.get_entry_picks(manager_id, entry.started_event)
        )
        transfers = api.get_entry_transfers(manager_id)
    except requests.RequestException as error:
        raise SquadUnavailableError(
            f"Could not read manager {manager_id}'s team from FPL "
            f"({type(error).__name__})."
        ) from None

    if next_gameweek != expected_gameweek:
        raise ValueError(
            f"FPL says gameweek {next_gameweek} is next but the forward "
            f"predictions start at {expected_gameweek}; the predictions "
            "are stale, so re-run before planning."
        )
    if settled.active_chip == FREE_HIT:
        raise ChipActiveError(
            f"GW{entry.current_event} was a free hit, so the squad it "
            "reports reverts rather than carrying in; plan this one by "
            "hand."
        )

    opening_prices = prices_at_gameweek(entry.started_event)
    opening_squad = {
        element: opening_prices[element]
        for element in opening.elements
        if element in opening_prices
    }
    missing = sorted(set(opening.elements) - set(opening_squad))
    if missing:
        raise OpeningBudgetError(
            f"No stored gameweek {entry.started_event} price for elements "
            f"{missing}, so the opening squad cannot be priced."
        )
    check_opening_budget(opening_squad, opening.bank)

    prices = purchase_prices(opening_squad, transfers)
    unpriced = sorted(set(settled.elements) - set(prices))
    if unpriced:
        raise OpeningBudgetError(
            f"Replaying transfers left elements {unpriced} unpriced; the "
            "reconstructed squad has drifted from the settled one."
        )

    squad = Squad(
        gameweek=expected_gameweek,
        free_transfers=free_transfers,
        bank=settled.bank,
        players=[
            OwnedPlayer(element=element, purchase_price=prices[element])
            for element in settled.elements
        ],
    )
    save_squad_snapshot(squad, season, folder=snapshot_folder)
    return squad
