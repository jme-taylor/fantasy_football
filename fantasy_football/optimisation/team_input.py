import json
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import ConfigDict, TypeAdapter
from pydantic.dataclasses import dataclass

from fantasy_football.features.roster import current_roster

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

config = ConfigDict(extra="forbid")


@dataclass(config=config, frozen=True)
class TeamFile:
    """A human-authored FPL team for a single gameweek.

    Attributes
    ----------
    gameweek : int
        The gameweek the team is for; the optimiser's start_gw.
    free_transfers : int
        Free transfers available at that gameweek.
    players : list[str]
        Full player names exactly as they appear in the data's ``name`` column.
    bank : int
        Money in the bank in tenths of a million. Defaults to 0.
    """

    gameweek: int
    free_transfers: int
    players: list[str]
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


def resolve_names_to_ids(
    names: list[str],
    season: str,
    connection: "DuckDBPyConnection | None" = None,
) -> list[int]:
    """Map full player names to FPL element ids for a season.

    Names resolve against the current roster -- the same set of players the
    optimiser can buy -- so a name that resolves is always a player the plan
    can actually hold. Resolving against played gameweeks instead would
    accept a name the optimiser has no price for, and reject a summer signing
    who has not played yet.

    Parameters
    ----------
    names : list[str]
        Full player names as they appear in the roster's ``name`` column.
    season : str
        Season to read, e.g. ``"2026-27"``.
    connection : duckdb.DuckDBPyConnection | None, optional
        An open connection. When None, one is opened per table read.

    Returns
    -------
    list[int]
        The element id for each input name, in order.

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

    unmatched = sorted(n for n in names if n not in name_to_ids)
    ambiguous = sorted(
        n for n in names if n in name_to_ids and len(name_to_ids[n]) > 1
    )
    if unmatched or ambiguous:
        raise ValueError(
            f"could not resolve names to ids: unmatched={unmatched}, "
            f"ambiguous={ambiguous}"
        )
    return [next(iter(name_to_ids[n])) for n in names]
