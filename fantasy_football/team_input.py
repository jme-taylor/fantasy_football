import json
from pathlib import Path

import polars as pl
from pydantic import ConfigDict, TypeAdapter
from pydantic.dataclasses import dataclass

from fantasy_football.constants import RAW_DATA_FOLDER

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


def resolve_names_to_ids(names: list[str], season: str) -> list[int]:
    """Map full player names to FPL element ids for a season.

    The mapping is built from that season's ``merged_gw.csv`` (the ``name``
    and ``element`` columns).

    Parameters
    ----------
    names : list[str]
        Full player names as they appear in the data's ``name`` column.
    season : str
        Season to read, e.g. ``"2025-26"``.

    Returns
    -------
    list[int]
        The element id for each input name, in order.

    Raises
    ------
    ValueError
        If any name has no row, or maps to more than one distinct element.
    """
    merged = pl.read_csv(
        RAW_DATA_FOLDER.joinpath(season, "gws", "merged_gw.csv")
    )
    name_to_ids: dict[str, set[int]] = {}
    for name, element in zip(
        merged["name"].to_list(), merged["element"].to_list(), strict=True
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
