import json
from pathlib import Path

from pydantic import ConfigDict, TypeAdapter
from pydantic.dataclasses import dataclass

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
