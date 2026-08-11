"""Shared seeding helpers for the per-position points model tests.

The defender and forwards models are the same machinery pointed at
different columns, so their tests seed the same rows and differ only in
which position they put on element 1 and which form values they assert.
These helpers take the position as an argument rather than being copied
per model.

Everything here is exposed as a fixture returning a callable, so test
modules pick up the helpers by name without importing across the test
tree.
"""

from collections.abc import Callable
from datetime import datetime, timedelta

import duckdb
import numpy as np
import polars as pl
import pytest

from fantasy_football.features.transformation import (
    _FALLBACK_IDENTITY_PREFIX,
)
from fantasy_football.modelling.points import (
    KEY_COLUMNS,
    TARGET,
    PositionPointsPredictor,
)
from fantasy_football.storage.tables import (
    PLAYER_MATCH,
    PLAYER_MATCH_OPTA,
    PLAYER_SEASON,
    PLAYER_WEEK,
    TABLES,
    TEAM_FIXTURE,
)

SEASON = "2025-26"
FIRST_KICKOFF = datetime(2025, 8, 16, 14, 0)
GW1_KICKOFF = FIRST_KICKOFF
GW2_KICKOFF = FIRST_KICKOFF + timedelta(days=7)
GW3_KICKOFF = FIRST_KICKOFF + timedelta(days=14)

# Real FCI_SLUG_TO_FPL entries, reused from tests/unit/features/test_team_form.py
# and test_match_form.py so the club-name join actually resolves.
UNITED, ARSENAL, SPURS, CHELSEA = "Man Utd", "Arsenal", "Spurs", "Chelsea"
SLUGS = {
    UNITED: "manchester-united",
    ARSENAL: "arsenal",
    SPURS: "tottenham-hotspur",
    CHELSEA: "chelsea",
}
# fpl_team_id only maps a team name to an id when some player_match row
# has that team as its opponent, so every club needs at least one
# player_match row naming it as the opposition -- see the extra Spurs and
# Arsenal player_match rows in the seeding helpers, which exist solely to
# mint the Man Utd and Chelsea mappings.
TEAM_IDS = {UNITED: 1, SPURS: 2, ARSENAL: 3, CHELSEA: 4}

# The club each element plays for across every seeding helper here.
CLUBS = {1: UNITED, 2: SPURS, 3: ARSENAL, 4: CHELSEA}


class ConstantPointsModel:
    """Predicts the same number for every row handed to it."""

    def __init__(self, points: float = 4.5) -> None:
        """Store the constant this model returns for every row."""
        self.points = points

    def predict(self, x) -> np.ndarray:
        """Return ``points`` once per input row."""
        return np.full(len(x), self.points)


def append_rows(table, connection, rows: list[dict]) -> None:
    """Append dict rows, filling unstated columns with nulls."""
    table.append(connection, table.conform(pl.DataFrame(rows)))


def opta_row(
    season: str,
    gw: int,
    element: int,
    match_id: str,
    xg: float,
    goals: int,
    conceded: int,
) -> dict:
    """Build a minimal player_match_opta row for one side of a fixture."""
    return {
        "season": season,
        "gw": gw,
        "element": element,
        "match_id": match_id,
        "competition": "prem",
        "minutes_played": 90,
        "xg": xg,
        "goals": goals,
        "team_goals_conceded": conceded,
    }


@pytest.fixture
def connection():
    """Yield an in-memory DuckDB connection with every table created."""
    conn = duckdb.connect(":memory:")
    for table in TABLES:
        conn.execute(table.ddl)
    yield conn
    conn.close()


@pytest.fixture
def seed_model_frame() -> Callable[..., None]:
    """Return a callable seeding the training-frame row set.

    Two prior-gameweek fixtures give Man Utd and Arsenal each their own,
    deliberately different, rolling defensive/attacking record before the
    gw2 fixture between them (the target row, element 1). Man Utd's own
    xg_against/goals_against/clean_sheet in gw1 (vs Spurs) and Arsenal's
    own xg_for/goals_for in gw1 (vs Chelsea) are picked to be
    unambiguously distinct, so a transposed own/opposition join is caught
    rather than coincidentally passing.
    """

    def seed(connection, position: str = "DEF") -> None:
        utd_v_spurs = f"25-26-prem-{SLUGS[UNITED]}-vs-{SLUGS[SPURS]}"
        ars_v_chelsea = f"25-26-prem-{SLUGS[ARSENAL]}-vs-{SLUGS[CHELSEA]}"
        utd_v_arsenal = f"25-26-prem-{SLUGS[UNITED]}-vs-{SLUGS[ARSENAL]}"
        other = "MID" if position != "MID" else "FWD"

        append_rows(
            PLAYER_SEASON,
            connection,
            [{"season": SEASON, "element": 1, "position": position}],
        )
        append_rows(
            PLAYER_WEEK,
            connection,
            [
                {
                    "season": SEASON,
                    "gw": gw,
                    "element": element,
                    "position": position if element == 1 else other,
                    "team": CLUBS[element],
                }
                for gw, element in [
                    (1, 1),
                    (1, 2),
                    (1, 3),
                    (1, 4),
                    (2, 1),
                    (2, 3),
                ]
            ],
        )
        append_rows(
            PLAYER_MATCH,
            connection,
            [
                {
                    "season": SEASON,
                    "gw": gw,
                    "element": element,
                    "opponent": opponent,
                    "is_home": is_home,
                    "minutes": 90,
                    "kickoff_time": kickoff,
                }
                for gw, element, opponent, is_home, kickoff in [
                    (1, 1, TEAM_IDS[SPURS], True, GW1_KICKOFF),
                    (2, 1, TEAM_IDS[ARSENAL], True, GW2_KICKOFF),
                    # These two exist only to mint the Man Utd and
                    # Chelsea team-id mappings in fpl_team_id (see the
                    # TEAM_IDS comment above).
                    (1, 2, TEAM_IDS[UNITED], False, GW1_KICKOFF),
                    (1, 3, TEAM_IDS[CHELSEA], True, GW1_KICKOFF),
                ]
            ],
        )
        append_rows(
            TEAM_FIXTURE,
            connection,
            [
                {
                    "season": SEASON,
                    "gw": gw,
                    "team": team,
                    "is_home": is_home,
                    "opposition": opposition,
                    "kickoff_time": kickoff,
                }
                for gw, team, is_home, opposition, kickoff in [
                    (1, UNITED, True, SPURS, GW1_KICKOFF),
                    (1, SPURS, False, UNITED, GW1_KICKOFF),
                    (1, ARSENAL, True, CHELSEA, GW1_KICKOFF),
                    (1, CHELSEA, False, ARSENAL, GW1_KICKOFF),
                    (2, UNITED, True, ARSENAL, GW2_KICKOFF),
                    (2, ARSENAL, False, UNITED, GW2_KICKOFF),
                ]
            ],
        )
        append_rows(
            PLAYER_MATCH_OPTA,
            connection,
            [
                # gw1: Man Utd 2-0 Spurs. Man Utd keep a clean sheet,
                # facing 0.3 xG against.
                opta_row(SEASON, 1, 1, utd_v_spurs, 2.0, 2, 0),
                opta_row(SEASON, 1, 2, utd_v_spurs, 0.3, 0, 2),
                # gw1: Arsenal 1-1 Chelsea. Arsenal's own attacking
                # record (1.0 xG for) is what must land in the
                # *opposition* columns of the gw2 fixture below when the
                # subject is a defender, and in the own columns when the
                # subject is an Arsenal forward.
                opta_row(SEASON, 1, 3, ars_v_chelsea, 1.0, 1, 1),
                opta_row(SEASON, 1, 4, ars_v_chelsea, 0.5, 1, 1),
                # gw2: Man Utd vs Arsenal, the target row. Values here
                # are irrelevant to the rolling columns -- the window
                # excludes the current match by construction.
                opta_row(SEASON, 2, 1, utd_v_arsenal, 1.5, 1, 1),
                opta_row(SEASON, 2, 3, utd_v_arsenal, 0.8, 1, 1),
            ],
        )

    return seed


@pytest.fixture
def seed_forward_history() -> Callable[..., None]:
    """Return a callable seeding two played gws plus an unplayed gw3.

    Every club's gw1 and gw2 figures are deliberately different, so the
    inclusive rolling value (the mean of both matches) is distinct from
    the exclusive one (gw1 alone). Derivations, all per club:

    * element 1 (a Man Utd player) records 0.1 xG in gw1 and 0.9 in gw2
      over 90 minutes each: inclusive 0.5 per 90, exclusive 0.1.
    * Man Utd concede 1.0 xG and 2 goals in gw1, then 0.2 xG and none in
      gw2: inclusive 0.6 / 1.0 / 0.5 clean-sheet rate, exclusive
      1.0 / 2.0 / 0.0. Man Utd make 0.1 xG and no goals in gw1, then 0.9
      and 1 in gw2: inclusive 0.5 / 0.5 xG and goals for.
    * Arsenal make 1.0 xG and 1 goal in gw1, then 3.0 and 3 in gw2:
      inclusive 2.0 / 2.0, exclusive 1.0 / 1.0. Arsenal concede 0.5 xG
      and 1 goal in gw1, then 0.4 and none in gw2: inclusive
      0.45 / 0.5 against.
    """

    def seed(connection, position: str = "DEF") -> None:
        utd_v_spurs = f"25-26-prem-{SLUGS[UNITED]}-vs-{SLUGS[SPURS]}"
        ars_v_chelsea = f"25-26-prem-{SLUGS[ARSENAL]}-vs-{SLUGS[CHELSEA]}"
        utd_v_chelsea = f"25-26-prem-{SLUGS[UNITED]}-vs-{SLUGS[CHELSEA]}"
        ars_v_spurs = f"25-26-prem-{SLUGS[ARSENAL]}-vs-{SLUGS[SPURS]}"
        other = "MID" if position != "MID" else "FWD"

        append_rows(
            PLAYER_SEASON,
            connection,
            [
                {
                    "season": SEASON,
                    "element": element,
                    "position": position if element == 1 else other,
                }
                for element in CLUBS
            ],
        )
        append_rows(
            PLAYER_WEEK,
            connection,
            [
                {
                    "season": SEASON,
                    "gw": gw,
                    "element": element,
                    "position": position if element == 1 else other,
                    "team": team,
                }
                for gw in (1, 2)
                for element, team in CLUBS.items()
            ],
        )
        append_rows(
            PLAYER_MATCH,
            connection,
            [
                {
                    "season": SEASON,
                    "gw": gw,
                    "element": element,
                    "opponent": opponent,
                    "is_home": is_home,
                    "minutes": 90,
                    "kickoff_time": kickoff,
                }
                for gw, element, opponent, is_home, kickoff in [
                    (1, 1, TEAM_IDS[SPURS], True, GW1_KICKOFF),
                    (1, 2, TEAM_IDS[UNITED], False, GW1_KICKOFF),
                    (1, 3, TEAM_IDS[CHELSEA], True, GW1_KICKOFF),
                    (1, 4, TEAM_IDS[ARSENAL], False, GW1_KICKOFF),
                    (2, 1, TEAM_IDS[CHELSEA], True, GW2_KICKOFF),
                    (2, 4, TEAM_IDS[UNITED], False, GW2_KICKOFF),
                    (2, 3, TEAM_IDS[SPURS], True, GW2_KICKOFF),
                    (2, 2, TEAM_IDS[ARSENAL], False, GW2_KICKOFF),
                ]
            ],
        )
        append_rows(
            TEAM_FIXTURE,
            connection,
            [
                {
                    "season": SEASON,
                    "gw": gw,
                    "team": team,
                    "is_home": is_home,
                    "opposition": opposition,
                    "kickoff_time": kickoff,
                }
                for gw, team, is_home, opposition, kickoff in [
                    (1, UNITED, True, SPURS, GW1_KICKOFF),
                    (1, SPURS, False, UNITED, GW1_KICKOFF),
                    (1, ARSENAL, True, CHELSEA, GW1_KICKOFF),
                    (1, CHELSEA, False, ARSENAL, GW1_KICKOFF),
                    (2, UNITED, True, CHELSEA, GW2_KICKOFF),
                    (2, CHELSEA, False, UNITED, GW2_KICKOFF),
                    (2, ARSENAL, True, SPURS, GW2_KICKOFF),
                    (2, SPURS, False, ARSENAL, GW2_KICKOFF),
                    (3, UNITED, True, ARSENAL, GW3_KICKOFF),
                    (3, ARSENAL, False, UNITED, GW3_KICKOFF),
                ]
            ],
        )
        append_rows(
            PLAYER_MATCH_OPTA,
            connection,
            [
                opta_row(SEASON, 1, 1, utd_v_spurs, 0.1, 0, 2),
                opta_row(SEASON, 1, 2, utd_v_spurs, 1.0, 2, 0),
                opta_row(SEASON, 1, 3, ars_v_chelsea, 1.0, 1, 1),
                opta_row(SEASON, 1, 4, ars_v_chelsea, 0.5, 1, 1),
                opta_row(SEASON, 2, 1, utd_v_chelsea, 0.9, 1, 0),
                opta_row(SEASON, 2, 4, utd_v_chelsea, 0.2, 0, 1),
                opta_row(SEASON, 2, 3, ars_v_spurs, 3.0, 3, 0),
                opta_row(SEASON, 2, 2, ars_v_spurs, 0.4, 0, 3),
            ],
        )

    return seed


@pytest.fixture
def forward_fixture() -> Callable[..., dict]:
    """Return a callable building one ``build_forward_fixtures`` row."""

    def build(
        gw: int = 3,
        element: int = 1,
        opponent: int = TEAM_IDS[ARSENAL],
        kickoff: datetime = GW3_KICKOFF,
        team: str = UNITED,
        season: str = SEASON,
        position: str = "DEF",
    ) -> dict:
        return {
            "season": season,
            "gw": gw,
            "element": element,
            "opponent": opponent,
            "kickoff_time": kickoff,
            "minutes": None,
            "position": position,
            "team": team,
            "value": 50,
            "chance_of_playing_this_round": 100,
        }

    return build


@pytest.fixture
def forward_frame() -> Callable[[list[dict]], pl.DataFrame]:
    """Return a callable typing fixture rows like the real thing."""

    def build(rows: list[dict]) -> pl.DataFrame:
        return pl.DataFrame(
            rows,
            schema={
                "season": pl.Utf8,
                "gw": pl.Int64,
                "element": pl.Int64,
                "opponent": pl.Int64,
                "kickoff_time": pl.Datetime("us"),
                "minutes": pl.Int64,
                "position": pl.Utf8,
                "team": pl.Utf8,
                "value": pl.Int64,
                "chance_of_playing_this_round": pl.Int64,
            },
        )

    return build


@pytest.fixture
def stub_view() -> Callable[..., None]:
    """Return a callable replacing a feature view with a hand-built frame."""

    def stub(connection, name: str, frame: pl.DataFrame) -> None:
        connection.register(f"{name}_stub", frame)
        connection.execute(
            f"CREATE OR REPLACE TEMP VIEW {name} AS SELECT * FROM {name}_stub"
        )

    return stub


@pytest.fixture
def stub_player_form(stub_view) -> Callable[..., pl.DataFrame]:
    """Return a callable stubbing ``player_match_form_inclusive``.

    Absent columns are null-filled. ``rolling_identity`` defaults to the
    view's own no-player-code fallback for the row's
    ``(season, element)``, so a stub that states no identity behaves
    exactly like a player with no ``player_season`` row.
    """

    def stub(
        connection, rows: dict, predictor: PositionPointsPredictor
    ) -> pl.DataFrame:
        frame = pl.DataFrame(rows).with_columns(
            [
                pl.lit(None, dtype=pl.Float64).alias(name)
                for name in predictor.PLAYER_FORM_COLUMNS
                if name not in rows
            ]
        )
        if "rolling_identity" not in rows:
            frame = frame.with_columns(
                rolling_identity=pl.lit(_FALLBACK_IDENTITY_PREFIX)
                + pl.col("season")
                + pl.lit("_")
                + pl.col("element").cast(pl.Utf8)
            )
        stub_view(connection, "player_match_form_inclusive", frame)
        return frame

    return stub


@pytest.fixture
def stub_team_form(stub_view) -> Callable[..., pl.DataFrame]:
    """Return a callable stubbing ``team_match_form_inclusive``."""

    def stub(
        connection, rows: dict, predictor: PositionPointsPredictor
    ) -> pl.DataFrame:
        # A position may read the same measure for both clubs -- the
        # midfielder reads all five -- and the view holds one copy.
        columns = dict.fromkeys(
            predictor.OWN_TEAM_COLUMNS + predictor.OPPOSITION_COLUMNS
        )
        frame = pl.DataFrame(rows).with_columns(
            [
                pl.lit(None, dtype=pl.Float64).alias(name)
                for name in columns
                if name not in rows
            ]
        )
        stub_view(connection, "team_match_form_inclusive", frame)
        return frame

    return stub


@pytest.fixture
def stub_two_season_form(stub_team_form, stub_view) -> Callable[..., None]:
    """Return a callable stubbing form and id views across two seasons.

    Arsenal and Chelsea both appear in 2024-25 and 2025-26, and the FPL
    ids are reshuffled between the two: id 1 is Chelsea in 2024-25 and
    Arsenal in 2025-26. Any join that resolves the opposition by id
    without the season -- or that resolves it by id at all -- can pick
    up the wrong club's form. Chelsea played more recently, so such a
    join as-of matches its 3.0 rather than Arsenal's 0.5.
    """

    def stub(connection, predictor: PositionPointsPredictor) -> None:
        stub_team_form(
            connection,
            {
                "season": ["2024-25", "2024-25", SEASON, SEASON],
                "team": [ARSENAL, CHELSEA, ARSENAL, CHELSEA],
                "kickoff_time": [
                    GW1_KICKOFF - timedelta(days=365),
                    GW1_KICKOFF - timedelta(days=364),
                    GW1_KICKOFF,
                    GW2_KICKOFF,
                ],
                **{
                    column: [9.0, 9.5, 0.5, 3.0]
                    for column in predictor.OPPOSITION_COLUMNS
                },
            },
            predictor,
        )
        stub_view(
            connection,
            "fpl_team_id",
            pl.DataFrame(
                {
                    "season": ["2024-25", "2024-25", SEASON, SEASON],
                    "team_id": [1, 4, 1, 4],
                    "team": [CHELSEA, ARSENAL, ARSENAL, CHELSEA],
                }
            ),
        )

    return stub


@pytest.fixture
def synthetic_frame() -> Callable[..., pl.DataFrame]:
    """Return a callable building a model frame spanning several gameweeks."""

    def build(
        predictor: PositionPointsPredictor,
        n_gws: int = 14,
        n_players: int = 12,
        season: str = "2026-27",
    ) -> pl.DataFrame:
        rows = [
            {
                "season": season,
                "gw": gw,
                "element": element,
                "opponent": (element % 5) + 1,
                TARGET: float(element % 7),
                **{name: float(element + gw) for name in predictor.FEATURES},
            }
            for gw in range(1, n_gws + 1)
            for element in range(1, n_players + 1)
        ]
        return pl.DataFrame(rows).select(
            KEY_COLUMNS + [TARGET] + predictor.FEATURES
        )

    return build
