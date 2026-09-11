"""Tests for the optimiser's database-backed input loader."""

import logging
from datetime import datetime

import polars as pl
import pytest

from fantasy_football.optimisation.inputs import (
    MissingPositionPredictionsError,
    load_component_breakdown,
    load_optimiser_inputs,
)
from fantasy_football.storage import database
from fantasy_football.storage.database import get_connection
from fantasy_football.storage.tables import (
    BACKFILL_KIND,
    FORWARD_KIND,
    PLAYER_SEASON,
    PLAYER_SNAPSHOT,
    POINTS_COMPONENT,
    POINTS_PREDICTION,
)

SEASON = "2026-27"

# One player per position, so a squad-shaped coverage guard has something
# to find in every position it checks.
ROSTER = (
    (1, "GK", "Arsenal", 45, "Aaron", "Keeper"),
    (2, "DEF", "Arsenal", 50, "Bill", "Back"),
    (3, "MID", "Chelsea", 80, "Carl", "Middle"),
    (4, "FWD", "Chelsea", 95, "Dan", "Forward"),
)


def _seed(
    tmp_path,
    monkeypatch,
    predictions: list[dict],
    roster: tuple = ROSTER,
    season: str = SEASON,
    departed_element: int | None = None,
) -> None:
    """Seed snapshot, identity and prediction rows into a tmp database."""
    captured = datetime(2026, 8, 1, 12, 0)
    snapshot = pl.DataFrame(
        [
            {
                "season": season,
                "captured_at": captured,
                "element": element,
                "value": value,
                "team": team,
                "position": position,
                "chance_of_playing_this_round": 100,
                "status": "u" if element == departed_element else "a",
            }
            for element, position, team, value, _, _ in roster
        ]
    )
    identity = pl.DataFrame(
        [
            {
                "season": season,
                "element": element,
                "player_code": 1000 + element,
                "web_name": second,
                "first_name": first,
                "second_name": second,
                "position": position,
                "team_code": element,
                "birth_date": None,
                "region": None,
                "team_join_date": None,
            }
            for element, position, _, _, first, second in roster
        ],
        schema_overrides={
            "birth_date": pl.Date,
            "region": pl.Int64,
            "team_join_date": pl.Date,
        },
    )
    db_path = tmp_path / "inputs.duckdb"
    monkeypatch.setattr(database, "DATABASE_PATH", db_path)
    connection = get_connection(db_path)
    try:
        PLAYER_SNAPSHOT.upsert_current(connection, snapshot, season)
        PLAYER_SEASON.upsert_current(connection, identity, season)
        if predictions:
            POINTS_PREDICTION.upsert_current(
                connection, pl.DataFrame(predictions), season
            )
    finally:
        connection.close()


def _prediction(
    element: int,
    gw: int,
    points: float,
    *,
    opponent: int = 10,
    position: str = "MID",
    kind: str = FORWARD_KIND,
    season: str = SEASON,
) -> dict:
    """Build one points_prediction row."""
    return {
        "season": season,
        "gw": gw,
        "element": element,
        "opponent": opponent,
        "position": position,
        "predicted_points": points,
        "model_version": "1",
        "prediction_kind": kind,
    }


def _full_cover(gws: list[int], points: float = 2.0) -> list[dict]:
    """Build a forward prediction for every roster player in every gw."""
    return [
        _prediction(element, gw, points, position=position)
        for element, position, *_ in ROSTER
        for gw in gws
    ]


def test_returns_one_row_per_player_and_gameweek(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every roster player appears once in every requested gameweek."""
    _seed(tmp_path, monkeypatch, _full_cover([5, 6]))

    frame = load_optimiser_inputs(SEASON, [5, 6])

    assert frame.height == len(ROSTER) * 2
    assert sorted(frame.columns) == [
        "element",
        "gw",
        "is_departed",
        "name",
        "position",
        "predicted_points",
        "team",
        "value",
    ]
    assert sorted(frame["gw"].unique().to_list()) == [5, 6]


def test_departed_player_is_in_the_universe_scoring_zero(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Predictions made before a player left do not survive his departure."""
    _seed(
        tmp_path,
        monkeypatch,
        _full_cover([5]),
        departed_element=4,
    )

    frame = load_optimiser_inputs(SEASON, [5])

    forward = frame.filter(pl.col("element") == 4)
    assert forward["is_departed"].item() is True
    assert forward["predicted_points"].item() == 0.0
    assert frame.filter(~pl.col("is_departed")).height == len(ROSTER) - 1


def test_carries_roster_name_club_position_and_price(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Identity and price columns come from the roster, not the predictions."""
    _seed(tmp_path, monkeypatch, _full_cover([5]))

    frame = load_optimiser_inputs(SEASON, [5])
    keeper = frame.filter(pl.col("element") == 1).row(0, named=True)

    assert keeper["name"] == "Aaron Keeper"
    assert keeper["position"] == "GK"
    assert keeper["team"] == "Arsenal"
    assert keeper["value"] == 45


def test_double_gameweek_fixtures_are_summed(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two fixtures in one gameweek add up rather than one overwriting."""
    predictions = _full_cover([5]) + [
        _prediction(3, 5, 4.5, opponent=99, position="MID")
    ]
    _seed(tmp_path, monkeypatch, predictions)

    frame = load_optimiser_inputs(SEASON, [5])
    midfielder = frame.filter(pl.col("element") == 3).row(0, named=True)

    assert midfielder["predicted_points"] == pytest.approx(6.5)


def test_roster_player_without_predictions_scores_zero(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A player with no row is kept at zero, not dropped from the universe."""
    predictions = [row for row in _full_cover([5]) if row["element"] != 4] + [
        _prediction(9, 5, 7.0, position="FWD")
    ]
    roster = (*ROSTER, (9, "FWD", "Everton", 60, "Eddie", "Extra"))
    _seed(tmp_path, monkeypatch, predictions, roster=roster)

    frame = load_optimiser_inputs(SEASON, [5])
    forward = frame.filter(pl.col("element") == 4).row(0, named=True)

    assert forward["predicted_points"] == pytest.approx(0.0)
    assert frame.height == len(roster)


def test_players_outside_the_roster_are_excluded(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A prediction for someone with no price cannot enter the problem."""
    predictions = _full_cover([5]) + [_prediction(404, 5, 9.0)]
    _seed(tmp_path, monkeypatch, predictions)

    frame = load_optimiser_inputs(SEASON, [5])

    assert 404 not in frame["element"].to_list()


def test_only_the_requested_gameweeks_are_returned(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gameweeks outside the horizon are filtered out."""
    _seed(tmp_path, monkeypatch, _full_cover([5, 6, 7]))

    frame = load_optimiser_inputs(SEASON, [6])

    assert frame["gw"].unique().to_list() == [6]


def test_backfill_predictions_are_ignored(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only forward-kind rows score; backfill rows describe played weeks."""
    predictions = _full_cover([5]) + [
        _prediction(3, 5, 50.0, opponent=99, kind=BACKFILL_KIND)
    ]
    _seed(tmp_path, monkeypatch, predictions)

    frame = load_optimiser_inputs(SEASON, [5])
    midfielder = frame.filter(pl.col("element") == 3).row(0, named=True)

    assert midfielder["predicted_points"] == pytest.approx(2.0)


def test_other_seasons_are_ignored(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A prediction from another season never reaches the horizon."""
    predictions = _full_cover([5]) + [
        _prediction(3, 5, 50.0, opponent=99, season="2025-26")
    ]
    _seed(tmp_path, monkeypatch, predictions)

    frame = load_optimiser_inputs(SEASON, [5])
    midfielder = frame.filter(pl.col("element") == 3).row(0, named=True)

    assert midfielder["predicted_points"] == pytest.approx(2.0)


def test_position_with_no_predictions_in_a_gameweek_raises(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A whole position missing from a gameweek means a model did not run."""
    predictions = [
        row
        for row in _full_cover([5, 6])
        if not (row["gw"] == 6 and row["position"] == "GK")
    ]
    _seed(tmp_path, monkeypatch, predictions)

    with pytest.raises(MissingPositionPredictionsError) as excinfo:
        load_optimiser_inputs(SEASON, [5, 6])

    assert "GK" in str(excinfo.value)
    assert "6" in str(excinfo.value)


def test_individual_missing_players_are_warned_not_raised(
    tmp_path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """One uncovered player logs coverage; it does not stop the run."""
    predictions = [row for row in _full_cover([5]) if row["element"] != 4] + [
        _prediction(9, 5, 7.0, position="FWD")
    ]
    roster = (*ROSTER, (9, "FWD", "Everton", 60, "Eddie", "Extra"))
    _seed(tmp_path, monkeypatch, predictions, roster=roster)

    with caplog.at_level(
        logging.WARNING, logger="fantasy_football.optimisation.inputs"
    ):
        load_optimiser_inputs(SEASON, [5])

    assert "FWD" in caplog.text


def test_empty_roster_raises(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a roster there are no prices, so nothing can be optimised."""
    _seed(tmp_path, monkeypatch, [], roster=ROSTER, season="2025-26")

    with pytest.raises(ValueError, match="no roster"):
        load_optimiser_inputs(SEASON, [5])


def _component(
    element: int,
    gw: int,
    component: str,
    points: float,
    *,
    opponent: int = 10,
    position: str = "MID",
    kind: str = FORWARD_KIND,
    season: str = SEASON,
) -> dict:
    """Build one points_component row."""
    return {
        "season": season,
        "gw": gw,
        "element": element,
        "opponent": opponent,
        "position": position,
        "prediction_kind": kind,
        "component": component,
        "points": points,
        "model_version": "1",
        "diagnostics": None,
    }


def _seed_components(tmp_path, monkeypatch, components: list[dict]) -> None:
    """Seed points_component rows into a tmp database."""
    db_path = tmp_path / "components.duckdb"
    monkeypatch.setattr(database, "DATABASE_PATH", db_path)
    connection = get_connection(db_path)
    try:
        POINTS_COMPONENT.upsert_current(
            connection, pl.DataFrame(components), SEASON
        )
    finally:
        connection.close()


def test_breakdown_keys_components_by_player_and_gameweek(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each player-gameweek carries a points value per component."""
    _seed_components(
        tmp_path,
        monkeypatch,
        [
            _component(3, 5, "goals", 1.2),
            _component(3, 5, "assists", 0.8),
            _component(3, 6, "goals", 1.5),
        ],
    )

    breakdown = load_component_breakdown(SEASON, [5, 6])

    assert breakdown[(3, 5)].components == {"goals": 1.2, "assists": 0.8}
    assert breakdown[(3, 6)].components == {"goals": 1.5}


def test_breakdown_sums_double_gameweek_legs_and_counts_fixtures(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two fixtures add up, and the fixture count says the sum is a double."""
    _seed_components(
        tmp_path,
        monkeypatch,
        [
            _component(3, 5, "goals", 1.2, opponent=10),
            _component(3, 5, "goals", 0.9, opponent=11),
            _component(4, 5, "goals", 1.0, opponent=10),
        ],
    )

    breakdown = load_component_breakdown(SEASON, [5])

    assert breakdown[(3, 5)].components["goals"] == pytest.approx(2.1)
    assert breakdown[(3, 5)].fixtures == 2
    assert breakdown[(4, 5)].fixtures == 1


def test_breakdown_ignores_backfill_rows(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Backfill components describe played weeks and never enter the report."""
    _seed_components(
        tmp_path,
        monkeypatch,
        [
            _component(3, 5, "goals", 1.2),
            _component(3, 5, "goals", 9.0, opponent=11, kind=BACKFILL_KIND),
        ],
    )

    breakdown = load_component_breakdown(SEASON, [5])

    assert breakdown[(3, 5)].components["goals"] == pytest.approx(1.2)
    assert breakdown[(3, 5)].fixtures == 1


def test_breakdown_ignores_other_seasons_and_gameweeks(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the requested season and horizon reach the report."""
    _seed_components(
        tmp_path,
        monkeypatch,
        [
            _component(3, 5, "goals", 1.2),
            _component(3, 7, "goals", 4.0),
        ],
    )
    connection = get_connection(tmp_path / "components.duckdb")
    try:
        POINTS_COMPONENT.upsert_current(
            connection,
            pl.DataFrame([_component(3, 5, "goals", 9.0, season="2025-26")]),
            "2025-26",
        )
    finally:
        connection.close()

    breakdown = load_component_breakdown(SEASON, [5])

    assert set(breakdown) == {(3, 5)}
    assert breakdown[(3, 5)].components["goals"] == pytest.approx(1.2)


def test_breakdown_omits_players_with_no_component_rows(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unpredicted player is absent rather than present with zeroes."""
    _seed_components(tmp_path, monkeypatch, [_component(3, 5, "goals", 1.2)])

    breakdown = load_component_breakdown(SEASON, [5])

    assert (4, 5) not in breakdown
