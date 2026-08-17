"""Composition of component rows into a single points prediction."""

import logging

import polars as pl
import pytest

from fantasy_football.modelling.components import (
    POSITION_COMPONENTS,
    Component,
    compose,
    compose_points,
    composite_version,
    write_appearance_components,
)
from fantasy_football.storage.tables import (
    BACKFILL_KIND,
    FORWARD_KIND,
    MINUTES_PREDICTION,
    PLAYER_SEASON,
    POINTS_COMPONENT,
    POINTS_PREDICTION,
)
from tests.unit.modelling.conftest import append_rows

SEASON = "2025-26"


def component_row(
    component: Component,
    points: float,
    *,
    element: int = 1,
    gw: int = 1,
    opponent: int = 2,
    position: str = "MID",
    kind: str = BACKFILL_KIND,
    version: str = "3",
    diagnostics: str | None = None,
) -> dict[str, object]:
    """Build one component row as a dict."""
    return {
        "season": SEASON,
        "gw": gw,
        "element": element,
        "opponent": opponent,
        "position": position,
        "prediction_kind": kind,
        "component": str(component),
        "points": points,
        "model_version": version,
        "diagnostics": diagnostics,
    }


def frame(*rows: dict[str, object]) -> pl.DataFrame:
    """Shape component dicts into the stored component schema."""
    if not rows:
        return pl.DataFrame(schema=POINTS_COMPONENT.schema)
    return POINTS_COMPONENT.coerce(pl.DataFrame(list(rows)))


# --- Version stamping ------------------------------------------------


def test_single_component_keeps_the_bare_version() -> None:
    """A lone component keeps its bare version string."""
    # A monolithic position must produce the exact string it produced
    # before decomposition existed, or the backfill rebuild check
    # compares a composite against a bare production version and
    # rebuilds every run.
    assert composite_version({Component.TOTAL: "7"}) == "7"


def test_several_components_are_named_and_sorted() -> None:
    """Several components fold into a named, sorted string."""
    version = composite_version(
        {Component.RESIDUAL: "2", Component.DEFCON: "5"}
    )
    assert version == "defcon=5|residual=2"


def test_ordering_of_the_input_does_not_change_the_output() -> None:
    """The version string does not depend on mapping order."""
    first = composite_version({Component.DEFCON: "5", Component.RESIDUAL: "2"})
    second = composite_version(
        {Component.RESIDUAL: "2", Component.DEFCON: "5"}
    )
    assert first == second


def test_no_components_is_an_error() -> None:
    """Versioning a prediction with no components is refused."""
    with pytest.raises(ValueError, match="no components"):
        composite_version({})


# --- Composition -----------------------------------------------------


def test_single_component_passes_the_points_through() -> None:
    """One component composes to exactly its own points."""
    composed = compose(frame(component_row(Component.TOTAL, 4.5)))
    assert composed["predicted_points"].to_list() == [4.5]
    assert composed["model_version"].to_list() == ["3"]


def test_output_matches_the_points_prediction_schema() -> None:
    """Composed rows are shaped for points_prediction."""
    composed = compose(frame(component_row(Component.TOTAL, 4.5)))
    assert composed.columns == POINTS_PREDICTION.columns


def test_components_sum_within_a_fixture_leg() -> None:
    """Components of one fixture leg sum into one row."""
    composed = compose(
        frame(
            component_row(Component.APPEARANCE, 1.8, version="1"),
            component_row(Component.DEFCON, 0.9, version="2"),
            component_row(Component.RESIDUAL, 2.3, version="3"),
        )
    )
    assert composed.height == 1
    assert composed["predicted_points"].to_list() == [pytest.approx(5.0)]
    assert composed["model_version"].to_list() == [
        "appearance=1|defcon=2|residual=3"
    ]


def test_fixture_legs_stay_separate() -> None:
    """Two legs of a double gameweek stay two rows."""
    # A double gameweek is two legs that must not collapse into one
    # row, or the optimiser's per-gameweek sum halves.
    composed = compose(
        frame(
            component_row(Component.TOTAL, 4.0, opponent=2),
            component_row(Component.TOTAL, 3.0, opponent=5),
        )
    )
    assert sorted(composed["predicted_points"].to_list()) == [3.0, 4.0]


def test_prediction_kinds_stay_separate() -> None:
    """Backfill and forward rows do not collapse together."""
    composed = compose(
        frame(
            component_row(Component.TOTAL, 4.0, kind=BACKFILL_KIND),
            component_row(Component.TOTAL, 3.0, kind=FORWARD_KIND),
        )
    )
    assert composed.height == 2


def test_positions_stay_separate() -> None:
    """Each position keeps its own composed row."""
    composed = compose(
        frame(
            component_row(Component.TOTAL, 4.0, position="MID"),
            component_row(Component.TOTAL, 3.0, position="FWD", element=2),
        )
    )
    assert set(composed["position"].to_list()) == {"MID", "FWD"}


def test_empty_input_yields_an_empty_frame_of_the_right_shape() -> None:
    """No components gives an empty frame, not an error."""
    composed = compose(frame())
    assert composed.is_empty()
    assert composed.columns == POINTS_PREDICTION.columns


def test_unknown_component_is_an_error() -> None:
    """A component name that is not declared is refused."""
    rows = frame(component_row(Component.TOTAL, 4.0)).with_columns(
        component=pl.lit("bonus")
    )
    with pytest.raises(ValueError, match="bonus"):
        compose(rows)


def test_a_leg_missing_a_component_is_dropped_not_partly_summed() -> None:
    """Half a decomposition yields no row, rather than a low one.

    Components genuinely cover different rows -- there is no CBIT to rate
    before 2024-25, and no head rates a five-minute cameo -- so an
    incomplete leg is expected rather than a fault. Summing the subset
    would under-predict and the optimiser would act on it; dropping the
    leg leaves nothing, which it reads as zero and passes over.
    """
    rows = frame(
        component_row(Component.APPEARANCE, 1.8, position="DEF"),
        component_row(Component.DEFCON, 0.9, position="DEF"),
    )

    composed = compose(rows, expected=POSITION_COMPONENTS)

    assert composed.is_empty()


def test_the_dropped_legs_name_the_component_they_are_missing(
    caplog,
) -> None:
    """Otherwise a whole position vanishing is a mystery, not a message.

    Declaring a new component drops every stored leg of that position
    until its backfill has run, and the optimiser reads the composed
    table -- so the cause has to be in the log.
    """
    rows = frame(
        component_row(Component.APPEARANCE, 1.8, position="DEF"),
        component_row(Component.DEFCON, 0.9, position="DEF"),
        component_row(Component.GOALS, 0.4, position="DEF"),
        component_row(Component.RESIDUAL, 2.3, position="DEF"),
    )

    with caplog.at_level(logging.WARNING):
        compose(rows, expected=POSITION_COMPONENTS)

    assert str(Component.ASSISTS) in caplog.text


def test_complete_legs_survive_when_a_neighbour_is_dropped() -> None:
    """One incomplete leg does not take the whole frame down with it."""
    rows = frame(
        component_row(Component.APPEARANCE, 1.8, position="DEF"),
        component_row(Component.DEFCON, 0.9, position="DEF"),
        component_row(Component.GOALS, 0.4, position="DEF"),
        component_row(Component.ASSISTS, 0.3, position="DEF"),
        component_row(Component.RESIDUAL, 2.3, position="DEF"),
        component_row(Component.APPEARANCE, 2.0, element=2, position="DEF"),
        component_row(Component.DEFCON, 0.1, element=2, position="DEF"),
    )

    composed = compose(rows, expected=POSITION_COMPONENTS)

    assert composed["element"].to_list() == [1]
    assert composed["predicted_points"].to_list() == [pytest.approx(5.7)]


def test_duplicate_component_for_one_leg_is_an_error() -> None:
    """One leg carrying a component twice is refused."""
    rows = frame(
        component_row(Component.TOTAL, 4.0, version="3"),
        component_row(Component.TOTAL, 1.0, version="4"),
    )
    with pytest.raises(ValueError, match="more than once"):
        compose(rows)


def test_components_sum_to_the_stored_total() -> None:
    """Every composed total equals the sum of its components."""
    rows = frame(
        component_row(Component.APPEARANCE, 1.8, version="1"),
        component_row(Component.DEFCON, 0.9, version="2"),
        component_row(Component.RESIDUAL, 2.3, version="3"),
        component_row(Component.APPEARANCE, 2.0, element=2, version="1"),
        component_row(Component.DEFCON, 0.1, element=2, version="2"),
        component_row(Component.RESIDUAL, 4.4, element=2, version="3"),
    )
    composed = compose(rows)
    expected = (
        rows.group_by("season", "gw", "element", "opponent")
        .agg(pl.col("points").sum())
        .sort("element")
    )
    got = composed.sort("element")
    assert got["predicted_points"].to_list() == pytest.approx(
        expected["points"].to_list()
    )


# --- The composition step --------------------------------------------


def test_writes_the_composed_total(connection) -> None:
    """compose_points stores the summed prediction."""
    POINTS_COMPONENT.append(
        connection, frame(component_row(Component.TOTAL, 4.5))
    )

    compose_points(connection)

    stored = POINTS_PREDICTION.load(connection)
    assert stored["predicted_points"].to_list() == [4.5]
    assert stored["model_version"].to_list() == ["3"]


def test_nothing_stored_leaves_the_table_alone(connection) -> None:
    """No components stored means nothing is written."""
    compose_points(connection)

    assert POINTS_PREDICTION.load(connection).is_empty()


def test_recomposing_replaces_rather_than_duplicates(connection) -> None:
    """Composing twice replaces the partition, not appends."""
    POINTS_COMPONENT.append(
        connection, frame(component_row(Component.TOTAL, 4.5))
    )
    compose_points(connection)
    compose_points(connection)

    assert POINTS_PREDICTION.load(connection).height == 1


def test_frozen_forward_rows_survive_recomposition(connection) -> None:
    """A frozen forward row survives a later recomposition."""
    # The freeze lives in the component table: a forward row written
    # for an already-played gameweek stays there, so recomposing the
    # partition reproduces it rather than dropping it.
    POINTS_COMPONENT.append(
        connection,
        frame(
            component_row(
                Component.TOTAL, 9.9, gw=2, kind=FORWARD_KIND, version="1"
            ),
            component_row(
                Component.TOTAL, 4.5, gw=3, kind=FORWARD_KIND, version="9"
            ),
        ),
    )

    compose_points(connection)

    stored = POINTS_PREDICTION.load(connection).sort("gw")
    assert stored["predicted_points"].to_list() == [9.9, 4.5]
    assert stored["model_version"].to_list() == ["1", "9"]


def test_kinds_land_in_separate_partitions(connection) -> None:
    """Each prediction kind is replaced independently."""
    POINTS_COMPONENT.append(
        connection,
        frame(
            component_row(Component.TOTAL, 4.5, kind=BACKFILL_KIND),
            component_row(Component.TOTAL, 6.5, kind=FORWARD_KIND),
        ),
    )

    compose_points(connection)

    stored = POINTS_PREDICTION.load(connection)
    assert sorted(stored["prediction_kind"].to_list()) == sorted(
        [BACKFILL_KIND, FORWARD_KIND]
    )


def test_incomplete_decomposition_writes_nothing(connection, mocker) -> None:
    """A half-written decomposition stores no prediction at all."""
    mocker.patch.dict(
        "fantasy_football.modelling.components.POSITION_COMPONENTS",
        {"DEF": (Component.APPEARANCE, Component.RESIDUAL)},
    )
    POINTS_COMPONENT.append(
        connection,
        frame(component_row(Component.APPEARANCE, 1.8, position="DEF")),
    )

    compose_points(connection)

    assert POINTS_PREDICTION.load(connection).is_empty()


def test_a_component_a_position_no_longer_declares_is_refused() -> None:
    """Rows left behind by an earlier pipeline shape cannot be summed in.

    A position decomposed after its monolithic rows were written would
    otherwise sum the old total on top of the components that replaced
    it, doubling the prediction with nothing failing.
    """
    rows = frame(
        component_row(Component.TOTAL, 4.0, position="DEF"),
        component_row(Component.APPEARANCE, 1.8, position="DEF"),
        component_row(Component.DEFCON, 0.9, position="DEF"),
        component_row(Component.RESIDUAL, 2.3, position="DEF"),
    )
    with pytest.raises(ValueError, match="no longer declares"):
        compose(rows, expected=POSITION_COMPONENTS)


def _seed_minutes(connection) -> None:
    """Seed one DEF minutes prediction and its player-season row."""
    append_rows(
        PLAYER_SEASON,
        connection,
        [{"season": SEASON, "element": 1, "position": "DEF"}],
    )
    append_rows(
        MINUTES_PREDICTION,
        connection,
        [
            {
                "season": SEASON,
                "gw": 1,
                "element": 1,
                "opponent": 2,
                "p_zero": 0.1,
                "p_partial": 0.3,
                "p_sixty_plus": 0.6,
                "expected_minutes": 60.0,
                "model_version": "5",
                "prediction_kind": BACKFILL_KIND,
            }
        ],
    )


def test_appearance_points_come_from_the_minutes_model(connection) -> None:
    """Appearance rows are written where the other components landed."""
    _seed_minutes(connection)
    POINTS_COMPONENT.append(
        connection,
        frame(
            component_row(Component.DEFCON, 0.9, position="DEF"),
            component_row(Component.RESIDUAL, 2.3, position="DEF"),
        ),
    )

    write_appearance_components(connection, BACKFILL_KIND)

    stored = POINTS_COMPONENT.load(connection).filter(
        pl.col("component") == str(Component.APPEARANCE)
    )
    assert stored["points"].to_list() == pytest.approx([0.3 + 2 * 0.6])
    # No model of its own, so the version is the minutes model's.
    assert stored["model_version"].to_list() == ["5"]


def test_appearance_is_not_written_where_no_component_landed(
    connection,
) -> None:
    """Appearance covers the other components' rows, not the minutes'.

    The models are narrower than the minutes forecast -- no CBIT before
    2024-25, no head rating a five-minute cameo. Writing appearance for
    every scored minute would emit points for legs that can never be
    composed, and every one of them would then be dropped, leaving the
    stored appearance rows describing predictions that do not exist.
    """
    _seed_minutes(connection)

    write_appearance_components(connection, BACKFILL_KIND)

    assert POINTS_COMPONENT.load(connection).is_empty()


def test_appearance_leaves_frozen_forward_legs_alone(connection) -> None:
    """A frozen forward leg keeps the appearance points it was frozen with.

    Reading the stored components inherits the freeze: the components
    below the first unplayed gameweek were not rewritten, so their keys
    are the only ones appearance is regenerated for.
    """
    _seed_minutes(connection)
    POINTS_COMPONENT.append(
        connection,
        frame(
            component_row(
                Component.DEFCON, 0.9, position="DEF", kind=FORWARD_KIND
            ),
            component_row(
                Component.RESIDUAL, 2.3, position="DEF", kind=FORWARD_KIND
            ),
        ),
    )

    write_appearance_components(connection, BACKFILL_KIND)

    # The only components stored are forward ones, so the backfill pass
    # writes nothing rather than reaching across the kinds.
    assert (
        POINTS_COMPONENT.load(connection)
        .filter(pl.col("component") == str(Component.APPEARANCE))
        .is_empty()
    )


def test_appearance_skips_positions_that_are_not_decomposed(
    connection,
) -> None:
    """A monolithic position's total already includes appearance points."""
    append_rows(
        PLAYER_SEASON,
        connection,
        [{"season": SEASON, "element": 1, "position": "MID"}],
    )
    POINTS_COMPONENT.append(
        connection, frame(component_row(Component.TOTAL, 4.0, position="MID"))
    )
    append_rows(
        MINUTES_PREDICTION,
        connection,
        [
            {
                "season": SEASON,
                "gw": 1,
                "element": 1,
                "opponent": 2,
                "p_zero": 0.1,
                "p_partial": 0.3,
                "p_sixty_plus": 0.6,
                "expected_minutes": 60.0,
                "model_version": "5",
                "prediction_kind": BACKFILL_KIND,
            }
        ],
    )

    write_appearance_components(connection, BACKFILL_KIND)

    assert (
        POINTS_COMPONENT.load(connection)
        .filter(pl.col("component") == str(Component.APPEARANCE))
        .is_empty()
    )
