"""The pooled yellow-cards rate head.

Two of these matter more than the rest, and both cover silent failures.

The head keeps rows the goals and assists heads throw away -- no-form
rows and cameos -- and that is deliberate rather than an oversight: card
history reaches back six seasons further than any FCI column, and the
filters those heads use would spend eight of ten training seasons. A
filter copied across from a sibling head would look right and quietly
undo the reason this one trains wide.

The other is the red card. The component pays for yellows and the
residual deducts yellows, so reds stay in the residual and are paid for
exactly once. Deducting them here without giving them a component would
pay for them twice, and nothing downstream would complain.
"""

import polars as pl
import pytest

from fantasy_football.modelling.components import (
    YELLOW_CARD_POINTS,
    Component,
)
from fantasy_football.modelling.folds import ExpandingGameweekFoldStrategy
from fantasy_football.modelling.points import position_dummy_names
from fantasy_football.modelling.yellow_cards import (
    SCORING_MINUTES_FLOOR,
    TEST_SEASONS,
    TRAINING_POSITIONS,
    TRAINING_SEASONS,
    YELLOW_CARDS_SPEC,
    YellowCardsRatePredictor,
)
from fantasy_football.storage.tables import (
    PLAYER_MATCH,
    PLAYER_MATCH_FPL,
    PLAYER_MATCH_OPTA,
    PLAYER_SEASON,
    PLAYER_WEEK,
    TEAM_FIXTURE,
)
from tests.unit.modelling.conftest import (
    ARSENAL,
    GW1_KICKOFF,
    SEASON,
    SLUGS,
    TEAM_IDS,
    UNITED,
    append_rows,
)

# Two gameweeks: the first builds the form window, the second is the row
# under test.
GWS = (1, 2)


def _predictor(cls, spec, connection):
    """Build a predictor against the test database."""
    return cls(
        experiment_name="test",
        params={},
        model_spec=spec,
        connection=connection,
        fold_strategy=ExpandingGameweekFoldStrategy(),
    )


def _seed_player(
    connection,
    *,
    element: int = 1,
    position: str = "DEF",
    minutes: int = 90,
    yellow_cards: tuple[int, int] = (0, 0),
    red_cards: tuple[int, int] = (0, 0),
    total_points: int = 6,
    with_opta: bool = True,
) -> None:
    """Seed one player's two-gameweek run for the given position.

    ``with_opta`` off leaves the FCI rows absent, which is what every
    season before 2024-25 looks like and what ``has_no_form`` reports.
    """
    match_id = f"25-26-prem-{SLUGS[UNITED]}-vs-{SLUGS[ARSENAL]}"
    append_rows(
        PLAYER_SEASON,
        connection,
        [{"season": SEASON, "element": element, "position": position}],
    )
    for gw in GWS:
        kickoff = GW1_KICKOFF.replace(day=GW1_KICKOFF.day + 7 * (gw - 1))
        append_rows(
            PLAYER_WEEK,
            connection,
            [
                {
                    "season": SEASON,
                    "gw": gw,
                    "element": element,
                    "position": position,
                    "team": UNITED,
                }
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
                    "opponent": TEAM_IDS[ARSENAL],
                    "is_home": True,
                    "minutes": minutes,
                    "total_points": total_points,
                    "yellow_cards": yellow_cards[gw - 1],
                    "red_cards": red_cards[gw - 1],
                    "kickoff_time": kickoff,
                }
            ],
        )
        if with_opta:
            append_rows(
                PLAYER_MATCH_OPTA,
                connection,
                [
                    {
                        "season": SEASON,
                        "gw": gw,
                        "element": element,
                        "match_id": match_id,
                        "competition": "prem",
                        "minutes_played": minutes,
                        "xg": 0.1,
                        "tackles": 3,
                        "fouls_committed": 2,
                    }
                ],
            )
        append_rows(
            PLAYER_MATCH_FPL,
            connection,
            [
                {
                    "season": SEASON,
                    "gw": gw,
                    "element": element,
                    "fixture": gw,
                    "opponent_team": TEAM_IDS[ARSENAL],
                    "minutes": minutes,
                    "assists": 0,
                    "goals_scored": 0,
                    "clean_sheets": 0,
                }
            ],
        )


def _seed_fixtures(connection) -> None:
    """Seed both clubs' fixtures, which the club-name bridge reads."""
    for gw in GWS:
        kickoff = GW1_KICKOFF.replace(day=GW1_KICKOFF.day + 7 * (gw - 1))
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
                for team, is_home, opposition in [
                    (UNITED, True, ARSENAL),
                    (ARSENAL, False, UNITED),
                ]
            ],
        )


# --- What the model predicts ------------------------------------------


def test_the_target_is_the_booking_count_per_ninety(connection) -> None:
    """Cards come off the match row, and the target is a rate."""
    _seed_fixtures(connection)
    _seed_player(connection, yellow_cards=(0, 1))
    predictor = _predictor(
        YellowCardsRatePredictor, YELLOW_CARDS_SPEC, connection
    )

    frame = predictor.build_training_data().filter(pl.col("gw") == 2)

    assert frame["yellow_cards_per_90"].item() == pytest.approx(1.0)
    assert frame["yellow_cards"].item() == 1


def test_a_booking_in_half_a_match_is_twice_the_rate(connection) -> None:
    """Per 90, not per match, like every sibling rate head."""
    _seed_fixtures(connection)
    _seed_player(connection, minutes=45, yellow_cards=(1, 1))
    predictor = _predictor(
        YellowCardsRatePredictor, YELLOW_CARDS_SPEC, connection
    )

    frame = predictor.build_training_data().filter(pl.col("gw") == 2)

    assert frame["yellow_cards_per_90"].item() == pytest.approx(2.0)


def test_a_leg_with_no_card_count_has_no_target(connection) -> None:
    """Null means no source filed cards, which cannot be fit.

    Distinct from a zero, which is a player who played and stayed out of
    the book.
    """
    _seed_fixtures(connection)
    _seed_player(connection, yellow_cards=(0, 1))
    connection.execute("UPDATE player_match SET yellow_cards = NULL")
    predictor = _predictor(
        YellowCardsRatePredictor, YELLOW_CARDS_SPEC, connection
    )

    assert predictor.build_training_data().is_empty()


# --- Who the model learns from ----------------------------------------


def test_every_outfield_position_reaches_the_training_frame(
    connection,
) -> None:
    """Pooled: a booking is too rare to fit from defenders alone."""
    _seed_fixtures(connection)
    for element, position in ((1, "DEF"), (2, "MID"), (3, "FWD")):
        _seed_player(connection, element=element, position=position)
    predictor = _predictor(
        YellowCardsRatePredictor, YELLOW_CARDS_SPEC, connection
    )

    frame = predictor.build_training_data()

    # Position arrives as indicator columns, not as a position column.
    for name in position_dummy_names(TRAINING_POSITIONS):
        assert frame[name].sum() > 0


def test_goalkeepers_are_left_out_of_the_fit(connection) -> None:
    """Booked for different reasons and at a fraction of the rate."""
    _seed_fixtures(connection)
    _seed_player(connection, element=1, position="DEF")
    _seed_player(connection, element=2, position="GK")
    predictor = _predictor(
        YellowCardsRatePredictor, YELLOW_CARDS_SPEC, connection
    )

    frame = predictor.build_training_data()

    # Only the defender's legs survive; a keeper has no indicator to set.
    assert frame.height == len(GWS)


def test_no_form_rows_are_kept_in_the_fit(connection) -> None:
    """The departure from the goals and assists heads, made explicit.

    ``has_no_form`` is anchored on xG, which FCI publishes and Vaastav
    does not, so every season before 2024-25 carries the flag. Filtering
    on it -- which both sibling heads do, correctly for them -- would
    throw away eight of this head's ten training seasons.
    """
    _seed_fixtures(connection)
    _seed_player(connection, yellow_cards=(0, 1), with_opta=False)
    predictor = _predictor(
        YellowCardsRatePredictor, YELLOW_CARDS_SPEC, connection
    )

    frame = predictor.build_training_data()

    assert not frame.is_empty()
    assert frame["has_no_form"].to_list() == [1.0] * frame.height


def test_a_cameo_booking_reaches_the_fit(connection) -> None:
    """No training minutes floor: a cameo booking is real evidence.

    The sibling heads floor at 30 minutes because a cameo's arithmetic
    rate has enormous leverage. Minutes weighting already cuts that here,
    and a booking off the bench is one of the more informative rows in
    the set.
    """
    _seed_fixtures(connection)
    _seed_player(
        connection, minutes=SCORING_MINUTES_FLOOR, yellow_cards=(1, 1)
    )
    predictor = _predictor(
        YellowCardsRatePredictor, YELLOW_CARDS_SPEC, connection
    )

    assert not predictor.build_training_data().is_empty()


def test_the_head_trains_wider_than_it_is_tested(connection) -> None:
    """Ten seasons of history, judged on the two it serves in."""
    assert set(TEST_SEASONS) < set(TRAINING_SEASONS)
    assert "2016-17" in TRAINING_SEASONS
    assert "2016-17" not in TEST_SEASONS


# --- What the component pays ------------------------------------------


def test_a_booking_costs_the_same_whatever_the_shirt() -> None:
    """Flat across every position, which is why it is not a table."""
    assert YELLOW_CARD_POINTS == -1.0
    assert YellowCardsRatePredictor.COMPONENT is Component.YELLOW_CARDS
    assert (
        YellowCardsRatePredictor.COMPONENT_IMPL.rate.points_per_event == -1.0
    )


# --- What the fold reports --------------------------------------------


def test_fold_metrics_score_the_fold_at_match_scale(connection) -> None:
    """Every metric the head declares comes back a real number.

    The frame ``fold_metrics`` receives has to carry the booking count,
    the minutes and the trailing rate: two of those are extra columns
    and one is a feature, so a change to either list breaks the scoring
    rather than the fit, and the fit is what the other tests watch.
    """
    _seed_fixtures(connection)
    _seed_player(connection, yellow_cards=(1, 1))
    predictor = _predictor(
        YellowCardsRatePredictor, YELLOW_CARDS_SPEC, connection
    )
    frame = predictor.build_training_data()

    metrics = predictor.fold_metrics(frame, [0.25] * frame.height).as_dict()

    assert set(metrics) == {
        "poisson_deviance",
        "brier",
        "base_rate_brier",
        "skill_score",
        "player_rate_skill",
        "booking_rate",
        "rate_mae",
        "top_decile_ratio",
    }
    assert all(value == value for value in metrics.values())  # no NaNs
    assert metrics["booking_rate"] == pytest.approx(1.0)


def test_player_rate_skill_falls_back_when_there_is_no_history(
    connection,
) -> None:
    """A player with no trailing rate is scored, not dropped.

    The first appearance of every player has a null rolling rate, and
    this head keeps those rows, so the naive baseline has to have an
    answer for them.
    """
    _seed_fixtures(connection)
    _seed_player(connection, yellow_cards=(1, 0), with_opta=False)
    predictor = _predictor(
        YellowCardsRatePredictor, YELLOW_CARDS_SPEC, connection
    )
    frame = predictor.build_training_data()

    metrics = predictor.fold_metrics(frame, [0.1] * frame.height)

    assert metrics.player_rate_skill == metrics.player_rate_skill


# --- The decomposition invariant --------------------------------------
