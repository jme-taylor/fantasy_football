"""The goalkeeper saves rate head.

What matters here is what would go wrong quietly. The target must be
FPL's save count and not a rate the component then floors a second time;
the scoring floor must match the keeper's sibling components or a leg
loses its whole composed prediction rather than just its saves; and the
training window must be the seasons the team-form features actually
exist in, since the rows outside it would be fit on median fills.
"""

import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from fantasy_football.modelling.components import (
    Component,
    SavesComponent,
)
from fantasy_football.modelling.conceding import (
    SCORING_MINUTES_FLOOR as CONCEDING_SCORING_FLOOR,
)
from fantasy_football.modelling.folds import ExpandingGameweekFoldStrategy
from fantasy_football.modelling.saves import (
    MINUTES_FLOOR,
    POSITION,
    SAVES_SPEC,
    SCORING_MINUTES_FLOOR,
    SavesRatePredictor,
)
from fantasy_football.storage.coverage import FCI_SEASONS
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

GWS = (1, 2)


@pytest.fixture
def predictor(connection) -> SavesRatePredictor:
    """Return a ``SavesRatePredictor`` bound to the test connection."""
    return SavesRatePredictor(
        experiment_name="test-saves",
        params={},
        model_spec=SAVES_SPEC,
        connection=connection,
        fold_strategy=ExpandingGameweekFoldStrategy(),
    )


def _seed_keeper(
    connection,
    *,
    element: int = 1,
    minutes: int = 90,
    saves: tuple[int, int] = (0, 0),
) -> None:
    """Seed one keeper's two-gameweek run.

    The Opta rows carry a deliberately different save count from the FPL
    ones, so a target reading the wrong feed is caught rather than
    coincidentally agreeing.
    """
    match_id = f"25-26-prem-{SLUGS[UNITED]}-vs-{SLUGS[ARSENAL]}"
    append_rows(
        PLAYER_SEASON,
        connection,
        [{"season": SEASON, "element": element, "position": POSITION}],
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
                    "position": POSITION,
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
                    "total_points": 3,
                    "yellow_cards": 0,
                    "red_cards": 0,
                    "kickoff_time": kickoff,
                }
            ],
        )
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
                    "saves": saves[gw - 1] + 7,
                    "team_goals_conceded": 1,
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
                    "saves": saves[gw - 1],
                    "goals_conceded": 1,
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


def test_target_reads_fpls_save_count(connection, predictor) -> None:
    """FPL settles the save points, so FPL's count is the target.

    The seed gives Opta a deliberately different count; it must not
    reach the target.
    """
    _seed_fixtures(connection)
    _seed_keeper(connection, saves=(1, 4))

    frame = predictor.build_training_data().sort("gw")

    assert frame["saves_per_90"].to_list() == pytest.approx([1.0, 4.0])
    assert frame["saves"].to_list() == [1, 4]


def test_the_target_is_per_ninety_not_per_match(connection, predictor) -> None:
    """A keeper subbed at the hour makes his saves in less than a match."""
    _seed_fixtures(connection)
    _seed_keeper(connection, minutes=60, saves=(2, 2))

    frame = predictor.build_training_data()

    assert frame["saves_per_90"].to_list() == pytest.approx([3.0, 3.0])


def test_a_leg_the_fpl_join_misses_has_no_target(
    connection, predictor
) -> None:
    """Null only ever means the join missed, and that row cannot be fit."""
    _seed_fixtures(connection)
    _seed_keeper(connection, saves=(1, 2))
    connection.execute("DELETE FROM player_match_fpl")

    assert predictor.build_training_data().is_empty()


def test_the_head_predicts_a_rate_and_not_points(connection) -> None:
    """The target is saves, not ``floor(saves / 3)``.

    Regressing on the floored value would apply the rounding twice --
    once in the target and again in the component -- and lose the
    fractional expectation that makes a keeper on 2.9 saves different
    from one on 2.1.
    """
    assert SavesRatePredictor.TARGET == "saves_per_90"
    assert isinstance(SavesRatePredictor.COMPONENT_IMPL, SavesComponent)


# --- Where it is allowed to look --------------------------------------


def test_component_and_spec_agree_on_position_and_component() -> None:
    """A spec disagreeing with the class rewrites another's partition."""
    assert POSITION == "GK"
    assert SAVES_SPEC.position == POSITION
    assert SAVES_SPEC.component == Component.SAVES
    assert SavesRatePredictor.COMPONENT == Component.SAVES


def test_training_window_is_the_team_form_coverage() -> None:
    """Team form is FCI-derived, so it does not predate FCI.

    The danger features this head leans on -- what the keeper's club
    concedes and what the opposition creates -- come from
    ``team_match_form``, which is built from FCI player rows. Training
    outside that window would median-fill every one of them.
    """
    assert SavesRatePredictor.TRAINING_SEASONS == FCI_SEASONS


def test_scoring_floor_matches_the_keepers_other_components() -> None:
    """A tighter floor here would cost the leg its whole prediction.

    Saves, conceding and appearance must cover the same legs: ``compose``
    drops a fixture leg missing any declared component rather than
    summing a partial one.
    """
    assert SCORING_MINUTES_FLOOR == CONCEDING_SCORING_FLOOR


def test_cameos_are_scored_but_never_learned_from(predictor) -> None:
    """A keeper's short appearance is a rate with enormous leverage.

    Two saves in ten minutes is eighteen per 90 against a population
    mean near three. The row still has to be scored, so the restriction
    is fit-only.
    """
    assert MINUTES_FLOOR > SCORING_MINUTES_FLOOR
    assert f"m.minutes >= {MINUTES_FLOOR}" in predictor.training_row_filter
    assert f"m.minutes >= {MINUTES_FLOOR}" not in predictor.row_filter


def test_reads_no_minutes_feature() -> None:
    """Minutes enter once, at composition, for every component alike."""
    assert SavesRatePredictor.MINUTES_COLUMNS == []
    for column in ("expected_minutes", "p_zero", "p_partial", "p_sixty_plus"):
        assert column not in SavesRatePredictor.FEATURES


def test_reads_the_danger_it_faces_from_both_sides(predictor) -> None:
    """Saves are shots faced, so both clubs' form is read.

    The keeper's own club for what it lets through, the opposition for
    what it creates.
    """
    assert SavesRatePredictor.OWN_TEAM_COLUMNS == [
        "xg_against_rolling_5",
        "goals_against_rolling_5",
    ]
    assert SavesRatePredictor.OPPOSITION_COLUMNS == [
        "xg_for_rolling_5",
        "goals_for_rolling_5",
    ]
    assert predictor.opposition_feature_names == [
        "opposition_xg_for_rolling_5",
        "opposition_goals_for_rolling_5",
    ]


def test_reads_the_keepers_own_save_rate() -> None:
    """Shots faced drives most of it, but the keeper is not nothing.

    A sweeper behind a high line faces fewer shots of a harder kind than
    his club's concession rate alone would suggest.
    """
    assert "saves_per90_rolling_5" in SavesRatePredictor.FEATURES


def test_outfield_rates_are_excluded() -> None:
    """A keeper records essentially none of these."""
    for column in (
        "xg_per90_rolling_5",
        "xa_per90_rolling_5",
        "tackles_per90_rolling_5",
        "clearances_per90_rolling_5",
    ):
        assert column not in SavesRatePredictor.FEATURES


def test_every_computed_column_reaches_the_feature_list(predictor) -> None:
    """No form column is computed and then silently dropped."""
    computed = (
        predictor.PLAYER_FORM_COLUMNS
        + predictor.OWN_TEAM_COLUMNS
        + predictor.opposition_feature_names
        + predictor.MINUTES_COLUMNS
    )

    assert [
        column for column in computed if column not in predictor.FEATURES
    ] == []


def test_fits_a_poisson_loss_rather_than_squared_error(predictor) -> None:
    """The target is a count feeding a Poisson survival function.

    A squared-error fit neither respects the non-negativity nor the
    mean-variance link, which is the conceding head's reasoning applied
    to the same shape of target.
    """
    model = predictor.make_pipeline().named_steps["model"]

    assert isinstance(model, HistGradientBoostingRegressor)
    assert model.loss == "poisson"
