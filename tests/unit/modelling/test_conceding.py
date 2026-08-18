"""The team conceding head and the component it feeds.

The tests that matter most are the ones pinning the grain. This is the
first head in the repo that trains on team-fixtures rather than player
appearances, and the two places that can go wrong are both silent: a
training frame that has quietly fanned out to player rows fits the same
team eleven times over, and a scoring frame whose keys drift from the
sibling components' deletes whole defender predictions at composition
rather than failing.
"""

from datetime import timedelta

import polars as pl
import pytest

from fantasy_football.features.views import register_feature_views
from fantasy_football.modelling.assists import (
    ASSISTS_SPEC,
    AssistsRatePredictor,
)
from fantasy_football.modelling.components import Component
from fantasy_football.modelling.conceding import (
    CONCEDING_SPEC,
    FORM_MATCHES_FLOOR,
    POSITION,
    ConcedingPredictor,
)
from fantasy_football.modelling.defcon import (
    DEFCON_SPEC,
    DefconRatePredictor,
)
from fantasy_football.modelling.folds import TrainTestSplitStrategy
from fantasy_football.modelling.goals import (
    GOALS_SPEC,
    GoalsRatePredictor,
)
from fantasy_football.storage.tables import (
    BACKFILL_KIND,
    PLAYER_MATCH,
    PLAYER_MATCH_FPL,
    PLAYER_MATCH_OPTA,
    PLAYER_SEASON,
    PLAYER_WEEK,
    TEAM_FIXTURE,
    TEST_CONCEDING_PREDICTION,
)
from tests.unit.modelling.conftest import (
    ARSENAL,
    CHELSEA,
    GW1_KICKOFF,
    SEASON,
    SLUGS,
    SPURS,
    TEAM_IDS,
    UNITED,
    append_rows,
    opta_row,
)

# Enough gameweeks that the last one clears the form floor. The head
# refuses to learn from a team whose rolling window is short, so a seed
# with fewer would train on nothing at all.
GWS = tuple(range(1, FORM_MATCHES_FLOOR + 2))
LAST_GW = GWS[-1]

# Man Utd's six fixtures, home and away against three clubs. Distinct
# opponents and orderings are what keeps every fixture's slug -- and so
# its match_id -- unique, which the team_match view relies on to pair the
# two sides of a fixture.
OPPONENTS: tuple[tuple[str, bool], ...] = (
    (SPURS, True),
    (SPURS, False),
    (CHELSEA, True),
    (CHELSEA, False),
    (ARSENAL, True),
    (ARSENAL, False),
)

# The subject club, whose conceding is what every test here reads.
SUBJECT = UNITED


def fixture(gw: int) -> tuple[str, bool, str]:
    """Return the opponent, home flag and match id for a gameweek."""
    opposition, is_home = OPPONENTS[gw - 1]
    home, away = (SUBJECT, opposition) if is_home else (opposition, SUBJECT)
    return opposition, is_home, f"25-26-prem-{SLUGS[home]}-vs-{SLUGS[away]}"


def kickoff(gw: int):
    """Return the kickoff for a gameweek, one week apart."""
    return GW1_KICKOFF + timedelta(days=7 * (gw - 1))


def _predictor(cls, spec, connection):
    """Build a predictor against the test database."""
    return cls(
        experiment_name="test",
        params={},
        model_spec=spec,
        connection=connection,
        fold_strategy=TrainTestSplitStrategy(),
    )


def _opponent_element(gw: int) -> int:
    """Return the element standing in for the opposition that gameweek."""
    return 100 + gw


def seed_league(
    connection,
    *,
    conceded: dict[int, tuple[int, int]] | None = None,
    minutes: int = 90,
    position: str = "DEF",
    total_points: int = 6,
    clean_sheets: int = 1,
    player_conceded: int = 0,
) -> None:
    """Seed the subject club's six fixtures, both sides of each.

    Element 1 is the subject's player throughout; a fresh element stands
    in for the opposition each gameweek, which is what gives ``team_match``
    two sides to pair and mints the club-name bridge.

    ``conceded`` maps a gameweek to the goals each side shipped, as
    ``(subject, opposition)``, defaulting to a nil-nil.
    """
    conceded = conceded or {}
    append_rows(
        PLAYER_SEASON,
        connection,
        [{"season": SEASON, "element": 1, "position": position}]
        + [
            {
                "season": SEASON,
                "element": _opponent_element(gw),
                "position": position,
            }
            for gw in GWS
        ],
    )
    for gw in GWS:
        opposition, is_home, match_id = fixture(gw)
        against, opposition_against = conceded.get(gw, (0, 0))
        other = _opponent_element(gw)
        append_rows(
            PLAYER_WEEK,
            connection,
            [
                {
                    "season": SEASON,
                    "gw": gw,
                    "element": element,
                    "position": position,
                    "team": team,
                }
                for element, team in [(1, SUBJECT), (other, opposition)]
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
                    "is_home": home,
                    "minutes": minutes,
                    "total_points": total_points,
                    "yellow_cards": 0,
                    "red_cards": 0,
                    "kickoff_time": kickoff(gw),
                }
                for element, opponent, home in [
                    (1, TEAM_IDS[opposition], is_home),
                    (other, TEAM_IDS[SUBJECT], not is_home),
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
                    "is_home": home,
                    "opposition": against_team,
                    "kickoff_time": kickoff(gw),
                }
                for team, home, against_team in [
                    (SUBJECT, is_home, opposition),
                    (opposition, not is_home, SUBJECT),
                ]
            ],
        )
        append_rows(
            PLAYER_MATCH_OPTA,
            connection,
            [
                opta_row(SEASON, gw, 1, match_id, 1.5, 0, against),
                # Rising with the gameweek, so the subject's own
                # xg_against window is distinguishable from any single
                # opponent's xg_for -- which is what makes a transposed
                # own/opposition join detectable rather than a coincidence.
                opta_row(
                    SEASON,
                    gw,
                    other,
                    match_id,
                    round(0.5 + 0.1 * gw, 1),
                    0,
                    opposition_against,
                ),
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
                    "opponent_team": opponent,
                    "minutes": minutes,
                    "goals_scored": 0,
                    "assists": 0,
                    "clean_sheets": clean_sheets,
                    "goals_conceded": player_conceded,
                    "defensive_contribution": 6,
                }
                for element, opponent in [
                    (1, TEAM_IDS[opposition]),
                    (other, TEAM_IDS[SUBJECT]),
                ]
            ],
        )


# --- What the head trains on -----------------------------------------


def test_the_training_frame_is_one_row_per_team_fixture(connection) -> None:
    """Team grain, not player grain.

    The whole premise of the head. Eleven players share one clean sheet,
    so a frame that has fanned out to player rows fits the same team
    number eleven times over and calls the extra rows evidence.
    """
    seed_league(connection)
    predictor = _predictor(ConcedingPredictor, CONCEDING_SPEC, connection)

    frame = predictor.build_training_data()

    assert set(frame.columns) >= {"season", "gw", "team", "opposition"}
    assert "element" not in frame.columns
    assert frame.height == frame.select("season", "gw", "team").n_unique()


def test_the_target_is_what_the_team_conceded(connection) -> None:
    """Read off the team counter, not summed from player rows.

    An own goal is conceded by a team and credited to no player, so a
    summed target would undercount exactly the fixtures it matters in.
    """
    seed_league(connection, conceded={LAST_GW: (2, 1)})
    predictor = _predictor(ConcedingPredictor, CONCEDING_SPEC, connection)

    frame = predictor.build_training_data().filter(pl.col("gw") == LAST_GW)

    # Only the subject has a long enough record to clear the form floor;
    # the opposition club is playing its second fixture of the seed.
    assert frame["team"].to_list() == [SUBJECT]
    assert frame["goals_conceded"].item() == 2


def test_the_head_refuses_to_learn_from_a_short_form_window(
    connection,
) -> None:
    """A window of one or two matches is noise, not a defensive record."""
    seed_league(connection)
    predictor = _predictor(ConcedingPredictor, CONCEDING_SPEC, connection)

    frame = predictor.build_training_data()

    assert frame.filter(pl.col("team") == SUBJECT)["gw"].to_list() == [LAST_GW]


def test_a_short_form_window_is_still_scored(connection) -> None:
    """An opening gameweek needs a prediction, however thin the form.

    A leg missing one component is dropped from the composed prediction
    outright, so what this head may not learn from it must still score.
    """
    seed_league(connection)
    predictor = _predictor(ConcedingPredictor, CONCEDING_SPEC, connection)

    scored = predictor.scoring_frame()

    assert sorted(scored.filter(pl.col("element") == 1)["gw"].to_list()) == (
        list(GWS)
    )


def test_a_club_with_no_match_data_never_reaches_the_fit(connection) -> None:
    """The contaminated fixture rows cannot become training rows.

    ``team_match`` is derived from the match feed, so a club sitting in
    the fixture list without a single played match has no target to
    learn from -- and, unlike a filter, that cannot be forgotten. At team
    grain a phantom fixture would be a whole training row carrying a
    real-looking nil.
    """
    seed_league(connection)
    append_rows(
        TEAM_FIXTURE,
        connection,
        [
            {
                "season": SEASON,
                "gw": gw,
                "team": "Coventry",
                "is_home": True,
                "opposition": SUBJECT,
                "kickoff_time": kickoff(gw),
            }
            for gw in GWS
        ],
    )
    predictor = _predictor(ConcedingPredictor, CONCEDING_SPEC, connection)

    frame = predictor.build_training_data()

    assert "Coventry" not in frame["team"].to_list()


# --- The decomposition holds ------------------------------------------


def test_the_fan_out_matches_the_sibling_components_row_for_row(
    connection,
) -> None:
    """No sibling may cover a leg the fan-out does not.

    ``compose`` drops a fixture leg missing any one of its position's
    components, so a fan-out narrower than its siblings deletes whole
    predictions without failing. The converse is harmless -- a leg the
    others cannot score was already being dropped -- which is why this
    is a subset check and not an equality: the defcon head really does
    cover fewer legs, having no count to rate on some of them.
    """
    seed_league(connection)
    keys = ["season", "gw", "element", "opponent"]
    sibling_specs = [
        (DefconRatePredictor, DEFCON_SPEC),
        (GoalsRatePredictor, GOALS_SPEC),
        (AssistsRatePredictor, ASSISTS_SPEC),
    ]
    conceding = _predictor(ConcedingPredictor, CONCEDING_SPEC, connection)

    covered = set(conceding.scoring_frame().select(keys).iter_rows())

    assert covered
    for cls, spec in sibling_specs:
        sibling = _predictor(cls, spec, connection)
        sibling_legs = set(sibling.scoring_frame().select(keys).iter_rows())
        assert sibling_legs, f"{cls.__name__} seeded no legs to compare"
        assert (
            not sibling_legs - covered
        ), f"{cls.__name__} covers legs the conceding fan-out misses"


def test_a_defender_is_fanned_out_once_in_a_double_gameweek(
    connection,
) -> None:
    """The kickoff is what disambiguates, not the gameweek.

    A club playing twice in one gameweek is the opponent of two different
    sides that week, so joining on the opponent alone matches both of
    their fixtures and every defender facing them fans out twice --
    doubling his conceding points against components that did not double.
    """
    seed_league(connection)
    second = kickoff(LAST_GW) + timedelta(days=2)
    opposition = fixture(LAST_GW)[0]
    # The opposition plays a second fixture that gameweek, against a club
    # the subject has nothing to do with.
    append_rows(
        TEAM_FIXTURE,
        connection,
        [
            {
                "season": SEASON,
                "gw": LAST_GW,
                "team": team,
                "is_home": is_home,
                "opposition": against,
                "kickoff_time": second,
            }
            for team, is_home, against in [
                (CHELSEA, True, opposition),
                (opposition, False, CHELSEA),
            ]
        ],
    )
    match_id = f"25-26-prem-{SLUGS[CHELSEA]}-vs-{SLUGS[opposition]}"
    append_rows(
        PLAYER_MATCH_OPTA,
        connection,
        [
            opta_row(SEASON, LAST_GW, 200, match_id, 1.0, 0, 1),
            opta_row(SEASON, LAST_GW, 201, match_id, 1.0, 0, 2),
        ],
    )
    append_rows(
        PLAYER_SEASON,
        connection,
        [
            {"season": SEASON, "element": element, "position": "DEF"}
            for element in (200, 201)
        ],
    )
    append_rows(
        PLAYER_WEEK,
        connection,
        [
            {
                "season": SEASON,
                "gw": LAST_GW,
                "element": element,
                "position": "DEF",
                "team": team,
            }
            for element, team in [(200, CHELSEA), (201, opposition)]
        ],
    )
    append_rows(
        PLAYER_MATCH,
        connection,
        [
            {
                "season": SEASON,
                "gw": LAST_GW,
                "element": element,
                "opponent": opponent,
                "is_home": is_home,
                "minutes": 90,
                "total_points": 2,
                "yellow_cards": 0,
                "red_cards": 0,
                "kickoff_time": second,
            }
            for element, opponent, is_home in [
                (200, TEAM_IDS[opposition], True),
                (201, TEAM_IDS[CHELSEA], False),
            ]
        ],
    )
    predictor = _predictor(ConcedingPredictor, CONCEDING_SPEC, connection)

    scored = predictor.scoring_frame()

    subject_leg = scored.filter(
        (pl.col("element") == 1) & (pl.col("gw") == LAST_GW)
    )
    assert subject_leg.height == 1


# --- Scoring forward --------------------------------------------------


def test_the_forward_frame_carries_each_side_of_the_unplayed_fixture(
    connection, forward_fixture, forward_frame
) -> None:
    """Own defensive form and opposition attacking form, not swapped.

    The seed gives the subject a heavier xG-against record than the
    opponents it is about to face have an xG-for one, so a transposed
    join is caught rather than coincidentally passing.
    """
    seed_league(connection)
    register_feature_views(connection)
    unplayed = LAST_GW + 1
    predictor = _predictor(ConcedingPredictor, CONCEDING_SPEC, connection)
    append_rows(
        TEAM_FIXTURE,
        connection,
        [
            {
                "season": SEASON,
                "gw": unplayed,
                "team": team,
                "is_home": is_home,
                "opposition": against,
                "kickoff_time": kickoff(unplayed),
            }
            for team, is_home, against in [
                (SUBJECT, True, SPURS),
                (SPURS, False, SUBJECT),
            ]
        ],
    )

    frame = predictor.build_forward_data(
        forward_frame(
            [
                forward_fixture(
                    gw=unplayed,
                    opponent=TEAM_IDS[SPURS],
                    kickoff=kickoff(unplayed),
                    team=SUBJECT,
                )
            ]
        )
    )

    assert frame.height == 1
    assert set(frame.columns) == set(
        ["season", "gw", "element", "opponent"] + predictor.FEATURES
    )
    # The subject's five most recent matches conceded 0.7, 0.8, 0.9, 1.0
    # and 1.1 expected goals; Spurs, met in gws 1 and 2, created 0.6 and
    # 0.7. Transposing the join would swap 0.9 for 0.65.
    assert frame["xg_against_rolling_5"].item() == pytest.approx(0.9)
    assert frame["opposition_xg_for_rolling_5"].item() == pytest.approx(0.65)


def test_the_forward_path_scores_only_rostered_fixtures(
    connection, forward_fixture, forward_frame
) -> None:
    """A contaminated club cannot re-enter on the forward path.

    The team fixtures come out of the rostered player fixtures rather
    than off the fixture list, so a club with no players to score has
    nothing to bring it back -- which is why the in-league filter is not
    written a second time here.
    """
    seed_league(connection)
    register_feature_views(connection)
    unplayed = LAST_GW + 1
    append_rows(
        TEAM_FIXTURE,
        connection,
        [
            {
                "season": SEASON,
                "gw": unplayed,
                "team": "Coventry",
                "is_home": True,
                "opposition": SPURS,
                "kickoff_time": kickoff(unplayed),
            }
        ],
    )
    predictor = _predictor(ConcedingPredictor, CONCEDING_SPEC, connection)

    frame = predictor.build_forward_data(
        forward_frame(
            [
                forward_fixture(
                    gw=unplayed,
                    opponent=TEAM_IDS[SPURS],
                    kickoff=kickoff(unplayed),
                    team=SUBJECT,
                    position="MID",
                )
            ]
        )
    )

    assert frame.is_empty()


def test_scoring_writes_conceding_component_rows(connection) -> None:
    """The head's output is one component of a defender's points."""
    seed_league(connection, conceded={gw: (1, 1) for gw in GWS})
    predictor = _predictor(ConcedingPredictor, CONCEDING_SPEC, connection)
    model = predictor.train_final(predictor.build_training_data())

    rows = predictor.build_prediction_rows(
        predictor.scoring_frame(), model, "1", BACKFILL_KIND
    )

    assert rows["component"].unique().to_list() == [str(Component.CONCEDING)]
    assert rows["position"].unique().to_list() == [POSITION]
    assert rows["points"].null_count() == 0


def test_fold_rows_are_shaped_for_the_team_grain_eval_table(
    connection,
) -> None:
    """Stored at team grain, not fanned out to the players who share it.

    Eleven identical rows per fixture would make the error analysis
    report a sample size the model never saw.
    """
    seed_league(connection, conceded={gw: (1, 1) for gw in GWS})
    predictor = _predictor(ConcedingPredictor, CONCEDING_SPEC, connection)
    frame = predictor.build_training_data()

    rows = predictor.fold_predictions(frame, [1.4] * frame.height)

    assert set(rows.columns) | {"run_id"} == set(
        TEST_CONCEDING_PREDICTION.columns
    )
    assert rows["actual_conceded"].to_list() == [1.0]


def test_both_payoffs_are_scored_separately(connection) -> None:
    """A clean-sheet bias and a deduction bias are different failures."""
    seed_league(connection, conceded={gw: (1, 1) for gw in GWS})
    predictor = _predictor(ConcedingPredictor, CONCEDING_SPEC, connection)
    frame = predictor.build_training_data()

    metrics = predictor.fold_metrics(frame, [2.0] * frame.height)

    # One fixture, one goal conceded: no clean sheet kept, and the
    # deduction actually paid was floor(1 / 2) = 0 against a rate of two.
    assert metrics.clean_sheet_rate == pytest.approx(0.0)
    assert metrics.calibration == pytest.approx(2.0)
    assert metrics.deduction_mae > 0.0
    assert set(metrics.as_dict()) >= {"deduction_mae", "clean_sheet_brier"}


def test_the_forward_path_registers_the_views_it_reads(
    connection, forward_fixture, forward_frame
) -> None:
    """Scoring forward must not depend on having trained first.

    The form views are temporary and die with the connection, so a
    predictor asked only to score would otherwise fail on a catalog error
    that names a view rather than the missing call.
    """
    seed_league(connection)
    unplayed = LAST_GW + 1
    append_rows(
        TEAM_FIXTURE,
        connection,
        [
            {
                "season": SEASON,
                "gw": unplayed,
                "team": team,
                "is_home": is_home,
                "opposition": against,
                "kickoff_time": kickoff(unplayed),
            }
            for team, is_home, against in [
                (SUBJECT, True, SPURS),
                (SPURS, False, SUBJECT),
            ]
        ],
    )
    predictor = _predictor(ConcedingPredictor, CONCEDING_SPEC, connection)

    frame = predictor.build_forward_data(
        forward_frame(
            [
                forward_fixture(
                    gw=unplayed,
                    opponent=TEAM_IDS[SPURS],
                    kickoff=kickoff(unplayed),
                    team=SUBJECT,
                )
            ]
        )
    )

    assert frame.height == 1
