-- Worst goals-rate prediction errors from the latest eval run.
--
-- The goals head's target is goals_per_90, so test_points_prediction's
-- predicted_points / actual_points hold *rates*, not points. The rate is
-- not the quantity of interest -- P(goal) is -- so this pulls the rate
-- back to match scale with the fixture's actual minutes, exactly as
-- GoalsRatePredictor.fold_metrics does, and scores the tail.
--
-- Ordered by minutes-weighted absolute rate error, which is what
-- rate_mae averages: a wild rate off a 30 minute cameo is noise.
WITH fpl_team_id AS (
    SELECT DISTINCT
        m.season,
        m.opponent AS team_id,
        tf.opposition AS team
    FROM player_match AS m
    JOIN player_week AS w
        ON w.season = m.season
        AND w.gw = m.gw
        AND w.element = m.element
    JOIN team_fixture AS tf
        ON tf.season = m.season
        AND tf.gw = m.gw
        AND tf.team = w.team
        AND tf.is_home = m.is_home
        AND tf.kickoff_time = m.kickoff_time
),
-- The goals head, the defcon head and the DEF residual head all write
-- position = 'DEF' (the goals model is pooled across DEF/MID/FWD but
-- only scores and writes the position it serves), and only run_id
-- separates them. goals_scored_per90_rolling_5 is in the goals feature
-- list and in no other, so it identifies the run.
latest_run AS (
    SELECT run_id
    FROM test_points_prediction
    WHERE position = 'DEF'
        AND json_extract(features, '$.goals_scored_per90_rolling_5') IS NOT NULL
    GROUP BY run_id
    ORDER BY max(gw) DESC, run_id DESC
    LIMIT 1
),
scored AS (
    SELECT
        p.run_id,
        p.season,
        p.gw,
        m.kickoff_time,
        p.element,
        w.name AS player,
        w.team AS player_team,
        opp.team AS opposition,
        m.minutes,
        p.predicted_points AS predicted_goals_per90,
        p.actual_points AS actual_goals_per90,
        greatest(p.predicted_points, 0) * m.minutes / 90.0 AS expected_goals,
        p.actual_points * m.minutes / 90.0 AS actual_goals,
        p.features
    FROM test_points_prediction AS p
    JOIN latest_run AS r
        ON r.run_id = p.run_id
    LEFT JOIN player_match AS m
        ON m.season = p.season
        AND m.gw = p.gw
        AND m.element = p.element
        AND m.opponent = p.opponent
    LEFT JOIN player_week AS w
        ON w.season = p.season
        AND w.gw = p.gw
        AND w.element = p.element
    LEFT JOIN fpl_team_id AS opp
        ON opp.season = p.season
        AND opp.team_id = p.opponent
    WHERE p.position = 'DEF'
),
tailed AS (
    SELECT
        s.*,
        round(s.actual_goals) >= 1 AS scored_goal,
        -- Poisson P(goals >= 1 | lambda = expected_goals), the same tail
        -- fold_metrics scores.
        1 - exp(-s.expected_goals) AS p_one_plus
    FROM scored AS s
)
SELECT
    run_id,
    season,
    gw,
    kickoff_time,
    element,
    player,
    player_team,
    opposition,
    minutes,
    predicted_goals_per90,
    actual_goals_per90,
    actual_goals_per90 - predicted_goals_per90 AS rate_error,
    abs(actual_goals_per90 - predicted_goals_per90) AS absolute_rate_error,
    -- What rate_mae actually averages.
    abs(actual_goals_per90 - predicted_goals_per90) * minutes
        AS weighted_absolute_rate_error,
    expected_goals,
    round(actual_goals) AS actual_goals,
    -- Six points a goal for a defender; the component converts by
    -- position, so this is the DEF rows' points scale.
    expected_goals * 6 AS expected_goal_points,
    p_one_plus,
    scored_goal::INT AS scored_goal,
    -- Per-row Brier contribution: the rows the tail is most wrong about,
    -- as opposed to the rows the rate is most wrong about.
    pow(scored_goal::INT - p_one_plus, 2) AS brier_contribution,
    json_extract(features, '$.is_home')::DOUBLE AS is_home,
    json_extract(features, '$.is_defender')::DOUBLE AS is_defender,
    json_extract(features, '$.is_midfielder')::DOUBLE AS is_midfielder,
    json_extract(features, '$.is_forward')::DOUBLE AS is_forward,
    json_extract(features, '$.has_no_form')::DOUBLE AS has_no_form,
    json_extract(features, '$.goals_scored_per90_rolling_5')::DOUBLE AS goals_scored_per90,
    json_extract(features, '$.xg_per90_rolling_5')::DOUBLE AS xg_per90,
    json_extract(features, '$.xa_per90_rolling_5')::DOUBLE AS xa_per90,
    json_extract(features, '$.expected_pen_attempts_per_90')::DOUBLE AS pen_attempts_per90,
    json_extract(features, '$.xg_for_rolling_5')::DOUBLE AS xg_for,
    json_extract(features, '$.goals_for_rolling_5')::DOUBLE AS goals_for,
    json_extract(features, '$.xg_against_rolling_5')::DOUBLE AS xg_against,
    json_extract(features, '$.goals_against_rolling_5')::DOUBLE AS goals_against
FROM tailed
ORDER BY weighted_absolute_rate_error DESC
LIMIT 100;
