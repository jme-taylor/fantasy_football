-- Worst FWD prediction errors from the latest eval run, with player,
-- opposition, kickoff and the model's features flattened out of JSON.
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
latest_run AS (
    SELECT run_id
    FROM test_points_prediction
    WHERE position = 'FWD'
    GROUP BY run_id
    ORDER BY max(gw) DESC, run_id DESC
    LIMIT 1
)
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
    p.position,
    p.predicted_points,
    p.actual_points,
    p.actual_points - p.predicted_points AS error,
    abs(p.actual_points - p.predicted_points) AS absolute_error,
    json_extract(p.features, '$.is_home')::DOUBLE AS is_home,
    json_extract(p.features, '$.expected_minutes')::DOUBLE AS expected_minutes,
    json_extract(p.features, '$.xg_per90_rolling_5')::DOUBLE AS xg_per90,
    json_extract(p.features, '$.xa_per90_rolling_5')::DOUBLE AS xa_per90,
    json_extract(p.features, '$.xg_for_rolling_5')::DOUBLE AS xg_for,
    json_extract(p.features, '$.goals_for_rolling_5')::DOUBLE AS goals_for,
    json_extract(p.features, '$.xg_against_rolling_5')::DOUBLE AS xg_against,
    json_extract(p.features, '$.goals_against_rolling_5')::DOUBLE AS goals_against,
    json_extract(p.features, '$.interceptions_per90_rolling_5')::DOUBLE AS interceptions_per90,
    json_extract(p.features, '$.tackles_per90_rolling_5')::DOUBLE AS tackles_per90,
    json_extract(p.features, '$.clearances_per90_rolling_5')::DOUBLE AS clearances_per90,
    json_extract(p.features, '$.blocks_per90_rolling_5')::DOUBLE AS blocks_per90,
    json_extract(p.features, '$.yellow_cards_per90_rolling_5')::DOUBLE AS yellow_per90,
    json_extract(p.features, '$.red_cards_per90_rolling_5')::DOUBLE AS red_per90,
    json_extract(p.features, '$.yellow_cards_season_to_date')::DOUBLE AS yellow_std,
    json_extract(p.features, '$.red_cards_season_to_date')::DOUBLE AS red_std
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
WHERE p.position = 'FWD'
ORDER BY absolute_error DESC
LIMIT 100;
