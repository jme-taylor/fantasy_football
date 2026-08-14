-- Worst defcon (CBIT rate) prediction errors from the latest eval run.
--
-- The defcon head's target is cbit_per_90, so test_points_prediction's
-- predicted_points / actual_points hold *rates*, not points. The rate is
-- not the quantity of interest -- P(CBIT >= 10) is -- so this pulls the
-- rate back to match scale with the fixture's actual minutes, exactly as
-- DefconRatePredictor.fold_metrics does, and scores the threshold.
--
-- Ordered by minutes-weighted absolute rate error, which is what
-- rate_mae averages: a wild rate off 16 minutes is arithmetic noise.
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
-- The defcon head and the DEF residual head both write position = 'DEF',
-- and only run_id separates them. cbit_per90_rolling_5 is in the defcon
-- feature list and in no other, so it identifies the run.
latest_run AS (
    SELECT run_id
    FROM test_points_prediction
    WHERE position = 'DEF'
        AND json_extract(features, '$.cbit_per90_rolling_5') IS NOT NULL
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
        p.predicted_points AS predicted_cbit_per90,
        p.actual_points AS actual_cbit_per90,
        greatest(p.predicted_points, 0) * m.minutes / 90.0 AS expected_cbit,
        p.actual_points * m.minutes / 90.0 AS actual_cbit,
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
        round(s.actual_cbit) >= 10 AS hit,
        -- Poisson P(CBIT >= 10 | lambda = expected_cbit), the same tail
        -- the composition scores. Written as 1 - P(<= 9) because DuckDB
        -- has no poisson_cdf.
        1 - list_sum(
            list_transform(
                range(0, 10),
                k -> exp(-s.expected_cbit)
                     * pow(s.expected_cbit, k)
                     / factorial(k::INTEGER)
            )
        ) AS p_ten_plus
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
    predicted_cbit_per90,
    actual_cbit_per90,
    actual_cbit_per90 - predicted_cbit_per90 AS rate_error,
    abs(actual_cbit_per90 - predicted_cbit_per90) AS absolute_rate_error,
    -- What rate_mae actually averages.
    abs(actual_cbit_per90 - predicted_cbit_per90) * minutes AS weighted_absolute_rate_error,
    expected_cbit,
    round(actual_cbit) AS actual_cbit,
    -- Poisson P(CBIT >= 10 | lambda = expected_cbit), the same tail the
    -- composition scores. Written as 1 - P(<= 9) because DuckDB has no
    -- poisson_cdf.
    1 - list_sum(
        list_transform(
            range(0, 10),
            k -> exp(-expected_cbit) * pow(expected_cbit, k) / factorial(k)
        )
    ) AS p_ten_plus,
    (round(actual_cbit) >= 10)::INT AS hit_ten_plus,
    -- Per-row Brier contribution: the rows the threshold model is most
    -- wrong about, as opposed to the rows the rate is most wrong about.
    pow(
        (round(actual_cbit) >= 10)::INT - (
            1 - list_sum(
                list_transform(
                    range(0, 10),
                    k -> exp(-expected_cbit) * pow(expected_cbit, k) / factorial(k)
                )
            )
        ),
        2
    ) AS brier_contribution,
    json_extract(features, '$.is_home')::DOUBLE AS is_home,
    json_extract(features, '$.cbit_per90_rolling_5')::DOUBLE AS cbit_per90_rolling_5,
    json_extract(features, '$.cbit_ten_plus_rate_rolling_5')::DOUBLE AS cbit_ten_plus_rate_rolling_5,
    json_extract(features, '$.cbit_std_rolling_5')::DOUBLE AS cbit_std_rolling_5,
    json_extract(features, '$.tackles_per90_rolling_5')::DOUBLE AS tackles_per90,
    json_extract(features, '$.interceptions_per90_rolling_5')::DOUBLE AS interceptions_per90,
    json_extract(features, '$.clearances_per90_rolling_5')::DOUBLE AS clearances_per90,
    json_extract(features, '$.blocks_per90_rolling_5')::DOUBLE AS blocks_per90,
    json_extract(features, '$.xg_against_rolling_5')::DOUBLE AS xg_against,
    json_extract(features, '$.goals_against_rolling_5')::DOUBLE AS goals_against,
    json_extract(features, '$.xg_for_rolling_5')::DOUBLE AS xg_for,
    json_extract(features, '$.goals_for_rolling_5')::DOUBLE AS goals_for
FROM scored
ORDER BY weighted_absolute_rate_error DESC
LIMIT 100;
