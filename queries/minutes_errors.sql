-- Worst minutes predictions from the latest eval run, with player,
-- opposition, kickoff and the model's features flattened out of JSON.
--
-- The minutes model is a three-way bucket classifier, not a regressor,
-- so expected_minutes is a derived quantity: p_partial and p_sixty_plus
-- weighted by their bucket midpoints (30 and 75). Ordered by absolute
-- minutes error, which is what e_min_mae averages.
--
-- The two decision boundaries the composition actually leans on are
-- scored alongside it, exactly as boundary_metrics does. Appearance
-- (p_partial + p_sixty_plus) pays the appearance point and gates every
-- rate component; sixty-plus pays the second one and the clean sheet.
-- A row can sit near the top on minutes error while both boundaries are
-- called correctly -- a 60-minute cameo predicted as a full start -- so
-- the Brier contributions are the column to sort on when it is the
-- boundary rather than the magnitude that matters.
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
-- Only the minutes model writes here, so unlike test_points_prediction
-- the newest run needs no feature probe to tell it from another head's.
latest_run AS (
    SELECT run_id
    FROM test_minutes_prediction
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
        p.p_zero,
        p.p_partial,
        p.p_sixty_plus,
        -- The two boundaries the points composition reads.
        p.p_partial + p.p_sixty_plus AS p_appear,
        p.expected_minutes,
        p.actual_minutes,
        p.actual_bucket,
        (p.actual_bucket <> '0_minutes')::INT AS appeared,
        (p.actual_bucket = '60_minutes_plus')::INT AS played_sixty,
        p.features
    FROM test_minutes_prediction AS p
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
    actual_bucket,
    actual_minutes,
    expected_minutes,
    actual_minutes - expected_minutes AS error,
    -- What e_min_mae actually averages.
    abs(actual_minutes - expected_minutes) AS absolute_error,
    p_zero,
    p_partial,
    p_sixty_plus,
    p_appear,
    appeared,
    played_sixty,
    -- Per-row Brier contributions: the rows a boundary is most wrong
    -- about, as opposed to the rows the minutes are most wrong about.
    pow(appeared - p_appear, 2) AS brier_appear_contribution,
    pow(played_sixty - p_sixty_plus, 2) AS brier_sixty_contribution,
    features ->> '$.position' AS position,
    json_extract(features, '$.value')::DOUBLE AS value,
    json_extract(features, '$.value_share_of_team')::DOUBLE AS value_share_of_team,
    json_extract(features, '$.pos_value_rank')::DOUBLE AS pos_value_rank,
    json_extract(features, '$.players_same_pos')::DOUBLE AS players_same_pos,
    json_extract(features, '$.chance_of_playing_this_round')::DOUBLE AS chance_of_playing_this_round,
    json_extract(features, '$.fit_rivals_same_pos')::DOUBLE AS fit_rivals_same_pos,
    json_extract(features, '$.fit_rivals_ahead')::DOUBLE AS fit_rivals_ahead,
    json_extract(features, '$.avg_minutes_rolling_5')::DOUBLE AS avg_minutes_rolling_5,
    json_extract(features, '$.games_played_this_season')::DOUBLE AS games_played_this_season,
    json_extract(features, '$.prev_season_minutes')::DOUBLE AS prev_season_minutes,
    json_extract(features, '$.prev_season_start_rate')::DOUBLE AS prev_season_start_rate,
    json_extract(features, '$.prev_season_points_per_start')::DOUBLE AS prev_season_points_per_start,
    json_extract(features, '$.pl_seasons_played')::DOUBLE AS pl_seasons_played,
    json_extract(features, '$.seasons_since_last_pl')::DOUBLE AS seasons_since_last_pl,
    json_extract(features, '$.age_years')::DOUBLE AS age_years,
    json_extract(features, '$.is_pl_newcomer')::DOUBLE AS is_pl_newcomer,
    json_extract(features, '$.is_promoted_club')::DOUBLE AS is_promoted_club
FROM scored
ORDER BY absolute_error DESC
LIMIT 100;
