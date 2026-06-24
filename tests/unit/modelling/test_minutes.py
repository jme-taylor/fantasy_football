import polars as pl

from fantasy_football.modelling.minutes import (
    BUCKET_PARTIAL,
    BUCKET_SIXTY_PLUS,
    BUCKET_ZERO,
    FEATURES,
    build_feature_frame,
    build_model_frame,
    create_minutes_bucket,
)


def test_create_minutes_bucket_edges() -> None:
    """0 -> zero, 1..59 -> partial, 60+ -> sixty-plus (boundaries pinned)."""
    data = pl.DataFrame({"minutes": [0, 1, 59, 60, 90]})

    result = create_minutes_bucket(data)
    buckets = result["minutes_bucket"].to_list()

    assert buckets == [
        BUCKET_ZERO,
        BUCKET_PARTIAL,
        BUCKET_PARTIAL,
        BUCKET_SIXTY_PLUS,
        BUCKET_SIXTY_PLUS,
    ]


def _player_week() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": ["2022-23"] * 3,
            "gw": [1, 1, 1],
            "element": [1, 2, 3],
            "name": ["A", "B", "C"],
            "position": ["MID", "MID", "DEF"],
            "team": ["Arsenal", "Arsenal", "Arsenal"],
            "bonus": [0, 0, 0],
            "minutes": [90, 0, 45],
            "round": [1, 1, 1],
            "total_points": [6, 0, 2],
            "value": [70, 50, 40],
        }
    )


def _availability() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": ["2022-23"] * 3,
            "gw": [1, 1, 1],
            "element": [1, 2, 3],
            "chance_of_playing_this_round": [100, 75, 100],
        }
    )


def test_build_feature_frame_has_one_row_per_player_week() -> None:
    """Feature frame keeps the player-week grain and the feature columns."""
    frame = build_feature_frame(_player_week(), _availability())

    assert frame.height == 3
    for column in ["season", "gw", "element", "position", *FEATURES]:
        assert column in frame.columns
    # Arsenal MID pecking order: element 1 (value 70) ranks above element 2.
    ranks = {
        row["element"]: row["pos_value_rank"]
        for row in frame.iter_rows(named=True)
    }
    assert ranks[1] == 1
    assert ranks[2] == 2


def test_build_model_frame_joins_features_onto_matches() -> None:
    """Match rows get features by (season, gw, element) and a bucket target."""
    feature_frame = build_feature_frame(_player_week(), _availability())
    player_match = pl.DataFrame(
        {
            "season": ["2022-23"] * 3,
            "gw": [1, 1, 1],
            "element": [1, 2, 3],
            "opponent": [10, 10, 10],
            "is_home": [True, True, True],
            "minutes": [90, 0, 45],
            "total_points": [6, 0, 2],
        }
    )

    model_frame = build_model_frame(player_match, feature_frame)

    assert model_frame.height == 3
    buckets = {
        row["element"]: row["minutes_bucket"]
        for row in model_frame.iter_rows(named=True)
    }
    assert buckets[1] == BUCKET_SIXTY_PLUS
    assert buckets[2] == BUCKET_ZERO
    assert buckets[3] == BUCKET_PARTIAL
    for column in ["minutes", "minutes_bucket", *FEATURES]:
        assert column in model_frame.columns
