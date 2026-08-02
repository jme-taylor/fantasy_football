from fantasy_football.modelling.folds import gameweek_folds, season_folds


def test_season_folds_expands_over_sorted_seasons():
    """Folds expand across seasons regardless of input order."""
    assert season_folds(["2025-26", "2024-25", "2026-27"]) == [
        (["2024-25"], "2025-26"),
        (["2024-25", "2025-26"], "2026-27"),
    ]


def test_gameweek_folds_orders_chronologically_across_seasons():
    """Keys sort chronologically across season boundaries."""
    keys = [("2026-27", 1), ("2025-26", 37), ("2025-26", 38)]
    folds = gameweek_folds(keys, min_train_gws=1)
    assert folds == [
        ([("2025-26", 37)], ("2025-26", 38)),
        ([("2025-26", 37), ("2025-26", 38)], ("2026-27", 1)),
    ]


def test_gameweek_folds_respects_min_train_gws():
    """The first fold's training side has at least min_train_gws."""
    keys = [("2025-26", gw) for gw in range(1, 6)]
    assert gameweek_folds(keys, min_train_gws=3) == [
        ([("2025-26", 1), ("2025-26", 2), ("2025-26", 3)], ("2025-26", 4)),
        (
            [
                ("2025-26", 1),
                ("2025-26", 2),
                ("2025-26", 3),
                ("2025-26", 4),
            ],
            ("2025-26", 5),
        ),
    ]


def test_gameweek_folds_deduplicates_repeated_keys():
    """Repeated keys count once when building folds."""
    keys = [("2025-26", 1), ("2025-26", 1), ("2025-26", 2)]
    assert gameweek_folds(keys, min_train_gws=1) == [
        ([("2025-26", 1)], ("2025-26", 2))
    ]


def test_gameweek_folds_empty_when_too_few_gameweeks():
    """No folds are built when there are fewer than min_train_gws."""
    assert gameweek_folds([("2025-26", 1)], min_train_gws=10) == []
