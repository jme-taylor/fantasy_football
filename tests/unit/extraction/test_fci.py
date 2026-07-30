from datetime import datetime, timezone

import polars as pl
import pytest
from pytest_mock import MockerFixture

from fantasy_football.extraction.extractor import GitHubAPIClient
from fantasy_football.extraction.fci import FciExtractor, build_merged_gw

# Two players across two gameweeks; player 2 has a double gameweek in GW2.
SNAPSHOTS = pl.DataFrame(
    {
        "gw": [1, 1, 2, 2],
        "id": [1, 2, 1, 2],
        "first_name": ["David", "Erling", "David", "Erling"],
        "second_name": ["Raya", "Haaland", "Raya", "Haaland"],
        "now_cost": [6.0, 14.0, 6.1, 14.2],
        "event_points": [6, 9, 2, 13],
        "bonus": [1, 3, 1, 7],  # cumulative season bonus
    },
    schema_overrides={"gw": pl.Int32},
)
MATCHSTATS = pl.DataFrame(
    {
        "gw": [1, 1, 2, 2, 2],
        "player_id": [
            1,
            2,
            1,
            2,
            2,
        ],  # player 2 plays twice in GW2 (real PL DGW)
        "match_id": [
            "25-26-prem-arsenal-vs-chelsea",
            "25-26-prem-arsenal-vs-chelsea",
            "25-26-prem-arsenal-vs-spurs",
            "25-26-prem-man-city-vs-spurs",
            "25-26-prem-man-city-vs-wolves",
        ],
        "minutes_played": [90, 90, 90, 80, 30],
    },
    schema_overrides={"gw": pl.Int32},
)
PLAYERS = pl.DataFrame(
    {
        "player_id": [1, 2],
        "position": ["Goalkeeper", "Forward"],
        "team_code": [3, 43],
    }
)
TEAM_CODE_TO_NAME = {3: "Arsenal", 43: "Man City"}

# Per-gameweek team_code: player 2 is at Arsenal (3) in GW1, Man City (43) GW2.
PLAYER_GW_TEAM = pl.DataFrame(
    {
        "gw": [1, 1, 2, 2],
        "element": [1, 2, 1, 2],
        "team_code": [3, 3, 3, 43],
    },
    schema={"gw": pl.Int64, "element": pl.Int64, "team_code": pl.Int64},
)


def test_build_merged_gw_columns_and_shape() -> None:
    """Adapter returns exactly the merged_gw columns, one row per player-gw."""
    result = build_merged_gw(
        SNAPSHOTS, MATCHSTATS, PLAYERS, PLAYER_GW_TEAM, TEAM_CODE_TO_NAME
    )
    assert set(result.columns) == {
        "name",
        "position",
        "team",
        "bonus",
        "element",
        "minutes",
        "round",
        "total_points",
        "GW",
        "value",
    }
    assert result.height == 4


def test_build_merged_gw_maps_core_fields() -> None:
    """Core fields map from the FCI snapshot for a known player-gw."""
    result = build_merged_gw(
        SNAPSHOTS, MATCHSTATS, PLAYERS, PLAYER_GW_TEAM, TEAM_CODE_TO_NAME
    ).sort(["GW", "element"])
    raya_gw1 = result.filter(
        (pl.col("element") == 1) & (pl.col("GW") == 1)
    ).row(0, named=True)
    assert raya_gw1["name"] == "David Raya"
    assert raya_gw1["position"] == "GK"
    assert raya_gw1["team"] == "Arsenal"
    assert raya_gw1["total_points"] == 6
    assert raya_gw1["value"] == 60  # 6.0 * 10
    assert raya_gw1["round"] == 1


def test_build_merged_gw_sums_double_gameweek_minutes() -> None:
    """Minutes are summed across a player's matches within one gameweek."""
    result = build_merged_gw(
        SNAPSHOTS, MATCHSTATS, PLAYERS, PLAYER_GW_TEAM, TEAM_CODE_TO_NAME
    )
    haaland_gw2 = result.filter(
        (pl.col("element") == 2) & (pl.col("GW") == 2)
    ).row(0, named=True)
    assert haaland_gw2["minutes"] == 110  # 80 + 30


def test_build_merged_gw_bonus_is_event_level_diff() -> None:
    """Event bonus is the cumulative-snapshot difference; first gw kept as-is."""
    result = build_merged_gw(
        SNAPSHOTS, MATCHSTATS, PLAYERS, PLAYER_GW_TEAM, TEAM_CODE_TO_NAME
    )
    # Player 2 cumulative bonus 3 -> 7, so GW2 event bonus = 4.
    haaland_gw2 = result.filter(
        (pl.col("element") == 2) & (pl.col("GW") == 2)
    ).row(0, named=True)
    assert haaland_gw2["bonus"] == 4
    # First GW uses the cumulative value as-is.
    haaland_gw1 = result.filter(
        (pl.col("element") == 2) & (pl.col("GW") == 1)
    ).row(0, named=True)
    assert haaland_gw1["bonus"] == 3


def test_build_merged_gw_excludes_non_prem_match_minutes() -> None:
    """Cup / European minutes are not summed into the gameweek total."""
    matchstats = pl.DataFrame(
        {
            "gw": [1, 1],
            "player_id": [1, 1],
            "match_id": [
                "25-26-prem-arsenal-vs-chelsea",
                "25-26-efl-cup-arsenal-vs-brighton",
            ],
            "minutes_played": [90, 90],
        },
        schema_overrides={"gw": pl.Int32},
    )
    result = build_merged_gw(
        SNAPSHOTS, matchstats, PLAYERS, PLAYER_GW_TEAM, TEAM_CODE_TO_NAME
    )
    raya_gw1 = result.filter(
        (pl.col("element") == 1) & (pl.col("GW") == 1)
    ).row(0, named=True)
    assert raya_gw1["minutes"] == 90  # PL only, not 180


def test_build_merged_gw_fills_missing_minutes_with_zero() -> None:
    """Players with no match rows in a gameweek get zero minutes."""
    snaps = SNAPSHOTS.clone()
    # No matchstats row for player 1 in GW2 -> minutes should be 0.
    matchstats = MATCHSTATS.filter(
        ~((pl.col("gw") == 2) & (pl.col("player_id") == 1))
    )
    result = build_merged_gw(
        snaps, matchstats, PLAYERS, PLAYER_GW_TEAM, TEAM_CODE_TO_NAME
    )
    raya_gw2 = result.filter(
        (pl.col("element") == 1) & (pl.col("GW") == 2)
    ).row(0, named=True)
    assert raya_gw2["minutes"] == 0


def test_build_merged_gw_uses_per_gameweek_team() -> None:
    """Team comes from the per-gameweek snapshot, not static team_code."""
    result = build_merged_gw(
        SNAPSHOTS, MATCHSTATS, PLAYERS, PLAYER_GW_TEAM, TEAM_CODE_TO_NAME
    )
    # Player 2 (static team_code 43 = Man City) was at Arsenal in GW1.
    p2_gw1 = result.filter((pl.col("element") == 2) & (pl.col("GW") == 1)).row(
        0, named=True
    )
    assert p2_gw1["team"] == "Arsenal"
    p2_gw2 = result.filter((pl.col("element") == 2) & (pl.col("GW") == 2)).row(
        0, named=True
    )
    assert p2_gw2["team"] == "Man City"


def test_build_merged_gw_falls_back_to_static_team() -> None:
    """A player absent from the per-gameweek table keeps the static team."""
    # Drop player 1 from the per-gameweek table entirely.
    partial = PLAYER_GW_TEAM.filter(pl.col("element") != 1)
    result = build_merged_gw(
        SNAPSHOTS, MATCHSTATS, PLAYERS, partial, TEAM_CODE_TO_NAME
    )
    p1_gw1 = result.filter((pl.col("element") == 1) & (pl.col("GW") == 1)).row(
        0, named=True
    )
    assert p1_gw1["team"] == "Arsenal"  # static team_code 3


def test_gw_number_from_path() -> None:
    """Gameweek number is parsed from a By-Gameweek path."""
    extractor = FciExtractor(
        api_client=GitHubAPIClient(
            api_key="k", owner="o", repo="r", branch="main"
        )
    )
    path = "data/2025-2026/By Gameweek/GW7/player_gameweek_stats.csv"
    assert extractor._gw_number_from_path(path) == 7


def test_list_gameweeks_extracts_sorted_unique_gws(
    mocker: MockerFixture,
) -> None:
    """list_gameweeks returns sorted unique GW numbers for the given season."""
    extractor = FciExtractor(
        api_client=GitHubAPIClient(
            api_key="k", owner="o", repo="r", branch="main"
        )
    )
    tree = {
        "tree": [
            {
                "path": "data/2025-2026/By Gameweek/GW2/player_gameweek_stats.csv"
            },
            {"path": "data/2025-2026/By Gameweek/GW2/playermatchstats.csv"},
            {
                "path": "data/2025-2026/By Gameweek/GW1/player_gameweek_stats.csv"
            },
            {"path": "data/2025-2026/players.csv"},
            {
                "path": "data/2024-2025/By Gameweek/GW1/player_gameweek_stats.csv"
            },
        ]
    }
    mocker.patch.object(
        extractor.api_client, "get_all_repo_files", return_value=tree
    )
    assert extractor.list_gameweeks("2025-2026") == [1, 2]


def test_fetch_season_frames_raises_when_no_gameweeks(
    mocker: MockerFixture,
) -> None:
    """A season with no gameweek folders raises a clear error."""
    extractor = FciExtractor(
        api_client=GitHubAPIClient(
            api_key="k", owner="o", repo="r", branch="main"
        ),
        fpl_api=mocker.Mock(),
    )
    mocker.patch.object(extractor, "list_gameweeks", return_value=[])
    with pytest.raises(ValueError, match="No gameweek data found"):
        extractor.fetch_season_frames("2025-2026")


def test_fetch_season_frames_uses_supplied_gameweeks(
    mocker: MockerFixture,
) -> None:
    """An explicit gameweek list is used instead of calling list_gameweeks.

    Parameters
    ----------
    mocker : MockerFixture
        Pytest fixture for mocking.
    """
    extractor = FciExtractor(
        api_client=GitHubAPIClient(
            api_key="k", owner="o", repo="r", branch="main"
        ),
        fpl_api=mocker.Mock(),
    )
    listed = mocker.patch.object(
        extractor, "list_gameweeks", return_value=[1, 2, 3]
    )

    def fake_read_csv(path: str) -> pl.DataFrame:
        """Return a minimally-shaped frame for whichever file is asked for.

        Parameters
        ----------
        path : str
            Repo-relative CSV path.

        Returns
        -------
        pl.DataFrame
            A one-row frame carrying the columns the adapter selects.
        """
        if "player_gameweek_stats" in path:
            return pl.DataFrame(
                {
                    "id": [1],
                    "first_name": ["David"],
                    "second_name": ["Raya"],
                    "now_cost": [6.0],
                    "event_points": [6],
                    "bonus": [1],
                }
            )
        if "playermatchstats" in path:
            return pl.DataFrame(
                {
                    "player_id": [1],
                    "minutes_played": [90],
                    "match_id": ["26-27-prem-arsenal-vs-chelsea"],
                }
            )
        return pl.DataFrame(
            {"player_id": [1], "position": ["Goalkeeper"], "team_code": [3]}
        )

    read = mocker.patch.object(
        extractor, "_read_csv", side_effect=fake_read_csv
    )

    extractor.fetch_season_frames("2026-2027", gameweeks=[2])

    listed.assert_not_called()
    paths = [call.args[0] for call in read.call_args_list]
    assert any("GW2/player_gameweek_stats.csv" in p for p in paths)
    assert not any("GW1/" in p for p in paths)
    assert not any("GW3/" in p for p in paths)


def test_build_current_season_merged_gw_upserts_to_db(
    mocker: MockerFixture, tmp_path
) -> None:
    """End-to-end build upserts player-week rows readable via load_player_week."""
    from fantasy_football.storage.database import (
        PLAYER_WEEK_COLUMNS,
        get_connection,
        load_player_week,
    )

    extractor = FciExtractor(
        api_client=GitHubAPIClient(
            api_key="k", owner="o", repo="r", branch="main"
        ),
        fpl_api=mocker.Mock(),
    )
    mocker.patch.object(
        extractor,
        "fetch_season_frames",
        return_value=(SNAPSHOTS, MATCHSTATS, PLAYERS),
    )
    mocker.patch.object(
        extractor, "_team_code_to_name", return_value=TEAM_CODE_TO_NAME
    )
    mocker.patch.object(
        extractor.fpl_cache,
        "build_player_gw_team",
        return_value=PLAYER_GW_TEAM,
    )
    mocker.patch.object(extractor, "list_gameweeks", return_value=[1, 2])
    mocker.patch.object(
        extractor.fpl_cache, "played_gameweeks", return_value=[1, 2]
    )

    connection = get_connection(tmp_path / "t.duckdb")
    try:
        extractor.build_current_season_merged_gw("2025-26", connection)
        stored = load_player_week(connection)
    finally:
        connection.close()

    assert stored.columns == PLAYER_WEEK_COLUMNS
    assert stored.height == 4
    assert stored["season"].unique().to_list() == ["2025-26"]


def test_fetch_season_frames_drops_inconsistent_unused_columns(
    mocker: MockerFixture,
) -> None:
    """Per-GW files with extra inconsistent-dtype columns still concatenate.

    FCI's ``player_gameweek_stats`` files infer unused rank/per-90 columns as
    ``Float64`` in some gameweeks and ``String`` in others. ``fetch_season_frames``
    must keep only the columns the adapter needs so the diagonal concat does
    not raise a ``SchemaError``.

    Parameters
    ----------
    mocker : MockerFixture
        Pytest fixture for mocking.
    """
    extractor = FciExtractor(
        api_client=GitHubAPIClient(
            api_key="k", owner="o", repo="r", branch="main"
        ),
        fpl_api=mocker.Mock(),
    )
    mocker.patch.object(extractor, "list_gameweeks", return_value=[1, 2])

    def fake_read_csv(path: str) -> pl.DataFrame:
        if "player_gameweek_stats" in path:
            is_gw1 = "GW1/" in path
            # Unused column: Float64 in GW1, String in GW2 (mirrors real FCI).
            unused = [1.0, 2.0] if is_gw1 else ["", ""]
            # Needed column drift: bonus is Int64 in GW1, Float64 in GW2.
            bonus = [1, 3] if is_gw1 else [1.0, 3.0]
            return pl.DataFrame(
                {
                    "id": [1, 2],
                    "first_name": ["David", "Erling"],
                    "second_name": ["Raya", "Haaland"],
                    "now_cost": [6.0, 14.0],
                    "event_points": [6, 9],
                    "bonus": bonus,
                    "creativity_rank": unused,
                }
            )
        if "playermatchstats" in path:
            return pl.DataFrame(
                {
                    "player_id": [1, 2],
                    "minutes_played": [90, 90],
                    "match_id": ["m", "m"],
                }
            )
        return pl.DataFrame(
            {
                "player_id": [1, 2],
                "position": ["Goalkeeper", "Forward"],
                "team_code": [3, 43],
            }
        )

    mocker.patch.object(extractor, "_read_csv", side_effect=fake_read_csv)

    snapshots, matchstats, _players = extractor.fetch_season_frames(
        "2025-2026"
    )

    assert snapshots.height == 4  # 2 players x 2 gameweeks
    assert "creativity_rank" not in snapshots.columns
    assert set(snapshots.columns) >= {
        "gw",
        "id",
        "first_name",
        "second_name",
        "now_cost",
        "event_points",
        "bonus",
    }
    # bonus drifts Int64/Float64 across gameweeks but is normalised to Int64.
    assert snapshots.schema["bonus"] == pl.Int64
    assert matchstats.height == 4


def _preseason_extractor(mocker: MockerFixture) -> FciExtractor:
    """Build an extractor whose season has 38 folders but nothing played.

    Parameters
    ----------
    mocker : MockerFixture
        Pytest fixture for mocking.

    Returns
    -------
    FciExtractor
        Extractor with FCI and fplcache calls stubbed.
    """
    extractor = FciExtractor(
        api_client=GitHubAPIClient(
            api_key="k", owner="o", repo="r", branch="main"
        ),
        fpl_api=mocker.Mock(),
    )
    mocker.patch.object(
        extractor, "list_gameweeks", return_value=list(range(1, 39))
    )
    mocker.patch.object(
        extractor.fpl_cache, "played_gameweeks", return_value=[]
    )
    return extractor


def test_build_current_season_ingests_nothing_before_season_starts(
    mocker: MockerFixture, tmp_path
) -> None:
    """Pre-season: no rows are written and the database is left untouched.

    Parameters
    ----------
    mocker : MockerFixture
        Pytest fixture for mocking.
    tmp_path : pathlib.Path
        Pytest temporary directory fixture.
    """
    from fantasy_football.storage.database import (
        get_connection,
        load_player_week,
    )

    extractor = _preseason_extractor(mocker)
    fetch = mocker.patch.object(extractor, "fetch_season_frames")
    build_team = mocker.patch.object(
        extractor.fpl_cache, "build_player_gw_team"
    )

    connection = get_connection(tmp_path / "t.duckdb")
    try:
        result = extractor.build_current_season_merged_gw(
            "2026-27", connection
        )
        stored = load_player_week(connection)
    finally:
        connection.close()

    assert result.height == 0
    assert stored.height == 0
    fetch.assert_not_called()
    build_team.assert_not_called()


def test_build_current_season_does_not_wipe_existing_rows(
    mocker: MockerFixture, tmp_path
) -> None:
    """A no-op run must not delete rows a previous run stored.

    ``upsert_current_season`` is delete-then-insert, so calling it with an
    empty frame would blank the season. It must be skipped entirely.

    Parameters
    ----------
    mocker : MockerFixture
        Pytest fixture for mocking.
    tmp_path : pathlib.Path
        Pytest temporary directory fixture.
    """
    from fantasy_football.storage.database import (
        get_connection,
        load_player_week,
        upsert_current_season,
    )

    merged = build_merged_gw(
        SNAPSHOTS, MATCHSTATS, PLAYERS, PLAYER_GW_TEAM, TEAM_CODE_TO_NAME
    )
    seeded = merged.rename({"GW": "gw"}).with_columns(
        pl.lit("2026-27").alias("season")
    )

    extractor = _preseason_extractor(mocker)
    mocker.patch.object(extractor, "fetch_season_frames")

    connection = get_connection(tmp_path / "t.duckdb")
    try:
        upsert_current_season(connection, seeded, "2026-27")
        extractor.build_current_season_merged_gw("2026-27", connection)
        stored = load_player_week(connection)
    finally:
        connection.close()

    assert stored.height == 4


def test_build_current_season_does_not_wipe_rows_when_no_prem_matches(
    mocker: MockerFixture, tmp_path
) -> None:
    """A non-empty candidate list with no prem matches leaves rows intact.

    Deadlines can pass for gameweeks FCI has not populated with Premier
    League match data yet. ``_gameweeks_with_prem_matches`` then returns an
    empty list even though ``played_gameweeks`` was non-empty, and that
    second early return must also skip ``upsert_current_season`` rather
    than wiping the season with an empty frame.

    Parameters
    ----------
    mocker : MockerFixture
        Pytest fixture for mocking.
    tmp_path : pathlib.Path
        Pytest temporary directory fixture.
    """
    from fantasy_football.storage.database import (
        get_connection,
        load_player_week,
        upsert_current_season,
    )

    merged = build_merged_gw(
        SNAPSHOTS, MATCHSTATS, PLAYERS, PLAYER_GW_TEAM, TEAM_CODE_TO_NAME
    )
    seeded = merged.rename({"GW": "gw"}).with_columns(
        pl.lit("2026-27").alias("season")
    )

    non_prem_matchstats = pl.DataFrame(
        {
            "gw": [1, 2],
            "player_id": [1, 1],
            "match_id": [
                "26-27-fa-cup-arsenal-vs-chelsea",
                "26-27-fa-cup-arsenal-vs-chelsea",
            ],
            "minutes_played": [90, 90],
        },
        schema_overrides={"gw": pl.Int32},
    )

    extractor = FciExtractor(
        api_client=GitHubAPIClient(
            api_key="k", owner="o", repo="r", branch="main"
        ),
        fpl_api=mocker.Mock(),
    )
    mocker.patch.object(extractor, "list_gameweeks", return_value=[1, 2])
    mocker.patch.object(
        extractor.fpl_cache, "played_gameweeks", return_value=[1, 2]
    )
    mocker.patch.object(
        extractor,
        "fetch_season_frames",
        return_value=(SNAPSHOTS, non_prem_matchstats, PLAYERS),
    )
    build_team = mocker.patch.object(
        extractor.fpl_cache, "build_player_gw_team"
    )

    connection = get_connection(tmp_path / "t.duckdb")
    try:
        upsert_current_season(connection, seeded, "2026-27")
        extractor.build_current_season_merged_gw("2026-27", connection)
        stored = load_player_week(connection)
    finally:
        connection.close()

    assert stored.height == 4
    build_team.assert_not_called()


def test_build_current_season_drops_gameweeks_with_no_match_rows(
    mocker: MockerFixture, tmp_path
) -> None:
    """A deadline-passed gameweek FCI has not populated yet is excluded.

    GW2's snapshot rows exist but no Premier League match rows back them, so
    they must not become player-week rows.

    Parameters
    ----------
    mocker : MockerFixture
        Pytest fixture for mocking.
    tmp_path : pathlib.Path
        Pytest temporary directory fixture.
    """
    from fantasy_football.storage.database import (
        get_connection,
        load_player_week,
    )

    extractor = FciExtractor(
        api_client=GitHubAPIClient(
            api_key="k", owner="o", repo="r", branch="main"
        ),
        fpl_api=mocker.Mock(),
    )
    gw1_only = MATCHSTATS.filter(pl.col("gw") == 1)
    mocker.patch.object(extractor, "list_gameweeks", return_value=[1, 2])
    mocker.patch.object(
        extractor.fpl_cache, "played_gameweeks", return_value=[1, 2]
    )
    mocker.patch.object(
        extractor,
        "fetch_season_frames",
        return_value=(SNAPSHOTS, gw1_only, PLAYERS),
    )
    mocker.patch.object(
        extractor, "_team_code_to_name", return_value=TEAM_CODE_TO_NAME
    )
    build_team = mocker.patch.object(
        extractor.fpl_cache,
        "build_player_gw_team",
        return_value=PLAYER_GW_TEAM.filter(pl.col("gw") == 1),
    )

    connection = get_connection(tmp_path / "t.duckdb")
    try:
        extractor.build_current_season_merged_gw("2026-27", connection)
        stored = load_player_week(connection)
    finally:
        connection.close()

    assert stored["gw"].unique().to_list() == [1]
    assert stored.height == 2
    assert build_team.call_args.args[1] == [1]


def test_build_current_season_excludes_non_prem_only_gameweeks(
    mocker: MockerFixture, tmp_path
) -> None:
    """A gameweek whose only match rows are cup ties is not ingested.

    Parameters
    ----------
    mocker : MockerFixture
        Pytest fixture for mocking.
    tmp_path : pathlib.Path
        Pytest temporary directory fixture.
    """
    from fantasy_football.storage.database import (
        get_connection,
        load_player_week,
    )

    cup_only = pl.DataFrame(
        {
            "gw": [2],
            "player_id": [1],
            "match_id": ["26-27-fa-cup-arsenal-vs-chelsea"],
            "minutes_played": [90],
        },
        schema_overrides={"gw": pl.Int32},
    )
    matchstats = pl.concat(
        [MATCHSTATS.filter(pl.col("gw") == 1), cup_only], how="vertical"
    )

    extractor = FciExtractor(
        api_client=GitHubAPIClient(
            api_key="k", owner="o", repo="r", branch="main"
        ),
        fpl_api=mocker.Mock(),
    )
    mocker.patch.object(extractor, "list_gameweeks", return_value=[1, 2])
    mocker.patch.object(
        extractor.fpl_cache, "played_gameweeks", return_value=[1, 2]
    )
    mocker.patch.object(
        extractor,
        "fetch_season_frames",
        return_value=(SNAPSHOTS, matchstats, PLAYERS),
    )
    mocker.patch.object(
        extractor, "_team_code_to_name", return_value=TEAM_CODE_TO_NAME
    )
    mocker.patch.object(
        extractor.fpl_cache,
        "build_player_gw_team",
        return_value=PLAYER_GW_TEAM.filter(pl.col("gw") == 1),
    )

    connection = get_connection(tmp_path / "t.duckdb")
    try:
        extractor.build_current_season_merged_gw("2026-27", connection)
        stored = load_player_week(connection)
    finally:
        connection.close()

    assert stored["gw"].unique().to_list() == [1]


def test_build_current_season_passes_now_to_played_gameweeks(
    mocker: MockerFixture, tmp_path
) -> None:
    """An injected ``now`` reaches the deadline filter.

    Parameters
    ----------
    mocker : MockerFixture
        Pytest fixture for mocking.
    tmp_path : pathlib.Path
        Pytest temporary directory fixture.
    """
    from fantasy_football.storage.database import get_connection

    extractor = _preseason_extractor(mocker)
    played = extractor.fpl_cache.played_gameweeks
    mocker.patch.object(extractor, "fetch_season_frames")
    now = datetime(2026, 7, 30, 12, 0, tzinfo=timezone.utc)

    connection = get_connection(tmp_path / "t.duckdb")
    try:
        extractor.build_current_season_merged_gw(
            "2026-27", connection, now=now
        )
    finally:
        connection.close()

    assert played.call_args.kwargs["now"] == now
