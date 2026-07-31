import logging

import polars as pl

from fantasy_football.constants import DATA_FOLDER, ROLLING_WINDOW
from fantasy_football.storage.tables import PLAYER_SEASON, PLAYER_WEEK

logger = logging.getLogger(__name__)

TRANSFORMED_DATA_FOLDER = DATA_FOLDER.joinpath("transformed")

KNOWN_POSITIONS: tuple[str, ...] = ("GK", "DEF", "MID", "FWD")

# Prefix used to build a fallback rolling-window identity for rows with a
# null player_code. It must contain a non-digit character: a real
# player_code is rendered as bare digits, so a string starting with a
# non-digit character can never equal one, no matter how the two integer
# ranges overlap.
_FALLBACK_IDENTITY_PREFIX = "no_player_code_element_"


def rolling_column_name(rolling_column: str, rolling_window: int) -> str:
    """Return the output column name produced by a rolling-average step."""
    return f"{rolling_column}_rolling_{rolling_window}"


def load_gw_data() -> pl.DataFrame:
    """Load all player-week data across every season, with ``player_code``.

    Reads the entire ``player_week`` table and left-joins the ``player_season``
    identity dimension, so downstream windows can partition on a key that is
    stable across seasons rather than on the display name. Positions are already
    normalised (``GKP`` collapsed to ``GK``) at write time.

    Returns
    -------
    pl.DataFrame
        One row per (player, gameweek) for all seasons, with ``season``, ``gw``
        and ``player_code`` columns.
    """
    return PLAYER_WEEK.load().join(
        PLAYER_SEASON.load().select(["season", "element", "player_code"]),
        on=["season", "element"],
        how="left",
        coalesce=True,
    )


def add_rolling_identity_column(data: pl.DataFrame) -> pl.DataFrame:
    """Add a collision-safe identity key for the rolling-points window.

    ``player_code`` is nullable: a row whose ``(season, element)`` has no
    matching ``player_season`` row -- for example because
    ``load_player_identity_data`` caught a failed source fetch and skipped
    that season -- keeps a null ``player_code``. Polars pools every null
    value in a ``.over()`` partition into a single group, so grouping the
    rolling window directly on ``player_code`` would silently merge every
    such player in a season into one shared series -- the same bug this
    module exists to fix, just triggered by a missing identity rather than a
    shared display name.

    ``element`` is unique only *within* a season -- it is reused across
    seasons -- so a fallback keyed on ``element`` alone would pool two
    different players who both lack a ``player_code`` in different seasons
    into one shared series the moment the window stops partitioning on
    ``season`` (as it now does; see ``create_rolling_points_data``). The
    fallback is therefore scoped to ``(season, element)``: it is a sound
    identity for these rows because ``element`` is unique within a season,
    and pinning it to that season means a fallback player never claims a
    cross-season identity they have no real evidence for -- their window
    simply restarts each season, which is the conservative behaviour given
    the alternative is silently averaging one stranger's points into
    another's history.

    ``element`` and ``player_code`` are drawn from overlapping integer
    ranges, so naively coalescing them risks an unmatched player colliding
    with an unrelated real ``player_code``. This stores the identity as a
    string instead: a real ``player_code`` is rendered as its bare digits,
    while a fallback is prefixed with ``_FALLBACK_IDENTITY_PREFIX``, which
    starts with a non-digit character. A digit-only string can never equal a
    string carrying a non-digit prefix, so the two spaces cannot collide
    regardless of the underlying integer or season values.

    Parameters
    ----------
    data : pl.DataFrame
        Frame with ``player_code``, ``element`` and ``season`` columns.

    Returns
    -------
    pl.DataFrame
        ``data`` with a ``rolling_identity`` string column added.
    """
    return data.with_columns(
        pl.when(pl.col("player_code").is_not_null())
        .then(pl.col("player_code").cast(pl.Utf8))
        .otherwise(
            pl.lit(_FALLBACK_IDENTITY_PREFIX)
            + pl.col("season")
            + pl.lit("_")
            + pl.col("element").cast(pl.Utf8)
        )
        .alias("rolling_identity")
    )


def create_rolling_average_column(
    data: pl.DataFrame,
    grouping_columns: list[str],
    rolling_column: str,
    rolling_window: int,
) -> pl.DataFrame:
    """Create a rolling average column over a given window size.

    The frame is sorted by season and gameweek, then ``rolling_column`` is
    averaged over ``rolling_window`` rows within each ``grouping_columns``
    partition.

    Parameters
    ----------
    data : pl.DataFrame
        The data to calculate the rolling average on.
    grouping_columns : list[str]
        The columns to partition the window by. Include ``season`` to stop a
        window spanning the summer break.
    rolling_column : str
        The column to calculate the rolling average on.
    rolling_window : int
        The window size to calculate the rolling average over.

    Returns
    -------
    pl.DataFrame
        The original dataframe with the rolling average column added.

    """
    rolling_average_column_name = rolling_column_name(
        rolling_column, rolling_window
    )
    data = data.sort(["season", "gw"]).with_columns(
        pl.col(rolling_column)
        .rolling_mean(window_size=rolling_window, min_periods=1)
        .over(grouping_columns)
        .alias(rolling_average_column_name)
    )
    return data


def fill_missing_values_by_position(
    data: pl.DataFrame, column_to_fill: str
) -> pl.DataFrame:
    """Fill missing values in a column by the average of the position.

    Parameters
    ----------
    data : pl.DataFrame
        The data to fill missing values on.
    column_to_fill : str
        The column to fill missing values for.

    Returns
    -------
    pl.DataFrame
        The original dataframe with the missing values filled.

    """
    positions_in_data = set(data.get_column("position").unique().to_list())
    unknown_positions = positions_in_data - set(KNOWN_POSITIONS)
    if unknown_positions:
        logger.warning(
            "fill_missing_values_by_position encountered unknown "
            "position(s) %s; rows with these positions will not have "
            "nulls in '%s' filled.",
            sorted(unknown_positions),
            column_to_fill,
        )
    for position in KNOWN_POSITIONS:
        position_data = data.filter(pl.col("position") == position)
        position_average = (
            position_data.select(pl.col(column_to_fill)).mean().item(0, 0)
        )
        data = data.with_columns(
            pl.when(
                (pl.col("position") == position)
                & (pl.col(column_to_fill).is_null())
            )
            .then(pl.lit(position_average))
            .otherwise(pl.col(column_to_fill))
            .alias(column_to_fill)
        )
    return data


def create_rolling_points_data(
    current_season: str, rolling_window: int = ROLLING_WINDOW
) -> None:
    """Create a rolling average column for player points over a given window.

    This function first creates a whole history of game week data by loading
    all previous seasons and the current season. It then calculates the rolling
    average of total points over a window size of `rolling_window`. Finally,
    it fills any missing values by the average of the position. Doesn't return
    anything, but writes the data to a CSV file in the transformed data folder
    named "rolling_points.csv".

    The rolling window uses ``min_periods=1``, so a row with fewer than
    ``rolling_window`` prior games in its partition gets a genuine (if thin)
    average over whatever exists rather than a null -- meaning fewer rows
    fall through to ``fill_missing_values_by_position``'s positional
    fallback than before that was added.

    The window partitions on ``rolling_identity`` alone, so it spans the
    summer break: a new season's opening gameweeks average in the tail of the
    previous one. This is what gives a pre-season gameweek any form to draw
    on at all, and it matches the cross-season rolling-minutes window. The
    frame is sorted ``["season", "gw"]`` and season strings sort
    chronologically, so within-partition row order is correct.

    Parameters
    ----------
    current_season : str
        The current season we are working with. Should be in a YYYY-YY format,
        e.g. "2020-21"
    rolling_window : int, optional
        The window size to calculate the rolling average over. Defaults to 5.

    """
    gw_data = load_gw_data()
    rolling_column = rolling_column_name("total_points", rolling_window)
    gw_data = add_rolling_identity_column(gw_data)
    gw_data = create_rolling_average_column(
        gw_data, ["rolling_identity"], "total_points", rolling_window
    )
    gw_data = gw_data.drop("rolling_identity")
    gw_data = fill_missing_values_by_position(gw_data, rolling_column)
    TRANSFORMED_DATA_FOLDER.mkdir(exist_ok=True, parents=True)
    gw_data.write_csv(TRANSFORMED_DATA_FOLDER.joinpath("rolling_points.csv"))
