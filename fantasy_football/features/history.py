"""Prior-season features, joined across seasons through ``player_code``.

FPL reassigns ``element`` ids each season, so any feature that reaches back
past the summer break has to go through the stable ``player_code`` in the
``player_season`` dimension. Everything here describes seasons strictly before
the row's own season, so none of it leaks: a completed season is entirely in
the past relative to every gameweek of the season being scored.
"""

import logging

import polars as pl

logger = logging.getLogger(__name__)

# Columns add_history_features appends, in the order it appends them.
HISTORY_FEATURES: list[str] = [
    "prev_season_minutes",
    "prev_season_start_rate",
    "prev_season_points_per_start",
    "pl_seasons_played",
    "seasons_since_last_pl",
    "is_pl_newcomer",
]

# A match counts as a start for the purposes of prev_season_start_rate when it
# reaches the same 60-minute threshold the minutes model predicts.
START_MINUTES: int = 60


def _season_start_year(column: str) -> pl.Expr:
    """Return an expression parsing ``"2023-24"`` into the integer ``2023``.

    Seasons are stored as strings, but gap arithmetic needs a number. The first
    four characters are always the starting year.

    Parameters
    ----------
    column : str
        Name of the short-form season-string column.

    Returns
    -------
    pl.Expr
        An ``Int64`` expression.
    """
    return pl.col(column).str.slice(0, 4).cast(pl.Int64)


def _season_totals(
    player_match: pl.DataFrame, player_season: pl.DataFrame
) -> pl.DataFrame:
    """Aggregate per-match rows into one row per (player_code, season).

    Aggregating from ``player_match`` rather than ``player_week`` is deliberate:
    ``player_week`` collapses double gameweeks, so a start rate computed from it
    would treat two matches in one gameweek as one.

    Parameters
    ----------
    player_match : pl.DataFrame
        Per-fixture rows with ``season``, ``element``, ``minutes`` and
        ``total_points``.
    player_season : pl.DataFrame
        Identity rows with ``season``, ``element`` and ``player_code``.

    Returns
    -------
    pl.DataFrame
        Columns ``player_code``, ``season``, ``season_start``,
        ``season_minutes``, ``season_start_rate``, ``season_points_per_start``.
    """
    joined = player_match.join(
        player_season.select(["season", "element", "player_code"]),
        on=["season", "element"],
        how="inner",
        coalesce=True,
    )
    return (
        joined.group_by(["player_code", "season"])
        .agg(
            pl.col("minutes").sum().alias("season_minutes"),
            (pl.col("minutes") >= START_MINUTES)
            .mean()
            .alias("season_start_rate"),
            pl.col("total_points")
            .filter(pl.col("minutes") >= START_MINUTES)
            .mean()
            .alias("season_points_per_start"),
        )
        .with_columns(_season_start_year("season").alias("season_start"))
    )


def add_history_features(
    player_week: pl.DataFrame,
    player_match: pl.DataFrame,
    player_season: pl.DataFrame,
) -> pl.DataFrame:
    """Attach ``player_code`` and prior-season features to player-week rows.

    For each row, the ``prev_season_*`` columns describe the most recent season
    strictly before it in which the player made an appearance -- which is not
    necessarily the immediately preceding season. ``seasons_since_last_pl``
    reports how stale that history is, so a model can discount it rather than
    treating a three-year-old season as current.

    Parameters
    ----------
    player_week : pl.DataFrame
        Player-week rows with ``season``, ``gw`` and ``element``.
    player_match : pl.DataFrame
        Per-fixture rows with ``season``, ``element``, ``minutes`` and
        ``total_points``.
    player_season : pl.DataFrame
        Identity rows from ``load_player_season``, with ``season``, ``element``
        and ``player_code``.

    Returns
    -------
    pl.DataFrame
        ``player_week`` with ``player_code`` and every column in
        ``HISTORY_FEATURES`` added. Rows whose ``(season, element)`` has no
        identity row keep a null ``player_code`` and are treated as newcomers.
    """
    frame = player_week.join(
        player_season.select(["season", "element", "player_code"]),
        on=["season", "element"],
        how="left",
        coalesce=True,
    ).with_columns(_season_start_year("season").alias("season_start"))

    totals = _season_totals(player_match, player_season)

    # Cross-join each player's row-seasons against their own played seasons,
    # keep only strictly-prior ones, then take the latest. join_where is not
    # available across all supported polars versions, so this is a plain join
    # on player_code followed by a filter.
    prior = (
        frame.select(["player_code", "season_start"])
        .unique()
        .filter(pl.col("player_code").is_not_null())
        .join(
            totals.rename({"season_start": "prior_start"}).drop("season"),
            on="player_code",
            how="inner",
            coalesce=True,
        )
        .filter(pl.col("prior_start") < pl.col("season_start"))
    )

    latest = (
        prior.sort("prior_start", descending=True)
        .group_by(["player_code", "season_start"])
        .agg(
            pl.col("season_minutes").first().alias("prev_season_minutes"),
            pl.col("season_start_rate")
            .first()
            .alias("prev_season_start_rate"),
            pl.col("season_points_per_start")
            .first()
            .alias("prev_season_points_per_start"),
            pl.col("prior_start").first().alias("last_played_start"),
            pl.len().alias("pl_seasons_played"),
        )
        .with_columns(
            (pl.col("season_start") - pl.col("last_played_start") - 1).alias(
                "seasons_since_last_pl"
            )
        )
        .drop("last_played_start")
    )

    return (
        frame.join(
            latest,
            on=["player_code", "season_start"],
            how="left",
            coalesce=True,
        )
        .with_columns(
            pl.col("pl_seasons_played").fill_null(0).cast(pl.Int64),
        )
        .with_columns(
            (pl.col("pl_seasons_played") == 0).alias("is_pl_newcomer"),
        )
        .drop("season_start")
    )
