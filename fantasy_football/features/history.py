import polars as pl

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
        joined.filter(pl.col("player_code").is_not_null())
        .group_by(["player_code", "season"])
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
    strictly before it in which the player had a ``player_match`` row -- that
    is, was in the squad for at least one fixture, not necessarily that they
    were on the pitch. Roughly 60% of ``player_match`` rows are 0-minute squad
    entries, so a bench-bound player correctly gets ``prev_season_minutes`` of
    0 rather than a null that would hide them from the model entirely. This is
    also not necessarily the immediately preceding season.
    ``seasons_since_last_pl`` reports how stale that history is, so a model can
    discount it rather than treating a three-year-old season as current.

    Parameters
    ----------
    player_week : pl.DataFrame
        Player-week rows with ``season``, ``gw`` and ``element``.
    player_match : pl.DataFrame
        Per-fixture rows with ``season``, ``element``, ``minutes`` and
        ``total_points``.
    player_season : pl.DataFrame
        Identity rows from ``PLAYER_SEASON.load()``, with ``season``,
        ``element`` and ``player_code``.

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


# Columns add_cold_start_features appends, in the order it appends them.
COLD_START_FEATURES: list[str] = [
    "age_years",
    "days_since_team_join",
    "is_promoted_club",
]

# Seasons start in August; anchoring age and join-recency at 1 August of the
# starting year keeps both comparable across seasons without needing a fixture
# date per row.
SEASON_ANCHOR_MONTH_DAY: tuple[int, int] = (8, 1)


def add_cold_start_features(
    data: pl.DataFrame, team_fixture: pl.DataFrame
) -> pl.DataFrame:
    """Add the signals that stand in for history when a player has none.

    A player arriving from another league has a ``player_code`` but no Premier
    League past, so every ``prev_season_*`` feature is null for them. These four
    signals are what remains: how old they are, how recently they joined their
    club, and whether that club itself is new to the division. Their FPL price
    -- already a model feature -- carries the rest.

    Parameters
    ----------
    data : pl.DataFrame
        Player rows with ``season``, ``team``, ``birth_date`` and
        ``team_join_date``.
    team_fixture : pl.DataFrame
        Fixture rows with ``season`` and ``team``, used to decide which clubs
        were in the division in the prior season.

    Returns
    -------
    pl.DataFrame
        ``data`` with every column in ``COLD_START_FEATURES`` added.
    """
    month, day = SEASON_ANCHOR_MONTH_DAY
    frame = data.with_columns(
        pl.date(_season_start_year("season"), month, day).alias(
            "_season_anchor"
        )
    )

    frame = frame.with_columns(
        (pl.col("_season_anchor") - pl.col("birth_date"))
        .dt.total_days()
        .truediv(365.25)
        .alias("age_years"),
        (pl.col("_season_anchor") - pl.col("team_join_date"))
        .dt.total_days()
        .cast(pl.Float64)
        .alias("days_since_team_join"),
    )

    # A club is promoted when it has fixtures this season but none last season.
    # The earliest season on record has no prior season to compare against, so
    # every club in it would look promoted; exclude it explicitly.
    seasons_with_fixtures = (
        team_fixture.select("season", "team")
        .unique()
        .with_columns(_season_start_year("season").alias("season_start"))
    )
    earliest_start = seasons_with_fixtures["season_start"].min()
    prior_presence = seasons_with_fixtures.select(
        pl.col("team"),
        (pl.col("season_start") + 1).alias("season_start"),
        pl.lit(True).alias("_in_prior_season"),
    )

    frame = (
        frame.with_columns(_season_start_year("season").alias("season_start"))
        .join(
            prior_presence,
            on=["team", "season_start"],
            how="left",
            coalesce=True,
        )
        .with_columns(
            pl.when(pl.col("team").is_null())
            .then(pl.lit(False))
            .when(pl.col("season_start") <= earliest_start)
            .then(pl.lit(False))
            .otherwise(pl.col("_in_prior_season").is_null())
            .alias("is_promoted_club")
        )
        .drop("_season_anchor", "season_start", "_in_prior_season")
    )
    return frame
