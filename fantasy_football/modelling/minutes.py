"""End-to-end minutes-played classifier: features, CV scoring, MLflow.

A single 3-class model predicts each player's minutes bucket for a match:
benched (``0_minutes``), partial (``1_to_59_minutes``) or a full-ish shift
(``60_minutes_plus``). It is scored on the two decision boundaries downstream
points models care about — probability of any appearance and probability of a
60+ minute appearance — cross-validated with an expanding window over seasons,
then refit on all seasons and logged to MLflow.
"""

import logging

import polars as pl

from fantasy_football.features.availability import (
    add_chance_of_playing,
    add_positional_availability,
)
from fantasy_football.features.valuation import (
    add_positional_value_rank,
    add_team_value,
)
from fantasy_football.storage.database import (
    load_player_availability,
    load_player_match,
    load_player_week,
)

logger = logging.getLogger(__name__)

# Minutes-bucket target labels.
BUCKET_ZERO = "0_minutes"
BUCKET_PARTIAL = "1_to_59_minutes"
BUCKET_SIXTY_PLUS = "60_minutes_plus"
MINUTES_BUCKETS = [BUCKET_ZERO, BUCKET_PARTIAL, BUCKET_SIXTY_PLUS]

# Model inputs.
NUM_FEATURES = [
    "value",
    "value_share_of_team",
    "pos_value_rank",
    "players_same_pos",
    "chance_of_playing_this_round",
    "fit_rivals_same_pos",
    "fit_rivals_ahead",
]
CAT_FEATURES = ["position"]
FEATURES = NUM_FEATURES + CAT_FEATURES

# Representative minutes per bucket, for the expected-minutes leverage metric.
MINUTE_MIDPOINTS = {
    BUCKET_ZERO: 0.0,
    BUCKET_PARTIAL: 30.0,
    BUCKET_SIXTY_PLUS: 75.0,
}
# FPL appearance points: 0 for no game, 1 for <60 mins, 2 for 60+.
APPEARANCE_POINTS = {
    BUCKET_ZERO: 0.0,
    BUCKET_PARTIAL: 1.0,
    BUCKET_SIXTY_PLUS: 2.0,
}


def create_minutes_bucket(
    data: pl.DataFrame, minutes_col: str = "minutes"
) -> pl.DataFrame:
    """Add a ``minutes_bucket`` target derived from a minutes column.

    Buckets are ``0_minutes`` (exactly zero), ``1_to_59_minutes`` (a partial
    appearance) and ``60_minutes_plus`` (a full-ish shift, the clean-sheet /
    appearance-point threshold).

    Parameters
    ----------
    data : pl.DataFrame
        Frame containing ``minutes_col``.
    minutes_col : str
        Name of the integer minutes column to bucket. Defaults to ``minutes``.

    Returns
    -------
    pl.DataFrame
        ``data`` with a ``minutes_bucket`` string column added.
    """
    return data.with_columns(
        pl.when(pl.col(minutes_col) == 0)
        .then(pl.lit(BUCKET_ZERO))
        .when(pl.col(minutes_col) < 60)
        .then(pl.lit(BUCKET_PARTIAL))
        .otherwise(pl.lit(BUCKET_SIXTY_PLUS))
        .alias("minutes_bucket")
    )


def build_feature_frame(
    player_week: pl.DataFrame, availability: pl.DataFrame
) -> pl.DataFrame:
    """Build the model feature frame at the player-week grain.

    Value features (``value_share_of_team``, ``pos_value_rank``,
    ``players_same_pos``) and availability features
    (``chance_of_playing_this_round``, ``fit_rivals_same_pos``,
    ``fit_rivals_ahead``) are computed once per ``(season, gw, element)`` so the
    per-gameweek counts are correct, then narrowed to the columns the model
    consumes plus the join keys and ``position``.

    Parameters
    ----------
    player_week : pl.DataFrame
        Player-week rows from :func:`load_player_week` (``season``, ``gw``,
        ``element``, ``position``, ``team``, ``value`` and more).
    availability : pl.DataFrame
        Availability rows from :func:`load_player_availability`.

    Returns
    -------
    pl.DataFrame
        One row per ``(season, gw, element)`` with ``position`` and every
        column in ``NUM_FEATURES``.
    """
    frame = add_team_value(player_week)
    frame = add_positional_value_rank(frame)
    frame = add_chance_of_playing(frame, availability)
    frame = add_positional_availability(frame)
    return frame.select(
        ["season", "gw", "element", "position", *NUM_FEATURES]
    )


def build_model_frame(
    player_match: pl.DataFrame, feature_frame: pl.DataFrame
) -> pl.DataFrame:
    """Join features onto match rows and derive the bucket target.

    Match-level minutes from ``player_match`` define the target (so double
    gameweeks contribute one row per match). Features are inner-joined on
    ``(season, gw, element)`` — match rows with no feature row are dropped.

    Parameters
    ----------
    player_match : pl.DataFrame
        Match rows from :func:`load_player_match` (``season``, ``gw``,
        ``element``, ``minutes`` and more).
    feature_frame : pl.DataFrame
        Output of :func:`build_feature_frame`.

    Returns
    -------
    pl.DataFrame
        Columns ``season``, ``gw``, ``element``, ``minutes``,
        ``minutes_bucket`` and every column in ``FEATURES``.
    """
    joined = player_match.select(
        ["season", "gw", "element", "minutes"]
    ).join(feature_frame, on=["season", "gw", "element"], how="inner")
    joined = create_minutes_bucket(joined)
    return joined.select(
        ["season", "gw", "element", "minutes", "minutes_bucket", *FEATURES]
    )


def assemble_model_frame() -> pl.DataFrame:
    """Load player data from the database and build the model frame.

    Returns
    -------
    pl.DataFrame
        The match-level model frame from :func:`build_model_frame`.
    """
    feature_frame = build_feature_frame(
        load_player_week(), load_player_availability()
    )
    return build_model_frame(load_player_match(), feature_frame)
