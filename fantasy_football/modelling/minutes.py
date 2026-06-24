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
