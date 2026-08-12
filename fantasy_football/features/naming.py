"""Shared naming helpers for the rolling feature windows.

The match-form and team-form modules both build rolling columns and both
need to agree on what those columns are called, and on how a player with no
``player_code`` is identified. Those two facts live here so neither module
owns them.
"""

# Prefix used to build a fallback rolling-window identity for rows with a
# null player_code. It must contain a non-digit character: a real
# player_code is rendered as bare digits, so a string starting with a
# non-digit character can never equal one, no matter how the two integer
# ranges overlap.
FALLBACK_IDENTITY_PREFIX = "no_player_code_element_"


def rolling_column_name(rolling_column: str, rolling_window: int) -> str:
    """Return the output column name produced by a rolling-average step."""
    return f"{rolling_column}_rolling_{rolling_window}"
