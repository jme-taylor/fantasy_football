import polars as pl

# Transfermarkt's granular positions, grouped into the pecking orders a
# club actually rotates within. A left winger competes with a right
# winger for a wide berth, not with a centre-back.
TM_POSITION_GROUPS: dict[str, str] = {
    "Attacking Midfield": "AM",
    "Centre-Back": "CB",
    "Centre-Forward": "ST",
    "Right-Back": "RB",
    "Goalkeeper": "GK",
    "Defensive Midfield": "CM",
    "Central Midfield": "CM",
    "Left-Back": "LB",
    "Left Midfield": "LM",
    "Left Winger": "LM",
    "Right Midfield": "RM",
    "Right Winger": "RM",
    "Second Striker": "AM",
}

# How far back a rival's arrival still counts as competition.
ARRIVAL_LOOKBACK_DAYS = 365


def gameweek_kickoff(match_stream: pl.DataFrame) -> pl.DataFrame:
    """Reduce a match stream to one as-of date per player-gameweek.

    The earliest kickoff in the gameweek is the state entering it, so
    both legs of a double gameweek share a date. Unplayed fixtures carry
    a real kickoff, which is what lets the calendar-driven features move
    forward while the match-derived ones freeze.

    Parameters
    ----------
    match_stream : pl.DataFrame
        Match-grain rows with ``season``, ``gw``, ``element`` and
        ``kickoff_time``.

    Returns
    -------
    pl.DataFrame
        One row per ``(season, gw, element)`` with an ``as_of_date``.
    """
    return (
        match_stream.group_by(["season", "gw", "element"])
        .agg(pl.col("kickoff_time").min().alias("as_of"))
        .with_columns(pl.col("as_of").dt.date().alias("as_of_date"))
        .drop("as_of")
    )


def add_tm_identity(
    data: pl.DataFrame,
    player_season: pl.DataFrame,
    tm_player_map: pl.DataFrame,
    tm_player: pl.DataFrame,
) -> pl.DataFrame:
    """Attach each player's Transfermarkt id, position and position group.

    The bridge is ``player_code`` -> ``tm_player_id``: FPL reassigns
    element ids each season, so the map is keyed on the stable code. A
    player with no Transfermarkt match keeps nulls rather than being
    dropped -- the model reads a missing value as its own signal.

    Parameters
    ----------
    data : pl.DataFrame
        Player-week rows with ``season``, ``gw`` and ``element``.
    player_season : pl.DataFrame
        Identity rows with ``season``, ``element`` and ``player_code``.
    tm_player_map : pl.DataFrame
        Bridge rows with ``player_code`` and ``tm_player_id``.
    tm_player : pl.DataFrame
        Transfermarkt players with ``tm_player_id`` and ``position``.

    Returns
    -------
    pl.DataFrame
        ``data`` with ``player_code``, ``tm_player_id``, ``tm_position``
        and ``tm_position_group`` added.
    """
    positions = tm_player.select(
        "tm_player_id", pl.col("position").alias("tm_position")
    ).with_columns(
        pl.col("tm_position")
        .replace_strict(TM_POSITION_GROUPS, default=None)
        .alias("tm_position_group")
    )
    return (
        data.join(
            player_season.select(["season", "element", "player_code"]),
            on=["season", "element"],
            how="left",
            coalesce=True,
        )
        .join(
            tm_player_map.select(["player_code", "tm_player_id"]),
            on="player_code",
            how="left",
            coalesce=True,
        )
        .join(positions, on="tm_player_id", how="left", coalesce=True)
    )


def _value_spells(tm_market_value: pl.DataFrame) -> pl.DataFrame:
    """Turn dated valuations into half-open intervals to join as-of."""
    return (
        tm_market_value.select(["tm_player_id", "value_date", "value_eur"])
        .sort(["tm_player_id", "value_date"])
        .with_columns(
            pl.col("value_date")
            .shift(-1)
            .over("tm_player_id")
            .alias("value_date_end")
        )
    )


def add_tm_value(
    data: pl.DataFrame,
    kickoffs: pl.DataFrame,
    tm_market_value: pl.DataFrame,
) -> pl.DataFrame:
    """Add the player's Transfermarkt valuation as of the fixture.

    Calendar-driven, so this reads the gameweek's own kickoff rather than
    freezing at the last played match: a valuation published in July is a
    known fact when a GW1 fixture is predicted in August, and freezing it
    would hand the model a stale May price.

    Parameters
    ----------
    data : pl.DataFrame
        Player-week rows carrying ``tm_player_id``.
    kickoffs : pl.DataFrame
        Output of :func:`gameweek_kickoff`.
    tm_market_value : pl.DataFrame
        Valuation rows with ``tm_player_id``, ``value_date``, ``value_eur``.

    Returns
    -------
    pl.DataFrame
        ``data`` with ``as_of_date`` and ``value_tm`` added. Players with
        no valuation on or before the date keep a null.
    """
    spells = _value_spells(tm_market_value)
    data = data.join(
        kickoffs, on=["season", "gw", "element"], how="left", coalesce=True
    )
    matched = (
        data.select(["season", "gw", "element", "tm_player_id", "as_of_date"])
        .filter(pl.col("tm_player_id").is_not_null())
        .join(spells, on="tm_player_id", how="inner")
        .filter(
            (pl.col("value_date") <= pl.col("as_of_date"))
            & (
                pl.col("value_date_end").is_null()
                | (pl.col("as_of_date") < pl.col("value_date_end"))
            )
        )
        .group_by(["season", "gw", "element"])
        .agg(pl.col("value_eur").last().alias("value_tm"))
    )
    return data.join(
        matched, on=["season", "gw", "element"], how="left", coalesce=True
    )


def add_tm_position_value_rank(data: pl.DataFrame) -> pl.DataFrame:
    """Rank players by Transfermarkt value within their club-position group.

    Mirrors :func:`add_positional_value_rank`, but partitions on the
    Transfermarkt position group and ranks on the Transfermarkt value.
    A missing valuation is not the same as being the cheapest, so those
    players count towards the group size but are not ranked within it.

    Parameters
    ----------
    data : pl.DataFrame
        Player-week rows with ``season``, ``gw``, ``team``,
        ``tm_position_group`` and ``value_tm``.

    Returns
    -------
    pl.DataFrame
        ``data`` with ``players_same_tm_pos``, ``tm_pos_value_share`` and
        ``tm_pos_value_rank_norm`` added.
    """
    partition = ["season", "gw", "team", "tm_position_group"]
    group_total = pl.col("value_tm").sum().over(partition)
    rank = (
        pl.when(pl.col("value_tm").is_null())
        .then(None)
        .otherwise(
            pl.col("value_tm")
            .rank(method="min", descending=True)
            .over(partition)
        )
        .cast(pl.Float64)
    )
    data = data.with_columns(
        pl.len().over(partition).cast(pl.Int64).alias("players_same_tm_pos"),
        pl.when(group_total > 0)
        .then(pl.col("value_tm") / group_total)
        .otherwise(None)
        .alias("tm_pos_value_share"),
        rank.alias("_tm_rank"),
    )
    # Rank is not comparable across group sizes: 3rd of 3 and 3rd of 8
    # differ. A player alone in their group is top of it, so 0.
    return data.with_columns(
        pl.when(pl.col("_tm_rank").is_null())
        .then(None)
        .when(pl.col("players_same_tm_pos") > 1)
        .then(
            (pl.col("_tm_rank") - 1.0) / (pl.col("players_same_tm_pos") - 1.0)
        )
        .otherwise(0.0)
        .alias("tm_pos_value_rank_norm")
    ).drop("_tm_rank")


def add_prev_game_minutes(
    data: pl.DataFrame,
    match_stream: pl.DataFrame,
    player_season: pl.DataFrame,
) -> pl.DataFrame:
    """Add the minutes played in each of the two previous matches.

    Keyed on ``player_code`` rather than ``element`` and ordered by real
    kickoff time, so the window is continuous across seasons and a
    reassigned element id never merges two people.

    Unplayed fixtures freeze: each carries the values from the player's
    last *played* match and does not advance, however far into the future
    the fixture is. ``days_since_prev_game`` freezes with them, so a
    forward row reports the gap between the last two played matches
    rather than the gap to its own kickoff.

    Parameters
    ----------
    data : pl.DataFrame
        Player-week rows with ``season``, ``gw`` and ``element``.
    match_stream : pl.DataFrame
        Match-grain rows with ``season``, ``gw``, ``element``,
        ``kickoff_time`` and ``minutes``. Unplayed fixtures carry a null
        ``minutes``.
    player_season : pl.DataFrame
        Identity rows with ``season``, ``element`` and ``player_code``.

    Returns
    -------
    pl.DataFrame
        ``data`` with ``prev_game_minutes``, ``prev_game_minutes_2`` and
        ``days_since_prev_game`` added.
    """
    outputs = [
        "prev_game_minutes",
        "prev_game_minutes_2",
        "days_since_prev_game",
    ]
    stream = (
        match_stream.select(
            ["season", "gw", "element", "kickoff_time", "minutes"]
        )
        .join(
            player_season.select(["season", "element", "player_code"]),
            on=["season", "element"],
            how="left",
            coalesce=True,
        )
        .filter(pl.col("player_code").is_not_null())
        # nulls_last, as in add_rolling_minutes: an unparsed kickoff must
        # sort as the most recent match, never the earliest.
        .sort("kickoff_time", nulls_last=True)
        .with_row_index("_row")
        .with_columns(pl.col("kickoff_time").dt.date().alias("_date"))
    )
    # The frozen values an unplayed fixture inherits, computed over the
    # played rows alone so intervening unplayed fixtures cannot shift the
    # lag, then forward-filled along each player's timeline.
    played = stream.filter(pl.col("minutes").is_not_null())
    carry = played.select(
        "_row",
        pl.col("minutes").alias("_carry_prev"),
        pl.col("minutes").shift(1).over("player_code").alias("_carry_prev_2"),
        (pl.col("_date") - pl.col("_date").shift(1))
        .dt.total_days()
        .over("player_code")
        .alias("_carry_days"),
    )
    stream = stream.join(carry, on="_row", how="left", coalesce=True)
    stream = stream.with_columns(
        [
            pl.col(column).forward_fill().over("player_code")
            for column in ("_carry_prev", "_carry_prev_2", "_carry_days")
        ]
    )
    played_prev = pl.col("minutes").shift(1).over("player_code")
    played_prev_2 = pl.col("minutes").shift(2).over("player_code")
    played_days = (
        (pl.col("_date") - pl.col("_date").shift(1))
        .dt.total_days()
        .over("player_code")
    )
    is_played = pl.col("minutes").is_not_null()
    stream = stream.with_columns(
        pl.when(is_played)
        .then(played_prev)
        .otherwise(pl.col("_carry_prev"))
        .alias("prev_game_minutes"),
        pl.when(is_played)
        .then(played_prev_2)
        .otherwise(pl.col("_carry_prev_2"))
        .alias("prev_game_minutes_2"),
        pl.when(is_played)
        .then(played_days)
        .otherwise(pl.col("_carry_days"))
        .alias("days_since_prev_game"),
    )
    entering = stream.group_by(["season", "gw", "element"]).agg(
        [
            pl.col(column)
            .sort_by("kickoff_time", nulls_last=True)
            .first()
            .alias(column)
            for column in outputs
        ]
    )
    return data.join(
        entering, on=["season", "gw", "element"], how="left", coalesce=True
    )


def add_position_minutes_to_date(data: pl.DataFrame) -> pl.DataFrame:
    """Add season-to-date minutes and share of the club's position minutes.

    Both accumulate strictly *before* the current gameweek, so a row never
    sees its own minutes and the value is known at prediction time. A
    forward gameweek has null minutes, which count as zero, so these
    freeze at the last played gameweek's totals.

    Parameters
    ----------
    data : pl.DataFrame
        Player-week rows with ``season``, ``gw``, ``element``, ``team``,
        ``tm_position_group`` and ``minutes``.

    Returns
    -------
    pl.DataFrame
        ``data`` with ``minutes_to_date`` and
        ``pct_position_minutes_to_date`` added.
    """
    played = pl.col("minutes").fill_null(0)
    data = data.with_columns(
        played.cum_sum()
        .over(["season", "element"], order_by="gw")
        .sub(played)
        .alias("minutes_to_date")
    )
    # Summed to one row per club-position-gameweek before accumulating, or
    # the running total would add the same gameweek once per player in it.
    team_group = ["season", "team", "tm_position_group"]
    team_position = (
        data.group_by([*team_group, "gw"])
        .agg(played.sum().alias("_gw_minutes"))
        .with_columns(
            pl.col("_gw_minutes")
            .cum_sum()
            .over(team_group, order_by="gw")
            .sub(pl.col("_gw_minutes"))
            .alias("_team_position_to_date")
        )
        .drop("_gw_minutes")
    )
    return (
        data.join(
            team_position,
            on=[*team_group, "gw"],
            how="left",
            coalesce=True,
        )
        .with_columns(
            pl.when(pl.col("_team_position_to_date") > 0)
            .then(pl.col("minutes_to_date") / pl.col("_team_position_to_date"))
            .otherwise(None)
            .alias("pct_position_minutes_to_date")
        )
        .drop("_team_position_to_date")
    )


def _previous_season(season: pl.Expr) -> pl.Expr:
    """Return the calendar-previous season label for ``season``.

    Calendar rather than the previous season present in the data:
    ``player_week`` jumps 2017-18 -> 2020-21, so a positional shift would
    call 2017-18 the prior season.
    """
    start = season.str.slice(0, 4).cast(pl.Int64) - 1
    return (
        start.cast(pl.Utf8)
        + "-"
        + ((start + 1) % 100).cast(pl.Utf8).str.pad_start(2, "0")
    )


def add_prev_season_position_share(data: pl.DataFrame) -> pl.DataFrame:
    """Add the player's share of their club's position minutes last season.

    Joined on ``player_code`` and ``team``, so a player who changed clubs
    keeps a null: last season's pecking order at a different club says
    little about this one's.

    Parameters
    ----------
    data : pl.DataFrame
        Player-week rows with ``season``, ``player_code``, ``team``,
        ``tm_position_group`` and ``minutes``.

    Returns
    -------
    pl.DataFrame
        ``data`` with ``prev_pct_position_minutes`` added.
    """
    season_totals = (
        data.filter(pl.col("player_code").is_not_null())
        .group_by(["season", "player_code", "team", "tm_position_group"])
        .agg(pl.col("minutes").fill_null(0).sum().alias("minutes"))
    )
    shares = season_totals.with_columns(
        pl.col("minutes")
        .sum()
        .over(["season", "team", "tm_position_group"])
        .alias("_team_minutes")
    ).with_columns(
        pl.when(pl.col("_team_minutes") > 0)
        .then(pl.col("minutes") / pl.col("_team_minutes"))
        .otherwise(None)
        .alias("prev_pct_position_minutes")
    )
    previous = shares.select(
        pl.col("season").alias("_prev_season"),
        "player_code",
        "team",
        "prev_pct_position_minutes",
    )
    return (
        data.with_columns(
            _previous_season(pl.col("season")).alias("_prev_season")
        )
        .join(
            previous,
            on=["_prev_season", "player_code", "team"],
            how="left",
            coalesce=True,
        )
        .drop("_prev_season")
    )


def _transfer_spells(
    tm_transfer: pl.DataFrame, tm_player: pl.DataFrame
) -> pl.DataFrame:
    """Turn a player's transfers into dated club spells with a position group."""
    positions = tm_player.select(
        "tm_player_id",
        pl.col("position")
        .replace_strict(TM_POSITION_GROUPS, default=None)
        .alias("tm_position_group"),
    )
    return (
        tm_transfer.filter(pl.col("transfer_date_parsed").is_not_null())
        .select(
            "tm_player_id",
            "joined_club",
            pl.col("transfer_date_parsed").alias("date_joined"),
        )
        .sort(["tm_player_id", "date_joined"])
        .with_columns(
            pl.col("date_joined")
            .shift(-1)
            .over("tm_player_id")
            .alias("date_left")
        )
        .join(positions, on="tm_player_id", how="inner")
    )


def add_position_arrivals(
    data: pl.DataFrame,
    tm_transfer: pl.DataFrame,
    tm_player: pl.DataFrame,
    tm_market_value: pl.DataFrame,
    lookback_days: int = ARRIVAL_LOOKBACK_DAYS,
) -> pl.DataFrame:
    """Add the competition a player's club has recently signed for their spot.

    A rival is another Transfermarkt player whose club spell covers the
    same club, in the same position group, joined within ``lookback_days``
    of the fixture. Calendar-driven like :func:`add_tm_value`, so this
    reads the gameweek's own kickoff -- a keeper signed in August is
    exactly what a GW1 prediction needs to see, and freezing the date at
    last season's final match would hide them.

    The player's own club comes from their Transfermarkt spell rather than
    the FPL ``team``, so no club crosswalk is needed to compare the two.

    Parameters
    ----------
    data : pl.DataFrame
        Player-week rows with ``tm_player_id``, ``as_of_date`` and
        ``value_tm``.
    tm_transfer : pl.DataFrame
        Transfer rows with ``tm_player_id``, ``joined_club`` and
        ``transfer_date_parsed``.
    tm_player : pl.DataFrame
        Transfermarkt players with ``tm_player_id`` and ``position``.
    tm_market_value : pl.DataFrame
        Valuation rows, used to price the rivals.
    lookback_days : int
        How far back an arrival still counts. Defaults to
        :data:`ARRIVAL_LOOKBACK_DAYS`.

    Returns
    -------
    pl.DataFrame
        ``data`` with ``rivals_joined_same_pos``,
        ``higher_value_rivals_joined_same_pos``, ``max_rival_value_eur``
        and ``days_since_rival_joined`` added. A player with no
        Transfermarkt spell keeps nulls; one with a spell and no arrivals
        gets zero counts.
    """
    spells = _transfer_spells(tm_transfer, tm_player)
    keys = ["season", "gw", "element"]
    rows = data.select(
        [*keys, "tm_player_id", "as_of_date", "value_tm"]
    ).filter(
        pl.col("tm_player_id").is_not_null()
        & pl.col("as_of_date").is_not_null()
    )
    # The club and position group the player is at on the fixture date.
    current = (
        rows.join(spells, on="tm_player_id", how="inner")
        .filter(
            (pl.col("date_joined") <= pl.col("as_of_date"))
            & (
                pl.col("date_left").is_null()
                | (pl.col("as_of_date") < pl.col("date_left"))
            )
        )
        .select(
            [
                *keys,
                "tm_player_id",
                "as_of_date",
                "value_tm",
                pl.col("joined_club").alias("club"),
                pl.col("tm_position_group"),
            ]
        )
        .unique(subset=keys, keep="first")
    )
    value_spells = _value_spells(tm_market_value).select(
        pl.col("tm_player_id").alias("rival_id"),
        pl.col("value_eur").alias("rival_value_eur"),
        "value_date",
        "value_date_end",
    )
    rivals = spells.select(
        pl.col("tm_player_id").alias("rival_id"),
        pl.col("joined_club").alias("club"),
        "tm_position_group",
        pl.col("date_joined").alias("rival_date_joined"),
    )
    arrivals = (
        current.join(rivals, on=["club", "tm_position_group"], how="inner")
        .filter(
            (pl.col("rival_id") != pl.col("tm_player_id"))
            & (pl.col("rival_date_joined") <= pl.col("as_of_date"))
            & (
                pl.col("rival_date_joined")
                > pl.col("as_of_date").dt.offset_by(f"-{lookback_days}d")
            )
        )
        .join(value_spells, on="rival_id", how="left", coalesce=True)
        .filter(
            pl.col("value_date").is_null()
            | (
                (pl.col("value_date") <= pl.col("as_of_date"))
                & (
                    pl.col("value_date_end").is_null()
                    | (pl.col("as_of_date") < pl.col("value_date_end"))
                )
            )
        )
        .group_by(keys)
        .agg(
            pl.len().cast(pl.Int64).alias("rivals_joined_same_pos"),
            (pl.col("rival_value_eur") > pl.col("value_tm"))
            .sum()
            .cast(pl.Int64)
            .alias("higher_value_rivals_joined_same_pos"),
            pl.col("rival_value_eur").max().alias("max_rival_value_eur"),
            (pl.col("as_of_date") - pl.col("rival_date_joined"))
            .dt.total_days()
            .min()
            .alias("days_since_rival_joined"),
        )
    )
    # A player with a club spell and no arrivals has zero rivals, which is
    # information. One with no spell at all keeps nulls.
    counted = current.select(keys).join(
        arrivals, on=keys, how="left", coalesce=True
    )
    counted = counted.with_columns(
        pl.col("rivals_joined_same_pos").fill_null(0),
        pl.col("higher_value_rivals_joined_same_pos").fill_null(0),
    )
    return data.join(counted, on=keys, how="left", coalesce=True)


def add_transfermarkt_features(
    data: pl.DataFrame,
    match_stream: pl.DataFrame,
    player_season: pl.DataFrame,
    tm_player: pl.DataFrame,
    tm_player_map: pl.DataFrame,
    tm_market_value: pl.DataFrame,
    tm_transfer: pl.DataFrame,
) -> pl.DataFrame:
    """Add every Transfermarkt-derived feature to a player-week frame.

    Two clocks run here. Features derived from matches the player has
    played freeze at their last played match, so a forward gameweek
    inherits the state entering it. Features derived from the calendar --
    market value and rival arrivals -- read the fixture's own kickoff,
    because a valuation or a signing is a known fact at prediction time
    and freezing it would hide the very signings the model exists to
    react to.

    Parameters
    ----------
    data : pl.DataFrame
        Player-week rows with ``season``, ``gw``, ``element``, ``team``
        and ``minutes``.
    match_stream : pl.DataFrame
        Match-grain rows, unplayed fixtures carrying a null ``minutes``.
    player_season : pl.DataFrame
        Identity rows from :meth:`PLAYER_SEASON.load`.
    tm_player : pl.DataFrame
        Rows from :meth:`TM_PLAYER.load`.
    tm_player_map : pl.DataFrame
        Rows from :meth:`TM_PLAYER_MAP.load`.
    tm_market_value : pl.DataFrame
        Rows from :meth:`TM_MARKET_VALUE.load`.
    tm_transfer : pl.DataFrame
        Rows from :meth:`TM_TRANSFER.load`.

    Returns
    -------
    pl.DataFrame
        ``data`` with every column in
        :data:`fantasy_football.modelling.minutes.TM_FEATURES` added.
    """
    kickoffs = gameweek_kickoff(match_stream)
    frame = add_tm_identity(data, player_season, tm_player_map, tm_player)
    frame = add_tm_value(frame, kickoffs, tm_market_value)
    frame = add_tm_position_value_rank(frame)
    frame = add_prev_game_minutes(frame, match_stream, player_season)
    frame = add_position_minutes_to_date(frame)
    frame = add_prev_season_position_share(frame)
    return add_position_arrivals(
        frame, tm_transfer, tm_player, tm_market_value
    )
