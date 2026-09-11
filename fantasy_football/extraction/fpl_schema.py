from typing import Any

from pydantic import BaseModel


class Overrides(BaseModel):  # noqa : DCO1
    rules: dict
    scoring: dict
    element_types: list
    pick_multiplier: Any | None = None


class Chip(BaseModel):  # noqa : DCO1
    id: int
    name: str
    number: int
    start_event: int
    stop_event: int
    chip_type: str
    overrides: Overrides


class Event(BaseModel):  # noqa : DCO1
    class ChipPlays(BaseModel):  # noqa : DCO1
        chip_name: str
        num_played: int

    id: int
    name: str
    deadline_time: str
    release_time: str | None = None
    average_entry_score: int
    finished: bool
    data_checked: bool
    highest_scoring_entry: int | None = None
    deadline_time_epoch: int
    deadline_time_game_offset: int
    highest_score: int | None = None
    is_previous: bool
    is_current: bool
    is_next: bool
    cup_leagues_created: bool
    h2h_ko_matches_created: bool
    can_enter: bool
    can_manage: bool
    released: bool
    ranked_count: int
    overrides: Overrides
    chip_plays: list[ChipPlays]
    most_selected: int | None = None
    most_transferred_in: int | None = None
    top_element: int | None = None
    transfers_made: int
    most_captained: int | None = None
    most_vice_captained: int | None = None


class GameSettings(BaseModel):  # noqa : DCO1
    league_join_private_max: int
    league_join_public_max: int
    league_max_size_public_classic: int
    league_max_size_public_h2h: int
    league_max_size_private_h2h: int
    league_max_ko_rounds_private_h2h: int
    league_prefix_public: str
    league_points_h2h_win: int
    league_points_h2h_lose: int
    league_points_h2h_draw: int
    league_ko_first_instead_of_random: bool
    cup_start_event_id: int | None = None
    cup_stop_event_id: int | None = None
    cup_qualifying_method: str | None = None  # guessed a string, could break
    cup_type: str | None = None  # guessed a string, could break
    featured_entries: list  # unsure of type within list
    element_sell_at_purchase_price: bool
    percentile_ranks: list[int]
    underdog_differential: int
    squad_squadplay: int
    squad_squadsize: int
    squad_special_min: int | None = None  # guessed a int, could break
    squad_special_max: int | None = None  # guessed a int, could break
    squad_team_limit: int
    squad_total_spend: int
    ui_currency_multiplier: int
    ui_use_special_shirts: bool
    ui_special_shirt_exclusions: list  # unsure of type within list
    stats_form_days: int
    sys_vice_captain_enabled: bool
    transfers_cap: int
    transfers_sell_on_fee: float
    max_extra_free_transfers: int
    league_h2h_tiebreak_stats: list[str]
    timezone: str | None = None


class GameConfig(BaseModel):  # noqa : DCO1
    class Settings(BaseModel):  # noqa : DCO1
        entry_per_event: bool
        timezone: str
        static_content_url: str
        price_change_deadlines: list[str]

    class Status(BaseModel):  # noqa : DCO1
        price_change_last_updated: str

    class Scoring(BaseModel):  # noqa : DCO1
        class PositionalScoring(BaseModel):  # noqa : DCO1
            DEF: int
            FWD: int
            GKP: int
            MID: int

        long_play: int
        short_play: int
        goals_conceded: PositionalScoring
        saves: int
        goals_scored: PositionalScoring
        assists: int
        clean_sheets: PositionalScoring
        penalties_saved: int
        penalties_missed: int
        yellow_cards: int
        red_cards: int
        own_goals: int
        bonus: int
        bps: int
        influence: int
        creativity: int
        threat: int
        ict_index: int
        special_multiplier: int
        tackles: int
        clearances_blocks_interceptions: int
        recoveries: int
        defensive_contribution: PositionalScoring
        starts: int
        mng_goals_scored: PositionalScoring
        mng_clean_sheets: PositionalScoring
        mng_win: PositionalScoring
        mng_draw: PositionalScoring
        mng_loss: int
        mng_underdog_win: PositionalScoring
        mng_underdog_draw: PositionalScoring
        expected_assists: int
        expected_goal_involvements: int
        expected_goals_conceded: int
        expected_goals: int

    settings: Settings
    rules: GameSettings
    status: Status
    scoring: Scoring


class Phase(BaseModel):  # noqa : DCO1
    id: int
    name: str
    start_event: int
    stop_event: int
    highest_score: int | None = None


class Teams(BaseModel):  # noqa : DCO1
    code: int
    draw: int
    form: int | None = None  # guessed a int, could break
    id: int
    loss: int
    name: str
    played: int
    points: int
    position: int
    short_name: str
    strength: int | None = None  # guessed a int, could break
    team_division: int | None = None  # guessed a int, could break
    unavailable: bool
    win: int
    link_url: str
    strength_overall_home: int
    strength_overall_away: int
    strength_attack_home: int
    strength_attack_away: int
    strength_defence_home: int
    strength_defence_away: int
    pulse_id: int


class ElementStat(BaseModel):  # noqa : DCO1
    label: str
    name: str


class ElementType(BaseModel):  # noqa : DCO1
    id: int
    plural_name: str
    plural_name_short: str
    singular_name: str
    singular_name_short: str
    squad_select: int
    squad_min_select: int | None = None  # guessed a int, could break
    squad_max_select: int | None = None  # guessed a int, could break
    squad_min_play: int
    squad_max_play: int
    ui_shirt_specific: bool
    sub_positions_locked: list[int]
    element_count: int


class Element(BaseModel):  # noqa : DCO1
    class PriceChangeProjection(BaseModel):  # noqa : DCO1
        offset: int
        projected_percent: str
        likelihood: int

    can_transact: bool
    can_select: bool
    chance_of_playing_this_round: int | None = (
        None  # guessed a int, could break
    )
    chance_of_playing_next_round: int | None = (
        None  # guessed a int, could break
    )
    code: int
    cost_change_event: int
    cost_change_event_fall: int
    cost_change_start: int
    cost_change_start_fall: int
    price_change_percent: str
    price_change_hourly_rate: int
    price_change_projections: list[PriceChangeProjection]
    price_change_locked_until: str | None = None  # guessed a str, could break
    price_change_calibrating: bool
    dreamteam_count: int
    element_type: int
    ep_next: str
    ep_this: str
    event_points: int
    first_name: str
    form: str
    id: int
    in_dreamteam: bool
    news: str
    news_added: str | None = None  # guessed a string, could break
    now_cost: int
    photo: str
    points_per_game: str
    removed: bool
    second_name: str
    selected_by_percent: float
    special: bool
    squad_number: int | None = None  # guessed a int, could break
    squad_select: int | None = None
    status: str
    team: int
    team_code: int
    total_points: int
    transfers_in: int
    transfers_in_event: int
    transfers_out: int
    transfers_out_event: int
    value_form: str
    value_season: str
    web_name: str
    known_name: str
    region: int | None = None
    team_join_date: str | None = None
    birth_date: str | None = None
    has_temporary_code: bool
    opta_code: str
    minutes: int
    goals_scored: int
    assists: int
    clean_sheets: int
    goals_conceded: int
    own_goals: int
    penalties_saved: int
    penalties_missed: int
    yellow_cards: int
    red_cards: int
    saves: int
    bonus: int
    bps: int
    influence: str
    creativity: str
    threat: str
    ict_index: str
    clearances_blocks_interceptions: int
    recoveries: int
    tackles: int
    defensive_contribution: int
    starts: int
    expected_goals: str
    expected_assists: str
    expected_goal_involvements: str
    expected_goals_conceded: str
    corners_and_indirect_freekicks_order: int | None = (
        None  # guessed a int, could break
    )
    corners_and_indirect_freekicks_text: str
    direct_freekicks_order: int | None = None  # guessed a int, could break
    direct_freekicks_text: str
    penalties_order: int | None = None  # guessed a int, could break
    penalties_text: str
    scout_risks: list
    scout_news_link: str
    influence_rank: int
    influence_rank_type: int
    creativity_rank: int
    creativity_rank_type: int
    threat_rank: int
    threat_rank_type: int
    ict_index_rank: int
    ict_index_rank_type: int
    expected_goals_per_90: float
    saves_per_90: float
    expected_assists_per_90: float
    expected_goal_involvements_per_90: float
    expected_goals_conceded_per_90: float
    goals_conceded_per_90: float
    now_cost_rank: int
    now_cost_rank_type: int
    points_per_game_rank: int
    points_per_game_rank_type: int
    selected_rank: int
    selected_rank_type: int
    starts_per_90: float
    clean_sheets_per_90: float
    defensive_contribution_per_90: float


class BootStrapResponse(BaseModel):
    """Respone from the FPL API for the bootstrap data."""

    chips: list[Chip]
    events: list[Event]
    game_settings: GameSettings
    game_config: GameConfig
    phases: list[Phase]
    teams: list[Teams]
    total_players: int
    element_stats: list[ElementStat]
    element_types: list[ElementType]
    elements: list[Element]


class FixtureResponse(BaseModel):
    """Class for storing a fixture response from the FPL API."""

    class FixtureStatistic(BaseModel):  # noqa : DCO1
        class TeamStatistic(BaseModel):  # noqa : DCO1
            value: int | None = None
            element: int | None = None

        identifier: str
        a: list[TeamStatistic]
        h: list[TeamStatistic]

    code: int
    event: int | None = None
    finished: bool
    finished_provisional: bool
    id: int
    kickoff_time: str | None = None
    minutes: int
    provisional_start_time: bool
    started: bool | None = None
    team_a: int
    team_a_score: int | None = None
    team_h: int
    team_h_score: int | None = None
    stats: list[FixtureStatistic]
    team_h_difficulty: int
    team_a_difficulty: int
    pulse_id: int


class FplFixtureResponses(BaseModel):  # noqa : DCO1
    fixtures: list[FixtureResponse]
