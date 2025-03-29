import pulp
import polars as pl
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

from fantasy_football.fpl_types import FplPlayer, FplSquad, FplSquadPlayer, PlayerExpectedPoints
from fantasy_football.constants import FPL_ID
from fantasy_football.fpl import get_players, get_manager_team_from_id

def optimize_team(
    players: List[FplPlayer], 
    expected_points: Dict[int, float], 
    total_budget: int = 1000, 
    existing_team: Optional[FplSquad] = None,
    free_transfers: int = 1,
    transfer_penalty: int = 4
) -> Tuple[List[FplPlayer], Dict[str, Any]]:
    """
    Optimize an FPL team selection using PuLP linear programming.
    
    Parameters
    ----------
    players : List[FplPlayer]
        List of all available players
    expected_points : Dict[int, float]
        Dictionary mapping player IDs to their expected points
    total_budget : int, optional
        Total budget in 0.1M units (1000 = £100M), by default 1000
    existing_team : Optional[FplSquad], optional
        Existing team if optimizing transfers, by default None
    free_transfers : int, optional
        Number of free transfers available, by default 1
    transfer_penalty : int, optional
        Points penalty per transfer above free allowance, by default 4
        
    Returns
    -------
    Tuple[List[FplPlayer], Dict[str, Any]]
        Tuple containing selected players and additional info (captain, vice captain, etc.)
    """
    # Create the optimization model
    model = pulp.LpProblem("FPL_Team_Selection", pulp.LpMaximize)
    
    # Create decision variables for each player (1 if selected, 0 if not)
    player_vars = {player.id: pulp.LpVariable(f"player_{player.id}", cat=pulp.LpBinary) for player in players}
    
    # Create captain and vice-captain variables
    captain_vars = {player.id: pulp.LpVariable(f"captain_{player.id}", cat=pulp.LpBinary) for player in players}
    vice_captain_vars = {player.id: pulp.LpVariable(f"vice_captain_{player.id}", cat=pulp.LpBinary) for player in players}
    
    # For transfers, we need variables for whether a player is transferred in or out
    if existing_team:
        existing_ids = [p.player.id for p in existing_team.players]
        transfer_in_vars = {player.id: pulp.LpVariable(f"transfer_in_{player.id}", cat=pulp.LpBinary) 
                           for player in players if player.id not in existing_ids}
        transfer_out_vars = {player.id: pulp.LpVariable(f"transfer_out_{player.id}", cat=pulp.LpBinary) 
                            for player in players if player.id in existing_ids}
    
    # Objective function: Maximize expected points
    # Basic expected points from all selected players
    model += pulp.lpSum([player_vars[player.id] * expected_points.get(player.id, 0) for player in players])
    
    # Add captain bonus (captain gets double points)
    model += pulp.lpSum([captain_vars[player.id] * expected_points.get(player.id, 0) for player in players])
    
    # Transfer penalty if applicable
    if existing_team:
        total_transfers = pulp.lpSum(list(transfer_in_vars.values()))
        # If transfers > free_transfers, apply penalty
        transfer_penalty_var = pulp.LpVariable("transfer_penalty", lowBound=0, cat=pulp.LpInteger)
        model += transfer_penalty_var >= total_transfers - free_transfers
        model += -transfer_penalty * transfer_penalty_var  # Subtract penalty from objective
    
    # Constraints
    
    # 1. Total number of players must be 15
    model += pulp.lpSum(list(player_vars.values())) == 15, "Total_players"
    
    # 2. Budget constraint
    model += pulp.lpSum([player_vars[player.id] * player.now_cost for player in players]) <= total_budget, "Budget"
    
    # 3. Position constraints: 2 GK, 5 DEF, 5 MID, 3 FWD
    gk_players = [player for player in players if player.element_type == 1]
    def_players = [player for player in players if player.element_type == 2]
    mid_players = [player for player in players if player.element_type == 3]
    fwd_players = [player for player in players if player.element_type == 4]
    
    model += pulp.lpSum([player_vars[player.id] for player in gk_players]) == 2, "GK_constraint"
    model += pulp.lpSum([player_vars[player.id] for player in def_players]) == 5, "DEF_constraint"
    model += pulp.lpSum([player_vars[player.id] for player in mid_players]) == 5, "MID_constraint"
    model += pulp.lpSum([player_vars[player.id] for player in fwd_players]) == 3, "FWD_constraint"
    
    # 4. Team constraint: Maximum 3 players from any team
    teams = set(player.team_id for player in players)
    for team_id in teams:
        team_players = [player for player in players if player.team_id == team_id]
        model += pulp.lpSum([player_vars[player.id] for player in team_players]) <= 3, f"Team_{team_id}_constraint"
    
    # 5. Captain and vice-captain constraints
    # Only one captain
    model += pulp.lpSum(list(captain_vars.values())) == 1, "One_captain"
    # Only one vice-captain
    model += pulp.lpSum(list(vice_captain_vars.values())) == 1, "One_vice_captain"
    # Captain and vice-captain must be different players
    for player in players:
        model += captain_vars[player.id] + vice_captain_vars[player.id] <= 1, f"Diff_captain_vice_{player.id}"
        # Only selected players can be captain or vice-captain
        model += captain_vars[player.id] <= player_vars[player.id], f"Captain_selected_{player.id}"
        model += vice_captain_vars[player.id] <= player_vars[player.id], f"Vice_captain_selected_{player.id}"
    
    # 6. Transfer constraints if optimizing an existing team
    if existing_team:
        # Map between player decisions and transfers
        for player in players:
            if player.id in existing_ids:
                # If player was in team and not selected now: transferred out
                model += player_vars[player.id] + transfer_out_vars[player.id] == 1, f"Transfer_out_constraint_{player.id}"
            else:
                # If player wasn't in team and selected now: transferred in
                if player.id in transfer_in_vars:
                    model += player_vars[player.id] == transfer_in_vars[player.id], f"Transfer_in_constraint_{player.id}"
        
        # Number of transfers in equals number of transfers out
        model += pulp.lpSum(list(transfer_in_vars.values())) == pulp.lpSum(list(transfer_out_vars.values())), "Transfer_balance"
    
    # Solve the model
    model.solve(pulp.PULP_CBC_CMD(msg=False))
    
    # Extract results
    selected_players = [player for player in players if player_vars[player.id].value() == 1]
    captain_id = next(player.id for player in players if captain_vars[player.id].value() == 1)
    vice_captain_id = next(player.id for player in players if vice_captain_vars[player.id].value() == 1)
    
    extra_info = {
        "captain_id": captain_id,
        "vice_captain_id": vice_captain_id,
        "total_expected_points": pulp.value(model.objective),
        "total_cost": sum(player.now_cost for player in selected_players),
    }
    
    if existing_team:
        transfers_in = [player for player in players 
                     if player.id in transfer_in_vars and transfer_in_vars[player.id].value() == 1]
        transfers_out = [player for player in players 
                      if player.id in transfer_out_vars and transfer_out_vars[player.id].value() == 1]
        extra_info["transfers_in"] = transfers_in
        extra_info["transfers_out"] = transfers_out
        extra_info["total_transfers"] = len(transfers_in)
        extra_info["transfer_penalty"] = transfer_penalty_var.value() * transfer_penalty if transfer_penalty_var.value() > 0 else 0
    
    return selected_players, extra_info

def select_optimal_lineup(players: List[FplPlayer], expected_points: Dict[int, float]) -> List[int]:
    """
    Select the optimal 11 players to start from the squad of 15
    
    Parameters
    ----------
    players : List[FplPlayer]
        The 15 players in the squad
    expected_points : Dict[int, float]
        Expected points for each player
        
    Returns
    -------
    List[int]
        List of player IDs for the starting 11
    """
    # Group players by position
    gk_players = [p for p in players if p.element_type == 1]
    def_players = [p for p in players if p.element_type == 2]
    mid_players = [p for p in players if p.element_type == 3]
    fwd_players = [p for p in players if p.element_type == 4]
    
    # Sort by expected points
    gk_players = sorted(gk_players, key=lambda p: expected_points.get(p.id, 0), reverse=True)
    def_players = sorted(def_players, key=lambda p: expected_points.get(p.id, 0), reverse=True)
    mid_players = sorted(mid_players, key=lambda p: expected_points.get(p.id, 0), reverse=True)
    fwd_players = sorted(fwd_players, key=lambda p: expected_points.get(p.id, 0), reverse=True)
    
    # Create a base valid formation: 1 GK, 3 DEF, 3 MID, 1 FWD (8 players fixed)
    lineup = [gk_players[0].id]  # 1 goalkeeper
    lineup.extend([p.id for p in def_players[:3]])  # At least 3 defenders
    lineup.extend([p.id for p in mid_players[:3]])  # At least 3 midfielders
    lineup.extend([p.id for p in fwd_players[:1]])  # At least 1 forward
    
    # Create a pool of remaining players to fill the remaining 3 spots
    remaining_pool = []
    remaining_pool.extend([(p.id, expected_points.get(p.id, 0)) for p in def_players[3:]])
    remaining_pool.extend([(p.id, expected_points.get(p.id, 0)) for p in mid_players[3:]])
    remaining_pool.extend([(p.id, expected_points.get(p.id, 0)) for p in fwd_players[1:]])
    
    # Sort by expected points and take top 3
    remaining_pool.sort(key=lambda x: x[1], reverse=True)
    lineup.extend([p_id for p_id, _ in remaining_pool[:3]])
    
    return lineup

def optimize_transfers_and_team(event: int, num_free_transfers: int = 1) -> dict:
    """
    Main function to optimize transfers and team selection for a given gameweek
    
    Parameters
    ----------
    event : int
        The gameweek number
    num_free_transfers : int, optional
        Number of free transfers available, by default 1
        
    Returns
    -------
    dict
        Dictionary with optimized team, transfers, and lineup information
    """
    # Get all available players
    all_players = get_players()
    
    # Get current team
    current_team = get_manager_team_from_id(FPL_ID, event-1, all_players)
    
    # Generate expected points (using current simple prediction model)
    # In reality, this would be replaced with your more sophisticated model later
    rolling_data = pl.read_csv("data/transformed/rolling_points.csv")
    
    expected_points = {}
    for player in all_players:
        player_data = rolling_data.filter(pl.col("element") == player.id).sort("gw", descending=True)
        if len(player_data) > 0:
            expected_points[player.id] = player_data.select("total_points_rolling_5")[0, 0]
        else:
            # Default value for players with no data
            expected_points[player.id] = 2.0  # Conservative estimate
    
    # Optimize team considering transfers
    selected_players, optimization_info = optimize_team(
        players=all_players,
        expected_points=expected_points,
        total_budget=1000,  # £100M
        existing_team=current_team,
        free_transfers=num_free_transfers,
        transfer_penalty=4
    )
    
    # Select optimal lineup from 15-player squad
    starting_lineup = select_optimal_lineup(selected_players, expected_points)
    
    # Find captain and vice-captain
    captain_id = optimization_info["captain_id"]
    vice_captain_id = optimization_info["vice_captain_id"]
    
    # Organize results
    return {
        "gameweek": event,
        "selected_squad": selected_players,
        "starting_lineup": starting_lineup,
        "captain_id": captain_id,
        "vice_captain_id": vice_captain_id,
        "expected_points": optimization_info["total_expected_points"],
        "transfers_in": optimization_info.get("transfers_in", []),
        "transfers_out": optimization_info.get("transfers_out", []),
        "transfer_penalty": optimization_info.get("transfer_penalty", 0)
    } 