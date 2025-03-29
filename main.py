import argparse

from fantasy_football.constants import CURRENT_SEASON
from fantasy_football.data_transformation import create_rolling_points_data
from fantasy_football.data_extraction import save_all_data_files, update_current_season_data
from fantasy_football.optimization import optimize_transfers_and_team

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--update-current-season",
        default=False,
        action="store_true",
        help="Update current season data only",
    )
    parser.add_argument(
        "--download-all",
        default=False,
        action="store_true",
        help="Download all data files",
    )
    parser.add_argument(
        "--optimize-team",
        type=int,
        help="Optimize team for a specific gameweek number",
    )
    parser.add_argument(
        "--free-transfers",
        type=int,
        default=1,
        help="Number of free transfers available",
    )
    args = parser.parse_args()
    
    if args.download_all:
        save_all_data_files()
    if args.update_current_season:
        update_current_season_data(CURRENT_SEASON)
        create_rolling_points_data(CURRENT_SEASON)
    if args.optimize_team:
        result = optimize_transfers_and_team(args.optimize_team, args.free_transfers)
        print_optimization_results(result)

def print_optimization_results(result: dict):
    """Print the optimization results in a readable format"""
    print(f"\n== Optimized Team for Gameweek {result['gameweek']} ==\n")
    
    # Print transfers if any
    if result['transfers_in']:
        print("Recommended Transfers:")
        for i, (player_in, player_out) in enumerate(zip(result['transfers_in'], result['transfers_out'])):
            print(f"  {i+1}. OUT: {player_out.first_name} {player_out.second_name} ({player_out.now_cost/10:.1f}m)")
            print(f"     IN: {player_in.first_name} {player_in.second_name} ({player_in.now_cost/10:.1f}m)")
        print(f"Total transfers: {len(result['transfers_in'])}")
        if result['transfer_penalty'] > 0:
            print(f"Transfer penalty: -{result['transfer_penalty']} points")
    else:
        print("No transfers recommended.")
    
    # Print squad
    print("\nFull Squad:")
    
    # Group by position
    goalkeepers = [p for p in result['selected_squad'] if p.element_type == 1]
    defenders = [p for p in result['selected_squad'] if p.element_type == 2]
    midfielders = [p for p in result['selected_squad'] if p.element_type == 3]
    forwards = [p for p in result['selected_squad'] if p.element_type == 4]
    
    # Print by position with indicator for starting lineup
    print("  Goalkeepers:")
    for player in goalkeepers:
        captain = " (C)" if player.id == result['captain_id'] else ""
        vice = " (V)" if player.id == result['vice_captain_id'] else ""
        starting = "*" if player.id in result['starting_lineup'] else " "
        print(f"    {starting} {player.web_name} - £{player.now_cost/10:.1f}m{captain}{vice}")
    
    print("  Defenders:")
    for player in defenders:
        captain = " (C)" if player.id == result['captain_id'] else ""
        vice = " (V)" if player.id == result['vice_captain_id'] else ""
        starting = "*" if player.id in result['starting_lineup'] else " "
        print(f"    {starting} {player.web_name} - £{player.now_cost/10:.1f}m{captain}{vice}")
    
    print("  Midfielders:")
    for player in midfielders:
        captain = " (C)" if player.id == result['captain_id'] else ""
        vice = " (V)" if player.id == result['vice_captain_id'] else ""
        starting = "*" if player.id in result['starting_lineup'] else " "
        print(f"    {starting} {player.web_name} - £{player.now_cost/10:.1f}m{captain}{vice}")
    
    print("  Forwards:")
    for player in forwards:
        captain = " (C)" if player.id == result['captain_id'] else ""
        vice = " (V)" if player.id == result['vice_captain_id'] else ""
        starting = "*" if player.id in result['starting_lineup'] else " "
        print(f"    {starting} {player.web_name} - £{player.now_cost/10:.1f}m{captain}{vice}")

    print(f"\nExpected points: {result['expected_points']:.2f}")
    print("* indicates starting lineup player")
    print("(C) indicates captain")
    print("(V) indicates vice-captain")

if __name__ == "__main__":
    main()
