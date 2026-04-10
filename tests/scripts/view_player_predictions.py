#!/usr/bin/env python3
"""
File: testing/view_player_predictions.py

View individual player predictions from the predictions file.
This shows you the actual ML predictions for each player in a clean, readable format.
"""

import sys
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_predictions(week=None, season=2025):
    """Load predictions for a specific week or the most recent one"""
    pred_dir = project_root / "data" / "nfl" / "predictions"

    if week:
        pred_file = pred_dir / f"predictions_{season}_week_{week}.parquet"
        if not pred_file.exists():
            print(f"❌ No predictions found for {season} Week {week}")
            return None
        print(f"📂 Loading: predictions_{season}_week_{week}.parquet\n")
        return pd.read_parquet(pred_file)
    else:
        pred_files = sorted(pred_dir.glob("predictions_*.parquet"))
        if not pred_files:
            print("❌ No prediction files found!")
            print("Run: python src/nfl/train_models.py")
            return None
        latest = pred_files[-1]
        print(f"📂 Loading: {latest.name}\n")
        return pd.read_parquet(latest)


def show_player_detail(df, player_name):
    """Show detailed predictions for a specific player"""

    # Search for player (case insensitive, partial match)
    player_df = df[df['player_name'].str.contains(player_name, case=False, na=False)]

    if len(player_df) == 0:
        print(f"❌ No player found matching '{player_name}'")
        return

    players = player_df['player_name'].unique()

    for player in players:
        player_data = player_df[player_df['player_name'] == player]

        # Get player info
        info = player_data.iloc[0]

        print("\n" + "="*80)
        print(f"  {player.upper()}")
        print("="*80)
        print(f"  Position: {info['position']:3s}  |  Team: {info['team']:3s}  |  ", end="")
        print(f"Opponent: {info['opponent']:3s}  |  Week: {info['week']}")
        print("="*80)

        # Get EVOB and POB data
        evob_data = player_data[player_data['model_type'] == 'evob'].copy()
        pob_data = player_data[player_data['model_type'] == 'pob'].copy()

        # Fantasy points prediction (main stat)
        fantasy_evob = evob_data[evob_data['stat'] == 'fantasy_points_ppr']
        fantasy_pob = pob_data[pob_data['stat'] == 'fantasy_points_ppr']

        if len(fantasy_evob) > 0:
            row = fantasy_evob.iloc[0]
            print("\n🎯 FANTASY POINTS (PPR)")
            print("─" * 80)
            print(f"  Predicted:  {row['predicted_value']:6.1f} pts")
            print(f"  Baseline:   {row['baseline']:6.1f} pts  (6-game rolling average)")
            print(f"  Difference: {row['predicted_diff']:+6.1f} pts")

            if pd.notna(row['confidence_lower']) and pd.notna(row['confidence_upper']):
                print(f"  Confidence: [{row['confidence_lower']:6.1f}, {row['confidence_upper']:6.1f}] pts")

        if len(fantasy_pob) > 0:
            row = fantasy_pob.iloc[0]
            prob_pct = row['probability_over'] * 100
            print(f"  Probability: {prob_pct:5.1f}% chance to exceed baseline")

        # Individual stat predictions
        other_stats = evob_data[evob_data['stat'] != 'fantasy_points_ppr']

        if len(other_stats) > 0:
            print("\n📊 INDIVIDUAL STAT PREDICTIONS")
            print("─" * 80)

            # Define stat display names and order
            stat_order = {
                'passing_yards': ('Passing Yards', 'yds'),
                'passing_tds': ('Passing TDs', 'TDs'),
                'passing_interceptions': ('Interceptions', 'INTs'),
                'rushing_yards': ('Rushing Yards', 'yds'),
                'rushing_tds': ('Rushing TDs', 'TDs'),
                'receiving_yards': ('Receiving Yards', 'yds'),
                'receiving_tds': ('Receiving TDs', 'TDs'),
                'receptions': ('Receptions', 'rec'),
                'fg_made': ('FG Made', 'FGs'),
                'fg_att': ('FG Attempts', 'att'),
            }

            for _, row in other_stats.iterrows():
                stat = row['stat']
                if stat in stat_order:
                    display_name, unit = stat_order[stat]
                    print(f"  {display_name:20s}: {row['predicted_value']:6.1f} {unit:4s} ", end="")
                    print(f"(baseline: {row['baseline']:6.1f}, {row['predicted_diff']:+6.1f})")

        print()


def show_top_performers(df, position=None, top_n=20):
    """Show top predicted fantasy performers"""

    # Filter for EVOB fantasy points only
    fantasy_df = df[(df['stat'] == 'fantasy_points_ppr') & (df['model_type'] == 'evob')].copy()

    if position:
        fantasy_df = fantasy_df[fantasy_df['position'] == position.upper()]
        title = f"TOP {top_n} {position.upper()} PREDICTIONS"
    else:
        title = f"TOP {top_n} OVERALL PREDICTIONS"

    if len(fantasy_df) == 0:
        print(f"❌ No predictions found")
        return

    # Sort by predicted value
    fantasy_df = fantasy_df.sort_values('predicted_value', ascending=False).head(top_n)

    print("\n" + "="*95)
    print(f"  {title}")
    print("="*95)
    print(f"{'#':<4} {'Player':<22} {'Pos':<5} {'Team':<5} {'Opp':<5} {'Pred':>7} {'Base':>7} {'Diff':>7} {'Prob':>7}")
    print("─"*95)

    # Get POB data for probabilities
    pob_df = df[(df['stat'] == 'fantasy_points_ppr') & (df['model_type'] == 'pob')]

    for i, (_, row) in enumerate(fantasy_df.iterrows(), 1):
        # Find matching POB prediction
        pob_row = pob_df[(pob_df['player_name'] == row['player_name']) &
                          (pob_df['week'] == row['week'])]

        prob_str = ""
        if len(pob_row) > 0:
            prob_pct = pob_row.iloc[0]['probability_over'] * 100
            prob_str = f"{prob_pct:5.1f}%"

        print(f"{i:<4} {row['player_name']:<22} {row['position']:<5} {row['team']:<5} {row['opponent']:<5} ", end="")
        print(f"{row['predicted_value']:7.1f} {row['baseline']:7.1f} {row['predicted_diff']:+7.1f} {prob_str:>7}")

    print()


def show_position_summary(df):
    """Show summary statistics by position"""

    fantasy_df = df[(df['stat'] == 'fantasy_points_ppr') & (df['model_type'] == 'evob')].copy()

    print("\n" + "="*80)
    print("  PREDICTION SUMMARY BY POSITION")
    print("="*80)

    summary_data = []

    for position in ['QB', 'RB', 'WR', 'TE', 'K']:
        pos_df = fantasy_df[fantasy_df['position'] == position]

        if len(pos_df) == 0:
            continue

        top_player = pos_df.loc[pos_df['predicted_value'].idxmax()]

        summary_data.append({
            'Position': position,
            'Players': len(pos_df),
            'Avg Pred': pos_df['predicted_value'].mean(),
            'Avg Base': pos_df['baseline'].mean(),
            'Top Player': top_player['player_name'],
            'Top Pred': top_player['predicted_value']
        })

    if summary_data:
        print(f"\n{'Pos':<5} {'Players':>8} {'Avg Pred':>10} {'Avg Base':>10}   {'Top Player':<20} {'Top Pred':>10}")
        print("─"*80)
        for row in summary_data:
            print(f"{row['Position']:<5} {row['Players']:8d} {row['Avg Pred']:10.1f} {row['Avg Base']:10.1f}   ", end="")
            print(f"{row['Top Player']:<20} {row['Top Pred']:10.1f}")

    print()


def show_team_leaders(df, team_code):
    """Show top predicted players for a specific team"""

    team_df = df[(df['team'] == team_code.upper()) &
                 (df['stat'] == 'fantasy_points_ppr') &
                 (df['model_type'] == 'evob')].copy()

    if len(team_df) == 0:
        print(f"❌ No predictions found for team '{team_code.upper()}'")
        return

    team_df = team_df.sort_values('predicted_value', ascending=False)

    info = team_df.iloc[0]

    print("\n" + "="*80)
    print(f"  {team_code.upper()} TEAM LEADERS - Week {info['week']} vs {info['opponent']}")
    print("="*80)
    print(f"{'#':<4} {'Player':<25} {'Pos':<5} {'Predicted':>10} {'Baseline':>10} {'Diff':>8}")
    print("─"*80)

    for i, (_, row) in enumerate(team_df.head(10).iterrows(), 1):
        print(f"{i:<4} {row['player_name']:<25} {row['position']:<5} ", end="")
        print(f"{row['predicted_value']:10.1f} {row['baseline']:10.1f} {row['predicted_diff']:+8.1f}")

    print()


def interactive_menu(df):
    """Interactive menu for exploring predictions"""

    while True:
        print("\n" + "="*80)
        print("  PREDICTION VIEWER MENU")
        print("="*80)
        print("  1. View top overall performers")
        print("  2. View top performers by position")
        print("  3. Search for specific player")
        print("  4. View team leaders")
        print("  5. Position summary")
        print("  6. Exit")
        print("─"*80)

        choice = input("\nEnter choice (1-6): ").strip()

        if choice == '1':
            n = input("How many players? [20]: ").strip() or "20"
            show_top_performers(df, top_n=int(n))

        elif choice == '2':
            pos = input("Position (QB/RB/WR/TE/K): ").strip().upper()
            n = input("How many players? [20]: ").strip() or "20"
            show_top_performers(df, position=pos, top_n=int(n))

        elif choice == '3':
            name = input("Enter player name (or part of name): ").strip()
            if name:
                show_player_detail(df, name)

        elif choice == '4':
            team = input("Enter team code (e.g., KC, SF, BUF): ").strip().upper()
            if team:
                show_team_leaders(df, team)

        elif choice == '5':
            show_position_summary(df)

        elif choice == '6':
            print("\n👋 Goodbye!\n")
            break

        else:
            print("❌ Invalid choice")


def main():
    """Main function"""

    print("\n" + "="*80)
    print("  NFL PLAYER PREDICTIONS VIEWER")
    print("="*80)

    # Check if week was specified as command line argument
    week = None
    if len(sys.argv) > 1:
        try:
            week = int(sys.argv[1])
            print(f"  Loading predictions for Week {week}")
        except ValueError:
            print(f"  Invalid week: {sys.argv[1]}")
            return 1

    # Load predictions
    df = load_predictions(week=week)

    if df is None:
        return 1

    info = df.iloc[0]
    print(f"  Season: {info['season']}  |  Week: {info['week']}  |  Total Predictions: {len(df)}")
    print(f"  Players: {df['player_name'].nunique()}  |  Positions: {', '.join(sorted(df['position'].unique()))}")
    print("="*80)

    # Show quick overview
    show_position_summary(df)
    show_top_performers(df, top_n=10)

    # Start interactive menu
    interactive_menu(df)

    return 0


if __name__ == "__main__":
    sys.exit(main())
