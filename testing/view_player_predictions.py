#!/usr/bin/env python3
"""
File: testing/view_player_predictions.py

View individual player predictions from the predictions file.
This shows you the actual ML predictions for each player.
"""

import sys
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_latest_predictions():
    """Load the most recent predictions file"""
    pred_dir = project_root / "data" / "nfl" / "predictions"
    pred_files = sorted(pred_dir.glob("predictions_*.parquet"))

    if not pred_files:
        print("❌ No prediction files found!")
        print("Run: python src/nfl/train_models.py")
        return None

    latest = pred_files[-1]
    print(f"📂 Loading: {latest.name}\n")

    return pd.read_parquet(latest)


def show_player_predictions(df, player_name=None, position=None, team=None, top_n=10):
    """
    Show predictions for specific player(s) or top performers.

    Args:
        df: Predictions DataFrame
        player_name: Specific player name (partial match OK)
        position: Filter by position (QB, RB, WR, TE, K)
        team: Filter by team code
        top_n: Number of top players to show (if no specific player)
    """

    # Filter by criteria
    filtered = df.copy()

    if player_name:
        filtered = filtered[filtered['player_name'].str.contains(player_name, case=False, na=False)]

    if position:
        filtered = filtered[filtered['position'] == position.upper()]

    if team:
        filtered = filtered[filtered['team'] == team.upper()]

    if len(filtered) == 0:
        print("❌ No players found matching criteria")
        return

    # Separate EVOB and POB predictions
    evob_df = filtered[filtered['model_type'] == 'evob'].copy()
    pob_df = filtered[filtered['model_type'] == 'pob'].copy()

    # If looking for specific player, show all their predictions
    if player_name:
        players = filtered['player_name'].unique()

        for player in players:
            print("="*70)
            print(f"PREDICTIONS FOR: {player}")
            print("="*70)

            player_evob = evob_df[evob_df['player_name'] == player]
            player_pob = pob_df[pob_df['player_name'] == player]

            if len(player_evob) > 0:
                player_info = player_evob.iloc[0]
                print(f"Position: {player_info['position']}")
                print(f"Team: {player_info['team']}")
                print(f"Opponent: {player_info['opponent']}")
                print(f"Week: {player_info['week']}")
                print()

            # Show EVOB predictions (Value predictions)
            if len(player_evob) > 0:
                print("VALUE PREDICTIONS (EVOB - Expected Value Over Baseline):")
                print("-" * 70)
                for _, row in player_evob.iterrows():
                    stat = row['stat']
                    predicted = row['predicted_value']
                    baseline = row['baseline']
                    diff = row['predicted_diff']
                    conf_lower = row.get('confidence_lower', 0)
                    conf_upper = row.get('confidence_upper', 0)

                    print(f"  {stat:25s}: {predicted:6.1f} (baseline: {baseline:6.1f}, diff: {diff:+6.1f})")
                    print(f"  {'':25s}  Confidence: [{conf_lower:6.1f}, {conf_upper:6.1f}]")
                print()

            # Show POB predictions (Probability predictions)
            if len(player_pob) > 0:
                print("PROBABILITY PREDICTIONS (POB - Probability Over Baseline):")
                print("-" * 70)
                for _, row in player_pob.iterrows():
                    stat = row['stat']
                    prob = row['probability_over']
                    baseline = row['baseline']

                    print(f"  {stat:25s}: {prob*100:5.1f}% chance to beat baseline ({baseline:.1f})")
                print()

            print()

    else:
        # Show top performers by fantasy points
        fantasy_evob = evob_df[evob_df['stat'] == 'fantasy_points_ppr'].copy()

        if len(fantasy_evob) == 0:
            print("❌ No fantasy point predictions found")
            return

        # Sort by predicted value
        fantasy_evob = fantasy_evob.sort_values('predicted_value', ascending=False)

        print(f"TOP {top_n} PREDICTED FANTASY PERFORMERS:")
        print("="*90)
        print(f"{'Rank':<5} {'Player':<20} {'Pos':<5} {'Team':<5} {'Predicted':<10} {'Baseline':<10} {'Diff':<8}")
        print("-"*90)

        for i, (_, row) in enumerate(fantasy_evob.head(top_n).iterrows(), 1):
            print(f"{i:<5} {row['player_name']:<20} {row['position']:<5} {row['team']:<5} "
                  f"{row['predicted_value']:<10.1f} {row['baseline']:<10.1f} {row['predicted_diff']:+8.1f}")

        print()


def show_position_summary(df):
    """Show summary statistics by position"""
    print("="*70)
    print("PREDICTION SUMMARY BY POSITION")
    print("="*70)

    # Get fantasy point predictions only
    fantasy_df = df[(df['stat'] == 'fantasy_points_ppr') & (df['model_type'] == 'evob')].copy()

    for position in ['QB', 'RB', 'WR', 'TE', 'K']:
        pos_df = fantasy_df[fantasy_df['position'] == position]

        if len(pos_df) == 0:
            continue

        print(f"\n{position}:")
        print(f"  Total Players: {len(pos_df)}")
        print(f"  Avg Predicted: {pos_df['predicted_value'].mean():.1f} pts")
        print(f"  Top Predicted: {pos_df['predicted_value'].max():.1f} pts "
              f"({pos_df.loc[pos_df['predicted_value'].idxmax(), 'player_name']})")
        print(f"  Avg Baseline:  {pos_df['baseline'].mean():.1f} pts")

    print()


def main():
    """Main function"""
    print("\n")
    print("="*70)
    print("NFL PLAYER PREDICTIONS VIEWER")
    print("="*70)
    print()

    # Load predictions
    df = load_latest_predictions()

    if df is None:
        return 1

    print(f"Total Predictions: {len(df)}")
    print(f"Players: {df['player_name'].nunique()}")
    print(f"Positions: {df['position'].unique().tolist()}")
    print()

    # Show position summary
    show_position_summary(df)

    # Examples of different queries
    print("="*70)
    print("EXAMPLE QUERIES")
    print("="*70)
    print()

    # Example 1: Top 10 overall fantasy performers
    show_player_predictions(df, top_n=10)

    # Example 2: Top QBs
    print("\nTOP 5 QBS:")
    print("-"*70)
    show_player_predictions(df, position='QB', top_n=5)

    # Example 3: Specific player (if you know a player name)
    # Uncomment and modify to search for specific players:
    # print("\nSPECIFIC PLAYER SEARCH:")
    # print("-"*70)
    # show_player_predictions(df, player_name="Mahomes")

    print("="*70)
    print("HOW TO USE THIS SCRIPT")
    print("="*70)
    print()
    print("To search for specific players, modify the main() function:")
    print()
    print("  # Search by player name (partial match):")
    print("  show_player_predictions(df, player_name='Mahomes')")
    print()
    print("  # Search by position:")
    print("  show_player_predictions(df, position='RB', top_n=20)")
    print()
    print("  # Search by team:")
    print("  show_player_predictions(df, team='KC', top_n=10)")
    print()
    print("  # Combine filters:")
    print("  show_player_predictions(df, position='WR', team='KC')")
    print()
    print("="*70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
