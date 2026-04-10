#!/usr/bin/env python3
"""
File: testing/compare_player_predictions.py

Compare individual player predictions between V1 and V2 models.
Shows side-by-side comparison of predictions for specific players across positions.
"""

import sys
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_predictions(version, season, week):
    """Load predictions for a specific version, season, and week"""
    pred_dir = project_root / "data" / "nfl" / "predictions" / version
    pred_file = pred_dir / f"predictions_{season}_week_{week}.parquet"

    if not pred_file.exists():
        return None

    return pd.read_parquet(pred_file)


def compare_player(player_name, v1_df, v2_df, v1_version, v2_version):
    """Compare predictions for a specific player between V1 and V2"""

    # Search for player (case insensitive, partial match)
    v1_player = v1_df[v1_df['player_name'].str.contains(player_name, case=False, na=False)]
    v2_player = v2_df[v2_df['player_name'].str.contains(player_name, case=False, na=False)]

    if len(v1_player) == 0 or len(v2_player) == 0:
        print(f"❌ Player '{player_name}' not found in both versions")
        return False

    # Get exact player names (should be the same)
    v1_players = v1_player['player_name'].unique()
    v2_players = v2_player['player_name'].unique()

    # Find common players
    common_players = set(v1_players) & set(v2_players)

    if not common_players:
        print(f"❌ No matching players found for '{player_name}'")
        return False

    for player in common_players:
        v1_data = v1_player[v1_player['player_name'] == player]
        v2_data = v2_player[v2_player['player_name'] == player]

        # Get player info
        info = v1_data.iloc[0]

        print("\n" + "="*100)
        print(f"  {player.upper()}")
        print("="*100)
        print(f"  Position: {info['position']:3s}  |  Team: {info['team']:3s}  |  ", end="")
        print(f"Opponent: {info['opponent']:3s}  |  Week: {info['week']}  |  Season: {info['season']}")
        print("="*100)

        # Compare fantasy points predictions
        v1_evob = v1_data[(v1_data['model_type'] == 'evob') &
                          (v1_data['stat'] == 'fantasy_points_ppr')]
        v2_evob = v2_data[(v2_data['model_type'] == 'evob') &
                          (v2_data['stat'] == 'fantasy_points_ppr')]

        v1_pob = v1_data[(v1_data['model_type'] == 'pob') &
                         (v1_data['stat'] == 'fantasy_points_ppr')]
        v2_pob = v2_data[(v2_data['model_type'] == 'pob') &
                         (v2_data['stat'] == 'fantasy_points_ppr')]

        if len(v1_evob) > 0 and len(v2_evob) > 0:
            v1_row = v1_evob.iloc[0]
            v2_row = v2_evob.iloc[0]

            print("\n🎯 FANTASY POINTS (PPR) COMPARISON")
            print("─" * 100)
            print(f"{'Metric':<25} {'V1 (' + v1_version + ')':>25} {'V2 (' + v2_version + ')':>25} {'Difference':>20}")
            print("─" * 100)
            print(f"{'Predicted':<25} {v1_row['predicted_value']:25.1f} {v2_row['predicted_value']:25.1f} {v2_row['predicted_value'] - v1_row['predicted_value']:+20.1f}")
            print(f"{'Baseline':<25} {v1_row['baseline']:25.1f} {v2_row['baseline']:25.1f} {v2_row['baseline'] - v1_row['baseline']:+20.1f}")
            print(f"{'Diff from Baseline':<25} {v1_row['predicted_diff']:+25.1f} {v2_row['predicted_diff']:+25.1f} {v2_row['predicted_diff'] - v1_row['predicted_diff']:+20.1f}")

            if pd.notna(v1_row['confidence_lower']) and pd.notna(v2_row['confidence_lower']):
                v1_range = v1_row['confidence_upper'] - v1_row['confidence_lower']
                v2_range = v2_row['confidence_upper'] - v2_row['confidence_lower']
                print(f"{'Confidence Range':<25} {v1_range:25.1f} {v2_range:25.1f} {v2_range - v1_range:+20.1f}")

        if len(v1_pob) > 0 and len(v2_pob) > 0:
            v1_row = v1_pob.iloc[0]
            v2_row = v2_pob.iloc[0]

            v1_prob = v1_row['probability_over'] * 100
            v2_prob = v2_row['probability_over'] * 100

            print(f"{'Probability Over Base':<25} {v1_prob:24.1f}% {v2_prob:24.1f}% {v2_prob - v1_prob:+19.1f}%")

        # Compare individual stats
        v1_stats = v1_data[(v1_data['model_type'] == 'evob') &
                           (v1_data['stat'] != 'fantasy_points_ppr')]
        v2_stats = v2_data[(v2_data['model_type'] == 'evob') &
                           (v2_data['stat'] != 'fantasy_points_ppr')]

        if len(v1_stats) > 0 and len(v2_stats) > 0:
            print("\n📊 INDIVIDUAL STAT PREDICTIONS")
            print("─" * 100)
            print(f"{'Stat':<25} {'V1 Predicted':>15} {'V2 Predicted':>15} {'Difference':>15} {'V1 Diff':>15} {'V2 Diff':>15}")
            print("─" * 100)

            # Get common stats
            v1_stat_names = set(v1_stats['stat'].values)
            v2_stat_names = set(v2_stats['stat'].values)
            common_stats = v1_stat_names & v2_stat_names

            for stat_name in sorted(common_stats):
                v1_stat = v1_stats[v1_stats['stat'] == stat_name].iloc[0]
                v2_stat = v2_stats[v2_stats['stat'] == stat_name].iloc[0]

                diff = v2_stat['predicted_value'] - v1_stat['predicted_value']

                print(f"{stat_name:<25} {v1_stat['predicted_value']:15.1f} {v2_stat['predicted_value']:15.1f} ", end="")
                print(f"{diff:+15.1f} {v1_stat['predicted_diff']:+15.1f} {v2_stat['predicted_diff']:+15.1f}")

        print()

    return True


def compare_position_sample(position, v1_df, v2_df, v1_version, v2_version, n=5):
    """Compare top N players from a specific position"""

    # Filter for position and fantasy points
    v1_pos = v1_df[(v1_df['position'] == position) &
                   (v1_df['stat'] == 'fantasy_points_ppr') &
                   (v1_df['model_type'] == 'evob')].copy()

    v2_pos = v2_df[(v2_df['position'] == position) &
                   (v2_df['stat'] == 'fantasy_points_ppr') &
                   (v2_df['model_type'] == 'evob')].copy()

    if len(v1_pos) == 0 or len(v2_pos) == 0:
        print(f"❌ No {position} predictions found in both versions")
        return

    # Get top N players from V2
    v2_pos = v2_pos.sort_values('predicted_value', ascending=False).head(n)
    top_players = v2_pos['player_name'].values

    print("\n" + "="*100)
    print(f"  TOP {n} {position} PLAYERS - PREDICTION COMPARISON")
    print("="*100)
    print(f"{'Player':<22} {'V1 Pred':>10} {'V2 Pred':>10} {'Change':>10} {'V1 Base':>10} {'V2 Base':>10} {'V1 Prob':>10} {'V2 Prob':>10}")
    print("─" * 100)

    for player in top_players:
        v1_player = v1_pos[v1_pos['player_name'] == player]
        v2_player = v2_pos[v2_pos['player_name'] == player]

        if len(v1_player) > 0 and len(v2_player) > 0:
            v1_row = v1_player.iloc[0]
            v2_row = v2_player.iloc[0]

            # Get probability data
            v1_pob = v1_df[(v1_df['player_name'] == player) &
                          (v1_df['stat'] == 'fantasy_points_ppr') &
                          (v1_df['model_type'] == 'pob')]
            v2_pob = v2_df[(v2_df['player_name'] == player) &
                          (v2_df['stat'] == 'fantasy_points_ppr') &
                          (v2_df['model_type'] == 'pob')]

            v1_prob = v1_pob.iloc[0]['probability_over'] * 100 if len(v1_pob) > 0 else 0
            v2_prob = v2_pob.iloc[0]['probability_over'] * 100 if len(v2_pob) > 0 else 0

            change = v2_row['predicted_value'] - v1_row['predicted_value']

            print(f"{player:<22} {v1_row['predicted_value']:10.1f} {v2_row['predicted_value']:10.1f} ", end="")
            print(f"{change:+10.1f} {v1_row['baseline']:10.1f} {v2_row['baseline']:10.1f} ", end="")
            print(f"{v1_prob:9.1f}% {v2_prob:9.1f}%")

    print()


def main():
    """Main function"""

    if len(sys.argv) < 4:
        print("Usage: python compare_player_predictions.py <season> <week> <v1_version> <v2_version> [player_names...]")
        print("\nExample:")
        print("  python compare_player_predictions.py 2025 11 v1_baseline_mae5.14 v2_variance_trends_mae4.66")
        print("  python compare_player_predictions.py 2025 11 v1_baseline_mae5.14 v2_variance_trends_mae4.66 Mahomes Henry")
        return 1

    season = int(sys.argv[1])
    week = int(sys.argv[2])
    v1_version = sys.argv[3]
    v2_version = sys.argv[4]

    print("\n" + "="*100)
    print("  V1 vs V2 PLAYER PREDICTION COMPARISON")
    print("="*100)
    print(f"  Season: {season}  |  Week: {week}")
    print(f"  V1: {v1_version}")
    print(f"  V2: {v2_version}")
    print("="*100)

    # Load predictions
    print("\nLoading predictions...")
    v1_df = load_predictions(v1_version, season, week)
    v2_df = load_predictions(v2_version, season, week)

    if v1_df is None:
        print(f"❌ V1 predictions not found: {v1_version}")
        return 1

    if v2_df is None:
        print(f"❌ V2 predictions not found: {v2_version}")
        return 1

    print(f"✅ V1: {len(v1_df)} predictions, {v1_df['player_name'].nunique()} players")
    print(f"✅ V2: {len(v2_df)} predictions, {v2_df['player_name'].nunique()} players")

    # If specific players were provided, compare them
    if len(sys.argv) > 5:
        player_names = sys.argv[5:]
        for player_name in player_names:
            compare_player(player_name, v1_df, v2_df, v1_version, v2_version)
    else:
        # Compare top players from each position
        print("\n📊 Comparing top players from each position...")
        for position in ['QB', 'RB', 'WR', 'TE', 'K']:
            compare_position_sample(position, v1_df, v2_df, v1_version, v2_version, n=5)

    return 0


if __name__ == "__main__":
    sys.exit(main())
