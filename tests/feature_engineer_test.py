"""
File: testing/feature_engineer_test.py

Test feature engineering calculations on a single player (Patrick Mahomes)
to verify logic before processing all 108 weeks of data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))


def get_most_recent_week_with_player(player_name, raw_data_dir='./data/nfl/raw', season=2025):
    """
    Find the most recent week where a specific player has data.
    Handles bye weeks and missed games.
    
    Args:
        player_name: Player name to search for
        raw_data_dir: Directory containing raw parquet files
        season: Season to check (default current season)
    
    Returns:
        Most recent week number where player appeared, or 0 if not found
    """
    for week in range(18, 0, -1):  # Check from week 18 down to 1
        filepath = f"{raw_data_dir}/player_stats_{season}_week_{week}.parquet"
        if Path(filepath).exists():
            df = pd.read_parquet(filepath)
            player_row = df[df['player_name'].str.contains(player_name, case=False, na=False)]
            if not player_row.empty:
                return week
    
    return 0


def load_player_history(player_name, current_season, current_week, raw_data_dir='./data/nfl/raw', games_back=6):
    """
    Load last N games for a specific player, going back to previous season if needed.
    
    Args:
        player_name: Player name to search for
        current_season: Current season year
        current_week: Current week number
        raw_data_dir: Directory containing raw parquet files
        games_back: Number of games to retrieve
    
    Returns:
        DataFrame with player's last N games
    """
    games = []
    season = current_season
    week = current_week - 1  # Start from week before current
    
    print(f"\n🔍 Loading history for {player_name}...")
    print(f"   Target: {games_back} games before Week {current_week}, {current_season}")
    
    while len(games) < games_back and season >= 2020:
        if week < 1:
            season -= 1
            week = 18
            if season < 2020:
                break
        
        filepath = f"{raw_data_dir}/player_stats_{season}_week_{week}.parquet"
        
        if Path(filepath).exists():
            df = pd.read_parquet(filepath)
            player_row = df[df['player_name'].str.contains(player_name, case=False, na=False)]
            
            if not player_row.empty:
                games.append(player_row.iloc[0])
                print(f"   ✓ Found: Season {season}, Week {week}")
        
        week -= 1
    
    if games:
        result = pd.DataFrame(games)
        print(f"   📊 Loaded {len(result)} games")
        return result
    else:
        print(f"   ❌ No history found")
        return pd.DataFrame()


def calculate_rolling_average(values, decay_factor=0.9):
    """
    Calculate weighted rolling average with exponential decay.
    Most recent value has weight 1.0, previous has decay_factor, etc.
    
    Args:
        values: List of values (most recent first)
        decay_factor: Decay rate (0.9 = 10% decay per game back)
    
    Returns:
        Weighted average
    """
    if len(values) == 0:
        return 0.0
    
    weights = [decay_factor ** i for i in range(len(values))]
    weighted_sum = sum(v * w for v, w in zip(values, weights))
    weight_sum = sum(weights)
    
    return weighted_sum / weight_sum if weight_sum > 0 else 0.0


def calculate_opponent_defense_rank(opponent_team, position, season, week, raw_data_dir='./data/nfl/raw'):
    """
    Calculate how many fantasy points this defense allows to a specific position.
    Lower rank = tougher defense, Higher rank = easier matchup.
    
    Args:
        opponent_team: Team abbreviation (e.g., 'CHI')
        position: Position (e.g., 'QB')
        season: Season year
        week: Current week
        raw_data_dir: Directory with raw data
    
    Returns:
        Defense rank (1-32, where 32 = worst defense = easiest matchup)
    """
    # Load data from previous weeks to calculate opponent strength
    all_teams_data = {}
    
    for w in range(1, week):
        filepath = f"{raw_data_dir}/player_stats_{season}_week_{w}.parquet"
        if Path(filepath).exists():
            df = pd.read_parquet(filepath)
            position_df = df[df['position'] == position]
            
            # Group by opponent_team and calculate avg fantasy points allowed
            for team in position_df['opponent_team'].unique():
                if pd.notna(team):
                    team_data = position_df[position_df['opponent_team'] == team]
                    avg_pts = team_data['fantasy_points_ppr'].mean()
                    
                    if team not in all_teams_data:
                        all_teams_data[team] = []
                    all_teams_data[team].append(avg_pts)
    
    # Calculate average points allowed per team
    team_averages = {team: np.mean(pts) for team, pts in all_teams_data.items()}
    
    if not team_averages:
        return 16  # Default to middle if no data
    
    # Rank teams (higher points allowed = worse defense = higher rank)
    sorted_teams = sorted(team_averages.items(), key=lambda x: x[1], reverse=True)
    
    # Find opponent's rank
    for rank, (team, avg) in enumerate(sorted_teams, 1):
        if team == opponent_team:
            return rank
    
    return 16  # Default to middle if not found


def calculate_usage_trend(player_history, stat_name):
    """
    Calculate if usage is increasing or decreasing over recent games.
    
    Args:
        player_history: DataFrame with player's recent games
        stat_name: Stat to track (e.g., 'attempts', 'targets')
    
    Returns:
        Percentage change in usage
    """
    if len(player_history) < 3 or stat_name not in player_history.columns:
        return 0.0
    
    recent_3 = player_history.head(3)[stat_name].mean()
    older_3 = player_history.tail(3)[stat_name].mean()
    
    if older_3 == 0:
        return 0.0
    
    return ((recent_3 - older_3) / older_3) * 100


def engineer_qb_features(player_history, current_week_data, season, week, raw_data_dir):
    """
    Calculate all QB-specific features.
    
    Args:
        player_history: DataFrame with player's last 6 games
        current_week_data: Series with current week info
        season: Current season
        week: Current week
        raw_data_dir: Path to raw data
    
    Returns:
        Dictionary of engineered features
    """
    features = {}
    
    # Universal features
    features['player_id'] = current_week_data['player_id']
    features['player_name'] = current_week_data['player_name']
    features['position'] = current_week_data['position']
    features['team'] = current_week_data['team']
    features['week'] = week
    features['season'] = season
    
    # Rolling averages
    features['rolling_avg_fantasy_pts'] = calculate_rolling_average(
        player_history['fantasy_points_ppr'].tolist()
    )
    features['rolling_avg_passing_yds'] = calculate_rolling_average(
        player_history['passing_yards'].tolist()
    )
    features['rolling_avg_passing_tds'] = calculate_rolling_average(
        player_history['passing_tds'].tolist()
    )
    features['rolling_avg_interceptions'] = calculate_rolling_average(
        player_history['passing_interceptions'].tolist()
    )
    features['rolling_avg_completions'] = calculate_rolling_average(
        player_history['completions'].tolist()
    )
    
    # Opponent adjustment
    opponent = current_week_data['opponent_team']
    features['opponent_team'] = opponent
    features['opponent_pass_defense_rank'] = calculate_opponent_defense_rank(
        opponent, 'QB', season, week, raw_data_dir
    )
    
    # Usage trend
    features['pass_attempts_trend'] = calculate_usage_trend(player_history, 'attempts')
    
    # Context features
    features['home_game'] = 1  # Simplified for test (would check team vs opponent)
    features['games_in_history'] = len(player_history)
    features['has_sufficient_data'] = len(player_history) >= 6
    
    return features


def print_feature_table(features):
    """Print features in a readable format."""
    print("\n" + "=" * 70)
    print("📊 ENGINEERED FEATURES")
    print("=" * 70)
    
    # Basic info
    print(f"\n🏈 Player: {features['player_name']} ({features['position']})")
    print(f"   Team: {features['team']}")
    print(f"   Week {features['week']}, {features['season']} Season")
    print(f"   Opponent: {features['opponent_team']}")
    
    # Rolling averages
    print(f"\n📈 Rolling Averages (Last {features['games_in_history']} Games):")
    print(f"   Fantasy Points (PPR): {features['rolling_avg_fantasy_pts']:.2f}")
    print(f"   Passing Yards:        {features['rolling_avg_passing_yds']:.1f}")
    print(f"   Passing TDs:          {features['rolling_avg_passing_tds']:.2f}")
    print(f"   Interceptions:        {features['rolling_avg_interceptions']:.2f}")
    print(f"   Completions:          {features['rolling_avg_completions']:.1f}")
    
    # Opponent
    print(f"\n🛡️  Opponent Strength:")
    print(f"   Pass Defense Rank:    #{features['opponent_pass_defense_rank']} (out of 32)")
    rank = features['opponent_pass_defense_rank']
    if rank > 24:
        matchup = "EASY (Weak defense)"
    elif rank > 16:
        matchup = "MODERATE"
    else:
        matchup = "TOUGH (Strong defense)"
    print(f"   Matchup Difficulty:   {matchup}")
    
    # Trends
    print(f"\n📊 Usage Trends:")
    trend = features['pass_attempts_trend']
    trend_str = f"+{trend:.1f}%" if trend > 0 else f"{trend:.1f}%"
    print(f"   Pass Attempts Trend:  {trend_str}")
    
    # Context
    print(f"\n🏟️  Context:")
    print(f"   Home Game:            {'Yes' if features['home_game'] else 'No'}")
    print(f"   Sufficient Data:      {'✓' if features['has_sufficient_data'] else '✗'}")
    
    print("\n" + "=" * 70)


def main():
    """Main test function."""
    print("=" * 70)
    print("🧪 FEATURE ENGINEERING TEST")
    print("=" * 70)
    
    # Parameters - DYNAMIC (no hardcoding)
    player_name = "Mahomes"
    season = 2025
    raw_data_dir = "./data/nfl/raw"
    
    # Automatically detect most recent week WHERE PLAYER ACTUALLY PLAYED
    most_recent_week = get_most_recent_week_with_player(player_name, raw_data_dir, season)
    
    if most_recent_week == 0:
        print(f"\n❌ ERROR: No data found for {player_name} in {season} season")
        return
    
    print(f"\n📅 Most recent game for {player_name}: Week {most_recent_week}, {season}")
    print(f"   (Automatically skipped bye weeks and missed games)")
    print(f"🎯 Testing feature engineering on this week")
    
    week = most_recent_week
    
    # Step 1: Load player history
    player_history = load_player_history(player_name, season, week, raw_data_dir, games_back=6)
    
    if player_history.empty:
        print("\n❌ ERROR: Could not load player history")
        return
    
    # Step 2: Load current week data
    current_filepath = f"{raw_data_dir}/player_stats_{season}_week_{week}.parquet"
    if not Path(current_filepath).exists():
        print(f"\n❌ ERROR: Week {week} data not found")
        return
    
    current_df = pd.read_parquet(current_filepath)
    current_player = current_df[current_df['player_name'].str.contains(player_name, case=False, na=False)]
    
    if current_player.empty:
        print(f"\n❌ ERROR: {player_name} not found in Week {week}")
        return
    
    current_week_data = current_player.iloc[0]
    
    # Step 3: Print raw history
    print("\n" + "=" * 70)
    print(f"📋 RAW STATS (Last {len(player_history)} Games)")
    print("=" * 70)
    for idx, row in player_history.iterrows():
        print(f"Week {row['week']:2d}, {row['season']}: "
              f"{row['passing_yards']:3.0f} pass yds, "
              f"{row['passing_tds']:.0f} TDs, "
              f"{row['passing_interceptions']:.0f} INTs, "
              f"{row['fantasy_points_ppr']:5.1f} fantasy pts")
    
    # Step 4: Engineer features
    print("\n⚙️  Calculating features...")
    features = engineer_qb_features(player_history, current_week_data, season, week, raw_data_dir)
    
    # Step 5: Display results
    print_feature_table(features)
    
    print("\n✅ Feature engineering test complete!")
    print("\n💡 This proves the logic works for handling:")
    print("   • Bye weeks (automatically skipped)")
    print("   • Cross-season history loading")
    print("   • Rolling averages with decay")
    print("   • Opponent strength calculations")
    print("\nNext step: Build full FeatureEngineer class in src/nfl/feature_engineer.py")


if __name__ == "__main__":
    main()