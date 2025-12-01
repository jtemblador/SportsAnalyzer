"""
File: src/nfl/feature_engineer.py

FeatureEngineer class for calculating predictive features from raw NFL stats.
Processes all historical data (2020-2025) and saves engineered features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


class FeatureEngineer:
    """
    Transforms raw player statistics into predictive features for ML models.
    Handles rolling averages, opponent adjustments, usage trends, and context.
    """
    
    def __init__(self, raw_data_dir='./data/nfl/raw', cleaned_data_dir='./data/nfl/cleaned'):
        """
        Initialize the FeatureEngineer.
        
        Args:
            raw_data_dir: Directory containing raw parquet files
            cleaned_data_dir: Directory to save engineered features
        """
        self.raw_data_dir = raw_data_dir
        self.cleaned_data_dir = cleaned_data_dir
        
        # Create cleaned directory if it doesn't exist
        Path(self.cleaned_data_dir).mkdir(parents=True, exist_ok=True)
        
        print("✓ FeatureEngineer initialized")
        print(f"  Raw data: {self.raw_data_dir}")
        print(f"  Cleaned data: {self.cleaned_data_dir}")
    
    # ===== UTILITY METHODS =====
    
    def load_player_history(self, player_id, current_season, current_week, games_back=6):
        """
        Load last N games for a player (cross-season if needed).
        
        Args:
            player_id: Player ID
            current_season: Current season
            current_week: Current week
            games_back: Number of games to retrieve
        
        Returns:
            DataFrame with player's last N games
        """
        games = []
        season = current_season
        week = current_week - 1
        
        while len(games) < games_back and season >= 2020:
            if week < 1:
                season -= 1
                week = 18
                if season < 2020:
                    break
            
            filepath = f"{self.raw_data_dir}/player_stats_{season}_week_{week}.parquet"
            
            if Path(filepath).exists():
                df = pd.read_parquet(filepath)
                player_row = df[df['player_id'] == player_id]
                
                if not player_row.empty:
                    games.append(player_row.iloc[0])
            
            week -= 1
        
        return pd.DataFrame(games) if games else pd.DataFrame()
    
    def calculate_rolling_average(self, values, decay_factor=0.9):
        """
        Calculate weighted rolling average with exponential decay.
        
        Args:
            values: List of values (most recent first)
            decay_factor: Decay rate (0.9 = 10% decay per game)
        
        Returns:
            Weighted average
        """
        if len(values) == 0:
            return 0.0
        
        weights = [decay_factor ** i for i in range(len(values))]
        weighted_sum = sum(v * w for v, w in zip(values, weights))
        weight_sum = sum(weights)
        
        return weighted_sum / weight_sum if weight_sum > 0 else 0.0
    
    def get_opponent_defense_rank(self, opponent_team, position, season, week):
        """
        Calculate opponent defense rank based on fantasy points allowed.
        
        Args:
            opponent_team: Team abbreviation
            position: Position (QB, RB, WR, TE)
            season: Season year
            week: Current week
        
        Returns:
            Defense rank (1-32, higher = easier matchup)
        """
        all_teams_data = {}
        
        for w in range(1, week):
            filepath = f"{self.raw_data_dir}/player_stats_{season}_week_{w}.parquet"
            if Path(filepath).exists():
                df = pd.read_parquet(filepath)
                position_df = df[df['position'] == position]
                
                for team in position_df['opponent_team'].unique():
                    if pd.notna(team):
                        team_data = position_df[position_df['opponent_team'] == team]
                        avg_pts = team_data['fantasy_points_ppr'].mean()
                        
                        if team not in all_teams_data:
                            all_teams_data[team] = []
                        all_teams_data[team].append(avg_pts)
        
        team_averages = {team: np.mean(pts) for team, pts in all_teams_data.items()}
        
        if not team_averages:
            return 16
        
        sorted_teams = sorted(team_averages.items(), key=lambda x: x[1], reverse=True)
        
        for rank, (team, avg) in enumerate(sorted_teams, 1):
            if team == opponent_team:
                return rank
        
        return 16
    
    def calculate_usage_trend(self, player_history, stat_name):
        """
        Calculate percentage change in usage over recent games.
        
        Args:
            player_history: DataFrame with player history
            stat_name: Stat to track (attempts, targets, carries)
        
        Returns:
            Percentage change
        """
        if len(player_history) < 3 or stat_name not in player_history.columns:
            return 0.0
        
        recent_3 = player_history.head(3)[stat_name].mean()
        older_3 = player_history.tail(3)[stat_name].mean()
        
        if older_3 == 0:
            return 0.0
        
        return ((recent_3 - older_3) / older_3) * 100
    
    # ===== POSITION-SPECIFIC FEATURE BUILDERS =====
    
    def engineer_qb_features(self, player_row, player_history, season, week):
        """Calculate QB-specific features."""
        features = self._base_features(player_row, season, week)
        
        # Rolling averages
        features['rolling_avg_passing_yds'] = self.calculate_rolling_average(
            player_history['passing_yards'].tolist()
        )
        features['rolling_avg_passing_tds'] = self.calculate_rolling_average(
            player_history['passing_tds'].tolist()
        )
        features['rolling_avg_interceptions'] = self.calculate_rolling_average(
            player_history['passing_interceptions'].tolist()
        )
        features['rolling_avg_completions'] = self.calculate_rolling_average(
            player_history['completions'].tolist()
        )
        features['rolling_avg_rushing_yds'] = self.calculate_rolling_average(
            player_history['rushing_yards'].tolist()
        )
        
        # Opponent
        features['opponent_pass_defense_rank'] = self.get_opponent_defense_rank(
            player_row['opponent_team'], 'QB', season, week
        )
        
        # Usage trend
        features['pass_attempts_trend'] = self.calculate_usage_trend(
            player_history, 'attempts'
        )
        
        return features
    
    def engineer_rb_features(self, player_row, player_history, season, week):
        """Calculate RB-specific features."""
        features = self._base_features(player_row, season, week)
        
        # Rolling averages
        features['rolling_avg_rushing_yds'] = self.calculate_rolling_average(
            player_history['rushing_yards'].tolist()
        )
        features['rolling_avg_rushing_tds'] = self.calculate_rolling_average(
            player_history['rushing_tds'].tolist()
        )
        features['rolling_avg_carries'] = self.calculate_rolling_average(
            player_history['carries'].tolist()
        )
        features['rolling_avg_receptions'] = self.calculate_rolling_average(
            player_history['receptions'].tolist()
        )
        features['rolling_avg_receiving_yds'] = self.calculate_rolling_average(
            player_history['receiving_yards'].tolist()
        )
        
        # Opponent
        features['opponent_rush_defense_rank'] = self.get_opponent_defense_rank(
            player_row['opponent_team'], 'RB', season, week
        )
        
        # Usage trends
        features['carry_share_trend'] = self.calculate_usage_trend(
            player_history, 'carries'
        )
        features['target_trend'] = self.calculate_usage_trend(
            player_history, 'targets'
        )
        
        return features
    
    def engineer_wr_features(self, player_row, player_history, season, week):
        """Calculate WR-specific features."""
        features = self._base_features(player_row, season, week)
        
        # Rolling averages
        features['rolling_avg_receiving_yds'] = self.calculate_rolling_average(
            player_history['receiving_yards'].tolist()
        )
        features['rolling_avg_receiving_tds'] = self.calculate_rolling_average(
            player_history['receiving_tds'].tolist()
        )
        features['rolling_avg_receptions'] = self.calculate_rolling_average(
            player_history['receptions'].tolist()
        )
        features['rolling_avg_targets'] = self.calculate_rolling_average(
            player_history['targets'].tolist()
        )
        features['rolling_avg_air_yards'] = self.calculate_rolling_average(
            player_history['receiving_air_yards'].tolist()
        )
        
        # Opponent
        features['opponent_pass_defense_rank'] = self.get_opponent_defense_rank(
            player_row['opponent_team'], 'WR', season, week
        )
        
        # Usage trends
        features['target_share_trend'] = self.calculate_usage_trend(
            player_history, 'target_share'
        )
        features['air_yards_share_trend'] = self.calculate_usage_trend(
            player_history, 'air_yards_share'
        )
        
        return features
    
    def engineer_te_features(self, player_row, player_history, season, week):
        """Calculate TE-specific features."""
        features = self._base_features(player_row, season, week)
        
        # Rolling averages (same as WR but different patterns)
        features['rolling_avg_receiving_yds'] = self.calculate_rolling_average(
            player_history['receiving_yards'].tolist()
        )
        features['rolling_avg_receiving_tds'] = self.calculate_rolling_average(
            player_history['receiving_tds'].tolist()
        )
        features['rolling_avg_receptions'] = self.calculate_rolling_average(
            player_history['receptions'].tolist()
        )
        features['rolling_avg_targets'] = self.calculate_rolling_average(
            player_history['targets'].tolist()
        )
        
        # Opponent
        features['opponent_pass_defense_rank'] = self.get_opponent_defense_rank(
            player_row['opponent_team'], 'TE', season, week
        )
        
        # Usage trends
        features['target_share_trend'] = self.calculate_usage_trend(
            player_history, 'target_share'
        )
        
        return features
    
    def engineer_k_features(self, player_row, player_history, season, week):
        """Calculate K-specific features."""
        features = self._base_features(player_row, season, week)
        
        # Rolling averages
        features['rolling_avg_fg_made'] = self.calculate_rolling_average(
            player_history['fg_made'].tolist()
        )
        features['rolling_avg_fg_att'] = self.calculate_rolling_average(
            player_history['fg_att'].tolist()
        )
        features['rolling_avg_pat_made'] = self.calculate_rolling_average(
            player_history['pat_made'].tolist()
        )
        
        # No opponent rank for kickers (team-dependent)
        features['opponent_defense_rank'] = 16  # Neutral
        
        return features
    
    def _base_features(self, player_row, season, week):
        """Create base features common to all positions."""
        return {
            'player_id': player_row['player_id'],
            'player_name': player_row['player_name'],
            'position': player_row['position'],
            'team': player_row['team'],
            'opponent_team': player_row['opponent_team'],
            'week': week,
            'season': season,
            'rolling_avg_fantasy_pts': 0.0,  # Will be calculated
            'rolling_avg_fantasy_ppr': 0.0,
            'games_in_history': 0,
            'has_sufficient_data': False
        }
    
    # ===== MAIN PIPELINE =====
    
    def engineer_week_features(self, season, week):
        """
        Process one week - calculate features for all players.
        
        Args:
            season: Season year
            week: Week number
        
        Returns:
            DataFrame with engineered features, or None if failed
        """
        filepath = f"{self.raw_data_dir}/player_stats_{season}_week_{week}.parquet"
        
        if not Path(filepath).exists():
            return None
        
        # Load raw data for this week
        raw_df = pd.read_parquet(filepath)
        
        all_features = []
        
        # Process each player
        for idx, player_row in raw_df.iterrows():
            player_id = player_row['player_id']
            position = player_row['position']
            
            # Load player history
            player_history = self.load_player_history(player_id, season, week)
            
            # Skip if no history
            if player_history.empty:
                continue
            
            # Calculate base rolling averages
            base_features = {
                'rolling_avg_fantasy_pts': self.calculate_rolling_average(
                    player_history['fantasy_points'].tolist()
                ),
                'rolling_avg_fantasy_ppr': self.calculate_rolling_average(
                    player_history['fantasy_points_ppr'].tolist()
                ),
                'games_in_history': len(player_history),
                'has_sufficient_data': len(player_history) >= 6
            }
            
            # Engineer position-specific features
            if position == 'QB':
                features = self.engineer_qb_features(player_row, player_history, season, week)
            elif position == 'RB':
                features = self.engineer_rb_features(player_row, player_history, season, week)
            elif position == 'WR':
                features = self.engineer_wr_features(player_row, player_history, season, week)
            elif position == 'TE':
                features = self.engineer_te_features(player_row, player_history, season, week)
            elif position == 'K':
                features = self.engineer_k_features(player_row, player_history, season, week)
            else:
                continue  # Skip other positions for now
            
            # Update base features
            features.update(base_features)
            
            all_features.append(features)
        
        # Convert to DataFrame
        if all_features:
            features_df = pd.DataFrame(all_features)
            
            # Save to cleaned directory
            output_path = f"{self.cleaned_data_dir}/features_{season}_week_{week}.parquet"
            features_df.to_parquet(output_path, index=False)
            
            return features_df
        
        return None
    
    def engineer_all_features(self, start_season=2020, end_season=2025):
        """
        Process all weeks from start_season to end_season.
        
        Args:
            start_season: Starting season year
            end_season: Ending season year
        """
        print("\n" + "=" * 70)
        print("🔧 FEATURE ENGINEERING PIPELINE")
        print("=" * 70)
        print(f"Processing seasons {start_season} to {end_season}")
        print(f"Output directory: {self.cleaned_data_dir}")
        print()
        
        total_processed = 0
        total_skipped = 0
        
        for season in range(start_season, end_season + 1):
            for week in range(1, 19):
                # Check if already exists
                output_path = f"{self.cleaned_data_dir}/features_{season}_week_{week}.parquet"
                if Path(output_path).exists():
                    total_skipped += 1
                    continue
                
                # Check if raw data exists
                raw_path = f"{self.raw_data_dir}/player_stats_{season}_week_{week}.parquet"
                if not Path(raw_path).exists():
                    continue
                
                print(f"Processing {season} Week {week}...", end=" ")
                
                result = self.engineer_week_features(season, week)
                
                if result is not None:
                    print(f"✓ ({len(result)} players)")
                    total_processed += 1
                else:
                    print("✗ (no data)")
        
        print("\n" + "=" * 70)
        print("✅ FEATURE ENGINEERING COMPLETE")
        print("=" * 70)
        print(f"Processed: {total_processed} weeks")
        print(f"Skipped (already exist): {total_skipped} weeks")
        print(f"Output: {self.cleaned_data_dir}")
        print("=" * 70)


# Allow running this file directly
if __name__ == "__main__":
    engineer = FeatureEngineer()
    engineer.engineer_all_features(start_season=2020, end_season=2025)