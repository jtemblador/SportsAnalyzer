"""
File: src/nfl/v3_feature_engineer.py

FeatureEngineer class for V3 - Enhanced with EPA, efficiency metrics, and position-specific decay.

VERSION: V3
Improvements over V2:
- EPA features (Expected Points Added) - #1 predictive stat in NFL analytics
- Efficiency metrics: PACR, RACR, WOPR, CPOE
- Target share & air yards share features
- YAC (Yards After Catch) features
- Position-specific decay factors:
    QB: 0.90 (more stable week-to-week)
    RB: 0.85 (moderate reactivity)
    WR: 0.85 (moderate reactivity)
    TE: 0.80 (highly volatile, catch hot streaks)
    K:  0.90 (stable usage)
- Kicker distance features (40-49, 50+)

Expected features: 50-55 columns (V2 had 42)

Used by: v3_retrain.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


class V3FeatureEngineer:
    """
    V3 Feature Engineering with EPA, efficiency metrics, and position-specific optimization.
    """

    # Position-specific decay factors
    POSITION_DECAY = {
        'QB': 0.90,  # More stable - less reactive to variance
        'RB': 0.85,  # Moderate reactivity
        'WR': 0.85,  # Moderate reactivity
        'TE': 0.80,  # Highly volatile - catch hot streaks faster
        'K': 0.90,   # Stable usage patterns
    }

    def __init__(self, raw_data_dir='./data/nfl/raw', features_dir='./data/nfl/features', version='v3_epa_efficiency'):
        """
        Initialize the V3 FeatureEngineer.

        Args:
            raw_data_dir: Directory containing raw parquet files
            features_dir: Base directory for features (will create versioned subdirectory)
            version: Feature version identifier
        """
        self.raw_data_dir = raw_data_dir
        self.features_base_dir = features_dir
        self.version = version

        # Create versioned subdirectory
        self.cleaned_data_dir = str(Path(features_dir) / version)
        Path(self.cleaned_data_dir).mkdir(parents=True, exist_ok=True)

        print("✓ V3 FeatureEngineer initialized")
        print(f"  Version: {version}")
        print(f"  Raw data: {self.raw_data_dir}")
        print(f"  Feature output: {self.cleaned_data_dir}")
        print(f"  Position decay factors: {self.POSITION_DECAY}")

    # ===== UTILITY METHODS =====

    def load_player_history(self, player_id, current_season, current_week, games_back=6):
        """
        Load last N games for a player (cross-season if needed).
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

    def calculate_rolling_average(self, values, decay_factor=0.85):
        """
        Calculate weighted rolling average with exponential decay.
        NOTE: V3 uses position-specific decay passed as parameter.
        """
        if len(values) == 0:
            return 0.0

        weights = [decay_factor ** i for i in range(len(values))]
        weighted_sum = sum(v * w for v, w in zip(values, weights))
        weight_sum = sum(weights)

        return weighted_sum / weight_sum if weight_sum > 0 else 0.0

    def safe_rolling_average(self, player_history, column_name, decay_factor=0.85):
        """
        Calculate rolling average with safe column access.
        Returns 0.0 if column doesn't exist or has no valid data.
        """
        if column_name not in player_history.columns:
            return 0.0

        values = player_history[column_name].fillna(0).tolist()
        return self.calculate_rolling_average(values, decay_factor)

    def calculate_variance(self, values):
        """Calculate standard deviation for consistency/boom-bust indicator."""
        if len(values) < 2:
            return 0.0
        return np.std(values)

    def safe_variance(self, player_history, column_name):
        """Calculate variance with safe column access."""
        if column_name not in player_history.columns:
            return 0.0
        values = player_history[column_name].fillna(0).tolist()
        return self.calculate_variance(values)

    def calculate_recent_trend(self, values, recent_n=3):
        """
        Calculate trend from recent N games vs previous games.
        Positive = improving, Negative = declining
        """
        if len(values) < recent_n + 1:
            return 0.0

        recent_avg = np.mean(values[:recent_n])
        old_avg = np.mean(values[recent_n:])

        if old_avg == 0:
            return 0.0

        return (recent_avg - old_avg) / old_avg

    def safe_recent_trend(self, player_history, column_name, recent_n=3):
        """Calculate recent trend with safe column access."""
        if column_name not in player_history.columns:
            return 0.0
        values = player_history[column_name].fillna(0).tolist()
        return self.calculate_recent_trend(values, recent_n)

    def get_opponent_defense_rank(self, opponent_team, position, season, week):
        """Calculate opponent defense rank based on fantasy points allowed."""
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
        """Calculate percentage change in usage over recent games."""
        if len(player_history) < 3 or stat_name not in player_history.columns:
            return 0.0

        recent_3 = player_history.head(3)[stat_name].mean()
        older_3 = player_history.tail(3)[stat_name].mean()

        if older_3 == 0:
            return 0.0

        return ((recent_3 - older_3) / older_3) * 100

    # ===== POSITION-SPECIFIC FEATURE BUILDERS =====

    def engineer_qb_features(self, player_row, player_history, season, week):
        """
        Calculate QB-specific features.
        V3 NEW: EPA, CPOE, PACR, YAC features + decay=0.90
        """
        features = self._base_features(player_row, season, week)
        decay = self.POSITION_DECAY['QB']

        # === V2 FEATURES (rolling averages with QB-specific decay) ===
        features['rolling_avg_passing_yds'] = self.safe_rolling_average(
            player_history, 'passing_yards', decay
        )
        features['rolling_avg_passing_tds'] = self.safe_rolling_average(
            player_history, 'passing_tds', decay
        )
        features['rolling_avg_interceptions'] = self.safe_rolling_average(
            player_history, 'passing_interceptions', decay
        )
        features['rolling_avg_completions'] = self.safe_rolling_average(
            player_history, 'completions', decay
        )
        features['rolling_avg_rushing_yds'] = self.safe_rolling_average(
            player_history, 'rushing_yards', decay
        )

        # Variance features
        features['fantasy_pts_variance'] = self.safe_variance(
            player_history, 'fantasy_points_ppr'
        )
        features['passing_yds_variance'] = self.safe_variance(
            player_history, 'passing_yards'
        )

        # Trend features
        features['fantasy_pts_trend'] = self.safe_recent_trend(
            player_history, 'fantasy_points_ppr'
        )
        features['passing_yds_trend'] = self.safe_recent_trend(
            player_history, 'passing_yards'
        )

        # Opponent
        features['opponent_pass_defense_rank'] = self.get_opponent_defense_rank(
            player_row['opponent_team'], 'QB', season, week
        )

        # Usage trend
        features['pass_attempts_trend'] = self.calculate_usage_trend(
            player_history, 'attempts'
        )

        # === V3 NEW FEATURES ===

        # EPA (Expected Points Added) - THE key efficiency metric
        features['rolling_avg_passing_epa'] = self.safe_rolling_average(
            player_history, 'passing_epa', decay
        )

        # CPOE (Completion Percentage Over Expected) - accuracy metric
        features['rolling_avg_cpoe'] = self.safe_rolling_average(
            player_history, 'passing_cpoe', decay
        )

        # PACR (Pass Air Conversion Ratio) - efficiency converting air yards
        features['rolling_avg_pacr'] = self.safe_rolling_average(
            player_history, 'pacr', decay
        )

        # YAC (Yards After Catch by receivers) - indicates receiver quality
        features['rolling_avg_passing_yac'] = self.safe_rolling_average(
            player_history, 'passing_yards_after_catch', decay
        )

        return features

    def engineer_rb_features(self, player_row, player_history, season, week):
        """
        Calculate RB-specific features.
        V3 NEW: rushing_epa + decay=0.85
        """
        features = self._base_features(player_row, season, week)
        decay = self.POSITION_DECAY['RB']

        # === V2 FEATURES ===
        features['rolling_avg_rushing_yds'] = self.safe_rolling_average(
            player_history, 'rushing_yards', decay
        )
        features['rolling_avg_rushing_tds'] = self.safe_rolling_average(
            player_history, 'rushing_tds', decay
        )
        features['rolling_avg_carries'] = self.safe_rolling_average(
            player_history, 'carries', decay
        )
        features['rolling_avg_receptions'] = self.safe_rolling_average(
            player_history, 'receptions', decay
        )
        features['rolling_avg_receiving_yds'] = self.safe_rolling_average(
            player_history, 'receiving_yards', decay
        )

        # Variance features
        features['fantasy_pts_variance'] = self.safe_variance(
            player_history, 'fantasy_points_ppr'
        )
        features['rushing_yds_variance'] = self.safe_variance(
            player_history, 'rushing_yards'
        )

        # Trend features
        features['fantasy_pts_trend'] = self.safe_recent_trend(
            player_history, 'fantasy_points_ppr'
        )
        features['carries_trend'] = self.safe_recent_trend(
            player_history, 'carries'
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

        # === V3 NEW FEATURES ===

        # EPA (Expected Points Added) - efficiency per carry
        features['rolling_avg_rushing_epa'] = self.safe_rolling_average(
            player_history, 'rushing_epa', decay
        )

        # Receiving EPA for pass-catching backs
        features['rolling_avg_receiving_epa'] = self.safe_rolling_average(
            player_history, 'receiving_epa', decay
        )

        return features

    def engineer_wr_features(self, player_row, player_history, season, week):
        """
        Calculate WR-specific features.
        V3 NEW: EPA, RACR, WOPR, YAC, target_share, air_yards_share + decay=0.85
        """
        features = self._base_features(player_row, season, week)
        decay = self.POSITION_DECAY['WR']

        # === V2 FEATURES ===
        features['rolling_avg_receiving_yds'] = self.safe_rolling_average(
            player_history, 'receiving_yards', decay
        )
        features['rolling_avg_receiving_tds'] = self.safe_rolling_average(
            player_history, 'receiving_tds', decay
        )
        features['rolling_avg_receptions'] = self.safe_rolling_average(
            player_history, 'receptions', decay
        )
        features['rolling_avg_targets'] = self.safe_rolling_average(
            player_history, 'targets', decay
        )
        features['rolling_avg_air_yards'] = self.safe_rolling_average(
            player_history, 'receiving_air_yards', decay
        )

        # Variance features
        features['fantasy_pts_variance'] = self.safe_variance(
            player_history, 'fantasy_points_ppr'
        )
        features['receiving_yds_variance'] = self.safe_variance(
            player_history, 'receiving_yards'
        )

        # Trend features
        features['fantasy_pts_trend'] = self.safe_recent_trend(
            player_history, 'fantasy_points_ppr'
        )
        features['targets_trend'] = self.safe_recent_trend(
            player_history, 'targets'
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

        # === V3 NEW FEATURES ===

        # EPA (Expected Points Added) - THE key efficiency metric
        features['rolling_avg_receiving_epa'] = self.safe_rolling_average(
            player_history, 'receiving_epa', decay
        )

        # RACR (Receiver Air Conversion Ratio) - efficiency metric
        features['rolling_avg_racr'] = self.safe_rolling_average(
            player_history, 'racr', decay
        )

        # WOPR (Weighted Opportunity Rating) - opportunity metric
        features['rolling_avg_wopr'] = self.safe_rolling_average(
            player_history, 'wopr', decay
        )

        # YAC (Yards After Catch) - elusiveness metric
        features['rolling_avg_receiving_yac'] = self.safe_rolling_average(
            player_history, 'receiving_yards_after_catch', decay
        )

        # Target share - team context adjusted
        features['rolling_avg_target_share'] = self.safe_rolling_average(
            player_history, 'target_share', decay
        )

        # Air yards share - deep threat indicator
        features['rolling_avg_air_yards_share'] = self.safe_rolling_average(
            player_history, 'air_yards_share', decay
        )

        # Target share variance - consistency metric
        features['target_share_variance'] = self.safe_variance(
            player_history, 'target_share'
        )

        return features

    def engineer_te_features(self, player_row, player_history, season, week):
        """
        Calculate TE-specific features.
        V3 NEW: EPA, RACR, YAC, target_share + decay=0.80 (more reactive for volatile TEs)
        """
        features = self._base_features(player_row, season, week)
        decay = self.POSITION_DECAY['TE']

        # === V2 FEATURES ===
        features['rolling_avg_receiving_yds'] = self.safe_rolling_average(
            player_history, 'receiving_yards', decay
        )
        features['rolling_avg_receiving_tds'] = self.safe_rolling_average(
            player_history, 'receiving_tds', decay
        )
        features['rolling_avg_receptions'] = self.safe_rolling_average(
            player_history, 'receptions', decay
        )
        features['rolling_avg_targets'] = self.safe_rolling_average(
            player_history, 'targets', decay
        )

        # Variance features
        features['fantasy_pts_variance'] = self.safe_variance(
            player_history, 'fantasy_points_ppr'
        )
        features['receiving_yds_variance'] = self.safe_variance(
            player_history, 'receiving_yards'
        )

        # Trend features
        features['fantasy_pts_trend'] = self.safe_recent_trend(
            player_history, 'fantasy_points_ppr'
        )
        features['targets_trend'] = self.safe_recent_trend(
            player_history, 'targets'
        )

        # Opponent
        features['opponent_pass_defense_rank'] = self.get_opponent_defense_rank(
            player_row['opponent_team'], 'TE', season, week
        )

        # Usage trends
        features['target_share_trend'] = self.calculate_usage_trend(
            player_history, 'target_share'
        )

        # === V3 NEW FEATURES ===

        # EPA (Expected Points Added)
        features['rolling_avg_receiving_epa'] = self.safe_rolling_average(
            player_history, 'receiving_epa', decay
        )

        # RACR (Receiver Air Conversion Ratio)
        features['rolling_avg_racr'] = self.safe_rolling_average(
            player_history, 'racr', decay
        )

        # YAC (Yards After Catch) - elusiveness metric
        features['rolling_avg_receiving_yac'] = self.safe_rolling_average(
            player_history, 'receiving_yards_after_catch', decay
        )

        # Target share - team context adjusted
        features['rolling_avg_target_share'] = self.safe_rolling_average(
            player_history, 'target_share', decay
        )

        # Target share variance - consistency metric
        features['target_share_variance'] = self.safe_variance(
            player_history, 'target_share'
        )

        return features

    def engineer_k_features(self, player_row, player_history, season, week):
        """
        Calculate K-specific features.
        V3 NEW: Distance features (40-49, 50+) + decay=0.90
        """
        features = self._base_features(player_row, season, week)
        decay = self.POSITION_DECAY['K']

        # === V2 FEATURES ===
        features['rolling_avg_fg_made'] = self.safe_rolling_average(
            player_history, 'fg_made', decay
        )
        features['rolling_avg_fg_att'] = self.safe_rolling_average(
            player_history, 'fg_att', decay
        )
        features['rolling_avg_pat_made'] = self.safe_rolling_average(
            player_history, 'pat_made', decay
        )

        # No opponent rank for kickers (team-dependent)
        features['opponent_defense_rank'] = 16  # Neutral

        # === V3 NEW FEATURES ===

        # Distance features - 40-49 yard FGs
        features['rolling_avg_fg_made_40_49'] = self.safe_rolling_average(
            player_history, 'fg_made_40_49', decay
        )

        # 50+ yard FGs (combine 50-59 and 60+)
        fg_50_plus = []
        if 'fg_made_50_59' in player_history.columns and 'fg_made_60_' in player_history.columns:
            for _, row in player_history.iterrows():
                fg_50 = row.get('fg_made_50_59', 0) or 0
                fg_60 = row.get('fg_made_60_', 0) or 0
                fg_50_plus.append(fg_50 + fg_60)
            features['rolling_avg_fg_made_50_plus'] = self.calculate_rolling_average(fg_50_plus, decay)
        else:
            features['rolling_avg_fg_made_50_plus'] = 0.0

        # Average FG distance made
        features['rolling_avg_fg_made_distance'] = self.safe_rolling_average(
            player_history, 'fg_made_distance', decay
        )

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
            'rolling_avg_fantasy_pts': 0.0,
            'rolling_avg_fantasy_ppr': 0.0,
            'games_in_history': 0,
            'has_sufficient_data': False
        }

    # ===== MAIN PIPELINE =====

    def engineer_week_features(self, season, week):
        """
        Process one week - calculate features for all players.
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

            # Get position-specific decay
            decay = self.POSITION_DECAY.get(position, 0.85)

            # Calculate base rolling averages with position-specific decay
            base_features = {
                'rolling_avg_fantasy_pts': self.safe_rolling_average(
                    player_history, 'fantasy_points', decay
                ),
                'rolling_avg_fantasy_ppr': self.safe_rolling_average(
                    player_history, 'fantasy_points_ppr', decay
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
                continue

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

    def engineer_all_features(self, start_season=2020, end_season=2025, force_regenerate=False):
        """
        Process all weeks from start_season to end_season.

        Args:
            start_season: Starting season year
            end_season: Ending season year
            force_regenerate: If True, regenerate even if file exists
        """
        print("\n" + "=" * 70)
        print("🔧 V3 FEATURE ENGINEERING PIPELINE")
        print("=" * 70)
        print(f"Processing seasons {start_season} to {end_season}")
        print(f"Output directory: {self.cleaned_data_dir}")
        print(f"Position decay factors: {self.POSITION_DECAY}")
        print()

        total_processed = 0
        total_skipped = 0

        for season in range(start_season, end_season + 1):
            for week in range(1, 19):
                # Check if already exists
                output_path = f"{self.cleaned_data_dir}/features_{season}_week_{week}.parquet"
                if Path(output_path).exists() and not force_regenerate:
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
        print("✅ V3 FEATURE ENGINEERING COMPLETE")
        print("=" * 70)
        print(f"Processed: {total_processed} weeks")
        print(f"Skipped (already exist): {total_skipped} weeks")
        print(f"Output: {self.cleaned_data_dir}")
        print("=" * 70)


# Allow running this file directly for testing
if __name__ == "__main__":
    # Test on a single week first
    engineer = V3FeatureEngineer(version='v3_epa_efficiency_test')

    print("\n🧪 Testing V3 feature engineering on 2025 Week 13...")
    result = engineer.engineer_week_features(2025, 13)

    if result is not None:
        print(f"\n✅ Test successful! Generated {len(result)} player features")
        print(f"Feature columns ({len(result.columns)}): {list(result.columns)}")

        # Show sample features by position
        for pos in ['QB', 'RB', 'WR', 'TE', 'K']:
            pos_df = result[result['position'] == pos]
            if len(pos_df) > 0:
                print(f"\n{pos} features ({len(pos_df)} players):")
                print(f"  Columns: {len([c for c in result.columns if c not in ['player_id', 'player_name', 'position', 'team', 'opponent_team', 'week', 'season']])} engineered features")
    else:
        print("❌ Test failed - no features generated")
