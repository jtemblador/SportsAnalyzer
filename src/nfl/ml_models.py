"""
File: src/nfl/ml_models_fixed.py

Fixed version with XGBoost compatibility
Just rename this to ml_models.py to use
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import joblib
import json
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


class POBModel:
    """
    Probability Over Baseline - Binary Classification
    Predicts probability that a player will exceed their rolling average
    """
    
    def __init__(self, position: str, target_stat: str = 'fantasy_points_ppr'):
        """
        Initialize POB model for a specific position and stat.
        
        Args:
            position: Player position (QB, RB, WR, TE, K)
            target_stat: Stat to predict over/under for
        """
        self.position = position
        self.target_stat = target_stat
        self.models = {}
        self.feature_columns = []
        self.scaler = StandardScaler()
        self.metadata = {
            'position': position,
            'target_stat': target_stat,
            'created_date': datetime.now().isoformat(),
            'model_versions': {}
        }
        
        # Initialize ensemble models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ensemble of classification models"""
        self.models = {
            'xgboost': XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.01,
                objective='binary:logistic',
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=42,
                early_stopping_rounds=20  # Set here for new API
            ),
            'lightgbm': LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.01,
                objective='binary',
                metric='binary_logloss',
                random_state=42,
                verbose=-1
            ),
            'catboost': CatBoostClassifier(
                iterations=200,
                depth=6,
                learning_rate=0.01,
                loss_function='Logloss',
                random_state=42,
                verbose=False
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=6,
                random_state=42
            )
        }
    
    def prepare_target(self, df: pd.DataFrame) -> pd.Series:
        """
        Create binary target: 1 if player exceeds their rolling average, 0 otherwise.

        Args:
            df: DataFrame with features and actual stats

        Returns:
            Binary target series
        """
        # Get the actual target column
        target_col = f'{self.target_stat}_target'

        if target_col not in df.columns:
            raise ValueError(f"Target column {target_col} not found in dataframe")

        # Map stat names to their actual feature column names
        stat_to_feature_map = {
            'fantasy_points_ppr': 'rolling_avg_fantasy_ppr',
            'fantasy_points': 'rolling_avg_fantasy_pts',
            'passing_yards': 'rolling_avg_passing_yds',
            'passing_tds': 'rolling_avg_passing_tds',
            'passing_interceptions': 'rolling_avg_interceptions',
            'rushing_yards': 'rolling_avg_rushing_yds',
            'rushing_tds': 'rolling_avg_rushing_tds',
            'receiving_yards': 'rolling_avg_receiving_yds',
            'receiving_tds': 'rolling_avg_receiving_tds',
            'receptions': 'rolling_avg_receptions',
            'carries': 'rolling_avg_carries',
            'targets': 'rolling_avg_targets',
            'completions': 'rolling_avg_completions',
            'attempts': 'rolling_avg_attempts',
            'fg_made': 'rolling_avg_fg_made',
            'fg_att': 'rolling_avg_fg_att'
        }

        # Get baseline column using mapping or construct default
        baseline_col = stat_to_feature_map.get(self.target_stat, f'rolling_avg_{self.target_stat}')

        if baseline_col not in df.columns:
            raise ValueError(f"Baseline column {baseline_col} not found in dataframe")

        # Create binary target
        actual_values = df[target_col]
        baseline = df[baseline_col]

        return (actual_values > baseline).astype(int)
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, float]:
        """
        Train ensemble of models.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Dictionary of validation scores by model
        """
        self.feature_columns = X_train.columns.tolist()
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        val_scores = {}
        
        for name, model in self.models.items():
            print(f"  Training {name}...")
            
            # Train model with API-specific handling
            if name == 'xgboost':
                model.fit(
                    X_train_scaled, y_train,
                    eval_set=[(X_val_scaled, y_val)],
                    verbose=False
                )
            elif name == 'lightgbm':
                model.fit(
                    X_train_scaled, y_train,
                    eval_set=[(X_val_scaled, y_val)],
                    callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
                )
            else:
                model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = model.predict(X_val_scaled)
            val_scores[name] = {
                'accuracy': accuracy_score(y_val, y_pred),
                'precision': precision_score(y_val, y_pred, zero_division=0),
                'recall': recall_score(y_val, y_pred, zero_division=0),
                'f1': f1_score(y_val, y_pred, zero_division=0)
            }
            
            print(f"    Accuracy: {val_scores[name]['accuracy']:.3f}, F1: {val_scores[name]['f1']:.3f}")
        
        self.metadata['model_versions'] = val_scores
        self.metadata['training_date'] = datetime.now().isoformat()
        
        return val_scores
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using ensemble averaging.
        
        Args:
            X: Features for prediction
            
        Returns:
            Probability array (0-1)
        """
        X_scaled = self.scaler.transform(X[self.feature_columns])
        
        predictions = []
        for name, model in self.models.items():
            pred_proba = model.predict_proba(X_scaled)[:, 1]
            predictions.append(pred_proba)
        
        # Average ensemble predictions
        return np.mean(predictions, axis=0)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model performance on test set"""
        y_pred_proba = self.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }
    
    def save_model(self, filepath: str):
        """Save model and metadata"""
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'metadata': self.metadata
        }
        # Ensure parent directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, filepath)
        print(f"  POB model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from file"""
        model_data = joblib.load(filepath)
        self.models = model_data['models']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.metadata = model_data['metadata']


class EVOBModel:
    """
    Expected Value Over Baseline - Regression
    Predicts the difference from rolling average
    """
    
    def __init__(self, position: str, target_stat: str = 'fantasy_points_ppr'):
        """
        Initialize EVOB model for a specific position and stat.
        
        Args:
            position: Player position (QB, RB, WR, TE, K)
            target_stat: Stat to predict differential for
        """
        self.position = position
        self.target_stat = target_stat
        self.models = {}
        self.feature_columns = []
        self.scaler = StandardScaler()
        self.metadata = {
            'position': position,
            'target_stat': target_stat,
            'created_date': datetime.now().isoformat(),
            'model_versions': {}
        }
        
        # Initialize ensemble models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ensemble of regression models"""
        self.models = {
            'xgboost': XGBRegressor(
                n_estimators=300,
                max_depth=7,
                learning_rate=0.01,
                objective='reg:squarederror',
                random_state=42,
                early_stopping_rounds=20  # Set here for new API
            ),
            'lightgbm': LGBMRegressor(
                n_estimators=300,
                max_depth=7,
                learning_rate=0.01,
                objective='regression',
                metric='rmse',
                random_state=42,
                verbose=-1
            ),
            'catboost': CatBoostRegressor(
                iterations=300,
                depth=7,
                learning_rate=0.01,
                loss_function='RMSE',
                random_state=42,
                verbose=False
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=7,
                random_state=42
            )
        }
    
    def prepare_target(self, df: pd.DataFrame) -> pd.Series:
        """
        Create regression target: difference from rolling average.

        Args:
            df: DataFrame with features and actual stats

        Returns:
            Differential target series
        """
        # Get the actual target column
        target_col = f'{self.target_stat}_target'

        if target_col not in df.columns:
            raise ValueError(f"Target column {target_col} not found in dataframe")

        # Map stat names to their actual feature column names
        stat_to_feature_map = {
            'fantasy_points_ppr': 'rolling_avg_fantasy_ppr',
            'fantasy_points': 'rolling_avg_fantasy_pts',
            'passing_yards': 'rolling_avg_passing_yds',
            'passing_tds': 'rolling_avg_passing_tds',
            'passing_interceptions': 'rolling_avg_interceptions',
            'rushing_yards': 'rolling_avg_rushing_yds',
            'rushing_tds': 'rolling_avg_rushing_tds',
            'receiving_yards': 'rolling_avg_receiving_yds',
            'receiving_tds': 'rolling_avg_receiving_tds',
            'receptions': 'rolling_avg_receptions',
            'carries': 'rolling_avg_carries',
            'targets': 'rolling_avg_targets',
            'completions': 'rolling_avg_completions',
            'attempts': 'rolling_avg_attempts',
            'fg_made': 'rolling_avg_fg_made',
            'fg_att': 'rolling_avg_fg_att'
        }

        # Get baseline column using mapping or construct default
        baseline_col = stat_to_feature_map.get(self.target_stat, f'rolling_avg_{self.target_stat}')

        if baseline_col not in df.columns:
            raise ValueError(f"Baseline column {baseline_col} not found in dataframe")

        # Calculate differential
        actual_values = df[target_col]
        baseline = df[baseline_col]

        return actual_values - baseline
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, float]:
        """
        Train ensemble of models.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets

        Returns:
            Dictionary of validation scores by model
        """
        self.feature_columns = X_train.columns.tolist()

        # Check if targets have sufficient variance
        if y_train.std() < 0.01:
            print(f"    ⚠️ Warning: No variance in targets (std={y_train.std():.4f}). Skipping training.")
            return {}

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        val_scores = {}

        for name, model in self.models.items():
            print(f"  Training {name}...")

            try:
                # Train model with API-specific handling
                if name == 'xgboost':
                    model.fit(
                        X_train_scaled, y_train,
                        eval_set=[(X_val_scaled, y_val)],
                        verbose=False
                    )
                elif name == 'lightgbm':
                    model.fit(
                        X_train_scaled, y_train,
                        eval_set=[(X_val_scaled, y_val)],
                        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
                    )
                else:
                    model.fit(X_train_scaled, y_train)

                # Evaluate
                y_pred = model.predict(X_val_scaled)
                val_scores[name] = {
                    'mae': mean_absolute_error(y_val, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_val, y_pred)),
                    'r2': r2_score(y_val, y_pred)
                }

                print(f"    MAE: {val_scores[name]['mae']:.3f}, RMSE: {val_scores[name]['rmse']:.3f}")

            except Exception as e:
                print(f"    ⚠️ {name} training failed: {str(e)[:50]}... Skipping.")
        
        self.metadata['model_versions'] = val_scores
        self.metadata['training_date'] = datetime.now().isoformat()
        
        return val_scores
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using ensemble averaging.
        
        Args:
            X: Features for prediction
            
        Returns:
            Point differential predictions
        """
        X_scaled = self.scaler.transform(X[self.feature_columns])
        
        predictions = []
        for name, model in self.models.items():
            pred = model.predict(X_scaled)
            predictions.append(pred)
        
        # Average ensemble predictions
        return np.mean(predictions, axis=0)
    
    def predict_with_intervals(self, X: pd.DataFrame, confidence: float = 0.9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions with confidence intervals.
        
        Args:
            X: Features for prediction
            confidence: Confidence level (default 0.9)
            
        Returns:
            Tuple of (predictions, lower_bound, upper_bound)
        """
        X_scaled = self.scaler.transform(X[self.feature_columns])
        
        all_preds = []
        for name, model in self.models.items():
            pred = model.predict(X_scaled)
            all_preds.append(pred)
        
        all_preds = np.array(all_preds)
        
        # Calculate mean and confidence intervals
        mean_pred = np.mean(all_preds, axis=0)
        std_pred = np.std(all_preds, axis=0)
        
        # Use ensemble variance for intervals
        z_score = 1.96 if confidence == 0.95 else 1.645  # 95% or 90% CI
        lower = mean_pred - z_score * std_pred
        upper = mean_pred + z_score * std_pred
        
        return mean_pred, lower, upper
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model performance on test set"""
        y_pred = self.predict(X_test)
        
        return {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }
    
    def save_model(self, filepath: str):
        """Save model and metadata"""
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'metadata': self.metadata
        }
        joblib.dump(model_data, filepath)
        print(f"  EVOB model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from file"""
        model_data = joblib.load(filepath)
        self.models = model_data['models']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.metadata = model_data['metadata']


class StatPredictor:
    """
    Specialized predictor for individual stats (yards, TDs, etc.)
    Uses appropriate model types for different stat distributions
    """
    
    def __init__(self, position: str, stat_name: str):
        """
        Initialize stat-specific predictor.
        
        Args:
            position: Player position
            stat_name: Specific stat to predict
        """
        self.position = position
        self.stat_name = stat_name
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        
        # Choose model type based on stat
        self._initialize_model()
    
    def _initialize_model(self):
        """Choose appropriate model based on stat type"""

        # Count data (TDs, INTs) - use standard regression with non-negative constraint
        if any(x in self.stat_name.lower() for x in ['td', 'touchdown', 'int', 'sack', 'fumble']):
            self.model = XGBRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.01,
                objective='reg:squarederror',
                random_state=42,
                early_stopping_rounds=20
            )
        
        # Continuous data (yards) - use standard regression
        elif 'yard' in self.stat_name.lower():
            self.model = XGBRegressor(
                n_estimators=300,
                max_depth=7,
                learning_rate=0.01,
                objective='reg:squarederror',
                random_state=42,
                early_stopping_rounds=20
            )
        
        # Percentage data (completion %) - use logistic
        elif 'pct' in self.stat_name.lower() or 'percent' in self.stat_name.lower():
            self.model = XGBRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.01,
                objective='reg:logistic',
                random_state=42,
                early_stopping_rounds=20
            )
        
        # Default to standard regression
        else:
            self.model = XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.01,
                random_state=42,
                early_stopping_rounds=20
            )
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, float]:
        """Train the stat predictor"""
        self.feature_columns = X_train.columns.tolist()
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train with XGBoost's new API
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)],
            verbose=False
        )
        
        # Evaluate
        y_pred = self.model.predict(X_val_scaled)
        
        return {
            'mae': mean_absolute_error(y_val, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_val, y_pred)),
            'r2': r2_score(y_val, y_pred)
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        X_scaled = self.scaler.transform(X[self.feature_columns])
        predictions = self.model.predict(X_scaled)
        
        # Ensure non-negative for count data
        if any(x in self.stat_name.lower() for x in ['td', 'int', 'yard', 'attempt', 'completion', 'carry', 'target']):
            predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def save_model(self, filepath: str):
        """Save model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'stat_name': self.stat_name,
            'position': self.position
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str):
        """Load model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']


# Continue with NFLModelPipeline class (rest of the file remains the same)
class NFLModelPipeline:
    """
    Main pipeline for training and managing all NFL prediction models.
    Handles data loading, feature selection, training, and prediction storage.
    """
    
    def __init__(self, data_dir: str = "./data/nfl", model_dir: str = "./data/nfl/models"):
        """
        Initialize the model pipeline.
        
        Args:
            data_dir: Base directory for NFL data
            model_dir: Directory to save trained models
        """
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.feature_dir = self.data_dir / "cleaned"
        self.prediction_dir = self.data_dir / "predictions"
        self.raw_dir = self.data_dir / "raw"
        
        # Create directories
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.prediction_dir.mkdir(parents=True, exist_ok=True)
        
        # Position-specific feature sets
        self.position_features = {
            'QB': [
                'rolling_avg_passing_yds', 'rolling_avg_passing_tds',
                'rolling_avg_interceptions', 'rolling_avg_completions',
                'rolling_avg_rushing_yds', 'opponent_pass_defense_rank',
                'pass_attempts_trend', 'games_in_history'
            ],
            'RB': [
                'rolling_avg_rushing_yds', 'rolling_avg_rushing_tds',
                'rolling_avg_carries', 'rolling_avg_receptions',
                'rolling_avg_receiving_yds', 'opponent_rush_defense_rank',
                'carry_share_trend', 'target_trend', 'games_in_history'
            ],
            'WR': [
                'rolling_avg_receiving_yds', 'rolling_avg_receiving_tds',
                'rolling_avg_receptions', 'rolling_avg_targets',
                'rolling_avg_air_yards', 'opponent_pass_defense_rank',
                'target_share_trend', 'air_yards_share_trend', 'games_in_history'
            ],
            'TE': [
                'rolling_avg_receiving_yds', 'rolling_avg_receiving_tds',
                'rolling_avg_receptions', 'rolling_avg_targets',
                'opponent_pass_defense_rank', 'target_share_trend',
                'games_in_history'
            ],
            'K': [
                'rolling_avg_fg_made', 'rolling_avg_fg_att',
                'rolling_avg_pat_made', 'opponent_defense_rank',
                'games_in_history'
            ]
        }
        
        # Stats to predict for each position
        self.position_stats = {
            'QB': ['fantasy_points_ppr', 'passing_yards', 'passing_tds', 'passing_interceptions'],
            'RB': ['fantasy_points_ppr', 'rushing_yards', 'rushing_tds', 'receptions'],
            'WR': ['fantasy_points_ppr', 'receiving_yards', 'receiving_tds', 'receptions'],
            'TE': ['fantasy_points_ppr', 'receiving_yards', 'receiving_tds', 'receptions'],
            'K': ['fantasy_points_ppr', 'fg_made', 'fg_att']
        }
        
        # Store trained models
        self.models = {}
    
    def load_features_and_targets(self, start_season: int = 2020, end_season: int = 2025) -> pd.DataFrame:
        """
        Load all feature data and merge with actual stats for targets.
        
        Args:
            start_season: First season to load
            end_season: Last season to load
            
        Returns:
            Combined DataFrame with features and targets
        """
        print("Loading feature data...")
        
        all_data = []
        
        for season in range(start_season, end_season + 1):
            for week in range(1, 19):
                # Load features
                feature_file = self.feature_dir / f"features_{season}_week_{week}.parquet"
                if not feature_file.exists():
                    continue
                
                features_df = pd.read_parquet(feature_file)
                
                # Load actual stats for targets (next week)
                next_week = week + 1
                next_season = season
                
                if next_week > 18:
                    next_week = 1
                    next_season = season + 1
                
                stats_file = self.raw_dir / f"player_stats_{next_season}_week_{next_week}.parquet"
                if not stats_file.exists():
                    continue
                
                stats_df = pd.read_parquet(stats_file)
                
                # Get the stats we want to predict
                target_stats = ['fantasy_points', 'fantasy_points_ppr', 'passing_yards', 'passing_tds', 
                               'passing_interceptions', 'rushing_yards', 'rushing_tds', 
                               'receiving_yards', 'receiving_tds', 'receptions', 
                               'fg_made', 'fg_att', 'targets', 'carries', 'completions', 'attempts']
                
                # Only keep columns that exist in stats_df
                available_stats = [col for col in target_stats if col in stats_df.columns]
                
                # Merge to get targets
                merge_cols = ['player_id'] + available_stats
                stats_subset = stats_df[merge_cols].copy()
                
                # Rename stats columns to indicate they are targets
                for col in available_stats:
                    stats_subset[f'{col}_target'] = stats_subset[col]
                    stats_subset = stats_subset.drop(columns=[col])
                
                # Merge with features
                merged = features_df.merge(
                    stats_subset,
                    on='player_id',
                    how='left'
                )
                
                # Add metadata
                merged['target_week'] = next_week
                merged['target_season'] = next_season
                
                all_data.append(merged)
        
        if not all_data:
            print("No data found!")
            return pd.DataFrame()
        
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Filter for sufficient data only
        combined_df = combined_df[combined_df['has_sufficient_data'] == True]
        
        print(f"Loaded {len(combined_df)} records with features and targets")
        
        # Show what target columns we have
        target_cols = [col for col in combined_df.columns if col.endswith('_target')]
        print(f"Available target columns: {target_cols[:5]}...")
        
        return combined_df
    
    def train_position_models(self, position: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train all models for a specific position.
        
        Args:
            position: Position to train models for
            df: DataFrame with features and targets
            
        Returns:
            Dictionary of trained models and scores
        """
        print(f"\n{'='*60}")
        print(f"Training models for {position}")
        print(f"{'='*60}")
        
        # Filter for position
        pos_df = df[df['position'] == position].copy()
        
        if len(pos_df) < 100:
            print(f"Insufficient data for {position} ({len(pos_df)} records)")
            return {}
        
        print(f"Training on {len(pos_df)} {position} records")
        
        # Get features for this position
        feature_cols = [col for col in self.position_features[position] if col in pos_df.columns]
        
        # Remove rows with NaN in features
        for col in feature_cols:
            pos_df = pos_df[pos_df[col].notna()]
        
        # Time-based train/val/test split
        pos_df = pos_df.sort_values(['season', 'week'])
        
        train_size = int(0.7 * len(pos_df))
        val_size = int(0.15 * len(pos_df))
        
        train_df = pos_df.iloc[:train_size]
        val_df = pos_df.iloc[train_size:train_size+val_size]
        test_df = pos_df.iloc[train_size+val_size:]
        
        position_models = {}
        
        # Train models for each stat
        for stat in self.position_stats[position]:
            target_col = f'{stat}_target'
            
            if target_col not in train_df.columns:
                print(f"  Skipping {stat} - no target column")
                continue
            
            # Remove rows with NaN targets
            train_clean = train_df[train_df[target_col].notna()].copy()
            val_clean = val_df[val_df[target_col].notna()].copy()
            test_clean = test_df[test_df[target_col].notna()].copy()
            
            if len(train_clean) < 50:
                print(f"  Skipping {stat} - insufficient training data")
                continue
            
            print(f"\n  Training {stat} models...")
            
            X_train = train_clean[feature_cols]
            X_val = val_clean[feature_cols]
            X_test = test_clean[feature_cols]
            
            # Train EVOB model (regression)
            evob_model = EVOBModel(position, stat)
            
            # Prepare regression targets
            y_train = evob_model.prepare_target(train_clean)
            y_val = evob_model.prepare_target(val_clean)
            y_test = evob_model.prepare_target(test_clean)
            
            # Remove NaN from targets
            mask_train = y_train.notna()
            mask_val = y_val.notna()
            mask_test = y_test.notna()
            
            if mask_train.sum() < 50 or mask_val.sum() < 10:
                print(f"    Insufficient non-null data for {stat}")
                continue
            
            evob_scores = evob_model.train(
                X_train[mask_train], y_train[mask_train],
                X_val[mask_val], y_val[mask_val]
            )

            # Only evaluate and save if training was successful
            if not evob_scores:
                print(f"    Skipping {stat} - no variance in data")
                continue

            test_scores = evob_model.evaluate(X_test[mask_test], y_test[mask_test])

            print(f"    EVOB Test - MAE: {test_scores['mae']:.2f}, R2: {test_scores['r2']:.3f}")
            
            # Train POB model (classification) for fantasy points
            if 'fantasy' in stat:
                pob_model = POBModel(position, stat)
                
                # Prepare binary targets
                y_train_binary = pob_model.prepare_target(train_clean)
                y_val_binary = pob_model.prepare_target(val_clean)
                y_test_binary = pob_model.prepare_target(test_clean)
                
                # Remove NaN from binary targets
                mask_train = y_train_binary.notna()
                mask_val = y_val_binary.notna()
                mask_test = y_test_binary.notna()
                
                if mask_train.sum() > 50 and mask_val.sum() > 10:
                    pob_scores = pob_model.train(
                        X_train[mask_train], y_train_binary[mask_train],
                        X_val[mask_val], y_val_binary[mask_val]
                    )
                    test_scores_pob = pob_model.evaluate(X_test[mask_test], y_test_binary[mask_test])
                    
                    print(f"    POB Test - Accuracy: {test_scores_pob['accuracy']:.3f}, F1: {test_scores_pob['f1']:.3f}")
                    
                    position_models[f'{stat}_pob'] = pob_model
            
            # Train specialized stat predictor
            stat_model = StatPredictor(position, stat)
            y_train_stat = train_clean[target_col]
            y_val_stat = val_clean[target_col]
            
            mask_train = y_train_stat.notna()
            mask_val = y_val_stat.notna()
            
            if mask_train.sum() > 50 and mask_val.sum() > 10:
                stat_scores = stat_model.train(
                    X_train[mask_train], y_train_stat[mask_train],
                    X_val[mask_val], y_val_stat[mask_val]
                )
            
            # Store models
            position_models[f'{stat}_evob'] = evob_model
            position_models[f'{stat}_stat'] = stat_model
        
        return position_models
    
    def train_all_models(self):
        """Train models for all positions"""
        print("\n" + "="*70)
        print("NFL MODEL TRAINING PIPELINE")
        print("="*70)
        
        # Load all data
        df = self.load_features_and_targets()
        
        # Train for each position
        for position in ['QB', 'RB', 'WR', 'TE', 'K']:
            models = self.train_position_models(position, df)
            
            if models:
                self.models[position] = models
                
                # Save models
                for model_name, model in models.items():
                    model_path = self.model_dir / f"{position}_{model_name}.joblib"
                    model.save_model(str(model_path))
        
        print("\n" + "="*70)
        print("✅ MODEL TRAINING COMPLETE")
        print("="*70)
        print(f"Models saved to: {self.model_dir}")
    
    def generate_predictions(self, season: int, week: int):
        """
        Generate predictions for a specific week.
        
        Args:
            season: Season year
            week: Week number
        """
        print(f"\nGenerating predictions for {season} Week {week}...")
        
        # Load features for this week
        feature_file = self.feature_dir / f"features_{season}_week_{week}.parquet"
        if not feature_file.exists():
            print(f"Feature file not found for {season} Week {week}")
            return
        
        features_df = pd.read_parquet(feature_file)
        features_df = features_df[features_df['has_sufficient_data'] == True]
        
        predictions = []
        
        for position in self.models.keys():
            pos_df = features_df[features_df['position'] == position]
            
            if len(pos_df) == 0:
                continue
            
            feature_cols = [col for col in self.position_features[position] if col in pos_df.columns]

            if not feature_cols:
                print(f"  ⚠️  No features available for {position}, skipping...")
                continue

            X = pos_df[feature_cols]

            # Skip if no valid data
            if len(X) == 0:
                continue

            for model_name, model in self.models[position].items():
                stat_name = model_name.replace('_evob', '').replace('_pob', '').replace('_stat', '')

                if 'evob' in model_name:
                    try:
                        pred_diff = model.predict(X)
                    except Exception as e:
                        print(f"  ⚠️  Error predicting {model_name}: {str(e)[:50]}")
                        continue

                    # Map stat names to their actual feature column names
                    stat_to_feature_map = {
                        'fantasy_points_ppr': 'rolling_avg_fantasy_ppr',
                        'fantasy_points': 'rolling_avg_fantasy_pts',
                        'passing_yards': 'rolling_avg_passing_yds',
                        'passing_tds': 'rolling_avg_passing_tds',
                        'passing_interceptions': 'rolling_avg_interceptions',
                        'rushing_yards': 'rolling_avg_rushing_yds',
                        'rushing_tds': 'rolling_avg_rushing_tds',
                        'receiving_yards': 'rolling_avg_receiving_yds',
                        'receiving_tds': 'rolling_avg_receiving_tds',
                        'receptions': 'rolling_avg_receptions',
                        'carries': 'rolling_avg_carries',
                        'targets': 'rolling_avg_targets',
                        'completions': 'rolling_avg_completions',
                        'attempts': 'rolling_avg_attempts',
                        'fg_made': 'rolling_avg_fg_made',
                        'fg_att': 'rolling_avg_fg_att'
                    }

                    baseline_col = stat_to_feature_map.get(stat_name, f'rolling_avg_{stat_name}')

                    # Check if baseline column exists
                    if baseline_col not in pos_df.columns:
                        print(f"  ⚠️  Baseline column {baseline_col} not found for {stat_name}, skipping...")
                        continue

                    pred_value = pos_df[baseline_col].values + pred_diff

                    # Add predictions with confidence intervals
                    try:
                        pred_mean, pred_lower, pred_upper = model.predict_with_intervals(X)
                    except Exception as e:
                        print(f"  ⚠️  Error getting confidence intervals for {model_name}: {str(e)[:50]}")
                        # Use simple fallback if confidence intervals fail
                        pred_lower = pred_diff
                        pred_upper = pred_diff

                    for i, (idx, row) in enumerate(pos_df.iterrows()):
                        predictions.append({
                            'player_id': row['player_id'],
                            'player_name': row['player_name'],
                            'position': position,
                            'team': row['team'],
                            'opponent': row['opponent_team'],
                            'season': season,
                            'week': week + 1,  # Predicting next week
                            'stat': stat_name,
                            'model_type': 'evob',
                            'predicted_value': pred_value[i],
                            'predicted_diff': pred_diff[i],
                            'confidence_lower': pos_df[baseline_col].values[i] + pred_lower[i],
                            'confidence_upper': pos_df[baseline_col].values[i] + pred_upper[i],
                            'baseline': pos_df[baseline_col].values[i]
                        })

                elif 'pob' in model_name:
                    try:
                        prob = model.predict(X)
                    except Exception as e:
                        print(f"  ⚠️  Error predicting {model_name}: {str(e)[:50]}")
                        continue

                    # Map stat names to their actual feature column names (same as EVOB)
                    stat_to_feature_map = {
                        'fantasy_points_ppr': 'rolling_avg_fantasy_ppr',
                        'fantasy_points': 'rolling_avg_fantasy_pts',
                        'passing_yards': 'rolling_avg_passing_yds',
                        'passing_tds': 'rolling_avg_passing_tds',
                        'passing_interceptions': 'rolling_avg_interceptions',
                        'rushing_yards': 'rolling_avg_rushing_yds',
                        'rushing_tds': 'rolling_avg_rushing_tds',
                        'receiving_yards': 'rolling_avg_receiving_yds',
                        'receiving_tds': 'rolling_avg_receiving_tds',
                        'receptions': 'rolling_avg_receptions',
                        'carries': 'rolling_avg_carries',
                        'targets': 'rolling_avg_targets',
                        'completions': 'rolling_avg_completions',
                        'attempts': 'rolling_avg_attempts',
                        'fg_made': 'rolling_avg_fg_made',
                        'fg_att': 'rolling_avg_fg_att'
                    }

                    baseline_col = stat_to_feature_map.get(stat_name, f'rolling_avg_{stat_name}')

                    # Check if baseline column exists
                    if baseline_col not in pos_df.columns:
                        print(f"  ⚠️  Baseline column {baseline_col} not found for {stat_name}, skipping...")
                        continue

                    for i, (idx, row) in enumerate(pos_df.iterrows()):
                        predictions.append({
                            'player_id': row['player_id'],
                            'player_name': row['player_name'],
                            'position': position,
                            'team': row['team'],
                            'opponent': row['opponent_team'],
                            'season': season,
                            'week': week + 1,
                            'stat': stat_name,
                            'model_type': 'pob',
                            'probability_over': prob[i],
                            'baseline': row[baseline_col]
                        })
        
        # Save predictions
        if predictions:
            pred_df = pd.DataFrame(predictions)
            pred_file = self.prediction_dir / f"predictions_{season}_week_{week+1}.parquet"
            pred_df.to_parquet(pred_file, index=False)
            print(f"Saved {len(pred_df)} predictions to {pred_file}")
        
        return predictions


if __name__ == "__main__":
    # Example usage
    pipeline = NFLModelPipeline()
    pipeline.train_all_models()