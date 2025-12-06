"""
File: src/nfl/v4_ml_models.py

V4 ML Models with Position-Specific Hyperparameters.

VERSION: V4
Key Changes from V2/V3:
- Position-specific hyperparameters (QB needs deeper trees, K needs simpler)
- QB: max_depth=9, n_estimators=500, learning_rate=0.005 (complex patterns)
- K: max_depth=3, n_estimators=100 (simple, prevent overfitting)
- TE: max_depth=6 (moderate complexity)
- RB/WR: Standard settings (already work well)

Used by: v4_retrain.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import joblib
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler


class PositionSpecificEVOBModel:
    """
    Expected Value Over Baseline - Regression
    V4: Position-specific hyperparameters for better performance
    """

    def __init__(self, position: str, target_stat: str = 'fantasy_points_ppr'):
        self.position = position
        self.target_stat = target_stat
        self.models = {}
        self.feature_columns = []
        self.scaler = StandardScaler()
        self.metadata = {
            'position': position,
            'target_stat': target_stat,
            'created_date': datetime.now().isoformat(),
            'model_versions': {},
            'v4_position_specific': True
        }

        self._initialize_models()

    def _initialize_models(self):
        """Initialize models with position-specific hyperparameters"""

        if self.position == 'QB':
            # QB: High variance (5-40 pts), limited data (2,893 records)
            # Need deeper trees to capture complex patterns + regularization
            self.models = {
                'xgboost': XGBRegressor(
                    n_estimators=500,
                    max_depth=9,
                    learning_rate=0.005,
                    min_child_weight=5,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    gamma=0.1,
                    objective='reg:squarederror',
                    random_state=42,
                    early_stopping_rounds=20
                ),
                'lightgbm': LGBMRegressor(
                    n_estimators=500,
                    max_depth=9,
                    learning_rate=0.005,
                    num_leaves=255,
                    min_data_in_leaf=50,
                    objective='regression',
                    metric='rmse',
                    random_state=42,
                    verbose=-1
                ),
                'catboost': CatBoostRegressor(
                    iterations=500,
                    depth=9,
                    learning_rate=0.005,
                    l2_leaf_reg=5,
                    loss_function='RMSE',
                    random_state=42,
                    verbose=False
                )
            }

        elif self.position == 'K':
            # K: Low variance (0-15 pts), limited data (2,592 records)
            # Simple models prevent overfitting
            self.models = {
                'xgboost': XGBRegressor(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.01,
                    objective='reg:squarederror',
                    random_state=42,
                    early_stopping_rounds=20
                ),
                'lightgbm': LGBMRegressor(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.01,
                    objective='regression',
                    random_state=42,
                    verbose=-1
                ),
                'ridge': Ridge(alpha=1.0)
            }

        elif self.position == 'TE':
            # TE: Lower variance, benefits from simpler models
            self.models = {
                'xgboost': XGBRegressor(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.01,
                    min_child_weight=3,
                    objective='reg:squarederror',
                    random_state=42,
                    early_stopping_rounds=20
                ),
                'lightgbm': LGBMRegressor(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.01,
                    objective='regression',
                    random_state=42,
                    verbose=-1
                ),
                'catboost': CatBoostRegressor(
                    iterations=300,
                    depth=6,
                    learning_rate=0.01,
                    loss_function='RMSE',
                    random_state=42,
                    verbose=False
                )
            }

        else:
            # RB/WR: Standard settings (already work well in V2)
            self.models = {
                'xgboost': XGBRegressor(
                    n_estimators=300,
                    max_depth=7,
                    learning_rate=0.01,
                    objective='reg:squarederror',
                    random_state=42,
                    early_stopping_rounds=20
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
        """Create regression target: difference from rolling average."""
        target_col = f'{self.target_stat}_target'

        if target_col not in df.columns:
            raise ValueError(f"Target column {target_col} not found in dataframe")

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

        baseline_col = stat_to_feature_map.get(self.target_stat, f'rolling_avg_{self.target_stat}')

        if baseline_col not in df.columns:
            raise ValueError(f"Baseline column {baseline_col} not found in dataframe")

        return df[target_col] - df[baseline_col]

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, float]:
        """Train ensemble of models."""
        self.feature_columns = X_train.columns.tolist()

        if y_train.std() < 0.01:
            print(f"    Warning: No variance in targets (std={y_train.std():.4f}). Skipping.")
            return {}

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        val_scores = {}

        for name, model in self.models.items():
            print(f"  Training {name}...")

            try:
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
                elif name == 'ridge':
                    model.fit(X_train_scaled, y_train)
                else:
                    model.fit(X_train_scaled, y_train)

                y_pred = model.predict(X_val_scaled)
                val_scores[name] = {
                    'mae': mean_absolute_error(y_val, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_val, y_pred)),
                    'r2': r2_score(y_val, y_pred)
                }

                print(f"    MAE: {val_scores[name]['mae']:.3f}, RMSE: {val_scores[name]['rmse']:.3f}")

            except Exception as e:
                print(f"    {name} training failed: {str(e)[:50]}... Skipping.")

        self.metadata['model_versions'] = val_scores
        self.metadata['training_date'] = datetime.now().isoformat()

        return val_scores

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using ensemble averaging."""
        X_scaled = self.scaler.transform(X[self.feature_columns])

        predictions = []
        for name, model in self.models.items():
            pred = model.predict(X_scaled)
            predictions.append(pred)

        return np.mean(predictions, axis=0)

    def predict_with_intervals(self, X: pd.DataFrame, confidence: float = 0.9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Make predictions with confidence intervals."""
        X_scaled = self.scaler.transform(X[self.feature_columns])

        all_preds = []
        for name, model in self.models.items():
            pred = model.predict(X_scaled)
            all_preds.append(pred)

        all_preds = np.array(all_preds)

        mean_pred = np.mean(all_preds, axis=0)
        std_pred = np.std(all_preds, axis=0)

        z_score = 1.96 if confidence == 0.95 else 1.645
        lower = mean_pred - z_score * std_pred
        upper = mean_pred + z_score * std_pred

        return mean_pred, lower, upper

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model performance on test set."""
        y_pred = self.predict(X_test)

        return {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }

    def save_model(self, filepath: str):
        """Save model and metadata."""
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'metadata': self.metadata
        }
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, filepath)
        print(f"  EVOB model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load model from file."""
        model_data = joblib.load(filepath)
        self.models = model_data['models']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.metadata = model_data['metadata']


class PositionSpecificPOBModel:
    """
    Probability Over Baseline - Binary Classification
    V4: Position-specific hyperparameters
    """

    def __init__(self, position: str, target_stat: str = 'fantasy_points_ppr'):
        self.position = position
        self.target_stat = target_stat
        self.models = {}
        self.feature_columns = []
        self.scaler = StandardScaler()
        self.metadata = {
            'position': position,
            'target_stat': target_stat,
            'created_date': datetime.now().isoformat(),
            'model_versions': {},
            'v4_position_specific': True
        }

        self._initialize_models()

    def _initialize_models(self):
        """Initialize classification models with position-specific settings"""

        if self.position == 'QB':
            self.models = {
                'xgboost': XGBClassifier(
                    n_estimators=300,
                    max_depth=8,
                    learning_rate=0.005,
                    min_child_weight=5,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='binary:logistic',
                    use_label_encoder=False,
                    eval_metric='logloss',
                    random_state=42,
                    early_stopping_rounds=20
                ),
                'lightgbm': LGBMClassifier(
                    n_estimators=300,
                    max_depth=8,
                    learning_rate=0.005,
                    num_leaves=127,
                    min_data_in_leaf=50,
                    objective='binary',
                    metric='binary_logloss',
                    random_state=42,
                    verbose=-1
                )
            }

        elif self.position == 'K':
            self.models = {
                'xgboost': XGBClassifier(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.01,
                    objective='binary:logistic',
                    use_label_encoder=False,
                    eval_metric='logloss',
                    random_state=42,
                    early_stopping_rounds=20
                )
            }

        else:
            # RB/WR/TE: Standard settings
            self.models = {
                'xgboost': XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.01,
                    objective='binary:logistic',
                    use_label_encoder=False,
                    eval_metric='logloss',
                    random_state=42,
                    early_stopping_rounds=20
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
        """Create binary target: 1 if player exceeds their rolling average."""
        target_col = f'{self.target_stat}_target'

        if target_col not in df.columns:
            raise ValueError(f"Target column {target_col} not found in dataframe")

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

        baseline_col = stat_to_feature_map.get(self.target_stat, f'rolling_avg_{self.target_stat}')

        if baseline_col not in df.columns:
            raise ValueError(f"Baseline column {baseline_col} not found in dataframe")

        return (df[target_col] > df[baseline_col]).astype(int)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, float]:
        """Train ensemble of classification models."""
        self.feature_columns = X_train.columns.tolist()

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        val_scores = {}

        for name, model in self.models.items():
            print(f"  Training {name}...")

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
        """Make predictions using ensemble averaging."""
        X_scaled = self.scaler.transform(X[self.feature_columns])

        predictions = []
        for name, model in self.models.items():
            pred_proba = model.predict_proba(X_scaled)[:, 1]
            predictions.append(pred_proba)

        return np.mean(predictions, axis=0)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model performance on test set."""
        y_pred_proba = self.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)

        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }

    def save_model(self, filepath: str):
        """Save model and metadata."""
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'metadata': self.metadata
        }
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, filepath)
        print(f"  POB model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load model from file."""
        model_data = joblib.load(filepath)
        self.models = model_data['models']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.metadata = model_data['metadata']


class PositionSpecificStatPredictor:
    """
    Specialized predictor for individual stats (yards, TDs, etc.)
    V4: Position-specific hyperparameters
    """

    def __init__(self, position: str, stat_name: str):
        self.position = position
        self.stat_name = stat_name
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []

        self._initialize_model()

    def _initialize_model(self):
        """Choose appropriate model based on position and stat type."""

        # Position-specific base settings
        if self.position == 'QB':
            base_depth = 8
            base_estimators = 400
            base_lr = 0.005
        elif self.position == 'K':
            base_depth = 3
            base_estimators = 100
            base_lr = 0.01
        elif self.position == 'TE':
            base_depth = 5
            base_estimators = 250
            base_lr = 0.01
        else:
            base_depth = 6
            base_estimators = 200
            base_lr = 0.01

        # Adjust for stat type
        if any(x in self.stat_name.lower() for x in ['td', 'touchdown', 'int', 'sack', 'fumble']):
            self.model = XGBRegressor(
                n_estimators=base_estimators,
                max_depth=min(base_depth, 5),
                learning_rate=base_lr,
                objective='reg:squarederror',
                random_state=42,
                early_stopping_rounds=20
            )
        elif 'yard' in self.stat_name.lower():
            self.model = XGBRegressor(
                n_estimators=base_estimators + 100,
                max_depth=base_depth + 1,
                learning_rate=base_lr,
                objective='reg:squarederror',
                random_state=42,
                early_stopping_rounds=20
            )
        else:
            self.model = XGBRegressor(
                n_estimators=base_estimators,
                max_depth=base_depth,
                learning_rate=base_lr,
                random_state=42,
                early_stopping_rounds=20
            )

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, float]:
        """Train the stat predictor."""
        self.feature_columns = X_train.columns.tolist()

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)],
            verbose=False
        )

        y_pred = self.model.predict(X_val_scaled)

        return {
            'mae': mean_absolute_error(y_val, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_val, y_pred)),
            'r2': r2_score(y_val, y_pred)
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        X_scaled = self.scaler.transform(X[self.feature_columns])
        predictions = self.model.predict(X_scaled)

        if any(x in self.stat_name.lower() for x in ['td', 'int', 'yard', 'attempt', 'completion', 'carry', 'target']):
            predictions = np.maximum(predictions, 0)

        return predictions

    def save_model(self, filepath: str):
        """Save model."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'stat_name': self.stat_name,
            'position': self.position
        }
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, filepath)

    def load_model(self, filepath: str):
        """Load model."""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']


# Alias for backward compatibility with NFLModelPipeline
EVOBModel = PositionSpecificEVOBModel
POBModel = PositionSpecificPOBModel
StatPredictor = PositionSpecificStatPredictor
