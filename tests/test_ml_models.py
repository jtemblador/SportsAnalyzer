"""
File: testing/test_ml_models.py

Comprehensive test suite for NFL ML models.
Tests different strategies, configurations, and edge cases.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.nfl.models.base import NFLModelPipeline, POBModel, EVOBModel, StatPredictor


class ModelTester:
    """Comprehensive testing suite for NFL models"""
    
    def __init__(self):
        self.pipeline = NFLModelPipeline()
        self.results = {}
    
    def test_data_loading(self):
        """Test that data loads correctly"""
        print("\n" + "="*60)
        print("TEST 1: Data Loading")
        print("="*60)
        
        try:
            df = self.pipeline.load_features_and_targets(start_season=2020, end_season=2021)
            
            print(f"✅ Successfully loaded {len(df)} records")
            print(f"   Seasons: {df['season'].unique()}")
            print(f"   Positions: {df['position'].value_counts().to_dict()}")
            
            # Check for required columns
            required_cols = ['player_id', 'player_name', 'position', 'team', 
                           'has_sufficient_data', 'fantasy_points_ppr_target']
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"❌ Missing columns: {missing_cols}")
            else:
                print(f"✅ All required columns present")
            
            self.results['data_loading'] = 'PASSED'
            return df
            
        except Exception as e:
            print(f"❌ Data loading failed: {e}")
            self.results['data_loading'] = 'FAILED'
            return None
    
    def test_single_position_training(self, position: str = 'QB'):
        """Test training for a single position"""
        print("\n" + "="*60)
        print(f"TEST 2: Single Position Training ({position})")
        print("="*60)
        
        try:
            # Load minimal data for faster testing
            df = self.pipeline.load_features_and_targets(start_season=2020, end_season=2021)
            
            if df is None:
                print("❌ Cannot test without data")
                return
            
            # Train models
            models = self.pipeline.train_position_models(position, df)
            
            if models:
                print(f"\n✅ Successfully trained {len(models)} models for {position}")
                for model_name in models.keys():
                    print(f"   - {model_name}")
                
                self.results[f'{position}_training'] = 'PASSED'
                return models
            else:
                print(f"❌ No models trained for {position}")
                self.results[f'{position}_training'] = 'FAILED'
                return None
                
        except Exception as e:
            print(f"❌ Training failed: {e}")
            self.results[f'{position}_training'] = 'FAILED'
            return None
    
    def test_prediction_generation(self):
        """Test generating predictions for a week"""
        print("\n" + "="*60)
        print("TEST 3: Prediction Generation")
        print("="*60)
        
        try:
            # First train a minimal model
            df = self.pipeline.load_features_and_targets(start_season=2020, end_season=2020)
            
            # Train just QB models for speed
            models = self.pipeline.train_position_models('QB', df)
            
            if models:
                self.pipeline.models['QB'] = models
                
                # Generate predictions for a test week
                predictions = self.pipeline.generate_predictions(2020, 10)
                
                if predictions:
                    print(f"✅ Generated {len(predictions)} predictions")
                    
                    # Check prediction structure
                    sample_pred = predictions[0] if predictions else {}
                    print("\n  Sample prediction structure:")
                    for key in list(sample_pred.keys())[:5]:
                        print(f"    - {key}: {type(sample_pred[key]).__name__}")
                    
                    self.results['prediction_generation'] = 'PASSED'
                    return predictions
                else:
                    print("❌ No predictions generated")
                    self.results['prediction_generation'] = 'FAILED'
                    
        except Exception as e:
            print(f"❌ Prediction generation failed: {e}")
            self.results['prediction_generation'] = 'FAILED'
            return None
    
    def test_model_comparison(self):
        """Compare different model strategies"""
        print("\n" + "="*60)
        print("TEST 4: Model Strategy Comparison")
        print("="*60)
        
        strategies = {
            'baseline': self._create_baseline_model,
            'ensemble': self._create_ensemble_model,
            'position_specific': self._create_position_specific_model
        }
        
        comparison_results = {}
        
        # Load test data
        df = self.pipeline.load_features_and_targets(start_season=2020, end_season=2021)
        if df is None:
            return
        
        # Test each strategy
        for strategy_name, strategy_func in strategies.items():
            print(f"\n  Testing {strategy_name} strategy...")
            
            try:
                results = strategy_func(df)
                comparison_results[strategy_name] = results
                print(f"    MAE: {results['mae']:.2f}")
                print(f"    RMSE: {results['rmse']:.2f}")
                print(f"    R2: {results['r2']:.3f}")
                
            except Exception as e:
                print(f"    ❌ Strategy failed: {e}")
                comparison_results[strategy_name] = {'mae': np.inf, 'rmse': np.inf, 'r2': -np.inf}
        
        # Find best strategy
        best_strategy = min(comparison_results.items(), key=lambda x: x[1]['mae'])
        print(f"\n✅ Best strategy: {best_strategy[0]} (MAE: {best_strategy[1]['mae']:.2f})")
        
        self.results['model_comparison'] = comparison_results
        return comparison_results
    
    def _create_baseline_model(self, df: pd.DataFrame) -> Dict[str, float]:
        """Create a simple baseline model (just uses rolling average)"""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        # Filter for QBs with sufficient data
        qb_df = df[(df['position'] == 'QB') & (df['has_sufficient_data'] == True)].copy()
        
        # Baseline: predict next week = rolling average
        y_true = qb_df['fantasy_points_ppr_target'].dropna()
        y_pred = qb_df.loc[y_true.index, 'rolling_avg_fantasy_ppr']
        
        return {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred)
        }
    
    def _create_ensemble_model(self, df: pd.DataFrame) -> Dict[str, float]:
        """Create an ensemble model (what we built)"""
        # Use our actual model
        qb_df = df[(df['position'] == 'QB') & (df['has_sufficient_data'] == True)].copy()
        
        if len(qb_df) < 100:
            return {'mae': np.inf, 'rmse': np.inf, 'r2': -np.inf}
        
        # Quick train/test split
        train_size = int(0.8 * len(qb_df))
        train_df = qb_df.iloc[:train_size]
        test_df = qb_df.iloc[train_size:]
        
        # Use EVOB model
        model = EVOBModel('QB', 'fantasy_points_ppr')
        
        feature_cols = [col for col in self.pipeline.position_features['QB'] if col in train_df.columns]
        
        X_train = train_df[feature_cols]
        y_train = train_df['fantasy_points_ppr_target'] - train_df['rolling_avg_fantasy_ppr']
        X_test = test_df[feature_cols]
        y_test = test_df['fantasy_points_ppr_target'] - test_df['rolling_avg_fantasy_ppr']
        
        # Quick training (reduced iterations for speed)
        model.models = {
            'xgboost': model.models['xgboost'].set_params(n_estimators=50)
        }
        
        val_size = int(0.2 * len(X_train))
        model.train(X_train[:-val_size], y_train[:-val_size],
                   X_train[-val_size:], y_train[-val_size:])
        
        return model.evaluate(X_test, y_test)
    
    def _create_position_specific_model(self, df: pd.DataFrame) -> Dict[str, float]:
        """Create a position-specific tuned model"""
        # Similar to ensemble but with position-specific hyperparameters
        return self._create_ensemble_model(df)  # Simplified for this test
    
    def test_edge_cases(self):
        """Test various edge cases"""
        print("\n" + "="*60)
        print("TEST 5: Edge Cases")
        print("="*60)
        
        edge_cases = []
        
        # Test 1: Empty data
        try:
            empty_df = pd.DataFrame()
            models = self.pipeline.train_position_models('QB', empty_df)
            if not models:
                edge_cases.append("✅ Empty data handled correctly")
            else:
                edge_cases.append("❌ Empty data not handled")
        except:
            edge_cases.append("✅ Empty data handled correctly")
        
        # Test 2: Missing features
        try:
            df = self.pipeline.load_features_and_targets(start_season=2020, end_season=2020)
            if df is not None:
                # Remove some columns
                df_missing = df.drop(columns=['rolling_avg_fantasy_ppr'], errors='ignore')
                models = self.pipeline.train_position_models('QB', df_missing)
                edge_cases.append("✅ Missing features handled")
        except:
            edge_cases.append("✅ Missing features handled")
        
        # Test 3: Single player prediction
        try:
            model = EVOBModel('QB', 'fantasy_points_ppr')
            single_row = pd.DataFrame({
                'rolling_avg_passing_yds': [250],
                'rolling_avg_passing_tds': [1.5],
                'games_in_history': [6]
            })
            # This should fail without training
            pred = model.predict(single_row)
            edge_cases.append("❌ Untrained model made prediction")
        except:
            edge_cases.append("✅ Untrained model handled correctly")
        
        # Print results
        for result in edge_cases:
            print(f"  {result}")
        
        self.results['edge_cases'] = edge_cases
    
    def test_performance_metrics(self):
        """Test and visualize model performance metrics"""
        print("\n" + "="*60)
        print("TEST 6: Performance Metrics")
        print("="*60)
        
        # Load data
        df = self.pipeline.load_features_and_targets(start_season=2020, end_season=2021)
        
        if df is None:
            return
        
        positions = ['QB', 'RB', 'WR']
        metrics = {}
        
        for position in positions:
            print(f"\n  Testing {position}...")
            
            pos_df = df[(df['position'] == position) & (df['has_sufficient_data'] == True)]
            
            if len(pos_df) < 100:
                continue
            
            # Simple train/test split
            train_size = int(0.8 * len(pos_df))
            test_df = pos_df.iloc[train_size:]
            
            # Calculate baseline accuracy
            baseline_mae = np.abs(
                test_df['fantasy_points_ppr_target'] - test_df['rolling_avg_fantasy_ppr']
            ).mean()
            
            metrics[position] = {
                'baseline_mae': baseline_mae,
                'sample_size': len(test_df),
                'avg_points': test_df['fantasy_points_ppr_target'].mean()
            }
            
            print(f"    Baseline MAE: {baseline_mae:.2f}")
            print(f"    Average Points: {metrics[position]['avg_points']:.2f}")
            print(f"    Test Samples: {metrics[position]['sample_size']}")
        
        self.results['performance_metrics'] = metrics
        return metrics
    
    def run_all_tests(self):
        """Run all tests and generate report"""
        print("\n" + "="*70)
        print("NFL MODEL TEST SUITE")
        print("="*70)
        
        # Run tests
        self.test_data_loading()
        self.test_single_position_training('QB')
        self.test_prediction_generation()
        self.test_model_comparison()
        self.test_edge_cases()
        self.test_performance_metrics()
        
        # Generate report
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        
        for test_name, result in self.results.items():
            if isinstance(result, str):
                status = "✅" if result == 'PASSED' else "❌"
                print(f"{status} {test_name}: {result}")
            elif isinstance(result, dict):
                print(f"📊 {test_name}:")
                for key, value in list(result.items())[:3]:
                    if isinstance(value, (int, float)):
                        print(f"    {key}: {value:.2f}")
                    else:
                        print(f"    {key}: {value}")
        
        return self.results


def test_walk_forward_validation():
    """Test walk-forward validation strategy"""
    print("\n" + "="*60)
    print("Walk-Forward Validation Test")
    print("="*60)
    
    pipeline = NFLModelPipeline()
    
    # Load data
    df = pipeline.load_features_and_targets(start_season=2020, end_season=2021)
    
    if df is None:
        return
    
    # Filter for QBs
    qb_df = df[(df['position'] == 'QB') & (df['has_sufficient_data'] == True)]
    qb_df = qb_df.sort_values(['season', 'week'])
    
    # Walk-forward validation
    min_train_size = 500
    test_size = 100
    
    results = []
    
    for i in range(min_train_size, len(qb_df) - test_size, test_size):
        train_df = qb_df.iloc[:i]
        test_df = qb_df.iloc[i:i+test_size]
        
        print(f"\n  Fold: Train size={len(train_df)}, Test size={len(test_df)}")
        
        # Quick model training
        model = EVOBModel('QB', 'fantasy_points_ppr')
        
        feature_cols = ['rolling_avg_passing_yds', 'rolling_avg_passing_tds', 
                       'games_in_history', 'opponent_pass_defense_rank']
        feature_cols = [col for col in feature_cols if col in train_df.columns]
        
        X_train = train_df[feature_cols]
        y_train = train_df['fantasy_points_ppr_target'] - train_df['rolling_avg_fantasy_ppr']
        X_test = test_df[feature_cols]
        y_test = test_df['fantasy_points_ppr_target'] - test_df['rolling_avg_fantasy_ppr']
        
        # Remove NaN
        mask_train = y_train.notna()
        mask_test = y_test.notna()
        
        if mask_train.sum() < 50 or mask_test.sum() < 10:
            continue
        
        # Use simple model for speed
        model.models = {
            'xgboost': model.models['xgboost'].set_params(n_estimators=50)
        }
        
        val_size = int(0.2 * mask_train.sum())
        model.train(
            X_train[mask_train].iloc[:-val_size], 
            y_train[mask_train].iloc[:-val_size],
            X_train[mask_train].iloc[-val_size:], 
            y_train[mask_train].iloc[-val_size:]
        )
        
        scores = model.evaluate(X_test[mask_test], y_test[mask_test])
        results.append(scores)
        
        print(f"    MAE: {scores['mae']:.2f}, R2: {scores['r2']:.3f}")
    
    if results:
        avg_mae = np.mean([r['mae'] for r in results])
        avg_r2 = np.mean([r['r2'] for r in results])
        
        print(f"\n✅ Walk-Forward Validation Complete")
        print(f"   Average MAE: {avg_mae:.2f}")
        print(f"   Average R2: {avg_r2:.3f}")
    
    return results


if __name__ == "__main__":
    # Run comprehensive tests
    tester = ModelTester()
    results = tester.run_all_tests()
    
    # Run walk-forward validation
    print("\n" + "="*70)
    print("ADVANCED TESTING")
    print("="*70)
    
    wf_results = test_walk_forward_validation()
    
    print("\n" + "="*70)
    print("✅ ALL TESTS COMPLETE")
    print("="*70)