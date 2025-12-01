#!/usr/bin/env python3
"""
File: testing/model_testing/verify_models.py

Verify that all trained models can be loaded and make predictions.
This script tests the entire ML pipeline to ensure everything works.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.nfl.ml_models import POBModel, EVOBModel, StatPredictor


class ModelVerifier:
    """Verify all trained models work correctly"""

    def __init__(self):
        """Initialize paths"""
        self.project_root = Path(__file__).parent.parent.parent
        self.data_dir = self.project_root / "data" / "nfl"
        self.model_dir = self.data_dir / "models"
        self.feature_dir = self.data_dir / "cleaned"
        self.raw_dir = self.data_dir / "raw"
        self.prediction_dir = self.data_dir / "predictions"

        # Track verification results
        self.results = {
            'total_models': 0,
            'loaded_successfully': 0,
            'prediction_successful': 0,
            'failed_models': []
        }

    def verify_directories(self):
        """Verify all required directories exist"""
        print("="*70)
        print("DIRECTORY VERIFICATION")
        print("="*70)

        dirs = {
            'Project Root': self.project_root,
            'Data Directory': self.data_dir,
            'Models': self.model_dir,
            'Features (Cleaned)': self.feature_dir,
            'Raw Stats': self.raw_dir,
            'Predictions': self.prediction_dir
        }

        all_exist = True
        for name, path in dirs.items():
            exists = path.exists()
            status = "✅" if exists else "❌"
            print(f"{status} {name}: {path}")
            if not exists:
                all_exist = False

        print()
        return all_exist

    def count_files(self):
        """Count files in each directory"""
        print("="*70)
        print("FILE COUNTS")
        print("="*70)

        # Model files
        model_files = list(self.model_dir.glob("*.joblib"))
        print(f"📦 Models: {len(model_files)} .joblib files")

        # Feature files
        feature_files = list(self.feature_dir.glob("features_*.parquet"))
        print(f"🔧 Features: {len(feature_files)} feature files")

        # Raw stats files
        raw_files = list(self.raw_dir.glob("player_stats_*.parquet"))
        print(f"📊 Raw Stats: {len(raw_files)} stat files")

        # Prediction files
        pred_files = list(self.prediction_dir.glob("predictions_*.parquet"))
        print(f"🎯 Predictions: {len(pred_files)} prediction files")

        print()
        return model_files, feature_files

    def load_sample_features(self):
        """Load a sample of features for testing"""
        print("="*70)
        print("LOADING SAMPLE FEATURES")
        print("="*70)

        # Find most recent feature file
        feature_files = sorted(self.feature_dir.glob("features_*.parquet"))
        if not feature_files:
            print("❌ No feature files found!")
            return None

        latest_file = feature_files[-1]
        print(f"Loading: {latest_file.name}")

        df = pd.read_parquet(latest_file)
        print(f"✅ Loaded {len(df)} records")
        print(f"Columns: {len(df.columns)}")
        print(f"Positions: {df['position'].value_counts().to_dict()}")

        # Filter for sufficient data
        df_filtered = df[df['has_sufficient_data'] == True]
        print(f"✅ {len(df_filtered)} records with sufficient data")

        print()
        return df_filtered

    def verify_model(self, model_path: Path, features_df: pd.DataFrame):
        """
        Verify a single model can be loaded and make predictions.

        Args:
            model_path: Path to model file
            features_df: DataFrame with features for testing

        Returns:
            True if model works, False otherwise
        """
        model_name = model_path.stem
        parts = model_name.split('_')

        if len(parts) < 3:
            print(f"  ⚠️  Skipping {model_name} - unexpected name format")
            return False

        position = parts[0]
        model_type = parts[-1]  # evob, pob, or stat

        # Get sample data for this position
        pos_df = features_df[features_df['position'] == position]

        if len(pos_df) == 0:
            print(f"  ⚠️  No data for {position}")
            return False

        # Take first 5 records for testing
        test_df = pos_df.head(5)

        try:
            # Load model
            model_data = joblib.load(model_path)

            # Get feature columns
            feature_cols = model_data.get('feature_columns', [])
            if not feature_cols:
                print(f"  ❌ {model_name}: No feature columns found")
                return False

            # Check if features exist in data
            available_features = [col for col in feature_cols if col in test_df.columns]
            if len(available_features) < len(feature_cols):
                missing = set(feature_cols) - set(available_features)
                print(f"  ⚠️  {model_name}: Missing {len(missing)} features")

            if len(available_features) == 0:
                print(f"  ❌ {model_name}: No features available")
                return False

            # Extract features
            X_test = test_df[available_features]

            # Load based on model type
            if model_type == 'pob':
                model = POBModel(position)
                model.load_model(str(model_path))
                predictions = model.predict(X_test)
                print(f"  ✅ {model_name}: POB predictions = {predictions[:3]}")

            elif model_type == 'evob':
                stat = '_'.join(parts[1:-1])  # Everything between position and model_type
                model = EVOBModel(position, stat)
                model.load_model(str(model_path))
                predictions = model.predict(X_test)
                print(f"  ✅ {model_name}: EVOB predictions = {predictions[:3]}")

            elif model_type == 'stat':
                stat = '_'.join(parts[1:-1])
                model = StatPredictor(position, stat)
                model.load_model(str(model_path))
                predictions = model.predict(X_test)
                print(f"  ✅ {model_name}: Stat predictions = {predictions[:3]}")

            else:
                print(f"  ⚠️  {model_name}: Unknown model type '{model_type}'")
                return False

            return True

        except Exception as e:
            print(f"  ❌ {model_name}: Error - {str(e)[:60]}")
            return False

    def verify_all_models(self, features_df: pd.DataFrame):
        """Verify all models in the models directory"""
        print("="*70)
        print("MODEL VERIFICATION")
        print("="*70)

        model_files = sorted(self.model_dir.glob("*.joblib"))
        self.results['total_models'] = len(model_files)

        print(f"Found {len(model_files)} models to verify\n")

        # Group models by position
        by_position = {}
        for model_file in model_files:
            position = model_file.stem.split('_')[0]
            if position not in by_position:
                by_position[position] = []
            by_position[position].append(model_file)

        # Verify each position's models
        for position in sorted(by_position.keys()):
            print(f"\n{position} Models ({len(by_position[position])} models):")
            print("-" * 50)

            for model_file in sorted(by_position[position]):
                success = self.verify_model(model_file, features_df)

                if success:
                    self.results['prediction_successful'] += 1
                else:
                    self.results['failed_models'].append(model_file.name)

        print()

    def verify_predictions(self):
        """Verify prediction files exist and are valid"""
        print("="*70)
        print("PREDICTION FILE VERIFICATION")
        print("="*70)

        pred_files = list(self.prediction_dir.glob("predictions_*.parquet"))

        if not pred_files:
            print("⚠️  No prediction files found")
            print("   Run train_models.py to generate predictions")
            print()
            return

        print(f"Found {len(pred_files)} prediction file(s):\n")

        for pred_file in sorted(pred_files):
            try:
                df = pd.read_parquet(pred_file)
                print(f"✅ {pred_file.name}")
                print(f"   Records: {len(df)}")
                print(f"   Columns: {df.columns.tolist()[:5]}...")

                if 'model_type' in df.columns:
                    print(f"   Model types: {df['model_type'].value_counts().to_dict()}")

                print()

            except Exception as e:
                print(f"❌ {pred_file.name}: Error - {str(e)[:60]}")
                print()

    def print_summary(self):
        """Print verification summary"""
        print("="*70)
        print("VERIFICATION SUMMARY")
        print("="*70)

        total = self.results['total_models']
        successful = self.results['prediction_successful']
        failed = len(self.results['failed_models'])

        success_rate = (successful / total * 100) if total > 0 else 0

        print(f"Total Models: {total}")
        print(f"✅ Successful: {successful} ({success_rate:.1f}%)")
        print(f"❌ Failed: {failed}")

        if self.results['failed_models']:
            print(f"\nFailed models:")
            for model in self.results['failed_models']:
                print(f"  - {model}")

        print()

        if failed == 0:
            print("🎉 ALL MODELS VERIFIED SUCCESSFULLY!")
        else:
            print("⚠️  Some models failed verification")

        print("="*70)


def main():
    """Main verification function"""
    print("\n")
    print("="*70)
    print("NFL ML MODEL VERIFICATION")
    print("="*70)
    print()

    verifier = ModelVerifier()

    # Step 1: Verify directories
    if not verifier.verify_directories():
        print("❌ Some directories are missing. Cannot continue.")
        return 1

    # Step 2: Count files
    model_files, feature_files = verifier.count_files()

    if not model_files:
        print("❌ No model files found. Run train_models.py first.")
        return 1

    if not feature_files:
        print("❌ No feature files found. Run feature_engineer.py first.")
        return 1

    # Step 3: Load sample features
    features_df = verifier.load_sample_features()

    if features_df is None or len(features_df) == 0:
        print("❌ Could not load features. Cannot test models.")
        return 1

    # Step 4: Verify all models
    verifier.verify_all_models(features_df)

    # Step 5: Verify prediction files
    verifier.verify_predictions()

    # Step 6: Print summary
    verifier.print_summary()

    # Return exit code
    return 0 if len(verifier.results['failed_models']) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
