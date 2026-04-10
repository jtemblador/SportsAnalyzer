#!/usr/bin/env python3
"""
V4 Model Training Workflow - Position-Specific Hyperparameters

This script:
1. Uses V2 features (same as V2 - variance, trends, stronger decay)
2. Trains models with POSITION-SPECIFIC hyperparameters (v4_ml_models.py)
   - QB: Deeper trees (depth=9), more regularization
   - K: Simpler models (depth=3)
   - TE: Moderate complexity (depth=6)
   - RB/WR: Standard settings
3. Generates predictions for 2025 weeks
4. Validates accuracy and compares to V2

Run time: ~20-30 minutes total
Location: src/nfl/v4_retrain.py
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime

# Get project root (4 levels up: src/nfl/training/ -> src/nfl/ -> src/ -> project/)
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from nfl.features.v4_engineer import FeatureEngineer
from nfl.models.v4_models import PositionSpecificEVOBModel as EVOBModel
from nfl.models.v4_models import PositionSpecificPOBModel as POBModel
from nfl.models.v4_models import PositionSpecificStatPredictor as StatPredictor

# Import the pipeline but we'll override the model classes
from nfl.models.base import NFLModelPipeline


def print_header(title):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


class V4ModelPipeline(NFLModelPipeline):
    """
    V4 Pipeline that uses position-specific model classes.
    Overrides the training methods to use V4 model classes.
    """

    def train_position_models(self, position, df):
        """Train all models for a specific position using V4 position-specific models."""
        print(f"\n{'='*60}")
        print(f"Training models for {position} (V4 Position-Specific)")
        print(f"{'='*60}")

        pos_df = df[df['position'] == position].copy()

        if len(pos_df) < 100:
            print(f"Insufficient data for {position} ({len(pos_df)} records)")
            return {}

        print(f"Training on {len(pos_df)} {position} records")

        feature_cols = [col for col in self.position_features[position] if col in pos_df.columns]

        for col in feature_cols:
            pos_df = pos_df[pos_df[col].notna()]

        pos_df = pos_df.sort_values(['season', 'week'])

        train_size = int(0.7 * len(pos_df))
        val_size = int(0.15 * len(pos_df))

        train_df = pos_df.iloc[:train_size]
        val_df = pos_df.iloc[train_size:train_size+val_size]
        test_df = pos_df.iloc[train_size+val_size:]

        position_models = {}

        for stat in self.position_stats[position]:
            target_col = f'{stat}_target'

            if target_col not in train_df.columns:
                print(f"  Skipping {stat} - no target column")
                continue

            train_clean = train_df[train_df[target_col].notna()].copy()
            val_clean = val_df[val_df[target_col].notna()].copy()
            test_clean = test_df[test_df[target_col].notna()].copy()

            if len(train_clean) < 50:
                print(f"  Skipping {stat} - insufficient training data")
                continue

            print(f"\n  Training {stat} models (V4 hyperparameters)...")

            X_train = train_clean[feature_cols]
            X_val = val_clean[feature_cols]
            X_test = test_clean[feature_cols]

            # Use V4 position-specific EVOB model
            evob_model = EVOBModel(position, stat)

            y_train = evob_model.prepare_target(train_clean)
            y_val = evob_model.prepare_target(val_clean)
            y_test = evob_model.prepare_target(test_clean)

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

            if not evob_scores:
                print(f"    Skipping {stat} - no variance in data")
                continue

            test_scores = evob_model.evaluate(X_test[mask_test], y_test[mask_test])
            print(f"    EVOB Test - MAE: {test_scores['mae']:.2f}, R2: {test_scores['r2']:.3f}")

            # Train POB model for fantasy points
            if 'fantasy' in stat:
                pob_model = POBModel(position, stat)

                y_train_binary = pob_model.prepare_target(train_clean)
                y_val_binary = pob_model.prepare_target(val_clean)
                y_test_binary = pob_model.prepare_target(test_clean)

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

            position_models[f'{stat}_evob'] = evob_model
            position_models[f'{stat}_stat'] = stat_model

        return position_models


def main():
    start_time = datetime.now()

    print_header("V4 MODEL TRAINING WORKFLOW")
    print("V4 Improvements:")
    print("  Phase 1 - Vegas Odds Integration:")
    print("    ✓ Team implied totals (volume predictor)")
    print("    ✓ Point spreads (game script predictor)")
    print("    ✓ Over/under (pace indicator)")
    print("    ✓ Home/away context")
    print("    ✓ Position-specific volume indices")
    print()
    print("  Phase 2 - Position-Specific Hyperparameters:")
    print("    ✓ QB: Deeper trees (depth=9), more regularization")
    print("    ✓ K: Simpler models (depth=3, 100 estimators)")
    print("    ✓ TE: Moderate complexity (depth=6)")
    print("    ✓ RB/WR: Standard settings (already work well)")
    print()
    print("Expected improvements:")
    print("  - Phase 1: 0.3-0.5 MAE reduction (game context)")
    print("  - Phase 2: 0.2 MAE reduction (QB improvement)")
    print("  - Combined: 4.66 → ~4.2-4.4 MAE (10% better)")
    print()
    print("This will take approximately 20-30 minutes.")
    print("="*80)

    response = input("\nProceed with V4 training? (y/n): ").strip().lower()
    if response != 'y':
        print("Aborted.")
        return 0

    version = "v4_position_specific"

    # Step 0: Fetch Vegas team lines if not cached
    print_header("STEP 0/4: Checking Vegas Odds Data")
    print("Vegas team lines are required for V4 features.")
    print("Checking cache...\n")

    ODDS_API_KEY = 'c6d41f99d9fdabfa5f5abaf8df1c9084'

    from nfl.odds.fetcher import VegasLinesFetcher
    vegas_fetcher = VegasLinesFetcher(
        cache_dir=str(project_root / 'data/nfl/vegas_odds'),
        odds_api_key=ODDS_API_KEY
    )

    # Check if we have cached data
    cached_lines = vegas_fetcher.load_cached_team_lines()

    if cached_lines.empty:
        print("⚠ No Vegas lines cached. Fetching from API...")
        print("This will use ~1 API call (500/month limit)")
        fetch_response = input("Fetch Vegas lines now? (y/n): ").strip().lower()
        if fetch_response != 'y':
            print("❌ Cannot proceed without Vegas data for V4.")
            return 1

        vegas_fetcher.fetch_team_lines(force_refresh=False)
        cached_lines = vegas_fetcher.load_cached_team_lines()

        if cached_lines.empty:
            print("❌ Failed to fetch Vegas lines. Cannot train V4.")
            return 1
    else:
        weeks = sorted(cached_lines['week'].unique())
        print(f"✅ Found cached Vegas lines for {len(cached_lines)} games")
        print(f"   Weeks available: {weeks}")
        print(f"   Season: {cached_lines['season'].unique()}")

    print("\n✓ Vegas data ready!\n")

    # Step 1: Generate features (V2 features + Vegas odds)
    print_header("STEP 1/4: Generating V4 Features")
    print("V4 Features (50 total):")
    print("  - 42 from V2 (variance, trends, decay=0.85)")
    print("  - 8 NEW Vegas features (spread, totals, volume indices)")
    print("\n⚡ SMART RESUME: Already-generated weeks will be skipped automatically!")
    print("Time: ~5-10 minutes (faster if resuming)\n")

    engineer = FeatureEngineer(
        raw_data_dir=str(project_root / 'data/nfl/raw'),
        features_dir=str(project_root / 'data/nfl/features'),
        version=version,
        odds_api_key=ODDS_API_KEY  # Enable Vegas features
    )

    try:
        engineer.engineer_all_features(start_season=2020, end_season=2025)
        print("\n✅ Feature generation complete!")
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user (Ctrl+C)")
        print("✓ Progress saved - already-processed weeks are preserved")
        return 1
    except Exception as e:
        print(f"\n❌ Feature generation failed: {e}")
        return 1

    # Step 2: Train models with V4 position-specific hyperparameters
    print_header("STEP 2/4: Training V4 Models (Position-Specific)")
    print("Training with 50 features (42 V2 + 8 Vegas) using position-optimized hyperparameters...")
    print("  - QB: depth=9, estimators=500, lr=0.005")
    print("  - K: depth=3, estimators=100, lr=0.01")
    print("  - TE: depth=6, estimators=300, lr=0.01")
    print("  - RB/WR: depth=7, estimators=300, lr=0.01")
    print("\nTime: ~15-20 minutes\n")

    pipeline = V4ModelPipeline(
        data_dir=str(project_root / 'data/nfl'),
        model_dir=str(project_root / 'data/nfl/models'),
        version=version
    )

    try:
        pipeline.train_all_models()
        print("\n✅ Model training complete!")
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user (Ctrl+C)")
        return 1
    except Exception as e:
        print(f"\n❌ Model training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Load all trained models
    print("\n🔄 Loading all trained V4 models for prediction...")
    import joblib

    model_files = list(pipeline.model_dir.glob("*.joblib"))
    pipeline.models = {}

    for model_file in model_files:
        parts = model_file.stem.split('_')
        position = parts[0]
        model_type = parts[-1]

        if position not in pipeline.models:
            pipeline.models[position] = {}

        model_name = '_'.join(parts[1:])

        if model_type == 'pob':
            stat = '_'.join(parts[1:-1])
            model = POBModel(position, stat)
        elif model_type == 'evob':
            stat = '_'.join(parts[1:-1])
            model = EVOBModel(position, stat)
        elif model_type == 'stat':
            stat = '_'.join(parts[1:-1])
            model = StatPredictor(position, stat)
        else:
            continue

        model.load_model(str(model_file))
        pipeline.models[position][model_name] = model

    print(f"✓ Loaded {len(model_files)} models for {len(pipeline.models)} positions\n")

    # Step 3: Generate predictions for all available weeks
    print_header("STEP 3/4: Generating Predictions for All 2025 Weeks")

    # Detect which weeks have raw data available
    raw_data_dir = project_root / 'data/nfl/raw'
    available_weeks = []

    for week in range(1, 19):  # Check weeks 1-18
        raw_file = raw_data_dir / f'player_stats_2025_week_{week}.parquet'
        if raw_file.exists():
            available_weeks.append(week)

    if not available_weeks:
        print("⚠ No 2025 raw data found. Skipping predictions.")
    else:
        print(f"Found raw data for {len(available_weeks)} weeks: {available_weeks}")
        print(f"Generating predictions for weeks {min(available_weeks)}-{max(available_weeks)}...\n")

        success_count = 0
        for week in available_weeks:
            try:
                predictions = pipeline.generate_predictions(2025, week)
                if predictions is not None:
                    print(f"  ✓ Week {week}: {len(predictions)} predictions")
                    success_count += 1
            except Exception as e:
                print(f"  ⚠ Week {week}: {str(e)[:50]}")

        print(f"\n✅ Generated predictions for {success_count}/{len(available_weeks)} weeks!")

    # Step 4: Validate and compare
    print_header("STEP 4/4: Validating Accuracy")

    # Use the last 3 available weeks for validation (or all if less than 3)
    validation_weeks = available_weeks[-3:] if len(available_weeks) >= 3 else available_weeks

    if validation_weeks:
        print(f"Comparing V2 vs V4 on weeks {validation_weeks}...\n")

        subprocess.run([
            "python3", str(project_root / "testing/compare_versions.py"),
            "v2_variance_trends_mae4.66",
            version,
        ] + [str(w) for w in validation_weeks])
    else:
        print("⚠ No weeks available for validation. Skipping comparison.")

    # Done
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60

    print_header("V4 WORKFLOW COMPLETE")
    print(f"Total time: {duration:.1f} minutes")
    print()
    print("📊 Review the validation results above!")
    print()
    print("Key metrics to check:")
    print("  - V2 MAE: 4.66 points (current best)")
    print("  - V4 MAE: ??? points (check output above)")
    print("  - V2 QB MAE: 7.19 points")
    print("  - V4 QB MAE: ??? (should be ~6.5-6.8 if improved)")
    print()
    print("V4 Feature Set:")
    print("  - 50 total features (42 V2 + 8 Vegas)")
    print("  - Vegas features: spread, totals, implied points, volume indices")
    print("  - Position-specific hyperparameters: QB depth=9, K depth=3")
    print()
    print("If V4 shows improvement:")
    print(f"  mv data/nfl/features/{version} data/nfl/features/v4_vegas_odds_maeX.XX")
    print(f"  mv data/nfl/models/{version} data/nfl/models/v4_vegas_odds_maeX.XX")
    print(f"  mv data/nfl/predictions/{version} data/nfl/predictions/v4_vegas_odds_maeX.XX")
    print("="*80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
