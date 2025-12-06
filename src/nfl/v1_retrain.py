#!/usr/bin/env python3
"""
V1 Model Training Workflow - Complete pipeline for v1 baseline models.

This script:
1. Regenerates V1 baseline features (standard decay=0.9, rolling averages)
2. Trains models and saves to v1_baseline_mae5.14 folder
3. Generates predictions for ALL available 2025 weeks (auto-detects which weeks have data)
4. Validates accuracy on the most recent 3 weeks

V1 Features:
- Rolling averages with decay=0.9
- Usage trends
- Opponent defense ranks
- 34 feature columns

Run time: ~20-25 minutes total
Location: src/nfl/v1_retrain.py
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime

# Get project root (3 levels up: src/nfl/ -> src/ -> project/)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from nfl.feature_engineer import FeatureEngineer
from nfl.ml_models import NFLModelPipeline

def print_header(title):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

def main():
    start_time = datetime.now()

    print_header("V1 BASELINE MODEL TRAINING WORKFLOW")
    print("V1 Baseline features:")
    print("  ✓ Rolling averages (decay=0.9)")
    print("  ✓ Usage trends")
    print("  ✓ Opponent defense ranks")
    print("  ✓ 34 feature columns")
    print()
    print("This is the baseline model that achieved 5.14 MAE.")
    print("This will take approximately 20-25 minutes.")
    print("="*80)

    # Confirm
    response = input("\nProceed with v1 training? (y/n): ").strip().lower()
    if response != 'y':
        print("Aborted.")
        return 0

    # Step 1: Generate V1 features
    print_header("STEP 1/4: Generating V1 Baseline Features")
    print("Creating v1_baseline_mae5.14 feature set...")
    print("This will generate features with standard decay (0.9) and rolling averages.")
    print("\n⚡ SMART RESUME: Already-generated weeks will be skipped automatically!")
    print("Generating baseline features for ALL weeks (2020-2025)...")
    print("Time: ~5-10 minutes (faster if resuming)\n")

    version = "v1_baseline_mae5.14"

    engineer = FeatureEngineer(
        raw_data_dir=str(project_root / 'data/nfl/raw'),
        features_dir=str(project_root / 'data/nfl/features'),
        version=version
    )

    try:
        # This will automatically skip already-processed weeks
        engineer.engineer_all_features(start_season=2020, end_season=2025)
        print("\n✅ Feature generation complete!")
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user (Ctrl+C)")
        print("✓ Progress saved - already-processed weeks are preserved")
        print("✓ Run this script again to resume from where you left off")
        return 1
    except Exception as e:
        print(f"\n❌ Feature generation failed: {e}")
        return 1

    # Step 2: Train models
    print_header("STEP 2/4: Training Models")
    print("Training all 40 models with baseline features...")
    print("⚡ SMART RESUME: Already-trained models will be skipped automatically!")
    print("Time: ~10-15 minutes (faster if resuming)\n")

    pipeline = NFLModelPipeline(
        data_dir=str(project_root / 'data/nfl'),
        model_dir=str(project_root / 'data/nfl/models'),
        version=version  # Uses v1 features and saves to v1 models
    )

    try:
        pipeline.train_all_models()
        print("\n✅ Model training complete!")
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user (Ctrl+C)")
        print("✓ Progress saved - already-trained models are preserved")
        print("✓ Run this script again to resume from where you left off")
        return 1
    except Exception as e:
        print(f"\n❌ Model training failed: {e}")
        return 1

    # Load all trained models before generating predictions
    print("\n🔄 Loading all trained models for prediction...")
    from nfl.ml_models import POBModel, EVOBModel, StatPredictor
    import joblib

    model_files = list(pipeline.model_dir.glob("*.joblib"))
    pipeline.models = {}  # Reset to empty

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
                print(f"  ⚠ Week {week}: {str(e)}")

        print(f"\n✅ Generated predictions for {success_count}/{len(available_weeks)} weeks!")

    # Step 4: Validate accuracy
    print_header("STEP 4/4: Validating Accuracy")

    # Use the last 3 available weeks for validation (or all if less than 3)
    validation_weeks = available_weeks[-3:] if len(available_weeks) >= 3 else available_weeks

    if validation_weeks:
        print(f"Validating V1 baseline model on weeks {validation_weeks}...\n")

        # Run validation
        subprocess.run([
            "python3", str(project_root / "testing/validate_accuracy.py"),
            version,
        ] + [str(w) for w in validation_weeks])
    else:
        print("⚠ No weeks available for validation. Skipping.")

    # Done
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60

    print_header("WORKFLOW COMPLETE")
    print(f"Total time: {duration:.1f} minutes")
    print()
    print("📊 Review the validation results above!")
    print()
    if available_weeks:
        print(f"✓ Generated predictions for {len(available_weeks)} weeks: {min(available_weeks)}-{max(available_weeks)}")
        print(f"✓ Validated on weeks: {validation_weeks}")
    print()
    print("V1 Baseline MAE: 5.14 points")
    print()
    print("Predictions are saved to:")
    print(f"  data/nfl/predictions/{version}/")
    print()
    print("To compare with V2:")
    print(f"  python3 testing/compare_versions.py {version} v2_variance_trends_mae4.66 {' '.join(map(str, validation_weeks))}")
    print("="*80 + "\n")

    return 0

if __name__ == "__main__":
    sys.exit(main())
