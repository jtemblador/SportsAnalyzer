#!/usr/bin/env python3
"""
V2 Model Training Workflow - Complete pipeline for v2 models with improvements.

This script:
1. Regenerates features with improvements (variance, trends, stronger decay)
2. Trains models and saves to v2_variance_trends_mae4.66 folder
3. Generates predictions for Weeks 10-12
4. Validates accuracy and compares to v1

Run time: ~20-25 minutes total
Location: src/nfl/v2_retrain.py
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

    print_header("V2 MODEL TRAINING WORKFLOW")
    print("Improvements in v2:")
    print("  ✓ Stronger decay factor (0.85 vs 0.9) - emphasizes recent 3 games")
    print("  ✓ Variance features - identifies boom/bust players")
    print("  ✓ Recent trend features - captures hot/cold streaks")
    print()
    print("This will take approximately 20-25 minutes.")
    print("="*80)

    # Confirm
    response = input("\nProceed with v2 training? (y/n): ").strip().lower()
    if response != 'y':
        print("Aborted.")
        return 0

    # Step 1: Generate V2 features
    print_header("STEP 1/4: Generating V2 Features")
    print("Creating v2_variance_trends feature set...")
    print("This will generate features WITH variance and trend columns.")
    print("\n⚡ SMART RESUME: Already-generated weeks will be skipped automatically!")
    print("Generating improved features for ALL weeks (2020-2025)...")
    print("Time: ~5-10 minutes (faster if resuming)\n")

    version = "v2_variance_trends_mae4.66"

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
    print("Training all 40 models with improved features...")
    print("⚡ SMART RESUME: Already-trained models will be skipped automatically!")
    print("Time: ~10-15 minutes (faster if resuming)\n")

    pipeline = NFLModelPipeline(
        data_dir=str(project_root / 'data/nfl'),
        model_dir=str(project_root / 'data/nfl/models'),
        version=version  # Uses v2 features and saves to v2 models
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

    # Step 3: Generate test predictions
    print_header("STEP 3/4: Generating Test Predictions")
    print("Generating predictions for 2025 Weeks 10-12...\n")

    for week in [10, 11, 12]:
        try:
            predictions = pipeline.generate_predictions(2025, week)
            if predictions is not None:
                print(f"  ✓ Week {week}: {len(predictions)} predictions")
        except Exception as e:
            print(f"  ⚠ Week {week}: {str(e)}")

    print("\n✅ Predictions generated!")

    # Step 4: Validate and compare
    print_header("STEP 4/4: Validating Accuracy")
    print("Comparing v1 (baseline) vs v2 (improved) on Weeks 10-12...\n")

    # Run comparison
    subprocess.run([
        "python3", str(project_root / "testing/compare_versions.py"),
        "v1_baseline_mae5.14",
        version,
        "10", "11", "12"
    ])

    # Done
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60

    print_header("WORKFLOW COMPLETE")
    print(f"Total time: {duration:.1f} minutes")
    print()
    print("📊 Review the validation results above to see if v2 improved!")
    print()
    print("Key metrics to check:")
    print("  - v1 MAE: 5.14 points (baseline)")
    print("  - v2 MAE: ??? points (check output above)")
    print()
    print("If v2 shows improvement, you can rename the folders to include actual MAE:")
    print(f"  mv data/nfl/features/{version} data/nfl/features/v2_variance_trends_maeX.XX")
    print(f"  mv data/nfl/models/{version} data/nfl/models/v2_variance_trends_maeX.XX")
    print(f"  mv data/nfl/predictions/{version} data/nfl/predictions/v2_variance_trends_maeX.XX")
    print("="*80 + "\n")

    return 0

if __name__ == "__main__":
    sys.exit(main())
