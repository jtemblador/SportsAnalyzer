#!/usr/bin/env python3
"""
V3 Model Training Workflow - Complete pipeline for V3 models with EPA & efficiency features.

This script:
1. Regenerates features with V3 improvements (EPA, efficiency metrics, position-specific decay)
2. Trains models and saves to v3_epa_efficiency folder
3. Generates predictions for ALL available 2025 weeks (auto-detects which weeks have data)
4. Validates accuracy on the most recent 3 weeks and compares to v2

V3 Improvements over V2:
- EPA features (Expected Points Added) - #1 predictive stat
- Efficiency metrics: PACR, RACR, WOPR, CPOE
- Target share & air yards share
- YAC (Yards After Catch) features
- Position-specific decay factors:
    QB: 0.90 (stable), RB: 0.85, WR: 0.85, TE: 0.80 (volatile), K: 0.90
- Kicker distance features (40-49, 50+)

Run time: ~3-4 hours (feature generation is slow)
Location: src/nfl/v3_retrain.py
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime

# Get project root (4 levels up: src/nfl/training/ -> src/nfl/ -> src/ -> project/)
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from nfl.features.v3_engineer import V3FeatureEngineer
from nfl.models.base import NFLModelPipeline


def print_header(title):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def main():
    start_time = datetime.now()

    print_header("V3 MODEL TRAINING WORKFLOW")
    print("V3 Improvements over V2:")
    print("  ✓ EPA features (Expected Points Added) - #1 predictive stat")
    print("  ✓ Efficiency metrics: PACR, RACR, WOPR, CPOE")
    print("  ✓ Target share & air yards share features")
    print("  ✓ YAC (Yards After Catch) features")
    print("  ✓ Position-specific decay factors:")
    print("      QB: 0.90 (stable), RB: 0.85, WR: 0.85, TE: 0.80 (volatile), K: 0.90")
    print("  ✓ Kicker distance features (40-49, 50+)")
    print()
    print("Target: 4.0-4.2 MAE (V2 baseline: 4.66)")
    print()
    print("⚠️  This will take approximately 3-4 hours (feature generation is slow).")
    print("⚡ SMART RESUME: Progress is saved - you can Ctrl+C and resume later!")
    print("="*80)

    # Confirm
    response = input("\nProceed with V3 training? (y/n): ").strip().lower()
    if response != 'y':
        print("Aborted.")
        return 0

    # Version name (will update with actual MAE after validation)
    version = "v3_epa_efficiency"

    # Step 1: Generate V3 features
    print_header("STEP 1/4: Generating V3 Features")
    print("Creating v3_epa_efficiency feature set...")
    print()
    print("V3 Features include:")
    print("  - QB: passing_epa, cpoe, pacr, passing_yac (+ position decay=0.90)")
    print("  - RB: rushing_epa, receiving_epa (+ position decay=0.85)")
    print("  - WR: receiving_epa, racr, wopr, yac, target_share, air_yards_share (+ decay=0.85)")
    print("  - TE: receiving_epa, racr, yac, target_share (+ decay=0.80)")
    print("  - K:  fg_made_40_49, fg_made_50_plus, fg_made_distance (+ decay=0.90)")
    print()
    print("⚡ SMART RESUME: Already-generated weeks will be skipped automatically!")
    print("Generating improved features for ALL weeks (2020-2025)...")
    print("Time: ~3-4 hours (faster if resuming)\n")

    engineer = V3FeatureEngineer(
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
        import traceback
        traceback.print_exc()
        return 1

    # Step 2: Train models
    print_header("STEP 2/4: Training Models")
    print("Training all 40 models with V3 features...")
    print("⚡ SMART RESUME: Already-trained models will be skipped automatically!")
    print("Time: ~10-15 minutes (faster if resuming)\n")

    pipeline = NFLModelPipeline(
        data_dir=str(project_root / 'data/nfl'),
        model_dir=str(project_root / 'data/nfl/models'),
        version=version  # Uses V3 features and saves to V3 models
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
        import traceback
        traceback.print_exc()
        return 1

    # Load all trained models before generating predictions
    print("\n🔄 Loading all trained models for prediction...")
    from nfl.models.base import POBModel, EVOBModel, StatPredictor

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

    # Step 4: Validate and compare
    print_header("STEP 4/4: Validating Accuracy")

    # Use the last 3 available weeks for validation (or all if less than 3)
    validation_weeks = available_weeks[-3:] if len(available_weeks) >= 3 else available_weeks

    if validation_weeks:
        print(f"Comparing V2 (baseline) vs V3 (improved) on weeks {validation_weeks}...\n")

        # Run comparison against V2
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

    print_header("WORKFLOW COMPLETE")
    print(f"Total time: {duration:.1f} minutes")
    print()
    print("📊 Review the validation results above to see if V3 improved!")
    print()
    if available_weeks:
        print(f"✓ Generated predictions for {len(available_weeks)} weeks: {min(available_weeks)}-{max(available_weeks)}")
        print(f"✓ Validated on weeks: {validation_weeks}")
    print()
    print("Key metrics to check:")
    print("  - V2 MAE: 4.66 points (baseline)")
    print("  - V3 MAE: ??? points (check output above)")
    print()
    print("Expected V3 Improvements:")
    print("  - Overall MAE: 4.66 → 4.0-4.2 (10-15% improvement)")
    print("  - QB MAE: 7.19 → <6.5 (fixed with EPA/CPOE)")
    print()
    print("Predictions are saved to:")
    print(f"  data/nfl/predictions/{version}/")
    print("  - RB MAE: 4.73 → <4.3 (fixed with rushing_epa)")
    print("  - TE MAE: 3.57 → maintain (already excellent)")
    print("  - WR MAE: 4.57 → maintain or improve")
    print()
    print("If V3 shows improvement, rename the folders to include actual MAE:")
    print(f"  mv data/nfl/features/{version} data/nfl/features/v3_epa_efficiency_maeX.XX")
    print(f"  mv data/nfl/models/{version} data/nfl/models/v3_epa_efficiency_maeX.XX")
    print(f"  mv data/nfl/predictions/{version} data/nfl/predictions/v3_epa_efficiency_maeX.XX")
    print("="*80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
