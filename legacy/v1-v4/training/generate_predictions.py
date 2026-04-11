#!/usr/bin/env python3
"""
File: src/nfl/generate_all_predictions.py

Generate predictions for all available weeks with features.
This script loads existing trained models and generates predictions
without retraining.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.nfl.models.base import NFLModelPipeline


def main():
    """Generate predictions for all weeks that have features"""

    print("="*70)
    print("GENERATE PREDICTIONS FOR ALL WEEKS")
    print("="*70)
    print(f"Started at: {datetime.now()}")
    print()

    # Hard-coded version mappings (no external config needed)
    VERSIONS = {
        '1': 'v1_baseline_mae5.14',
        '2': 'v2_variance_trends_mae4.66',
        '3': 'v3_epa_efficiency',
        '4': 'v4_position_specific'
    }

    # Ask user which version to use
    print("Available model versions:")
    print("1. V1 Baseline (MAE 5.14)")
    print("2. V2 Variance & Trends (MAE 4.66)")
    print("3. V3 EPA & Efficiency (MAE 4.66)")
    print("4. V4 Position-Specific (MAE TBD)")
    print()

    version_choice = input("Which version to use? (1/2/3): ").strip()

    if version_choice not in VERSIONS:
        print("❌ Invalid version choice!")
        return 1

    version = VERSIONS[version_choice]
    print(f"\n✓ Using version: {version}")
    print()

    # Initialize pipeline with selected version
    pipeline = NFLModelPipeline(version=version)

    # Load existing trained models
    print("Loading trained models...")
    model_files = list(pipeline.model_dir.glob("*.joblib"))

    if not model_files:
        print("❌ No trained models found!")
        print("Run: python src/nfl/train_models.py first")
        return 1

    print(f"✅ Found {len(model_files)} trained models")
    print()

    # Load models for each position
    from src.nfl.models.base import POBModel, EVOBModel, StatPredictor
    import joblib

    print("Loading models into pipeline...")
    for model_file in model_files:
        parts = model_file.stem.split('_')
        position = parts[0]
        model_type = parts[-1]

        if position not in pipeline.models:
            pipeline.models[position] = {}

        # Load model
        model_name = '_'.join(parts[1:])  # Everything after position

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

    print(f"✅ Loaded models for {len(pipeline.models)} positions")
    print()

    # Find all weeks with features
    feature_files = sorted(pipeline.feature_dir.glob("features_*.parquet"))

    if not feature_files:
        print("❌ No feature files found!")
        return 1

    print(f"Found {len(feature_files)} weeks with features")

    # Show available weeks
    available_weeks = []
    for f in feature_files:
        parts = f.stem.split('_')
        if len(parts) >= 4:
            season = int(parts[1])
            week = int(parts[3])
            available_weeks.append((season, week))

    # Group by season
    seasons = {}
    for season, week in available_weeks:
        if season not in seasons:
            seasons[season] = []
        seasons[season].append(week)

    print("\nAvailable weeks by season:")
    for season in sorted(seasons.keys()):
        weeks = sorted(seasons[season])
        print(f"  {season}: Weeks {min(weeks)}-{max(weeks)} ({len(weeks)} weeks)")
    print()

    # Ask which weeks to generate predictions for
    print("Options:")
    print("1. Generate for ALL weeks (all seasons)")
    print("2. Generate for 2025 season only")
    print("3. Generate for specific season and week range")

    choice = input("\nChoice (1/2/3): ").strip()

    weeks_to_process = []

    if choice == '1':
        # All weeks (all seasons)
        weeks_to_process = available_weeks

    elif choice == '2':
        # 2025 season only
        weeks_to_process = [(s, w) for s, w in available_weeks if s == 2025]

    elif choice == '3':
        # Specific season and week range
        season = int(input("Season (e.g., 2025): "))
        start_week = int(input("Start week (e.g., 1): "))
        end_week = int(input("End week (e.g., 13): "))

        weeks_to_process = [(s, w) for s, w in available_weeks
                           if s == season and start_week <= w <= end_week]

    else:
        print("❌ Invalid choice!")
        return 1

    print()
    print(f"Will generate predictions for {len(weeks_to_process)} weeks")
    print()

    # Generate predictions
    successful = 0
    failed = 0

    for season, week in weeks_to_process:
        try:
            print(f"Generating predictions for {season} Week {week}...")
            predictions = pipeline.generate_predictions(season, week)

            if predictions:
                successful += 1
            else:
                print(f"  ⚠️  No predictions generated (missing features?)")
                failed += 1

        except Exception as e:
            print(f"  ❌ Failed: {str(e)[:60]}")
            failed += 1

    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print()
    print(f"Predictions saved to: {pipeline.prediction_dir}")
    print("="*70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
