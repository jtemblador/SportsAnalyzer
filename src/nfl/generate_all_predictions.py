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
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.nfl.ml_models import NFLModelPipeline


def main():
    """Generate predictions for all weeks that have features"""

    print("="*70)
    print("GENERATE PREDICTIONS FOR ALL WEEKS")
    print("="*70)
    print(f"Started at: {datetime.now()}")
    print()

    # Initialize pipeline
    pipeline = NFLModelPipeline()

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
    from src.nfl.ml_models import POBModel, EVOBModel, StatPredictor
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
    print()

    # Ask which weeks to generate predictions for
    print("Options:")
    print("1. Generate for ALL weeks")
    print("2. Generate for latest week only")
    print("3. Generate for specific range")

    choice = input("\nChoice (1/2/3): ").strip()

    weeks_to_process = []

    if choice == '1':
        # All weeks
        for f in feature_files:
            parts = f.stem.split('_')
            if len(parts) >= 4:
                season = int(parts[1])
                week = int(parts[3])
                weeks_to_process.append((season, week))

    elif choice == '2':
        # Latest week only
        latest = feature_files[-1]
        parts = latest.stem.split('_')
        if len(parts) >= 4:
            season = int(parts[1])
            week = int(parts[3])
            weeks_to_process.append((season, week))

    elif choice == '3':
        # Specific range
        start_week = int(input("Start week (e.g., 1): "))
        end_week = int(input("End week (e.g., 13): "))
        season = int(input("Season (e.g., 2025): "))

        for week in range(start_week, end_week + 1):
            weeks_to_process.append((season, week))

    else:
        print("Invalid choice")
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
