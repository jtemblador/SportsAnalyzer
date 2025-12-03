#!/usr/bin/env python3
"""
File: src/nfl/train_models.py

Train NFL ML models on all available data and generate predictions.
This script uses all 2020-2025 data for maximum accuracy.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.nfl.ml_models import NFLModelPipeline


def main():
    """Main training function"""

    print("="*70)
    print("NFL ML MODEL TRAINING (VERSIONED)")
    print("="*70)
    print(f"Started at: {datetime.now()}")
    print()

    # Allow version to be specified via command line
    version = "v2_variance_trends"
    if len(sys.argv) > 1:
        version = sys.argv[1]

    print(f"🏷️  Version: {version}")
    print()

    # Set up paths - use project root data directory
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data" / "nfl"
    model_dir = data_dir / "models"  # Models saved to data/nfl/models/[version]

    # Initialize pipeline with versioning
    pipeline = NFLModelPipeline(
        data_dir=str(data_dir),
        model_dir=str(model_dir),
        version=version
    )

    print("📊 Training on ALL available data (2020-2025)")
    print("-"*70)
    print()
    
    try:
        # Train all models
        pipeline.train_all_models()
        
        # Generate predictions for latest week
        print("\n" + "="*70)
        print("GENERATING PREDICTIONS")
        print("="*70)
        
        # Find the latest week with features
        feature_files = sorted(pipeline.feature_dir.glob("features_*.parquet"))
        if feature_files:
            latest_file = feature_files[-1]
            parts = latest_file.stem.split('_')
            if len(parts) >= 4:
                season = int(parts[1])
                week = int(parts[3])
                
                print(f"Generating predictions for Season {season}, Week {week+1}")
                predictions = pipeline.generate_predictions(season, week)
                
                if predictions:
                    # Show sample predictions
                    print("\n📈 Sample Predictions (first 5):")
                    print("-"*40)

                    for pred in predictions[:5]:
                        if pred['model_type'] == 'evob':
                            print(f"{pred['player_name']} ({pred['position']}):")
                            print(f"  {pred['stat']}: {pred['predicted_value']:.1f}")
                            print(f"  Confidence: [{pred['confidence_lower']:.1f}, {pred['confidence_upper']:.1f}]")
                            print()
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE")
        print("="*70)
        print(f"Finished at: {datetime.now()}")

        # Show model locations
        print(f"\nModels saved to: {model_dir}")
        print(f"Predictions saved to: {pipeline.prediction_dir}")

        # List saved models
        model_files = list(model_dir.glob("*.joblib"))
        if model_files:
            print(f"\nSaved {len(model_files)} model files:")
            for f in model_files[:5]:
                print(f"  - {f.name}")
            if len(model_files) > 5:
                print(f"  ... and {len(model_files) - 5} more")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())