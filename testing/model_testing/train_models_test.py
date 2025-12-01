#!/usr/bin/env python3
"""
File: src/nfl/train_models.py

Simple script to train NFL ML models and generate predictions.
Run this to train all models and save them.
"""

import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.nfl.ml_models import NFLModelPipeline


def main():
    """Main training function"""
    
    print("="*70)
    print("NFL ML MODEL TRAINING")
    print("="*70)
    print(f"Started at: {datetime.now()}")
    print()

    # Set up paths - use project root data directory
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data" / "nfl"
    model_dir = Path(__file__).parent / "data" / "models"

    # Create directories if they don't exist
    model_dir.mkdir(parents=True, exist_ok=True)

    # Initialize pipeline with correct paths
    pipeline = NFLModelPipeline(
        data_dir=str(data_dir),
        model_dir=str(model_dir)
    )
    
    # Option 1: Quick training (fewer seasons for testing)
    quick_mode = input("Run quick training mode? (y/n): ").lower() == 'y'
    
    if quick_mode:
        print("\n🚀 Quick Training Mode (2020-2021 data only)")
        print("-"*40)
        
        # Override the load function to use less data
        original_func = pipeline.load_features_and_targets
        pipeline.load_features_and_targets = lambda: original_func(2020, 2021)
    else:
        print("\n📊 Full Training Mode (2020-2025 data)")
        print("-"*40)
    
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
                    
                    for i, pred in enumerate(predictions[:5]):
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
        print(f"\nModels saved to: ./models/nfl/")
        print(f"Predictions saved to: ./data/nfl/predictions/")
        
        # List saved models
        model_files = list(Path("./models/nfl").glob("*.joblib"))
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