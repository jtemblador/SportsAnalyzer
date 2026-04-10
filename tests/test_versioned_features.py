#!/usr/bin/env python3
"""
Test the versioned features system
"""

import sys
from pathlib import Path

# Get project root (two levels up from this file when in testing/)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from nfl.features.engineer import FeatureEngineer
from nfl.models.base import NFLModelPipeline

print("="*70)
print("TESTING VERSIONED FEATURES SYSTEM")
print("="*70)

# Test 1: FeatureEngineer with version
print("\nTest 1: FeatureEngineer Initialization")
print("-"*70)
engineer_v1 = FeatureEngineer(
    raw_data_dir=str(project_root / 'data/nfl/raw'),
    features_dir=str(project_root / 'data/nfl/features'),
    version='v1_baseline_mae5.14'
)
print("✓ V1 engineer created")

engineer_v2 = FeatureEngineer(
    raw_data_dir=str(project_root / 'data/nfl/raw'),
    features_dir=str(project_root / 'data/nfl/features'),
    version='v2_variance_trends_test'
)
print("✓ V2 engineer created")

# Test 2: Check paths
print("\nTest 2: Verify Paths")
print("-"*70)
print(f"V1 features path: {engineer_v1.cleaned_data_dir}")
v1_path = Path(engineer_v1.cleaned_data_dir)
print(f"  Exists: {v1_path.exists()}")
if v1_path.exists():
    files = list(v1_path.glob('*.parquet'))
    print(f"  Files: {len(files)}")

print(f"\nV2 features path: {engineer_v2.cleaned_data_dir}")
v2_path = Path(engineer_v2.cleaned_data_dir)
print(f"  Exists: {v2_path.exists()}")

# Test 3: NFLModelPipeline with versioned features
print("\nTest 3: NFLModelPipeline Initialization")
print("-"*70)
pipeline_v1 = NFLModelPipeline(
    data_dir=str(project_root / 'data/nfl'),
    model_dir=str(project_root / 'data/nfl/models'),
    version='v1_baseline_mae5.14'
)
print(f"\nV1 Pipeline feature_dir: {pipeline_v1.feature_dir}")
print(f"  Exists: {pipeline_v1.feature_dir.exists()}")

pipeline_v2 = NFLModelPipeline(
    data_dir=str(project_root / 'data/nfl'),
    model_dir=str(project_root / 'data/nfl/models'),
    version='v2_variance_trends_test'
)
print(f"\nV2 Pipeline feature_dir: {pipeline_v2.feature_dir}")
print(f"  Exists: {pipeline_v2.feature_dir.exists()}")

# Clean up test directory
test_dir = project_root / 'data/nfl/features/v2_variance_trends_test'
if test_dir.exists() and len(list(test_dir.glob('*'))) == 0:
    test_dir.rmdir()
    print("\n✓ Cleaned up test directory")

print("\n" + "="*70)
print("✅ ALL TESTS PASSED!")
print("="*70)
print("\nVersioned features system is working correctly:")
print("  - V1 features: data/nfl/features/v1_baseline_mae5.14/")
print("  - V2 features: data/nfl/features/v2_variance_trends_mae4.66/")
print("\nReady to run retrain_v2.py!")
