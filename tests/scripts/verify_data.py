#!/usr/bin/env python3
"""
File: testing/verify_data.py

Verify that features and targets are being loaded correctly
Run this before training to ensure data pipeline is working
"""

import sys
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.nfl.models.base import NFLModelPipeline


def verify_data_loading():
    """Verify the data loading process step by step"""
    
    print("="*70)
    print("DATA VERIFICATION DIAGNOSTIC")
    print("="*70)
    
    pipeline = NFLModelPipeline()
    
    # Step 1: Check directories exist
    print("\n1. Checking directories:")
    print(f"   Feature dir exists: {pipeline.feature_dir.exists()}")
    print(f"   Raw dir exists: {pipeline.raw_dir.exists()}")
    
    # Step 2: Check for feature files
    feature_files = list(pipeline.feature_dir.glob("features_*.parquet"))
    print(f"\n2. Feature files found: {len(feature_files)}")
    
    # Step 3: Check for raw stat files
    raw_files = list(pipeline.raw_dir.glob("player_stats_*.parquet"))
    print(f"   Raw stat files found: {len(raw_files)}")
    
    # Step 4: Test loading a single week
    print("\n3. Testing single week load:")
    
    # Find a good week to test (mid-season 2020)
    test_feature_file = pipeline.feature_dir / "features_2020_week_10.parquet"
    test_raw_file = pipeline.raw_dir / "player_stats_2020_week_11.parquet"
    
    if test_feature_file.exists() and test_raw_file.exists():
        # Load feature file
        features_df = pd.read_parquet(test_feature_file)
        print(f"   Features shape: {features_df.shape}")
        print(f"   Feature columns sample: {list(features_df.columns[:5])}")
        
        # Load raw file
        raw_df = pd.read_parquet(test_raw_file)
        print(f"   Raw stats shape: {raw_df.shape}")
        print(f"   Raw columns sample: {list(raw_df.columns[:5])}")
        
        # Check what stats are available
        stat_cols = ['fantasy_points', 'fantasy_points_ppr', 'passing_yards', 
                    'rushing_yards', 'receiving_yards']
        available_stats = [col for col in stat_cols if col in raw_df.columns]
        print(f"   Available target stats: {available_stats}")
        
        # Test merge
        print("\n4. Testing merge:")
        merge_test = features_df.merge(
            raw_df[['player_id'] + available_stats[:2]],
            on='player_id',
            how='left',
            suffixes=('', '_target')
        )
        
        print(f"   Merged shape: {merge_test.shape}")
        print(f"   Players with targets: {merge_test[available_stats[0]].notna().sum()}/{len(merge_test)}")
        
        # Show sample merged data
        print("\n5. Sample merged data:")
        sample = merge_test[merge_test['position'] == 'QB'].head(2)
        if len(sample) > 0:
            display_cols = ['player_name', 'position', 'rolling_avg_fantasy_ppr'] + available_stats[:2]
            print(sample[display_cols].to_string())
    else:
        print("   Test files not found!")
    
    # Step 5: Test full pipeline load
    print("\n6. Testing full pipeline load (2020 only):")
    try:
        df = pipeline.load_features_and_targets(start_season=2020, end_season=2020)
        
        if not df.empty:
            print(f"   ✅ Successfully loaded {len(df)} records")
            
            # Check target columns
            target_cols = [col for col in df.columns if col.endswith('_target')]
            print(f"   Target columns found: {len(target_cols)}")
            print(f"   Sample targets: {target_cols[:5]}")
            
            # Check data by position
            print("\n   Data by position:")
            for pos in ['QB', 'RB', 'WR', 'TE', 'K']:
                pos_df = df[df['position'] == pos]
                if len(pos_df) > 0:
                    has_fantasy_target = pos_df['fantasy_points_ppr_target'].notna().sum()
                    print(f"     {pos}: {len(pos_df)} records, {has_fantasy_target} with fantasy targets")
            
            # Check a specific player's data
            print("\n7. Sample player data:")
            qb_sample = df[(df['position'] == 'QB') & (df['fantasy_points_ppr_target'].notna())].head(1)
            if len(qb_sample) > 0:
                player = qb_sample.iloc[0]
                print(f"   Player: {player['player_name']}")
                print(f"   Week: {player['season']}-{player['week']}")
                print(f"   Rolling Avg Fantasy: {player['rolling_avg_fantasy_ppr']:.2f}")
                print(f"   Actual Next Week Fantasy: {player['fantasy_points_ppr_target']:.2f}")
                print(f"   Difference: {player['fantasy_points_ppr_target'] - player['rolling_avg_fantasy_ppr']:.2f}")
        else:
            print("   ❌ No data loaded!")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("VERIFICATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    verify_data_loading()