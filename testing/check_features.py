#!/usr/bin/env python3
"""
Quick diagnostic script to check feature data quality
Run this to understand what data is actually available
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def analyze_features():
    """Analyze the feature engineered data"""
    
    print("="*70)
    print("FEATURE DATA DIAGNOSTIC")
    print("="*70)
    
    # Check if cleaned directory exists
    cleaned_dir = Path("./data/nfl/cleaned")
    if not cleaned_dir.exists():
        print("❌ Cleaned directory doesn't exist!")
        return
    
    # Get all feature files
    feature_files = sorted(cleaned_dir.glob("features_*.parquet"))
    print(f"\n📁 Found {len(feature_files)} feature files")
    
    if not feature_files:
        print("❌ No feature files found!")
        return
    
    # Sample a few files to understand the data
    print("\n" + "="*70)
    print("ANALYZING SAMPLE FILES")
    print("="*70)
    
    for i, file in enumerate(feature_files[:3]):  # Check first 3 files
        print(f"\n📄 File: {file.name}")
        print("-"*40)
        
        try:
            df = pd.read_parquet(file)
            
            print(f"Shape: {df.shape[0]} players, {df.shape[1]} columns")
            print(f"Players with sufficient data: {df['has_sufficient_data'].sum()}/{len(df)}")
            
            # Check data completeness by position
            print("\nData completeness by position:")
            for pos in sorted(df['position'].unique()):
                pos_df = df[df['position'] == pos]
                
                # Get position-specific columns
                if pos == 'QB':
                    key_cols = ['rolling_avg_passing_yds', 'rolling_avg_passing_tds']
                elif pos == 'RB':
                    key_cols = ['rolling_avg_rushing_yds', 'rolling_avg_carries']
                elif pos in ['WR', 'TE']:
                    key_cols = ['rolling_avg_receiving_yds', 'rolling_avg_targets']
                elif pos == 'K':
                    key_cols = ['rolling_avg_fg_made', 'rolling_avg_pat_made']
                else:
                    key_cols = []
                
                # Check how many have non-null values
                if key_cols and all(col in df.columns for col in key_cols):
                    non_null = pos_df[key_cols[0]].notna().sum()
                    print(f"  {pos}: {len(pos_df)} players, {non_null} with data ({non_null/len(pos_df)*100:.1f}%)")
                else:
                    print(f"  {pos}: {len(pos_df)} players")
            
        except Exception as e:
            print(f"❌ Error reading file: {e}")
    
    # Load all data to get overall statistics
    print("\n" + "="*70)
    print("OVERALL DATA STATISTICS")
    print("="*70)
    
    all_dfs = []
    for file in feature_files:
        try:
            df = pd.read_parquet(file)
            all_dfs.append(df)
        except:
            pass
    
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        print(f"\nTotal records: {len(combined_df)}")
        print(f"Unique players: {combined_df['player_id'].nunique()}")
        print(f"Seasons: {sorted(combined_df['season'].unique())}")
        print(f"Weeks covered: {combined_df.groupby(['season', 'week']).size().shape[0]}")
        
        # Check overall data quality
        print("\n📊 Overall Data Quality:")
        print(f"Records with sufficient data: {combined_df['has_sufficient_data'].sum()} ({combined_df['has_sufficient_data'].mean()*100:.1f}%)")
        
        # Check null percentages for key columns
        print("\n🔍 Missing Data Analysis (% null):")
        key_feature_cols = [col for col in combined_df.columns if 'rolling_avg' in col or 'trend' in col or 'rank' in col]
        
        null_stats = {}
        for col in key_feature_cols[:10]:  # Check first 10 feature columns
            null_pct = combined_df[col].isna().mean() * 100
            null_stats[col] = null_pct
        
        for col, pct in sorted(null_stats.items(), key=lambda x: x[1]):
            print(f"  {col}: {pct:.1f}% null")
        
        # Understand why data might be missing
        print("\n🤔 Why might data be missing?")
        print("Checking games_in_history distribution:")
        hist_dist = combined_df['games_in_history'].value_counts().sort_index()
        print(f"  0 games: {hist_dist.get(0, 0)} players")
        print(f"  1-2 games: {combined_df[combined_df['games_in_history'].between(1,2)].shape[0]} players")
        print(f"  3-5 games: {combined_df[combined_df['games_in_history'].between(3,5)].shape[0]} players")
        print(f"  6+ games: {combined_df[combined_df['games_in_history'] >= 6].shape[0]} players")
        
        # Sample some data to see what it looks like
        print("\n" + "="*70)
        print("SAMPLE DATA (Players with 6+ games history)")
        print("="*70)
        
        good_data = combined_df[combined_df['games_in_history'] >= 6].head(5)
        if len(good_data) > 0:
            display_cols = ['player_name', 'position', 'team', 'season', 'week', 
                          'games_in_history', 'rolling_avg_fantasy_ppr']
            
            # Add position-specific columns if they exist
            for col in ['rolling_avg_passing_yds', 'rolling_avg_rushing_yds', 'rolling_avg_receiving_yds']:
                if col in good_data.columns:
                    display_cols.append(col)
            
            print(good_data[display_cols].to_string())
        
        # Check data availability by week
        print("\n" + "="*70)
        print("DATA AVAILABILITY BY SEASON/WEEK")
        print("="*70)
        
        week_stats = combined_df.groupby(['season', 'week']).agg({
            'player_id': 'count',
            'has_sufficient_data': 'sum'
        }).rename(columns={'player_id': 'total_players', 'has_sufficient_data': 'with_data'})
        
        week_stats['pct_with_data'] = week_stats['with_data'] / week_stats['total_players'] * 100
        
        print("\nFirst 10 weeks:")
        print(week_stats.head(10).to_string())
        
        print("\nLast 10 weeks:")
        print(week_stats.tail(10).to_string())

if __name__ == "__main__":
    analyze_features()