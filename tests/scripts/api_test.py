"""
Simple test to verify nflreadpy API is working
Run this: python test_nfl_api.py
"""

import nflreadpy as nfl
import pandas as pd

print("Testing nflreadpy API...")
print("=" * 60)

try:
    # Get current season and week
    print("\n📅 Getting current NFL season and week...")
    current_season = nfl.get_current_season()
    current_week = nfl.get_current_week()
    
    print(f"✓ Current Season: {current_season}")
    print(f"✓ Current Week: {current_week}")
    
    # Fetch player stats for 2024 season (using 2024 since 2025 may not have data yet)
    print(f"\n📥 Fetching player stats for 2024 season, week 1...")
    
    # Load player stats (returns Polars DataFrame)
    player_stats = nfl.load_player_stats([2025])
    
    # Convert to pandas
    player_stats_df = player_stats.to_pandas()
    
    # Filter to week 1
    player_stats_df = player_stats_df[player_stats_df['week'] == 1]
    
    print(f"✅ Successfully fetched {len(player_stats_df):,} player records")
    
    # Show ALL columns that actually came back
    print(f"\n📋 Total columns: {len(player_stats_df.columns)}")
    print("\nAll column names:")
    for i, col in enumerate(player_stats_df.columns, 1):
        print(f"  {i:2d}. {col}")
    
    # Show first few rows
    print("\n📊 First 5 rows of raw data:")
    print(player_stats_df.head())
    
    # Show full details for 2nd player
    print("\n" + "=" * 60)
    print("🔍 DETAILED VIEW: 2nd Player")
    print("=" * 60)
    second_player = player_stats_df.iloc[0]
    for col in player_stats_df.columns:
        print(f"{col:30} : {second_player[col]}")
    
    print("\n✅ API TEST PASSED - nflreadpy is working!")
    print("\n💡 Now we know exactly what columns the API provides")

except Exception as e:
    print(f"\n❌ API TEST FAILED")
    print(f"   Error: {str(e)}")
    print("\n   Possible issues:")
    print("   - No internet connection")
    print("   - nflreadpy not installed correctly")
    print("   - API endpoint is down")
    print("\n   Try installing: pip install nflreadpy")