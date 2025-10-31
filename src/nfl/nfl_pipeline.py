"""
File: src/nfl/nfl_pipeline.py

NFL data fetching and storing pipeline using nflreadpy
Stores data in: data/nfl/raw/
No cleaning - stores data exactly as received from API
"""

import nflreadpy as nfl
import pandas as pd
from pathlib import Path
from datetime import datetime


class NFLDataPipeline:
    """
    Fetches NFL data using nflreadpy and stores it locally.
    Data is stored as-is with no modifications.
    """
    
    def __init__(self, base_data_dir: str = "./data"):
        """
        Initialize the NFL pipeline.
        
        Args:
            base_data_dir: Base directory for all sports data (default: ./data)
        """
        # Set up paths for NFL data storage
        self.nfl_dir = f"{base_data_dir}/nfl"
        self.raw_dir = f"{self.nfl_dir}/raw"
        
        # Create directories if they don't exist
        Path(self.raw_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"✓ NFL Pipeline initialized")
        print(f"  Data will be stored in: {self.raw_dir}")
    
    def get_current_week(self):
        """
        Get the current NFL week.
        
        Returns:
            int: Current week number
        """
        current_week = nfl.get_current_week()
        return current_week
    
    def get_current_season(self):
        """
        Get the current NFL season year.
        
        Returns:
            int: Current season year
        """
        current_season = nfl.get_current_season()
        return current_season
    
    def get_last_downloaded_week(self):
        """
        Find the most recent season/week that has been downloaded.
        
        Returns:
            tuple: (season, week) of the last downloaded data, or (2024, 18) if none found
        """
        # Get all parquet files in raw directory
        parquet_files = list(Path(self.raw_dir).glob("player_stats_*.parquet"))
        
        if not parquet_files:
            # No files exist, start from end of 2024 season
            return (2024, 18)
        
        # Parse filenames to extract season and week
        max_season = 0
        max_week = 0
        
        for file in parquet_files:
            # Extract season and week from filename: player_stats_2024_week_1.parquet
            parts = file.stem.split('_')
            if len(parts) >= 4:
                try:
                    season = int(parts[2])
                    week = int(parts[4])
                    
                    # Find the latest season/week combination
                    if season > max_season or (season == max_season and week > max_week):
                        max_season = season
                        max_week = week
                except ValueError:
                    continue
        
        return (max_season, max_week)
    
    def check_file_exists(self, season, week):
        """
        Check if data for a specific season/week already exists.
        
        Args:
            season: Season year
            week: Week number
        
        Returns:
            bool: True if file exists, False otherwise
        """
        # Create expected filename
        filename = f"player_stats_{season}_week_{week}.parquet"
        filepath = f"{self.raw_dir}/{filename}"
        
        # Check if file exists
        return Path(filepath).exists()
        """
        Check if data for a specific season/week already exists.
        
        Args:
            season: Season year
            week: Week number
        
        Returns:
            bool: True if file exists, False otherwise
        """
        # Create expected filename
        filename = f"player_stats_{season}_week_{week}.parquet"
        filepath = f"{self.raw_dir}/{filename}"
        
        # Check if file exists
        return Path(filepath).exists()
    
    def fetch_player_stats(self, seasons, week=None):
        """
        Fetch player stats from nflreadpy.
        
        Args:
            seasons: List of seasons to fetch (e.g., [2024, 2025])
            week: Optional specific week number
        
        Returns:
            Pandas DataFrame with player stats
        """
        # Load player stats (this returns a Polars DataFrame)
        player_stats = nfl.load_player_stats(seasons)
        
        # Convert to pandas for easier handling
        player_stats_df = player_stats.to_pandas()
        
        # Filter by week if specified
        if week is not None:
            player_stats_df = player_stats_df[player_stats_df['week'] == week]
        
        return player_stats_df
    
    def save_data(self, data, season, week):
        """
        Save data to parquet file with consistent naming.
        
        Args:
            data: Pandas DataFrame to save
            season: Season year
            week: Week number
        
        Returns:
            Path to saved file
        """
        # Create filename following the naming convention
        filename = f"player_stats_{season}_week_{week}.parquet"
        filepath = f"{self.raw_dir}/{filename}"
        
        # Save as parquet (compressed format)
        data.to_parquet(filepath, index=False)
        
        print(f"💾 Saved data to: {filepath}")
        
        return filepath
    
    def print_data_summary(self, data, data_type="Player Stats"):
        """
        Print summary statistics about the fetched data.
        
        Args:
            data: Pandas DataFrame
            data_type: Type of data (for display)
        """
        print(f"\n" + "=" * 60)
        print(f"📊 {data_type} Summary")
        print("=" * 60)
        
        # Total records
        print(f"Total Records: {len(data):,}")
        
        # Total columns
        print(f"Total Columns: {len(data.columns)}")
        
        # Unique players (if player_name column exists)
        if 'player_name' in data.columns:
            unique_players = data['player_name'].nunique()
            print(f"Unique Players: {unique_players:,}")
        
        # Positions breakdown (if position column exists)
        if 'position' in data.columns and len(data) > 0:
            print(f"\nPositions:")
            position_counts = data['position'].value_counts()
            for position, count in position_counts.head(10).items():
                print(f"  {position}: {count}")
        
        # Teams breakdown (if recent_team column exists)
        if 'recent_team' in data.columns and len(data) > 0:
            unique_teams = data['recent_team'].nunique()
            print(f"\nUnique Teams: {unique_teams}")
        
        # Memory usage
        memory_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
        print(f"\nMemory Usage: {memory_mb:.2f} MB")
        
        print("=" * 60)
    
    def run_pipeline(self, season, week, silent_check=False):
        """
        Main pipeline: Fetch data and store it for a specific season/week.
        
        Args:
            season: Season year
            week: Week number
            silent_check: If True, suppresses "already exists" messages
        
        Returns:
            Pandas DataFrame with the fetched data, or None if already exists/no data
        """
        # Check if data already exists
        if self.check_file_exists(season, week):
            if not silent_check:
                print(f"⏭️  Data for Season {season}, Week {week} already exists. Skipping...")
            return None
        
        # Only print fetch message if we're actually going to try fetching
        print(f"📥 Fetching Season {season}, Week {week}...")
        
        # Fetch player stats
        player_stats = self.fetch_player_stats(seasons=[season], week=week)
        
        # Check if we got any data
        if len(player_stats) == 0:
            print(f"⚠️  No data available - week hasn't started yet")
            return None
        
        # Print summary
        self.print_data_summary(player_stats, f"Week {week}, Season {season}")
        
        # Save data with consistent naming
        filepath = self.save_data(player_stats, season, week)
        
        print(f"✅ Saved {len(player_stats):,} records")
        
        return player_stats


# Allow this file to be run independently
if __name__ == "__main__":
    print("=" * 60)
    print("NFL DATA FETCHER - Incremental Update")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = NFLDataPipeline()
    
    # Find the most recent season/week we have in our database
    last_season, last_week = pipeline.get_last_downloaded_week()
    
    print(f"\n📂 Last downloaded data: Season {last_season}, Week {last_week}")
    print(f"🔄 Checking for new data...")
    print()
    
    # Track how many new files we download
    total_fetched = 0
    
    # Calculate the next week to check (week after last downloaded)
    current_season = last_season
    current_week = last_week + 1
    
    # If we've gone past week 18, move to next season's week 1
    if current_week > 18:
        current_season += 1
        current_week = 1
    
    # Keep fetching new weeks until we hit a week with no data
    weeks_per_season = 18
    found_data = True  # Flag to track if we should keep checking
    
    # Loop until we find a week with no data or reach 2026
    while found_data and current_season <= 2025:
        # Silently check if this week's file already exists
        if pipeline.check_file_exists(current_season, current_week):
            # File exists, skip to next week (no output)
            current_week += 1
            if current_week > weeks_per_season:
                current_season += 1
                current_week = 1
            continue
        
        try:
            # Try to fetch data for this season/week
            # silent_check=True prevents duplicate "already exists" messages
            result = pipeline.run_pipeline(season=current_season, week=current_week, silent_check=True)
            
            # Check if we got data back
            if result is None:
                # No data available - this week hasn't started yet
                # Stop checking further weeks
                print(f"   → Week hasn't started yet. Stopping here.")
                found_data = False
            else:
                # Successfully fetched and saved data
                total_fetched += 1
                
                # Move to next week
                current_week += 1
                if current_week > weeks_per_season:
                    current_season += 1
                    current_week = 1
                
        except Exception as e:
            # If there's an error, stop checking
            print(f"⚠️  Error: {str(e)}")
            found_data = False
    
    # Print final summary
    print("\n" + "=" * 60)
    print("✅ UPDATE COMPLETE!")
    print("=" * 60)
    
    if total_fetched > 0:
        print(f"New files downloaded: {total_fetched}")
        print(f"Data stored in: {pipeline.raw_dir}")
    else:
        print(f"No new data available - you're up to date!")
        print(f"Last data: Season {last_season}, Week {last_week}")
    
    print("=" * 60)