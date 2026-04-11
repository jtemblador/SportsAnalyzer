"""
File: src/nfl/data/pipeline.py

NFL data fetching and storing pipeline using nflreadpy.
Orchestrates all dataset fetchers and stores data as Parquet files.
"""

import nflreadpy as nfl
from pathlib import Path

from src.nfl.data.fetch_schedules import ScheduleFetcher
from src.nfl.data.fetch_injuries import InjuryFetcher
from src.nfl.data.fetch_snap_counts import SnapCountFetcher
from src.nfl.data.fetch_nextgen_stats import NextGenStatsFetcher
from src.nfl.data.fetch_ff_opportunity import FFOpportunityFetcher
from src.nfl.data.fetch_pfr_advstats import PFRAdvStatsFetcher


class NFLDataPipeline:
    """
    Fetches NFL data using nflreadpy and stores it locally as Parquet files.
    Orchestrates all dataset fetchers via fetch_all().
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

        # Initialize dataset fetchers
        self.schedule_fetcher = ScheduleFetcher(data_dir=f"{self.nfl_dir}/schedules")
        self.injury_fetcher = InjuryFetcher(data_dir=f"{self.nfl_dir}/injuries")
        self.snap_count_fetcher = SnapCountFetcher(data_dir=f"{self.nfl_dir}/snap_counts")
        self.nextgen_stats_fetcher = NextGenStatsFetcher(data_dir=f"{self.nfl_dir}/nextgen_stats")
        self.ff_opportunity_fetcher = FFOpportunityFetcher(data_dir=f"{self.nfl_dir}/ff_opportunity")
        self.pfr_advstats_fetcher = PFRAdvStatsFetcher(data_dir=f"{self.nfl_dir}/pfr_advstats")

        print(f"✓ NFL Pipeline initialized")
        print(f"  Data will be stored in: {self.nfl_dir}")
    
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
        self.save_data(player_stats, season, week)

        print(f"✅ Saved {len(player_stats):,} records")
        
        return player_stats

    def fetch_all_player_stats(self, start_season=2018, end_season=None):
        """
        Fetch player stats for all seasons/weeks, skipping existing files.

        Args:
            start_season: First season to fetch
            end_season: Last season to fetch (default: current season)
        """
        if end_season is None:
            end_season = self.get_current_season()

        print("\nFetching player stats...")
        total_fetched = 0

        for season in range(start_season, end_season + 1):
            max_week = 18
            for week in range(1, max_week + 1):
                if self.check_file_exists(season, week):
                    continue
                result = self.run_pipeline(season=season, week=week, silent_check=True)
                if result is not None:
                    total_fetched += 1
                elif season == end_season:
                    # Current season, week hasn't happened yet — stop this season
                    break

        if total_fetched > 0:
            print(f"  Fetched {total_fetched} new weeks of player stats")
        else:
            print("  Player stats up to date")

    def fetch_all(self, start_season=2018, end_season=None):
        """
        Fetch ALL datasets for the given season range.
        Each fetcher skips data that already exists on disk.

        Args:
            start_season: First season to fetch (default: 2018 for warm-up data)
            end_season: Last season to fetch (default: current season)
        """
        if end_season is None:
            end_season = self.get_current_season()

        print("=" * 60)
        print("NFL DATA PIPELINE - Fetch All Datasets")
        print("=" * 60)
        print(f"Seasons: {start_season}-{end_season}")
        print()

        # 1. Player stats (existing, per-week files)
        self.fetch_all_player_stats(start_season, end_season)

        # 2. Schedules (Vegas lines, weather, game context)
        self.schedule_fetcher.fetch_all(start_season, end_season)

        # 3. Injuries (weekly injury reports, practice status)
        self.injury_fetcher.fetch_all(start_season, end_season)

        # 4. Snap counts (offense/defense/ST snap percentages)
        self.snap_count_fetcher.fetch_all(start_season, end_season)

        # 5-7. Next Gen Stats (passing, rushing, receiving)
        self.nextgen_stats_fetcher.fetch_all(start_season, end_season)

        # 8. Fantasy opportunity (expected fantasy points)
        self.ff_opportunity_fetcher.fetch_all(start_season, end_season)

        # 9-11. PFR Advanced Stats (pass, rush, rec)
        self.pfr_advstats_fetcher.fetch_all(start_season, end_season)

        # Future fetchers will be added here as tasks are completed:
        # 12. Team stats (Task 0.11)
        # 13. Depth charts (Task 0.12)

        print()
        print("=" * 60)
        print("All datasets up to date!")
        print("=" * 60)


# Allow this file to be run independently
if __name__ == "__main__":
    pipeline = NFLDataPipeline()
    pipeline.fetch_all()