"""
File: src/nfl/data/pipeline.py

NFL data fetching and storing pipeline using nflreadpy.
Orchestrates all dataset fetchers and stores data as Parquet files.

Usage:
    python src/nfl/data/pipeline.py           # Fetch all datasets 2018-2025
    python src/nfl/data/pipeline.py --latest  # Fetch current season only
"""

import nflreadpy as nfl

from src.nfl.data.fetch_players import PlayersFetcher
from src.nfl.data.fetch_player_stats import PlayerStatsFetcher
from src.nfl.data.fetch_schedules import ScheduleFetcher
from src.nfl.data.fetch_injuries import InjuryFetcher
from src.nfl.data.fetch_snap_counts import SnapCountFetcher
from src.nfl.data.fetch_nextgen_stats import NextGenStatsFetcher
from src.nfl.data.fetch_ff_opportunity import FFOpportunityFetcher
from src.nfl.data.fetch_pfr_advstats import PFRAdvStatsFetcher
from src.nfl.data.fetch_team_stats import TeamStatsFetcher
from src.nfl.data.fetch_depth_charts import DepthChartFetcher


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

        # Initialize dataset fetchers
        self.players_fetcher = PlayersFetcher(data_dir=f"{self.nfl_dir}/players")
        self.player_stats_fetcher = PlayerStatsFetcher(data_dir=f"{self.nfl_dir}/player_stats")
        self.schedule_fetcher = ScheduleFetcher(data_dir=f"{self.nfl_dir}/schedules")
        self.injury_fetcher = InjuryFetcher(data_dir=f"{self.nfl_dir}/injuries")
        self.snap_count_fetcher = SnapCountFetcher(data_dir=f"{self.nfl_dir}/snap_counts")
        self.nextgen_stats_fetcher = NextGenStatsFetcher(data_dir=f"{self.nfl_dir}/nextgen_stats")
        self.ff_opportunity_fetcher = FFOpportunityFetcher(data_dir=f"{self.nfl_dir}/ff_opportunity")
        self.pfr_advstats_fetcher = PFRAdvStatsFetcher(data_dir=f"{self.nfl_dir}/pfr_advstats")
        self.team_stats_fetcher = TeamStatsFetcher(data_dir=f"{self.nfl_dir}/team_stats")
        self.depth_chart_fetcher = DepthChartFetcher(data_dir=f"{self.nfl_dir}/depth_charts")

        print(f"✓ NFL Pipeline initialized")
        print(f"  Data will be stored in: {self.nfl_dir}")
    
    def get_current_week(self):
        """Get the current NFL week."""
        return nfl.get_current_week()

    def get_current_season(self):
        """Get the current NFL season year."""
        return nfl.get_current_season()

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

        # 0. Player reference table (ID mapping, static)
        self.players_fetcher.fetch_all(start_season, end_season)

        # 1. Player stats (per-season files)
        self.player_stats_fetcher.fetch_all(start_season, end_season)

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

        # 12. Team stats (team-level offensive/defensive EPA, yards, turnovers)
        self.team_stats_fetcher.fetch_all(start_season, end_season)

        # 13. Depth charts (starter/backup status, 2018-2024 only)
        self.depth_chart_fetcher.fetch_all(start_season, end_season)

        print()
        print("=" * 60)
        print("All datasets up to date!")
        print("=" * 60)

    def fetch_latest(self):
        """
        Fetch only the current season across all datasets.
        Use this during a live season to pull the latest week's data.
        """
        season = self.get_current_season()

        print("=" * 60)
        print(f"NFL DATA PIPELINE - Fetch Latest ({season} season)")
        print("=" * 60)
        print()

        # Player reference (re-fetch to pick up new players)
        self.players_fetcher.fetch(force=True)

        # Player stats — current season only (force re-fetch for latest weeks)
        self.player_stats_fetcher.fetch_season(season, force=True)

        # All other datasets — re-fetch current season (force=True for season-level files)
        self.schedule_fetcher.fetch_season(season, force=True)
        self.injury_fetcher.fetch_season(season, force=True)
        self.snap_count_fetcher.fetch_season(season, force=True)
        for stat_type in ['passing', 'rushing', 'receiving']:
            self.nextgen_stats_fetcher.fetch_season(season, stat_type, force=True)
        self.ff_opportunity_fetcher.fetch_season(season, force=True)
        for stat_type in ['pass', 'rush', 'rec']:
            self.pfr_advstats_fetcher.fetch_season(season, stat_type, force=True)
        self.team_stats_fetcher.fetch_season(season, force=True)
        self.depth_chart_fetcher.fetch_season(season)

        print()
        print("=" * 60)
        print(f"{season} season data up to date!")
        print("=" * 60)


# Allow this file to be run independently
if __name__ == "__main__":
    import sys

    if '--latest' in sys.argv:
        pipeline = NFLDataPipeline()
        pipeline.fetch_latest()
    else:
        pipeline = NFLDataPipeline()
        pipeline.fetch_all()