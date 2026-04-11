"""
File: src/nfl/data/fetch_player_stats.py

Fetches NFL weekly player stats from nflreadpy including:
- Passing, rushing, receiving, kicking stats
- Fantasy points (standard and PPR)
- EPA, CPOE, target share, air yards share, WOPR, etc.

One file per season (includes regular season + playoffs).

Stores data in: data/nfl/player_stats/
"""

import nflreadpy as nfl
import pandas as pd
from pathlib import Path


class PlayerStatsFetcher:
    """
    Fetches NFL player stats and stores as Parquet files.
    One file per season with all weeks (regular + playoffs).
    """

    def __init__(self, data_dir='./data/nfl/player_stats'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _file_path(self, season):
        return self.data_dir / f"player_stats_{season}.parquet"

    def fetch_season(self, season, force=False):
        """
        Fetch player stats for a single season (all weeks including playoffs).

        Args:
            season: Season year (e.g., 2024)
            force: If True, re-fetch even if file exists

        Returns:
            DataFrame with player stats, or None if skipped
        """
        filepath = self._file_path(season)

        if filepath.exists() and not force:
            return None

        print(f"  Fetching player stats for {season}...", end=" ")

        df = nfl.load_player_stats([season]).to_pandas()

        if df.empty:
            print("no data")
            return None

        df.to_parquet(filepath, index=False)
        print(f"{len(df)} records ({df['week'].max()} weeks)")

        return df

    def fetch_all(self, start_season=2018, end_season=2025):
        """
        Fetch player stats for a range of seasons.
        Skips seasons that already have data on disk.

        Args:
            start_season: First season to fetch
            end_season: Last season to fetch

        Returns:
            Dict of {season: row_count} for newly fetched seasons
        """
        print("Fetching NFL player stats...")
        fetched = {}

        for season in range(start_season, end_season + 1):
            result = self.fetch_season(season)
            if result is not None:
                fetched[season] = len(result)
            else:
                filepath = self._file_path(season)
                if filepath.exists():
                    print(f"  {season}: already exists, skipping")

        if fetched:
            total = sum(fetched.values())
            print(f"  Fetched {len(fetched)} seasons ({total} total records)")
        else:
            print("  All seasons up to date")

        return fetched

    def load_season(self, season):
        """
        Load previously fetched player stats for a season.

        Args:
            season: Season year

        Returns:
            DataFrame or None if not fetched yet
        """
        filepath = self._file_path(season)
        if filepath.exists():
            return pd.read_parquet(filepath)
        return None


if __name__ == "__main__":
    fetcher = PlayerStatsFetcher()
    fetcher.fetch_all()
