"""
File: src/nfl/data/fetch_ff_opportunity.py

Fetches NFL fantasy opportunity data from nflreadpy including:
- Expected fantasy points per player per week
- Actual vs expected differentials
- Team-level fantasy point shares

Stores data in: data/nfl/ff_opportunity/
"""

import nflreadpy as nfl
import pandas as pd
from pathlib import Path


class FFOpportunityFetcher:
    """
    Fetches NFL fantasy opportunity data and stores it as Parquet files.
    One file per season with weekly expected fantasy points.
    """

    def __init__(self, data_dir='./data/nfl/ff_opportunity'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _file_path(self, season):
        return self.data_dir / f"ff_opportunity_{season}.parquet"

    def fetch_season(self, season, force=False):
        """
        Fetch fantasy opportunity data for a single season.

        Args:
            season: Season year (e.g., 2024)
            force: If True, re-fetch even if file exists

        Returns:
            DataFrame with FF opportunity data, or None if skipped
        """
        filepath = self._file_path(season)

        if filepath.exists() and not force:
            return None

        print(f"  Fetching FF opportunity for {season}...", end=" ")

        df = nfl.load_ff_opportunity(seasons=[season]).to_pandas()

        if df.empty:
            print("no data")
            return None

        df.to_parquet(filepath, index=False)
        print(f"{len(df)} records")

        return df

    def fetch_all(self, start_season=2018, end_season=2025):
        """
        Fetch fantasy opportunity data for a range of seasons.
        Skips seasons that already have data on disk.

        Args:
            start_season: First season to fetch
            end_season: Last season to fetch

        Returns:
            Dict of {season: row_count} for newly fetched seasons
        """
        print("Fetching NFL fantasy opportunity...")
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
        Load previously fetched fantasy opportunity data for a season.

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
    fetcher = FFOpportunityFetcher()
    fetcher.fetch_all()
