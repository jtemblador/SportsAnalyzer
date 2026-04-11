"""
File: src/nfl/data/fetch_snap_counts.py

Fetches NFL weekly snap count data from nflreadpy including:
- Offense snap count and percentage per player
- Defense snap count and percentage per player
- Special teams snap count and percentage per player

Stores data in: data/nfl/snap_counts/
"""

import nflreadpy as nfl
import pandas as pd
from pathlib import Path


class SnapCountFetcher:
    """
    Fetches NFL snap count data and stores it as Parquet files.
    One file per season with all weekly snap counts.
    """

    def __init__(self, data_dir='./data/nfl/snap_counts'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _file_path(self, season):
        return self.data_dir / f"snap_counts_{season}.parquet"

    def fetch_season(self, season, force=False):
        """
        Fetch snap count data for a single season.

        Args:
            season: Season year (e.g., 2024)
            force: If True, re-fetch even if file exists

        Returns:
            DataFrame with snap count data, or None if skipped
        """
        filepath = self._file_path(season)

        if filepath.exists() and not force:
            return None

        print(f"  Fetching snap counts for {season}...", end=" ")

        df = nfl.load_snap_counts(seasons=[season]).to_pandas()

        if df.empty:
            print("no data")
            return None

        df.to_parquet(filepath, index=False)
        print(f"{len(df)} records")

        return df

    def fetch_all(self, start_season=2018, end_season=2025):
        """
        Fetch snap count data for a range of seasons.
        Skips seasons that already have data on disk.

        Args:
            start_season: First season to fetch
            end_season: Last season to fetch

        Returns:
            Dict of {season: row_count} for newly fetched seasons
        """
        print("Fetching NFL snap counts...")
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
        Load previously fetched snap count data for a season.

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
    fetcher = SnapCountFetcher()
    fetcher.fetch_all()
