"""
File: src/nfl/data/fetch_injuries.py

Fetches NFL weekly injury report data from nflreadpy including:
- Game-day status (Out, Doubtful, Questionable)
- Primary and secondary injuries
- Practice participation (Full, Limited, DNP)

Stores data in: data/nfl/injuries/
"""

import nflreadpy as nfl
import pandas as pd
from pathlib import Path


class InjuryFetcher:
    """
    Fetches NFL injury report data and stores it as Parquet files.
    One file per season with all weekly injury reports.
    """

    def __init__(self, data_dir='./data/nfl/injuries'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _file_path(self, season):
        return self.data_dir / f"injuries_{season}.parquet"

    def fetch_season(self, season, force=False):
        """
        Fetch injury data for a single season.

        Args:
            season: Season year (e.g., 2024)
            force: If True, re-fetch even if file exists

        Returns:
            DataFrame with injury data, or None if skipped
        """
        filepath = self._file_path(season)

        if filepath.exists() and not force:
            return None

        print(f"  Fetching injuries for {season}...", end=" ")

        df = nfl.load_injuries([season]).to_pandas()

        if df.empty:
            print("no data")
            return None

        df.to_parquet(filepath, index=False)
        print(f"{len(df)} reports")

        return df

    def fetch_all(self, start_season=2018, end_season=2025):
        """
        Fetch injury data for a range of seasons.
        Skips seasons that already have data on disk.

        Args:
            start_season: First season to fetch
            end_season: Last season to fetch

        Returns:
            Dict of {season: row_count} for newly fetched seasons
        """
        print("Fetching NFL injuries...")
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
            print(f"  Fetched {len(fetched)} seasons ({total} total reports)")
        else:
            print("  All seasons up to date")

        return fetched

    def load_season(self, season):
        """
        Load previously fetched injury data for a season.

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
    fetcher = InjuryFetcher()
    fetcher.fetch_all()
