"""
File: src/nfl/data/fetch_pfr_advstats.py

Fetches NFL PFR Advanced Stats from nflreadpy for 3 stat types:
- Pass: drops, bad throws, pressure rate, times hurried/blitzed/hit
- Rush: yards before/after contact, broken tackles
- Rec: drop rate, broken tackles, passer rating when targeted

Stores data in: data/nfl/pfr_advstats/
"""

import nflreadpy as nfl
import pandas as pd
from pathlib import Path

STAT_TYPES = ['pass', 'rush', 'rec']


class PFRAdvStatsFetcher:
    """
    Fetches NFL PFR Advanced Stats and stores as Parquet files.
    One file per season per stat type.
    """

    def __init__(self, data_dir='./data/nfl/pfr_advstats'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _file_path(self, season, stat_type):
        return self.data_dir / f"pfr_{stat_type}_{season}.parquet"

    def fetch_season(self, season, stat_type, force=False):
        """
        Fetch PFR advanced stats for a single season and stat type.

        Args:
            season: Season year (e.g., 2024)
            stat_type: One of 'pass', 'rush', 'rec'
            force: If True, re-fetch even if file exists

        Returns:
            DataFrame with PFR data, or None if skipped
        """
        filepath = self._file_path(season, stat_type)

        if filepath.exists() and not force:
            return None

        print(f"  Fetching PFR {stat_type} for {season}...", end=" ")

        df = nfl.load_pfr_advstats(seasons=[season], stat_type=stat_type).to_pandas()

        if df.empty:
            print("no data")
            return None

        df.to_parquet(filepath, index=False)
        print(f"{len(df)} records")

        return df

    def fetch_all(self, start_season=2018, end_season=2025):
        """
        Fetch all 3 PFR stat types for a range of seasons.
        Skips files that already exist on disk.

        Args:
            start_season: First season to fetch
            end_season: Last season to fetch

        Returns:
            Dict of {(season, stat_type): row_count} for newly fetched data
        """
        print("Fetching NFL PFR Advanced Stats...")
        fetched = {}

        for stat_type in STAT_TYPES:
            for season in range(start_season, end_season + 1):
                result = self.fetch_season(season, stat_type)
                if result is not None:
                    fetched[(season, stat_type)] = len(result)
                else:
                    filepath = self._file_path(season, stat_type)
                    if filepath.exists():
                        print(f"  {season} {stat_type}: already exists, skipping")

        if fetched:
            total = sum(fetched.values())
            print(f"  Fetched {len(fetched)} files ({total} total records)")
        else:
            print("  All PFR data up to date")

        return fetched

    def load_season(self, season, stat_type):
        """
        Load previously fetched PFR data for a season and stat type.

        Args:
            season: Season year
            stat_type: One of 'pass', 'rush', 'rec'

        Returns:
            DataFrame or None if not fetched yet
        """
        filepath = self._file_path(season, stat_type)
        if filepath.exists():
            return pd.read_parquet(filepath)
        return None


if __name__ == "__main__":
    fetcher = PFRAdvStatsFetcher()
    fetcher.fetch_all()
