"""
File: src/nfl/data/fetch_nextgen_stats.py

Fetches NFL Next Gen Stats from nflreadpy for all 3 stat types:
- Passing: time to throw, air yards, aggressiveness, CPOE
- Rushing: efficiency, yards over expected, time to LOS
- Receiving: separation, cushion, YAC above expectation

Week 0 rows are season aggregates. Weekly data is weeks 1+.

Stores data in: data/nfl/nextgen_stats/
"""

import nflreadpy as nfl
import pandas as pd
from pathlib import Path

STAT_TYPES = ['passing', 'rushing', 'receiving']


class NextGenStatsFetcher:
    """
    Fetches NFL Next Gen Stats and stores as Parquet files.
    One file per season per stat type.
    """

    def __init__(self, data_dir='./data/nfl/nextgen_stats'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _file_path(self, season, stat_type):
        return self.data_dir / f"ngs_{stat_type}_{season}.parquet"

    def fetch_season(self, season, stat_type, force=False):
        """
        Fetch Next Gen Stats for a single season and stat type.

        Args:
            season: Season year (e.g., 2024)
            stat_type: One of 'passing', 'rushing', 'receiving'
            force: If True, re-fetch even if file exists

        Returns:
            DataFrame with NGS data, or None if skipped
        """
        filepath = self._file_path(season, stat_type)

        if filepath.exists() and not force:
            return None

        print(f"  Fetching NGS {stat_type} for {season}...", end=" ")

        df = nfl.load_nextgen_stats(seasons=[season], stat_type=stat_type).to_pandas()

        if df.empty:
            print("no data")
            return None

        df.to_parquet(filepath, index=False)
        print(f"{len(df)} records")

        return df

    def fetch_all(self, start_season=2018, end_season=2025):
        """
        Fetch all 3 NGS stat types for a range of seasons.
        Skips files that already exist on disk.

        Args:
            start_season: First season to fetch
            end_season: Last season to fetch

        Returns:
            Dict of {(season, stat_type): row_count} for newly fetched data
        """
        print("Fetching NFL Next Gen Stats...")
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
            print("  All NGS data up to date")

        return fetched

    def load_season(self, season, stat_type):
        """
        Load previously fetched NGS data for a season and stat type.

        Args:
            season: Season year
            stat_type: One of 'passing', 'rushing', 'receiving'

        Returns:
            DataFrame or None if not fetched yet
        """
        filepath = self._file_path(season, stat_type)
        if filepath.exists():
            return pd.read_parquet(filepath)
        return None


if __name__ == "__main__":
    fetcher = NextGenStatsFetcher()
    fetcher.fetch_all()
