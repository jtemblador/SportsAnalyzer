"""
File: src/nfl/data/fetch_team_stats.py

Fetches NFL team-level weekly stats from nflreadpy including:
- Offensive/defensive EPA per week
- Passing, rushing, receiving yards and TDs
- Turnovers, sacks, penalties

Stores data in: data/nfl/team_stats/
"""

import nflreadpy as nfl
import pandas as pd
from pathlib import Path


class TeamStatsFetcher:
    """
    Fetches NFL team stats and stores as Parquet files.
    One file per season with weekly team-level stats.
    """

    def __init__(self, data_dir='./data/nfl/team_stats'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _file_path(self, season):
        return self.data_dir / f"team_stats_{season}.parquet"

    def fetch_season(self, season, force=False):
        """
        Fetch team stats for a single season.

        Args:
            season: Season year (e.g., 2024)
            force: If True, re-fetch even if file exists

        Returns:
            DataFrame with team stats, or None if skipped
        """
        filepath = self._file_path(season)

        if filepath.exists() and not force:
            return None

        print(f"  Fetching team stats for {season}...", end=" ")

        df = nfl.load_team_stats(seasons=[season]).to_pandas()

        if df.empty:
            print("no data")
            return None

        df.to_parquet(filepath, index=False)
        print(f"{len(df)} records")

        return df

    def fetch_all(self, start_season=2018, end_season=2025):
        """
        Fetch team stats for a range of seasons.
        Skips seasons that already have data on disk.

        Args:
            start_season: First season to fetch
            end_season: Last season to fetch

        Returns:
            Dict of {season: row_count} for newly fetched seasons
        """
        print("Fetching NFL team stats...")
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
        Load previously fetched team stats for a season.

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
    fetcher = TeamStatsFetcher()
    fetcher.fetch_all()
