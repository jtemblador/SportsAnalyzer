"""
File: src/nfl/data/fetch_depth_charts.py

Fetches NFL weekly depth chart data from nflreadpy including:
- Starter/backup/third-string designation (depth_team: 1/2/3)
- Formation (Offense/Defense/Special Teams)
- Depth position per player per week

Note: 2025 has an incompatible schema (different columns, no week/depth_team).
This fetcher only supports 2018-2024.

Stores data in: data/nfl/depth_charts/
"""

import nflreadpy as nfl
import pandas as pd
from pathlib import Path

# 2025 has a completely different schema — skip it
MAX_SUPPORTED_SEASON = 2024


class DepthChartFetcher:
    """
    Fetches NFL depth chart data and stores as Parquet files.
    One file per season with weekly depth chart positions.
    Supports 2018-2024 only (2025 schema is incompatible).
    """

    def __init__(self, data_dir='./data/nfl/depth_charts'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _file_path(self, season):
        return self.data_dir / f"depth_charts_{season}.parquet"

    def fetch_season(self, season, force=False):
        """
        Fetch depth chart data for a single season.

        Args:
            season: Season year (e.g., 2024)
            force: If True, re-fetch even if file exists

        Returns:
            DataFrame with depth chart data, or None if skipped
        """
        if season > MAX_SUPPORTED_SEASON:
            print(f"  {season}: skipping (schema incompatible after {MAX_SUPPORTED_SEASON})")
            return None

        filepath = self._file_path(season)

        if filepath.exists() and not force:
            return None

        print(f"  Fetching depth charts for {season}...", end=" ")

        df = nfl.load_depth_charts(seasons=[season]).to_pandas()

        if df.empty:
            print("no data")
            return None

        df.to_parquet(filepath, index=False)
        print(f"{len(df)} records")

        return df

    def fetch_all(self, start_season=2018, end_season=2025):
        """
        Fetch depth chart data for a range of seasons.
        Skips seasons that already exist and seasons after 2024.

        Args:
            start_season: First season to fetch
            end_season: Last season to fetch (capped at 2024)

        Returns:
            Dict of {season: row_count} for newly fetched seasons
        """
        print("Fetching NFL depth charts...")
        capped_end = min(end_season, MAX_SUPPORTED_SEASON)
        fetched = {}

        for season in range(start_season, capped_end + 1):
            result = self.fetch_season(season)
            if result is not None:
                fetched[season] = len(result)
            else:
                filepath = self._file_path(season)
                if filepath.exists():
                    print(f"  {season}: already exists, skipping")

        if end_season > MAX_SUPPORTED_SEASON:
            print(f"  Note: {MAX_SUPPORTED_SEASON + 1}+ skipped (incompatible schema)")

        if fetched:
            total = sum(fetched.values())
            print(f"  Fetched {len(fetched)} seasons ({total} total records)")
        else:
            print("  All seasons up to date")

        return fetched

    def load_season(self, season):
        """
        Load previously fetched depth chart data for a season.

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
    fetcher = DepthChartFetcher()
    fetcher.fetch_all()
