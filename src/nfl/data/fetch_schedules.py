"""
File: src/nfl/data/fetch_schedules.py

Fetches NFL schedule data from nflreadpy including:
- Game results (scores, overtime)
- Vegas lines (spread, total, moneylines)
- Weather (temp, wind, roof, surface)
- Rest days (home_rest, away_rest)
- Game context (stadium, coaches, referee, division game)

Stores data in: data/nfl/schedules/
"""

import nflreadpy as nfl
import pandas as pd
from pathlib import Path


class ScheduleFetcher:
    """
    Fetches NFL schedule data and stores it as Parquet files.
    One file per season with all games.
    """

    def __init__(self, data_dir='./data/nfl/schedules'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _file_path(self, season):
        return self.data_dir / f"schedules_{season}.parquet"

    def fetch_season(self, season, force=False):
        """
        Fetch schedule data for a single season.

        Args:
            season: Season year (e.g., 2024)
            force: If True, re-fetch even if file exists

        Returns:
            DataFrame with schedule data, or None if skipped
        """
        filepath = self._file_path(season)

        if filepath.exists() and not force:
            return None

        print(f"  Fetching schedules for {season}...", end=" ")

        df = nfl.load_schedules([season]).to_pandas()

        if df.empty:
            print("no data")
            return None

        # Calculate implied totals from spread and total line
        # spread_line convention: negative = home team favored
        # home_implied = (total + spread) / 2  — favored team gets more
        # away_implied = (total - spread) / 2
        df['home_implied_total'] = (df['total_line'] + df['spread_line']) / 2
        df['away_implied_total'] = (df['total_line'] - df['spread_line']) / 2

        df.to_parquet(filepath, index=False)
        print(f"{len(df)} games")

        return df

    def fetch_all(self, start_season=2018, end_season=2025):
        """
        Fetch schedule data for a range of seasons.
        Skips seasons that already have data on disk.

        Args:
            start_season: First season to fetch
            end_season: Last season to fetch

        Returns:
            Dict of {season: row_count} for newly fetched seasons
        """
        print("Fetching NFL schedules...")
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
            print(f"  Fetched {len(fetched)} seasons ({total} total games)")
        else:
            print("  All seasons up to date")

        return fetched

    def load_season(self, season):
        """
        Load previously fetched schedule data for a season.

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
    fetcher = ScheduleFetcher()
    fetcher.fetch_all()
