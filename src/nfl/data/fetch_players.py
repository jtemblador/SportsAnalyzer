"""
File: src/nfl/data/fetch_players.py

Fetches NFL player reference table from nflreadpy including:
- GSIS and PFR player IDs (for cross-dataset joining)
- Player name, position, team
- Draft info, physical attributes, experience

Filtered to players active since 2018 (our training window).
Single reference file, not per-season.

Stores data in: data/nfl/players/
"""

import nflreadpy as nfl
import pandas as pd
from pathlib import Path

MIN_LAST_SEASON = 2018


class PlayersFetcher:
    """
    Fetches NFL player reference data and stores as a single Parquet file.
    Filtered to players with last_season >= 2018.
    """

    def __init__(self, data_dir='./data/nfl/players'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _file_path(self):
        return self.data_dir / "players.parquet"

    def fetch(self, force=False):
        """
        Fetch player reference data, filtered to our season window.

        Args:
            force: If True, re-fetch even if file exists

        Returns:
            DataFrame with player data, or None if skipped
        """
        filepath = self._file_path()

        if filepath.exists() and not force:
            return None

        print(f"  Fetching player reference table...", end=" ")

        df = nfl.load_players().to_pandas()

        if df.empty:
            print("no data")
            return None

        # Filter to players active in our window
        df = df[df['last_season'] >= MIN_LAST_SEASON].copy()

        df.to_parquet(filepath, index=False)
        print(f"{len(df)} players (last_season >= {MIN_LAST_SEASON})")

        return df

    def fetch_all(self, start_season=2018, end_season=2025):
        """
        Fetch player reference data. Signature matches other fetchers
        for pipeline compatibility, but seasons are ignored since this
        is a single reference table.
        """
        print("Fetching NFL player reference...")
        result = self.fetch()
        if result is None:
            filepath = self._file_path()
            if filepath.exists():
                print(f"  Already exists, skipping")
        print("  Player reference up to date")
        return {}

    def load(self):
        """
        Load previously fetched player reference data.

        Returns:
            DataFrame or None if not fetched yet
        """
        filepath = self._file_path()
        if filepath.exists():
            return pd.read_parquet(filepath)
        return None

    def get_id_mapping(self):
        """
        Get GSIS <-> PFR ID mapping for cross-dataset joins.

        Returns:
            DataFrame with gsis_id, pfr_id, display_name, position columns.
            Only includes players with both IDs.
        """
        df = self.load()
        if df is None:
            return None
        mapped = df[df['gsis_id'].notna() & df['pfr_id'].notna()]
        return mapped[['gsis_id', 'pfr_id', 'display_name', 'position', 'last_season', 'latest_team']].copy()


if __name__ == "__main__":
    fetcher = PlayersFetcher()
    fetcher.fetch()
