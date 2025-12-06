#!/usr/bin/env python3
"""
File: src/nfl/odds_fetcher.py

Main orchestrator for fetching NFL Vegas lines from multiple sources.
Combines ESPN historical data (2020-2024) with Odds API current data (2025).

Usage:
    python3 src/nfl/odds_fetcher.py

Data cached to: data/nfl/vegas_lines/
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.nfl.v4_espnscraper import ESPNVegasScraper
from src.nfl.odds_api import OddsAPIClient
import pandas as pd
from datetime import datetime


class VegasLinesFetcher:
    """
    Main orchestrator for fetching and managing Vegas lines from multiple sources.
    """

    def __init__(self, cache_dir='./data/nfl/vegas_lines', odds_api_key=None):
        """
        Initialize the Vegas lines fetcher.

        Args:
            cache_dir: Directory to cache all fetched data
            odds_api_key: API key for The Odds API (optional)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.espn_scraper = ESPNVegasScraper(cache_dir=cache_dir)
        self.odds_api_key = odds_api_key

        if odds_api_key:
            self.odds_client = OddsAPIClient(api_key=odds_api_key, cache_dir=cache_dir)
        else:
            self.odds_client = None

        print(f"✓ Vegas Lines Fetcher initialized")
        print(f"  Cache directory: {self.cache_dir}")
        print(f"  Odds API: {'Enabled' if odds_api_key else 'Disabled (ESPN only)'}")

    def fetch_all_lines(self, historical_seasons=None, fetch_current=True, force_refresh=False):
        """
        Fetch all Vegas lines (historical + current).

        Args:
            historical_seasons: List of seasons to fetch (default: 2020-2024)
            fetch_current: Whether to fetch current 2025 lines
            force_refresh: If True, re-fetch even if cached

        Returns:
            DataFrame with all Vegas lines
        """
        if historical_seasons is None:
            historical_seasons = [2020, 2021, 2022, 2023, 2024]

        print(f"\n{'='*70}")
        print(f"VEGAS LINES FETCHER - FULL SYNC")
        print(f"{'='*70}")
        print(f"Historical seasons: {historical_seasons}")
        print(f"Fetch current (2025): {fetch_current}")
        print(f"Force refresh: {force_refresh}")
        print(f"{'='*70}\n")

        all_dataframes = []

        # Step 1: Fetch historical lines from ESPN
        if historical_seasons:
            print(f"STEP 1: Fetching Historical Lines (ESPN)")
            print(f"{'='*70}")

            historical_df = self.espn_scraper.fetch_historical_lines(
                seasons=historical_seasons,
                force_refresh=force_refresh
            )

            if not historical_df.empty:
                all_dataframes.append(historical_df)
                print(f"✅ Historical: {len(historical_df)} games")

        # Step 2: Fetch current lines from Odds API
        if fetch_current and self.odds_client:
            print(f"\nSTEP 2: Fetching Current Lines (Odds API)")
            print(f"{'='*70}")

            current_df = self.odds_client.fetch_current_lines(
                force_refresh=force_refresh
            )

            if not current_df.empty:
                all_dataframes.append(current_df)
                print(f"✅ Current: {len(current_df)} games")
        elif fetch_current and not self.odds_client:
            print(f"\n⚠ Odds API not enabled (no API key provided)")

        # Combine all data
        if all_dataframes:
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['game_id'])
            combined_df = combined_df.sort_values(['season', 'week'])

            print(f"\n{'='*70}")
            print(f"COMBINED RESULTS")
            print(f"{'='*70}")
            print(f"Total games: {len(combined_df)}")
            print(f"Seasons: {sorted(combined_df['season'].unique())}")
            print(f"Date range: {combined_df['game_date'].min()} to {combined_df['game_date'].max()}")
            print(f"{'='*70}\n")

            return combined_df
        else:
            print(f"\n⚠ No data fetched")
            return pd.DataFrame()

    def get_game_lines(self, team, season, week):
        """
        Get Vegas lines for a specific team/game.

        Args:
            team: Team abbreviation (e.g., 'KC', 'SF')
            season: Season year
            week: NFL week

        Returns:
            Dictionary with game lines, or None
        """
        # Load cached data
        historical_df = self.espn_scraper.load_cached_lines()
        current_df = self.odds_client.load_cached_lines() if self.odds_client else pd.DataFrame()

        # Combine
        all_lines = pd.concat([historical_df, current_df], ignore_index=True)

        # Filter for specific game
        game = all_lines[
            ((all_lines['home_team'] == team) | (all_lines['away_team'] == team)) &
            (all_lines['season'] == season) &
            (all_lines['week'] == week)
        ]

        if game.empty:
            return None

        game_row = game.iloc[0]

        # Determine if team is home or away
        is_home = game_row['home_team'] == team

        return {
            'team': team,
            'opponent': game_row['away_team'] if is_home else game_row['home_team'],
            'is_home': is_home,
            'spread': game_row['spread'] if is_home else -game_row['spread'],
            'team_implied_total': game_row['home_implied_total'] if is_home else game_row['away_implied_total'],
            'opponent_implied_total': game_row['away_implied_total'] if is_home else game_row['home_implied_total'],
            'over_under': game_row['over_under'],
            'game_date': game_row['game_date']
        }

    def load_all_lines(self):
        """
        Load all cached Vegas lines.

        Returns:
            DataFrame with all cached lines
        """
        historical_df = self.espn_scraper.load_cached_lines()
        current_df = self.odds_client.load_cached_lines() if self.odds_client else pd.DataFrame()

        all_lines = pd.concat([historical_df, current_df], ignore_index=True)
        all_lines = all_lines.drop_duplicates(subset=['game_id'])
        all_lines = all_lines.sort_values(['season', 'week'])

        return all_lines

    def get_summary(self):
        """
        Get summary of cached Vegas lines.
        """
        all_lines = self.load_all_lines()

        if all_lines.empty:
            print("⚠ No cached Vegas lines found")
            return

        print(f"\n{'='*70}")
        print(f"VEGAS LINES CACHE SUMMARY")
        print(f"{'='*70}")
        print(f"Total games: {len(all_lines)}")
        print(f"Seasons: {sorted(all_lines['season'].unique())}")
        print(f"Date range: {all_lines['game_date'].min()} to {all_lines['game_date'].max()}")
        print(f"\nGames per season:")

        for season in sorted(all_lines['season'].unique()):
            season_games = len(all_lines[all_lines['season'] == season])
            weeks = sorted(all_lines[all_lines['season'] == season]['week'].unique())
            print(f"  {season}: {season_games} games (weeks {min(weeks)}-{max(weeks)})")

        print(f"{'='*70}\n")


def main():
    """Main entry point for fetching Vegas lines"""
    print(f"\n{'='*70}")
    print(f"NFL VEGAS LINES FETCHER")
    print(f"{'='*70}")
    print(f"Started at: {datetime.now()}")
    print(f"{'='*70}\n")

    # Odds API key (from user)
    ODDS_API_KEY = 'c6d41f99d9fdabfa5f5abaf8df1c9084'

    # Initialize fetcher
    fetcher = VegasLinesFetcher(
        cache_dir='./data/nfl/vegas_lines',
        odds_api_key=ODDS_API_KEY
    )

    # Options
    print("What would you like to do?")
    print("1. Fetch ALL lines (historical 2020-2024 + current 2025)")
    print("2. Fetch historical only (2020-2024)")
    print("3. Fetch current only (2025)")
    print("4. Show cache summary")
    print("5. Test: Get specific game lines")

    choice = input("\nChoice (1/2/3/4/5): ").strip()

    if choice == '1':
        # Fetch everything
        df = fetcher.fetch_all_lines(
            historical_seasons=[2020, 2021, 2022, 2023, 2024],
            fetch_current=True,
            force_refresh=False  # Use cache if available
        )

        if not df.empty:
            print("\nSample data:")
            print(df.head(10))

    elif choice == '2':
        # Historical only
        df = fetcher.fetch_all_lines(
            historical_seasons=[2020, 2021, 2022, 2023, 2024],
            fetch_current=False,
            force_refresh=False
        )

    elif choice == '3':
        # Current only
        df = fetcher.fetch_all_lines(
            historical_seasons=[],
            fetch_current=True,
            force_refresh=False
        )

    elif choice == '4':
        # Show summary
        fetcher.get_summary()

    elif choice == '5':
        # Test specific game lookup
        team = input("Team abbreviation (e.g., KC): ").strip().upper()
        season = int(input("Season (e.g., 2025): "))
        week = int(input("Week (e.g., 1): "))

        lines = fetcher.get_game_lines(team, season, week)

        if lines:
            print(f"\n{'='*60}")
            print(f"Game Lines for {team} ({season} Week {week})")
            print(f"{'='*60}")
            print(f"Opponent: {lines['opponent']}")
            print(f"Location: {'Home' if lines['is_home'] else 'Away'}")
            print(f"Spread: {lines['spread']:+.1f}")
            print(f"Team Implied Total: {lines['team_implied_total']} pts")
            print(f"Opponent Implied Total: {lines['opponent_implied_total']} pts")
            print(f"Over/Under: {lines['over_under']} pts")
            print(f"Game Date: {lines['game_date']}")
            print(f"{'='*60}\n")
        else:
            print(f"\n⚠ No lines found for {team} in {season} Week {week}")

    else:
        print("❌ Invalid choice")

    print(f"\n{'='*70}")
    print(f"Done at: {datetime.now()}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
