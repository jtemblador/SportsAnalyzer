#!/usr/bin/env python3
"""
File: src/nfl/v4_odds_fetcher.py

Main orchestrator for fetching NFL Vegas lines from multiple sources.
Combines ESPN historical data (2020-2024) with Odds API current data (2025).

Usage:
    python3 src/nfl/v4_odds_fetcher.py

Data cached to: data/nfl/vegas_lines/
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.nfl.v4_espnscraper import ESPNVegasScraper
from src.nfl.v4_odds_api import OddsAPIClient
import pandas as pd
from datetime import datetime


class VegasLinesFetcher:
    """
    Main orchestrator for fetching and managing Vegas lines from multiple sources.
    """

    def __init__(self, cache_dir='./data/nfl/vegas_odds', odds_api_key=None):
        """
        Initialize the Vegas lines fetcher.

        Args:
            cache_dir: Directory to cache all fetched data
            odds_api_key: API key for The Odds API (optional)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Note: ESPN scraper deprecated (doesn't have historical betting lines)
        # self.espn_scraper = ESPNVegasScraper(cache_dir=cache_dir)
        self.odds_api_key = odds_api_key

        if odds_api_key:
            self.odds_client = OddsAPIClient(api_key=odds_api_key, cache_dir=cache_dir)
        else:
            self.odds_client = None

        print(f"✓ Vegas Odds Fetcher initialized")
        print(f"  Cache directory: {self.cache_dir}")
        print(f"  Odds API: {'Enabled' if odds_api_key else 'Disabled'}")

    def fetch_team_lines(self, force_refresh=False):
        """
        Fetch current team lines (spread, totals) from Odds API.

        Args:
            force_refresh: If True, re-fetch even if cached

        Returns:
            DataFrame with team lines
        """
        if not self.odds_client:
            print("❌ Odds API not enabled (no API key provided)")
            return pd.DataFrame()

        return self.odds_client.fetch_current_lines(force_refresh=force_refresh)

    def fetch_player_props(self, player_names, week, force_refresh=False):
        """
        Fetch player props for specific players only (smart caching).

        This is THE method to use for your workflow:
        1. Run predictions
        2. Get top 20-30 players
        3. Call this method to get their Vegas props
        4. Compare your predictions vs Vegas

        Args:
            player_names: List of player names (e.g., ['Patrick Mahomes', 'Christian McCaffrey'])
            week: NFL week number
            force_refresh: If True, re-fetch even if cached

        Returns:
            DataFrame with player props for requested players
        """
        if not self.odds_client:
            print("❌ Odds API not enabled (no API key provided)")
            return pd.DataFrame()

        return self.odds_client.fetch_props_for_specific_players(
            player_names=player_names,
            week=week,
            force_refresh=force_refresh
        )

    def load_cached_team_lines(self, week=None):
        """Load cached team lines."""
        if not self.odds_client:
            print("❌ Odds API not enabled")
            return pd.DataFrame()
        return self.odds_client.load_cached_lines(week=week)

    def load_cached_player_props(self, week=None):
        """Load cached player props."""
        if not self.odds_client:
            print("❌ Odds API not enabled")
            return pd.DataFrame()
        return self.odds_client.load_cached_player_props(week=week)

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
        all_lines = self.odds_client.load_cached_lines() if self.odds_client else pd.DataFrame()

        if all_lines.empty:
            return None

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
        Load all cached team lines.

        Returns:
            DataFrame with all cached lines
        """
        if not self.odds_client:
            return pd.DataFrame()

        all_lines = self.odds_client.load_cached_lines()
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


def test_specific_players():
    """Test fetching props for specific players (your use case)."""
    API_KEY = 'c6d41f99d9fdabfa5f5abaf8df1c9084'

    fetcher = VegasLinesFetcher(odds_api_key=API_KEY)

    # Simulate your workflow: get top predictions (players actually playing in week 14)
    top_players = [
        'Patrick Mahomes',
        'Josh Allen',
        'Travis Kelce',
        'Sam Darnold',
        'Lamar Jackson'
    ]

    week = 14

    print("\n" + "="*70)
    print("TESTING: Fetch props for specific players only")
    print("="*70)
    print(f"Players: {top_players}")
    print(f"Week: {week}")

    # This should use cache if available, only fetch missing players
    props_df = fetcher.fetch_player_props(
        player_names=top_players,
        week=week,
        force_refresh=False
    )

    if not props_df.empty:
        print("\n" + "="*70)
        print("RESULTS:")
        print("="*70)
        for player in top_players:
            player_props = props_df[props_df['player_name'] == player]
            if not player_props.empty:
                print(f"\n{player}:")
                for _, prop in player_props.iterrows():
                    print(f"  {prop['prop_type']}: {prop['prop_value']}")
            else:
                print(f"\n{player}: No props found")

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
    print("1. Fetch team lines only (spreads, totals)")
    print("2. Fetch props for specific players (your use case)")
    print("3. Show cache summary")
    print("4. Test specific players (Mahomes, CMC, etc.)")

    choice = input("\nChoice (1/2/3/4): ").strip()

    if choice == '1':
        # Fetch team lines
        df = fetcher.fetch_team_lines(force_refresh=False)

        if not df.empty:
            print("\nSample team lines:")
            print(df[['week', 'home_team', 'away_team', 'spread', 'over_under']].head(10))

    elif choice == '2':
        # Fetch props for specific players
        players_input = input("Enter player names (comma-separated): ").strip()
        players = [p.strip() for p in players_input.split(',')]
        week = int(input("Week number: ").strip())

        props_df = fetcher.fetch_player_props(
            player_names=players,
            week=week,
            force_refresh=False
        )

        if not props_df.empty:
            print("\nPlayer props:")
            for player in players:
                player_props = props_df[props_df['player_name'] == player]
                if not player_props.empty:
                    print(f"\n{player}:")
                    print(player_props[['prop_type', 'prop_value']].to_string(index=False))

    elif choice == '3':
        # Show summary
        fetcher.get_summary()

    elif choice == '4':
        # Test with specific players
        test_specific_players()
        return

    else:
        print("❌ Invalid choice")

    print(f"\n{'='*70}")
    print(f"Done at: {datetime.now()}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
