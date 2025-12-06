#!/usr/bin/env python3
"""
File: src/nfl/odds_api.py

The Odds API client for fetching current/future NFL Vegas lines (2025 season).
Uses The Odds API (https://the-odds-api.com) for real-time betting data.

Free tier: 500 requests/month
Data cached to: data/nfl/vegas_lines/current_lines_2025.parquet
"""

import requests
import pandas as pd
import time
from pathlib import Path
from datetime import datetime
import json


class OddsAPIClient:
    """Client for The Odds API - fetches current/future NFL betting lines"""

    BASE_URL = "https://api.the-odds-api.com/v4"
    SPORT = "americanfootball_nfl"

    def __init__(self, api_key, cache_dir='./data/nfl/vegas_lines'):
        """
        Initialize The Odds API client.

        Args:
            api_key: Your Odds API key
            cache_dir: Directory to cache fetched data
        """
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / 'current_lines_2025.parquet'
        self.log_file = self.cache_dir / 'odds_api_fetch_log.json'

        # Track API usage
        self.requests_used = 0
        self.requests_remaining = None

        print(f"✓ Odds API Client initialized")
        print(f"  Cache: {self.cache_file}")

    def check_api_key(self):
        """
        Test API key and check remaining requests.

        Returns:
            bool: True if API key is valid
        """
        url = f"{self.BASE_URL}/sports"
        params = {'apiKey': self.api_key}

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            # Check remaining requests from headers
            self.requests_remaining = response.headers.get('x-requests-remaining')
            self.requests_used = response.headers.get('x-requests-used')

            print(f"✅ API Key Valid")
            print(f"   Requests remaining: {self.requests_remaining}")
            print(f"   Requests used: {self.requests_used}")

            return True

        except requests.exceptions.RequestException as e:
            print(f"❌ API Key Error: {str(e)}")
            return False

    def fetch_odds(self, markets=['spreads', 'totals'], regions='us'):
        """
        Fetch current NFL odds from The Odds API.

        Args:
            markets: List of markets to fetch (spreads, totals, h2h)
            regions: Region for odds (us, uk, eu, au)

        Returns:
            List of event dictionaries with odds
        """
        url = f"{self.BASE_URL}/sports/{self.SPORT}/odds"

        params = {
            'apiKey': self.api_key,
            'regions': regions,
            'markets': ','.join(markets),
            'oddsFormat': 'american',  # American odds format
            'dateFormat': 'iso'
        }

        try:
            print(f"  Fetching current NFL odds...", end=' ')
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            # Update usage stats
            self.requests_remaining = response.headers.get('x-requests-remaining')
            self.requests_used = response.headers.get('x-requests-used')

            data = response.json()
            print(f"✓ {len(data)} games")
            print(f"   API requests remaining: {self.requests_remaining}")

            return data

        except requests.exceptions.RequestException as e:
            print(f"✗ Error: {str(e)}")
            return []

    def extract_vegas_lines(self, event):
        """
        Extract Vegas lines from Odds API event data.

        Args:
            event: Odds API event dictionary

        Returns:
            Dictionary with game info and Vegas lines, or None
        """
        try:
            game_id = event.get('id')
            game_date = event.get('commence_time')

            # Teams
            home_team = event.get('home_team')
            away_team = event.get('away_team')

            # Map full team names to abbreviations (ESPN format)
            team_abbr_map = {
                'Arizona Cardinals': 'ARI', 'Atlanta Falcons': 'ATL',
                'Baltimore Ravens': 'BAL', 'Buffalo Bills': 'BUF',
                'Carolina Panthers': 'CAR', 'Chicago Bears': 'CHI',
                'Cincinnati Bengals': 'CIN', 'Cleveland Browns': 'CLE',
                'Dallas Cowboys': 'DAL', 'Denver Broncos': 'DEN',
                'Detroit Lions': 'DET', 'Green Bay Packers': 'GB',
                'Houston Texans': 'HOU', 'Indianapolis Colts': 'IND',
                'Jacksonville Jaguars': 'JAX', 'Kansas City Chiefs': 'KC',
                'Las Vegas Raiders': 'LV', 'Los Angeles Chargers': 'LAC',
                'Los Angeles Rams': 'LAR', 'Miami Dolphins': 'MIA',
                'Minnesota Vikings': 'MIN', 'New England Patriots': 'NE',
                'New Orleans Saints': 'NO', 'New York Giants': 'NYG',
                'New York Jets': 'NYJ', 'Philadelphia Eagles': 'PHI',
                'Pittsburgh Steelers': 'PIT', 'San Francisco 49ers': 'SF',
                'Seattle Seahawks': 'SEA', 'Tampa Bay Buccaneers': 'TB',
                'Tennessee Titans': 'TEN', 'Washington Commanders': 'WSH'
            }

            home_abbr = team_abbr_map.get(home_team, home_team)
            away_abbr = team_abbr_map.get(away_team, away_team)

            # Extract bookmaker data (use consensus/average)
            bookmakers = event.get('bookmakers', [])

            if not bookmakers:
                return None

            # Take the first bookmaker (usually FanDuel or DraftKings)
            bookmaker = bookmakers[0]
            markets = bookmaker.get('markets', [])

            spread = None
            over_under = None

            # Extract spread and totals from markets
            for market in markets:
                market_key = market.get('key')

                if market_key == 'spreads':
                    # Find home team spread
                    outcomes = market.get('outcomes', [])
                    for outcome in outcomes:
                        if outcome.get('name') == home_team:
                            spread = outcome.get('point')
                            break

                elif market_key == 'totals':
                    # Total points (over/under)
                    outcomes = market.get('outcomes', [])
                    if outcomes:
                        over_under = outcomes[0].get('point')

            if spread is None or over_under is None:
                return None

            # Calculate implied totals
            home_implied = (over_under - spread) / 2
            away_implied = (over_under + spread) / 2

            return {
                'game_id': game_id,
                'game_date': game_date,
                'season': 2025,  # Current season
                'week': None,    # Will need to be determined
                'home_team': home_abbr,
                'away_team': away_abbr,
                'spread': spread,  # Home perspective
                'over_under': over_under,
                'home_implied_total': round(home_implied, 1),
                'away_implied_total': round(away_implied, 1),
                'odds_provider': bookmaker.get('title', 'Unknown')
            }

        except Exception as e:
            print(f"    ⚠ Error extracting lines: {str(e)[:40]}")
            return None

    def determine_nfl_week(self, game_date_str):
        """
        Determine NFL week from game date.

        Args:
            game_date_str: ISO format date string

        Returns:
            int: NFL week number (1-18)
        """
        game_date = datetime.fromisoformat(game_date_str.replace('Z', '+00:00'))

        # NFL 2025 season starts ~September 4, 2025
        season_start = datetime(2025, 9, 4)

        # Calculate days since season start
        days_diff = (game_date - season_start).days

        # Each week is ~7 days
        week = max(1, min(18, (days_diff // 7) + 1))

        return week

    def fetch_current_lines(self, force_refresh=False):
        """
        Fetch and cache current NFL lines for 2025 season.

        Args:
            force_refresh: If True, re-fetch even if cached

        Returns:
            DataFrame with current lines
        """
        # Check cache first
        if self.cache_file.exists() and not force_refresh:
            print(f"\n📦 Loading cached 2025 lines from {self.cache_file}")
            df = pd.read_parquet(self.cache_file)
            print(f"   ✓ {len(df)} games cached")
            return df

        # Fetch new data
        print(f"\n{'='*60}")
        print(f"Fetching 2025 NFL Lines from The Odds API")
        print(f"{'='*60}")

        # Check API key first
        if not self.check_api_key():
            print("❌ Cannot proceed without valid API key")
            return pd.DataFrame()

        # Fetch odds
        events = self.fetch_odds(markets=['spreads', 'totals'])

        all_lines = []

        for event in events:
            lines = self.extract_vegas_lines(event)

            if lines:
                # Determine NFL week from date
                lines['week'] = self.determine_nfl_week(lines['game_date'])
                all_lines.append(lines)

        df = pd.DataFrame(all_lines)

        if not df.empty:
            # Cache the results
            df.to_parquet(self.cache_file, index=False)

            # Log fetch
            log_data = {
                'last_fetch': datetime.now().isoformat(),
                'season': 2025,
                'total_games': len(df),
                'requests_remaining': self.requests_remaining,
                'cache_file': str(self.cache_file)
            }

            with open(self.log_file, 'w') as f:
                json.dump(log_data, f, indent=2)

            print(f"\n{'='*60}")
            print(f"SUMMARY")
            print(f"{'='*60}")
            print(f"✅ Total games fetched: {len(df)}")
            print(f"📅 Season: 2025")
            print(f"🔢 Weeks: {sorted(df['week'].unique())}")
            print(f"💾 Cached to: {self.cache_file}")
            print(f"📊 API requests remaining: {self.requests_remaining}")
            print(f"{'='*60}\n")

        return df

    def load_cached_lines(self):
        """
        Load cached 2025 lines.

        Returns:
            DataFrame with cached lines, or empty DataFrame if no cache
        """
        if self.cache_file.exists():
            return pd.read_parquet(self.cache_file)
        else:
            print(f"⚠ No cached data found at {self.cache_file}")
            return pd.DataFrame()


def main():
    """Test The Odds API client"""
    # API key from user
    API_KEY = 'c6d41f99d9fdabfa5f5abaf8df1c9084'

    client = OddsAPIClient(api_key=API_KEY)

    # Fetch current 2025 lines
    df = client.fetch_current_lines(force_refresh=False)

    if not df.empty:
        print("\nSample data:")
        print(df.head(10))
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nWeeks available: {sorted(df['week'].unique())}")
        print(f"Total games: {len(df)}")


if __name__ == "__main__":
    main()
