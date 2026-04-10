#!/usr/bin/env python3
"""
File: src/nfl/v4_odds_api.py

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

    def __init__(self, api_key, cache_dir='./data/nfl/vegas_odds'):
        """
        Initialize The Odds API client.

        Args:
            api_key: Your Odds API key
            cache_dir: Directory to cache fetched data
        """
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Subdirectories for organization
        self.team_lines_dir = self.cache_dir / 'team_lines'
        self.player_props_dir = self.cache_dir / 'player_props'
        self.team_lines_dir.mkdir(parents=True, exist_ok=True)
        self.player_props_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.cache_dir / 'fetch_log.json'

        # Track API usage
        self.requests_used = 0
        self.requests_remaining = None

        print(f"✓ Odds API Client initialized")
        print(f"  Team lines: {self.team_lines_dir}/")
        print(f"  Player props: {self.player_props_dir}/")

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
        from datetime import timezone

        game_date = datetime.fromisoformat(game_date_str.replace('Z', '+00:00'))

        # NFL 2025 season starts ~September 4, 2025 (make timezone-aware)
        season_start = datetime(2025, 9, 4, tzinfo=timezone.utc)

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
        # Check for existing cached files (load most recent)
        if not force_refresh:
            cached_files = sorted(self.team_lines_dir.glob('team_lines_week_*.parquet'))
            if cached_files:
                latest_cache = cached_files[-1]
                print(f"\n📦 Loading cached team lines from {latest_cache.name}")
                df = pd.read_parquet(latest_cache)
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
            # Cache by week with fetch date
            fetch_date = datetime.now().strftime('%Y-%m-%d')
            weeks = sorted(df['week'].unique())

            # Save each week separately
            for week in weeks:
                week_df = df[df['week'] == week]
                cache_file = self.team_lines_dir / f'team_lines_week_{week:02d}_fetched_{fetch_date}.parquet'
                week_df.to_parquet(cache_file, index=False)
                print(f"   💾 Week {week}: {len(week_df)} games → {cache_file.name}")

            # Log fetch
            log_data = {
                'last_fetch': datetime.now().isoformat(),
                'fetch_date': fetch_date,
                'season': 2025,
                'weeks': [int(w) for w in weeks],  # Convert to regular Python ints
                'total_games': len(df),
                'requests_remaining': self.requests_remaining
            }

            with open(self.log_file, 'w') as f:
                json.dump(log_data, f, indent=2)

            print(f"\n{'='*60}")
            print(f"SUMMARY")
            print(f"{'='*60}")
            print(f"✅ Total games fetched: {len(df)}")
            print(f"📅 Season: 2025")
            print(f"🔢 Weeks: {weeks}")
            print(f"💾 Cached to: {self.team_lines_dir}/")
            print(f"📊 API requests remaining: {self.requests_remaining}")
            print(f"{'='*60}\n")

        return df

    def load_cached_lines(self, week=None):
        """
        Load cached team lines.

        Args:
            week: Specific week to load (None = all weeks, load most recent for each)

        Returns:
            DataFrame with cached lines, or empty DataFrame if no cache
        """
        if week is not None:
            # Load specific week (most recent fetch)
            cached_files = sorted(self.team_lines_dir.glob(f'team_lines_week_{week:02d}_*.parquet'))
            if cached_files:
                return pd.read_parquet(cached_files[-1])
            else:
                print(f"⚠ No cached team lines for week {week}")
                return pd.DataFrame()
        else:
            # Load all weeks (most recent fetch for each week)
            all_lines = []
            week_files = {}

            for file in self.team_lines_dir.glob('team_lines_week_*.parquet'):
                # Extract week number from filename
                week_num = int(file.stem.split('_')[3])
                if week_num not in week_files or file > week_files[week_num]:
                    week_files[week_num] = file

            for week_num in sorted(week_files.keys()):
                all_lines.append(pd.read_parquet(week_files[week_num]))

            if all_lines:
                return pd.concat(all_lines, ignore_index=True)
            else:
                print(f"⚠ No cached team lines found in {self.team_lines_dir}")
                return pd.DataFrame()

    def fetch_player_props_for_event(self, event_id, event_info=None):
        """
        Fetch player props for a specific game.

        Args:
            event_id: The Odds API event ID
            event_info: Optional dict with game info (teams, date)

        Returns:
            List of dictionaries with player prop data
        """
        url = f"{self.BASE_URL}/sports/{self.SPORT}/events/{event_id}/odds"

        # Player prop markets we want
        markets = [
            'player_pass_tds', 'player_pass_yds', 'player_pass_completions',
            'player_rush_yds', 'player_rush_attempts',
            'player_receptions', 'player_reception_yds'
        ]

        params = {
            'apiKey': self.api_key,
            'regions': 'us',
            'markets': ','.join(markets),
            'oddsFormat': 'american'
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            # Update usage stats
            self.requests_remaining = response.headers.get('x-requests-remaining')
            self.requests_used = response.headers.get('x-requests-used')

            data = response.json()
            return self.extract_player_props(data, event_info)

        except requests.exceptions.RequestException as e:
            print(f"    ✗ Error fetching props for {event_id}: {str(e)[:50]}")
            return []

    def extract_player_props(self, event_data, event_info=None):
        """
        Extract player props from Odds API event data.

        Args:
            event_data: API response for event odds
            event_info: Optional game info dict

        Returns:
            List of player prop dictionaries
        """
        props = []

        try:
            game_id = event_data.get('id')
            game_date = event_data.get('commence_time')
            home_team = event_data.get('home_team')
            away_team = event_data.get('away_team')

            # Map team names to abbreviations (same as before)
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

            # Determine week
            week = self.determine_nfl_week(game_date) if game_date else None

            bookmakers = event_data.get('bookmakers', [])

            if not bookmakers:
                return props

            # Use first bookmaker (usually DraftKings or FanDuel)
            bookmaker = bookmakers[0]
            markets = bookmaker.get('markets', [])

            for market in markets:
                market_key = market.get('key')
                outcomes = market.get('outcomes', [])

                # Process each player in this market
                player_props_map = {}  # player_name -> prop_value

                for outcome in outcomes:
                    if outcome.get('name') == 'Over':
                        player_name = outcome.get('description')
                        prop_value = outcome.get('point')

                        if player_name and prop_value is not None:
                            player_props_map[player_name] = prop_value

                # Create prop records
                for player_name, prop_value in player_props_map.items():
                    props.append({
                        'game_id': game_id,
                        'game_date': game_date,
                        'season': 2025,
                        'week': week,
                        'home_team': home_abbr,
                        'away_team': away_abbr,
                        'player_name': player_name,
                        'prop_type': market_key,
                        'prop_value': prop_value,
                        'bookmaker': bookmaker.get('title', 'Unknown')
                    })

        except Exception as e:
            print(f"    ⚠ Error extracting player props: {str(e)[:50]}")

        return props

    def fetch_all_player_props(self, force_refresh=False):
        """
        Fetch player props for all current NFL games and cache by week.

        Args:
            force_refresh: If True, re-fetch even if cached

        Returns:
            DataFrame with all player props
        """
        print(f"\n{'='*70}")
        print(f"Fetching Player Props from The Odds API")
        print(f"{'='*70}")

        # First, get list of current games (reuse team lines data)
        team_lines = self.load_cached_lines()

        if team_lines.empty:
            print("⚠ No team lines cached. Fetch team lines first.")
            return pd.DataFrame()

        all_props = []
        games_by_week = team_lines.groupby('week')
        fetch_date = datetime.now().strftime('%Y-%m-%d')

        for week, week_games in games_by_week:
            props_cache_file = self.player_props_dir / f'player_props_week_{week:02d}_fetched_{fetch_date}.parquet'

            # Check if we already have props for this week (any date)
            existing_props = sorted(self.player_props_dir.glob(f'player_props_week_{week:02d}_*.parquet'))
            if existing_props and not force_refresh:
                latest_props = existing_props[-1]
                print(f"\n📦 Week {week}: Loading cached props from {latest_props.name}")
                cached_props = pd.read_parquet(latest_props)
                all_props.append(cached_props)
                continue

            print(f"\n🔄 Week {week}: Fetching props for {len(week_games)} games")
            week_props = []

            for idx, game in week_games.iterrows():
                game_id = game['game_id']
                print(f"  Game {idx+1}/{len(week_games)}: {game['away_team']} @ {game['home_team']}...", end=' ')

                props = self.fetch_player_props_for_event(game_id, game.to_dict())

                if props:
                    week_props.extend(props)
                    print(f"✓ {len(props)} props")
                else:
                    print(f"✗ No props")

                time.sleep(0.5)  # Rate limiting

            if week_props:
                # Cache this week's props
                week_df = pd.DataFrame(week_props)
                week_df.to_parquet(props_cache_file, index=False)
                all_props.append(week_df)
                print(f"  💾 Cached {len(week_props)} props to {props_cache_file.name}")

        if all_props:
            combined_df = pd.concat(all_props, ignore_index=True)

            print(f"\n{'='*70}")
            print(f"PLAYER PROPS SUMMARY")
            print(f"{'='*70}")
            print(f"✅ Total props: {len(combined_df)}")
            print(f"👥 Unique players: {combined_df['player_name'].nunique()}")
            print(f"📊 Prop types: {list(combined_df['prop_type'].unique())}")
            print(f"🔢 Weeks: {sorted(combined_df['week'].unique())}")
            print(f"📊 API requests remaining: {self.requests_remaining}")
            print(f"{'='*70}\n")

            return combined_df
        else:
            print("\n⚠ No player props fetched")
            return pd.DataFrame()

    def load_cached_player_props(self, week=None):
        """
        Load cached player props.

        Args:
            week: Specific week to load (None = all weeks, load most recent for each)

        Returns:
            DataFrame with cached player props
        """
        if week is not None:
            # Load specific week (most recent fetch)
            cached_files = sorted(self.player_props_dir.glob(f'player_props_week_{week:02d}_*.parquet'))
            if cached_files:
                return pd.read_parquet(cached_files[-1])
            else:
                print(f"⚠ No cached player props for week {week}")
                return pd.DataFrame()
        else:
            # Load all weeks (most recent fetch for each week)
            all_props = []
            week_files = {}

            for file in self.player_props_dir.glob('player_props_week_*.parquet'):
                # Extract week number from filename
                week_num = int(file.stem.split('_')[3])
                if week_num not in week_files or file > week_files[week_num]:
                    week_files[week_num] = file

            for week_num in sorted(week_files.keys()):
                all_props.append(pd.read_parquet(week_files[week_num]))

            if all_props:
                return pd.concat(all_props, ignore_index=True)
            else:
                print(f"⚠ No cached player props found in {self.player_props_dir}")
                return pd.DataFrame()

    def fetch_props_for_specific_players(self, player_names, week, force_refresh=False):
        """
        Fetch player props for SPECIFIC players only (saves API credits!).
        Perfect for validating top 20-30 predictions.

        Args:
            player_names: List of player names to fetch props for
            week: NFL week number
            force_refresh: If True, re-fetch even if cached

        Returns:
            DataFrame with player props for requested players only
        """
        print(f"\n{'='*70}")
        print(f"Fetching Props for {len(player_names)} Specific Players (Week {week})")
        print(f"{'='*70}")

        # Check cache first
        cached_props = self.load_cached_player_props(week=week)

        if not cached_props.empty and not force_refresh:
            # Filter to requested players
            cached_players = cached_props[cached_props['player_name'].isin(player_names)]

            if not cached_players.empty:
                print(f"✅ Found {len(cached_players['player_name'].unique())} players in cache")
                missing = set(player_names) - set(cached_players['player_name'].unique())

                if not missing:
                    print(f"✅ All requested players found in cache (0 API calls needed)")
                    return cached_players
                else:
                    print(f"⚠ Missing {len(missing)} players from cache: {list(missing)[:5]}...")
                    # Continue to fetch missing players
            else:
                print(f"⚠ No cached props found for requested players")
        else:
            print(f"📦 No cache for week {week}, will fetch fresh data")
            cached_players = pd.DataFrame()
            missing = set(player_names)

        # Get team lines to know which games to check
        team_lines = self.load_cached_lines(week=week)

        if team_lines.empty:
            print(f"❌ No team lines cached for week {week}. Fetch team lines first.")
            return cached_players if not cached_players.empty else pd.DataFrame()

        # Fetch props for each game and filter for our players
        print(f"\n🔄 Fetching props from {len(team_lines)} games...")
        new_props = []
        api_calls = 0

        for idx, game in team_lines.iterrows():
            game_id = game['game_id']

            # Fetch props for this game
            props = self.fetch_player_props_for_event(game_id, game.to_dict())
            api_calls += 1

            if props:
                # Filter to only players we care about
                game_props_df = pd.DataFrame(props)
                relevant_props = game_props_df[game_props_df['player_name'].isin(missing)]

                if not relevant_props.empty:
                    new_props.append(relevant_props)
                    found_players = relevant_props['player_name'].unique()
                    print(f"  ✓ Game {idx+1}/{len(team_lines)}: Found {len(found_players)} players - {list(found_players)}")
                    missing = missing - set(found_players)
                else:
                    print(f"  - Game {idx+1}/{len(team_lines)}: No relevant players")

            # Stop if we found everyone
            if not missing:
                print(f"\n✅ Found all {len(player_names)} players! Stopping early.")
                print(f"   API calls made: {api_calls}/{len(team_lines)} (saved {len(team_lines)-api_calls} calls)")
                break

            time.sleep(0.5)  # Rate limiting

        # Combine cached + new props
        all_results = []
        if not cached_players.empty:
            all_results.append(cached_players)
        if new_props:
            all_results.append(pd.concat(new_props, ignore_index=True))

        if all_results:
            final_df = pd.concat(all_results, ignore_index=True).drop_duplicates(
                subset=['player_name', 'prop_type'], keep='last'
            )

            # Update cache (merge with existing cache) only if we found new props
            if new_props:
                if not cached_props.empty:
                    # Merge new props with existing cache
                    updated_cache = pd.concat([cached_props, pd.concat(new_props, ignore_index=True)], ignore_index=True)
                    updated_cache = updated_cache.drop_duplicates(
                        subset=['player_name', 'prop_type'], keep='last'
                    )
                else:
                    updated_cache = pd.concat(new_props, ignore_index=True)

                # Save updated cache
                if not updated_cache.empty:
                    fetch_date = datetime.now().strftime('%Y-%m-%d')
                    cache_file = self.player_props_dir / f'player_props_week_{week:02d}_fetched_{fetch_date}.parquet'
                    updated_cache.to_parquet(cache_file, index=False)
                    print(f"\n💾 Updated cache: {cache_file.name}")

            print(f"\n{'='*70}")
            print(f"RESULTS")
            print(f"{'='*70}")
            print(f"✅ Players found: {len(final_df['player_name'].unique())}/{len(player_names)}")
            print(f"📊 Total props: {len(final_df)}")
            print(f"🔢 API calls made: {api_calls}")
            print(f"📊 API requests remaining: {self.requests_remaining}")

            if missing:
                print(f"⚠ Players not found: {list(missing)}")
            print(f"{'='*70}\n")

            return final_df
        else:
            print(f"\n⚠ No props found for any requested players")
            return pd.DataFrame()


def main():
    """Test The Odds API client"""
    # API key from user
    API_KEY = 'c6d41f99d9fdabfa5f5abaf8df1c9084'

    client = OddsAPIClient(api_key=API_KEY)

    print("\n" + "="*70)
    print("OPTIONS:")
    print("="*70)
    print("1. Fetch team lines only")
    print("2. Fetch player props (requires team lines first)")
    print("3. Show cached data summary")
    print("="*70)

    choice = input("\nChoice (1/2/3): ").strip()

    if choice == '1':
        # Fetch current 2025 team lines
        df = client.fetch_current_lines(force_refresh=False)

        if not df.empty:
            print("\nSample team lines:")
            print(df.head(10))
            print(f"\nWeeks available: {sorted(df['week'].unique())}")
            print(f"Total games: {len(df)}")

    elif choice == '2':
        # Fetch player props
        props_df = client.fetch_all_player_props(force_refresh=False)

        if not props_df.empty:
            print("\nSample player props:")
            print(props_df.head(20))
            print(f"\nProp types: {props_df['prop_type'].unique()}")
            print(f"Sample players: {props_df['player_name'].unique()[:10]}")

    elif choice == '3':
        # Show cache summary
        team_lines = client.load_cached_lines()
        player_props = client.load_cached_player_props()

        print(f"\n{'='*70}")
        print("CACHED DATA SUMMARY")
        print(f"{'='*70}")
        print(f"Team lines: {len(team_lines)} games")
        if not team_lines.empty:
            print(f"  Weeks: {sorted(team_lines['week'].unique())}")
        print(f"\nPlayer props: {len(player_props)} props")
        if not player_props.empty:
            print(f"  Players: {player_props['player_name'].nunique()}")
            print(f"  Prop types: {list(player_props['prop_type'].unique())}")
            print(f"  Weeks: {sorted(player_props['week'].unique())}")
        print(f"{'='*70}\n")

    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
