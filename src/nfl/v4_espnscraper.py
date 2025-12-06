#!/usr/bin/env python3
"""
File: src/nfl/v4_espnscraper.py

ESPN API scraper for historical NFL Vegas lines (2020-2024).
Uses ESPN's unofficial API to fetch betting data (spreads, totals).

Data cached to: data/nfl/vegas_lines/historical_lines.parquet
"""

import requests
import pandas as pd
import time
from pathlib import Path
from datetime import datetime, timedelta
import json


class ESPNVegasScraper:
    """Scrapes Vegas lines from ESPN's unofficial API"""

    BASE_URL = "https://site.web.api.espn.com/apis/site/v2/sports/football/nfl"

    def __init__(self, cache_dir='./data/nfl/vegas_lines'):
        """
        Initialize ESPN scraper.

        Args:
            cache_dir: Directory to cache fetched data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / 'historical_lines.parquet'
        self.log_file = self.cache_dir / 'espn_fetch_log.json'

        print(f"✓ ESPN Scraper initialized")
        print(f"  Cache: {self.cache_file}")

    def get_week_start_date(self, season, week):
        """
        Calculate approximate start date for NFL week.
        NFL season typically starts first Thursday in September.

        Args:
            season: Year (e.g., 2024)
            week: NFL week (1-18)

        Returns:
            Date string in YYYYMMDD format
        """
        # NFL season start dates (approximate first Thursday of September)
        season_starts = {
            2020: datetime(2020, 9, 10),
            2021: datetime(2021, 9, 9),
            2022: datetime(2022, 9, 8),
            2023: datetime(2023, 9, 7),
            2024: datetime(2024, 9, 5),
            2025: datetime(2025, 9, 4),
        }

        start_date = season_starts.get(season, datetime(season, 9, 7))
        week_date = start_date + timedelta(days=(week - 1) * 7)

        return week_date.strftime('%Y%m%d')

    def fetch_scoreboard(self, season, week):
        """
        Fetch NFL scoreboard for a specific week using ESPN API.

        Args:
            season: Year (e.g., 2024)
            week: NFL week (1-18)

        Returns:
            List of game data dictionaries
        """
        date_str = self.get_week_start_date(season, week)

        url = f"{self.BASE_URL}/scoreboard"
        params = {
            'dates': date_str,
            'limit': 100,
            'seasontype': 2,  # Regular season (2), Playoffs (3)
            'week': week
        }

        try:
            print(f"  Fetching {season} Week {week} (date: {date_str})...", end=' ')
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            events = data.get('events', [])

            print(f"✓ {len(events)} games")
            return events

        except requests.exceptions.RequestException as e:
            print(f"✗ Error: {str(e)[:50]}")
            return []

    def extract_vegas_lines(self, event):
        """
        Extract Vegas lines from ESPN event data.

        Args:
            event: ESPN event dictionary

        Returns:
            Dictionary with game info and Vegas lines, or None
        """
        try:
            # Basic game info
            game_id = event.get('id')
            game_date = event.get('date')

            # Teams
            competitions = event.get('competitions', [{}])[0]
            competitors = competitions.get('competitors', [])

            if len(competitors) < 2:
                return None

            # Home/Away teams
            home_team = None
            away_team = None

            for comp in competitors:
                team_abbr = comp.get('team', {}).get('abbreviation', '')
                if comp.get('homeAway') == 'home':
                    home_team = team_abbr
                else:
                    away_team = team_abbr

            if not home_team or not away_team:
                return None

            # Extract betting lines from "odds" field
            odds = competitions.get('odds', [])

            if not odds:
                # DEBUG: Check if odds exist at all in response
                # print(f"    [DEBUG] No odds for game {game_id} - {home_team} vs {away_team}")
                return None

            # ESPN typically has multiple books, take the first one
            odds_data = odds[0]

            # Extract spread (from home perspective)
            spread = odds_data.get('spread')  # Negative = home favored
            over_under = odds_data.get('overUnder')

            if spread is None or over_under is None:
                return None

            # Calculate implied totals
            # spread = home_total - away_total
            # over_under = home_total + away_total
            # Solving: home_total = (over_under - spread) / 2
            #          away_total = (over_under + spread) / 2
            home_implied = (over_under - spread) / 2
            away_implied = (over_under + spread) / 2

            return {
                'game_id': game_id,
                'game_date': game_date,
                'season': None,  # Will be set by caller
                'week': None,    # Will be set by caller
                'home_team': home_team,
                'away_team': away_team,
                'spread': spread,  # Home perspective (negative = home favored)
                'over_under': over_under,
                'home_implied_total': round(home_implied, 1),
                'away_implied_total': round(away_implied, 1),
                'odds_provider': odds_data.get('provider', {}).get('name', 'ESPN')
            }

        except Exception as e:
            print(f"    ⚠ Error extracting lines: {str(e)[:40]}")
            return None

    def fetch_season_lines(self, season, start_week=1, end_week=18):
        """
        Fetch Vegas lines for entire season.

        Args:
            season: Year (e.g., 2024)
            start_week: First week to fetch (default: 1)
            end_week: Last week to fetch (default: 18)

        Returns:
            DataFrame with Vegas lines
        """
        print(f"\n{'='*60}")
        print(f"Fetching {season} Season (Weeks {start_week}-{end_week})")
        print(f"{'='*60}")

        all_lines = []

        for week in range(start_week, end_week + 1):
            events = self.fetch_scoreboard(season, week)

            for event in events:
                lines = self.extract_vegas_lines(event)

                if lines:
                    lines['season'] = season
                    lines['week'] = week
                    all_lines.append(lines)

            # Polite rate limiting (ESPN is generous but let's be nice)
            time.sleep(0.5)

        df = pd.DataFrame(all_lines)

        print(f"\n✅ Fetched {len(df)} games with Vegas lines for {season}")

        return df

    def fetch_historical_lines(self, seasons, force_refresh=False):
        """
        Fetch and cache historical Vegas lines for multiple seasons.

        Args:
            seasons: List of seasons to fetch (e.g., [2020, 2021, 2022])
            force_refresh: If True, re-fetch even if cached

        Returns:
            DataFrame with all historical lines
        """
        # Check cache first
        if self.cache_file.exists() and not force_refresh:
            print(f"\n📦 Loading cached historical lines from {self.cache_file}")
            df = pd.read_parquet(self.cache_file)

            cached_seasons = sorted(df['season'].unique())
            print(f"   Cached seasons: {cached_seasons}")

            # Check if we need to fetch any new seasons
            missing_seasons = [s for s in seasons if s not in cached_seasons]

            if missing_seasons:
                print(f"   Missing seasons: {missing_seasons}")
                print(f"   Fetching missing data...\n")

                new_data = []
                for season in missing_seasons:
                    season_df = self.fetch_season_lines(season)
                    new_data.append(season_df)

                if new_data:
                    new_df = pd.concat(new_data, ignore_index=True)
                    df = pd.concat([df, new_df], ignore_index=True)
                    df = df.drop_duplicates(subset=['game_id'])
                    df = df.sort_values(['season', 'week'])

                    # Update cache
                    df.to_parquet(self.cache_file, index=False)
                    print(f"\n💾 Updated cache: {self.cache_file}")
            else:
                print(f"   ✓ All requested seasons cached!")

            return df

        # No cache or force refresh - fetch all
        print(f"\n🔄 Fetching historical lines for seasons: {seasons}")
        print(f"   This will take ~{len(seasons) * 2}-{len(seasons) * 3} minutes\n")

        all_data = []

        for season in seasons:
            season_df = self.fetch_season_lines(season)
            all_data.append(season_df)

        df = pd.concat(all_data, ignore_index=True)
        df = df.drop_duplicates(subset=['game_id'])
        df = df.sort_values(['season', 'week'])

        # Cache the results
        df.to_parquet(self.cache_file, index=False)

        # Log fetch
        log_data = {
            'last_fetch': datetime.now().isoformat(),
            'seasons': seasons,
            'total_games': len(df),
            'cache_file': str(self.cache_file)
        }

        with open(self.log_file, 'w') as f:
            json.dump(log_data, f, indent=2)

        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(f"✅ Total games fetched: {len(df)}")
        print(f"📅 Seasons: {sorted(df['season'].unique())}")
        print(f"💾 Cached to: {self.cache_file}")
        print(f"📋 Log saved: {self.log_file}")
        print(f"{'='*60}\n")

        return df

    def load_cached_lines(self):
        """
        Load cached historical lines.

        Returns:
            DataFrame with cached lines, or empty DataFrame if no cache
        """
        if self.cache_file.exists():
            return pd.read_parquet(self.cache_file)
        else:
            print(f"⚠ No cached data found at {self.cache_file}")
            return pd.DataFrame()


def main():
    """Test the ESPN scraper"""
    scraper = ESPNVegasScraper()

    # Fetch historical lines for 2020-2024
    df = scraper.fetch_historical_lines(
        seasons=[2020, 2021, 2022, 2023, 2024],
        force_refresh=False  # Use cache if available
    )

    if not df.empty:
        print("\nSample data:")
        print(df.head(10))
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nSeasons: {sorted(df['season'].unique())}")
        print(f"Total games: {len(df)}")


if __name__ == "__main__":
    main()
