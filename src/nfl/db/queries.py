"""
File: src/nfl/db/queries.py

Reusable SQL query functions for the NFL predictions database.
All functions return pandas DataFrames via pd.read_sql().

Usage:
    from src.nfl.db.queries import get_player_history, get_game_context

    # Get Mahomes' last 6 games before 2024 Week 5
    history = get_player_history('00-0033873', 2024, 5, games_back=6)

    # Get Vegas lines for KC in 2024 Week 1
    context = get_game_context(2024, 1, 'KC')
"""

import pandas as pd
from sqlalchemy import text

from src.nfl.db.connection import get_engine


def get_player_history(player_id, season, week, games_back=6):
    """
    Get a player's last N games of stats from weekly_stats.
    Handles cross-season lookback (e.g., week 2 of 2024 looks into 2023).

    Args:
        player_id: GSIS player ID (e.g., '00-0033873')
        season: Current season year
        week: Current week number
        games_back: Number of past games to retrieve (default: 6)

    Returns:
        DataFrame of player stats ordered by most recent first
    """
    engine = get_engine()
    query = text("""
        SELECT *
        FROM weekly_stats
        WHERE player_id = :player_id
          AND (season < :season OR (season = :season AND week < :week))
        ORDER BY season DESC, week DESC
        LIMIT :games_back
    """)
    return pd.read_sql(query, engine, params={
        'player_id': player_id,
        'season': season,
        'week': week,
        'games_back': games_back,
    })


def get_week_stats(season, week, position=None):
    """
    Get all player stats for a given week, optionally filtered by position.

    Args:
        season: Season year
        week: Week number
        position: Optional position filter (e.g., 'QB', 'RB', 'WR', 'TE', 'K')

    Returns:
        DataFrame of all player stats for that week
    """
    engine = get_engine()
    if position:
        query = text("""
            SELECT *
            FROM weekly_stats
            WHERE season = :season AND week = :week AND position = :position
        """)
        return pd.read_sql(query, engine, params={
            'season': season, 'week': week, 'position': position,
        })
    else:
        query = text("""
            SELECT *
            FROM weekly_stats
            WHERE season = :season AND week = :week
        """)
        return pd.read_sql(query, engine, params={
            'season': season, 'week': week,
        })


def get_player_injuries(player_id, season, week):
    """
    Get injury report status for a player in a given week.
    Empty DataFrame = player is healthy (not on injury report).

    Args:
        player_id: GSIS player ID
        season: Season year
        week: Week number

    Returns:
        DataFrame with injury report fields (may be empty if healthy)
    """
    engine = get_engine()
    query = text("""
        SELECT *
        FROM injuries
        WHERE gsis_id = :player_id AND season = :season AND week = :week
    """)
    return pd.read_sql(query, engine, params={
        'player_id': player_id, 'season': season, 'week': week,
    })


def get_snap_share(player_id, season, week):
    """
    Get snap count percentages for a player.
    Handles GSIS→PFR ID mapping via the players table since
    snap_counts uses pfr_player_id.

    Args:
        player_id: GSIS player ID
        season: Season year
        week: Week number

    Returns:
        DataFrame with snap count fields (may be empty if no mapping or no data)
    """
    engine = get_engine()
    query = text("""
        SELECT sc.*
        FROM snap_counts sc
        JOIN players p ON p.pfr_id = sc.pfr_player_id
        WHERE p.gsis_id = :player_id
          AND sc.season = :season
          AND sc.week = :week
    """)
    return pd.read_sql(query, engine, params={
        'player_id': player_id, 'season': season, 'week': week,
    })


def get_game_context(season, week, team):
    """
    Get Vegas lines, weather, and rest days for a team's game.
    Checks both home and away sides.

    Args:
        season: Season year
        week: Week number
        team: Team abbreviation (e.g., 'KC', 'BAL')

    Returns:
        DataFrame with game context (1 row if found). Includes a computed
        'is_home' column and 'implied_total' for the requested team.
    """
    engine = get_engine()
    query = text("""
        SELECT *,
            CASE WHEN home_team = :team THEN true ELSE false END AS is_home,
            CASE WHEN home_team = :team THEN home_implied_total
                 ELSE away_implied_total END AS implied_total,
            CASE WHEN home_team = :team THEN home_rest
                 ELSE away_rest END AS rest_days
        FROM games
        WHERE season = :season AND week = :week
          AND (home_team = :team OR away_team = :team)
    """)
    return pd.read_sql(query, engine, params={
        'season': season, 'week': week, 'team': team,
    })


def get_opponent_defense_rank(opponent_team, position, season, week):
    """
    Rank how tough an opponent's defense is against a position.
    Calculates average fantasy points allowed to the position over the
    season-to-date, then ranks across all 32 teams.

    Args:
        opponent_team: Team abbreviation of the opponent (e.g., 'KC')
        position: Position to evaluate against (e.g., 'QB', 'RB', 'WR', 'TE')
        season: Season year
        week: Current week (uses data from weeks < week)

    Returns:
        Dict with 'rank' (1=toughest, 32=easiest), 'avg_pts_allowed',
        and 'games_played'. Returns None if insufficient data.
    """
    engine = get_engine()
    # Calculate avg fantasy points allowed by each team's defense to the position
    query = text("""
        WITH defense_allowed AS (
            SELECT
                opponent_team AS defense_team,
                AVG(fantasy_points_ppr) AS avg_pts_allowed,
                COUNT(*) AS games_played
            FROM weekly_stats
            WHERE season = :season
              AND week < :week
              AND position = :position
              AND fantasy_points_ppr IS NOT NULL
            GROUP BY opponent_team
        ),
        ranked AS (
            SELECT *,
                RANK() OVER (ORDER BY avg_pts_allowed ASC) AS defense_rank
            FROM defense_allowed
        )
        SELECT defense_rank, avg_pts_allowed, games_played
        FROM ranked
        WHERE defense_team = :opponent_team
    """)
    df = pd.read_sql(query, engine, params={
        'season': season, 'week': week,
        'position': position, 'opponent_team': opponent_team,
    })
    if df.empty:
        return None
    row = df.iloc[0]
    return {
        'rank': int(row['defense_rank']),
        'avg_pts_allowed': float(row['avg_pts_allowed']),
        'games_played': int(row['games_played']),
    }


def get_nextgen_stats(player_id, season, week, stat_type):
    """
    Get Next Gen Stats metrics for a player.

    Args:
        player_id: GSIS player ID
        season: Season year
        week: Week number (use None for all weeks in the season)
        stat_type: 'passing', 'rushing', or 'receiving'

    Returns:
        DataFrame with NGS metrics
    """
    table_map = {
        'passing': 'ngs_passing',
        'rushing': 'ngs_rushing',
        'receiving': 'ngs_receiving',
    }
    table = table_map.get(stat_type)
    if table is None:
        raise ValueError(f"stat_type must be 'passing', 'rushing', or 'receiving', got '{stat_type}'")

    engine = get_engine()
    if week is not None:
        query = text(f"""
            SELECT *
            FROM {table}
            WHERE player_gsis_id = :player_id
              AND season = :season
              AND week = :week
              AND week > 0
        """)
        return pd.read_sql(query, engine, params={
            'player_id': player_id, 'season': season, 'week': week,
        })
    else:
        query = text(f"""
            SELECT *
            FROM {table}
            WHERE player_gsis_id = :player_id
              AND season = :season
              AND week > 0
            ORDER BY week
        """)
        return pd.read_sql(query, engine, params={
            'player_id': player_id, 'season': season,
        })
