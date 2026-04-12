# src/nfl/features/v5/master_table.py
"""
Builds the master player-week table by loading and joining all 13 datasets.
This is the foundation — every feature module reads from this table.
"""

import pandas as pd
from pathlib import Path


def _load_multi_season_parquets(data_dir, subdir, pattern, seasons):
    """Load and concatenate per-season Parquet files."""
    path = Path(data_dir) / subdir
    dfs = []
    for season in seasons:
        f = path / pattern.format(season=season)
        if f.exists():
            dfs.append(pd.read_parquet(f))
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def build_master_table(data_dir, seasons):
    """
    Build a player-week master table by joining all 13 datasets.

    Args:
        data_dir: Path containing subdirs for each dataset (e.g., 'data/nfl/')
        seasons: List of season years to load (e.g., [2020, 2021, ..., 2025])

    Returns:
        DataFrame with one row per (player_id, season, week) containing all
        available data. NULL where a dataset doesn't have that player/week.
    """
    data_dir = Path(data_dir)

    # 1. Start from weekly_stats — the most complete dataset
    ps = _load_multi_season_parquets(
        data_dir, 'player_stats', 'player_stats_{season}.parquet', seasons
    )
    # Filter null player_ids (garbage rows)
    ps = ps[ps['player_id'].notna()].copy()

    # 2. LEFT JOIN players to get PFR ID mapping
    players_path = data_dir / 'players' / 'players.parquet'
    if players_path.exists():
        players = pd.read_parquet(players_path)[['gsis_id', 'pfr_id']]
        ps = ps.merge(players, left_on='player_id', right_on='gsis_id', how='left')

    # 3. LEFT JOIN schedules for game context (Vegas, weather, rest)
    schedules = _load_multi_season_parquets(
        data_dir, 'schedules', 'schedules_{season}.parquet', seasons
    )
    if not schedules.empty:
        home_games = schedules[[
            'season', 'week', 'home_team', 'spread_line', 'total_line',
            'home_implied_total', 'away_implied_total', 'home_rest',
            'away_rest', 'temp', 'wind', 'roof', 'div_game',
        ]].copy()
        home_games['is_home'] = 1
        home_games = home_games.rename(columns={
            'home_team': 'team',
            'home_implied_total': 'team_implied_total',
            'away_implied_total': 'opponent_implied_total',
            'home_rest': 'team_rest',
            'away_rest': 'opponent_rest',
        })

        away_games = schedules[[
            'season', 'week', 'away_team', 'spread_line', 'total_line',
            'home_implied_total', 'away_implied_total', 'home_rest',
            'away_rest', 'temp', 'wind', 'roof', 'div_game',
        ]].copy()
        away_games['is_home'] = 0
        # Flip spread sign for away team
        away_games['spread_line'] = -away_games['spread_line']
        away_games = away_games.rename(columns={
            'away_team': 'team',
            'away_implied_total': 'team_implied_total',
            'home_implied_total': 'opponent_implied_total',
            'away_rest': 'team_rest',
            'home_rest': 'opponent_rest',
        })

        all_games = pd.concat([home_games, away_games], ignore_index=True)
        ps = ps.merge(all_games, on=['season', 'week', 'team'], how='left')

    # 4. LEFT JOIN injuries (via GSIS)
    injuries = _load_multi_season_parquets(
        data_dir, 'injuries', 'injuries_{season}.parquet', seasons
    )
    if not injuries.empty:
        inj_cols = ['gsis_id', 'season', 'week', 'report_status',
                    'practice_status', 'report_primary_injury']
        keep = [c for c in inj_cols if c in injuries.columns]
        injuries = injuries[keep]
        # Injuries can have multiple rows per (player, week) when the team
        # files multiple status updates during a week. Keep the last row per
        # (gsis_id, season, week) — this is typically the final game-day
        # status, which is the most predictive for fantasy outcomes.
        injuries = injuries.drop_duplicates(
            subset=['gsis_id', 'season', 'week'], keep='last'
        )
        ps = ps.merge(injuries, left_on=['player_id', 'season', 'week'],
                      right_on=['gsis_id', 'season', 'week'],
                      how='left', suffixes=('', '_inj'))
        ps = ps.drop(columns=['gsis_id_inj'], errors='ignore')

    # 5. LEFT JOIN snap_counts (via PFR ID)
    snap_counts = _load_multi_season_parquets(
        data_dir, 'snap_counts', 'snap_counts_{season}.parquet', seasons
    )
    if not snap_counts.empty and 'pfr_id' in ps.columns:
        sc_cols = ['pfr_player_id', 'season', 'week', 'offense_snaps',
                   'offense_pct', 'st_pct']
        keep = [c for c in sc_cols if c in snap_counts.columns]
        snap_counts = snap_counts[keep]
        # Dedup guard: prevent row explosion if source has dup (player,season,week)
        snap_counts = snap_counts.drop_duplicates(
            subset=['pfr_player_id', 'season', 'week']
        )
        ps = ps.merge(snap_counts, left_on=['pfr_id', 'season', 'week'],
                      right_on=['pfr_player_id', 'season', 'week'],
                      how='left')
        ps = ps.drop(columns=['pfr_player_id'], errors='ignore')

    # 6. LEFT JOIN ff_opportunity (via GSIS, filter null player_id)
    ff_opp = _load_multi_season_parquets(
        data_dir, 'ff_opportunity', 'ff_opportunity_{season}.parquet', seasons
    )
    if not ff_opp.empty:
        ff_opp = ff_opp[ff_opp['player_id'].notna()].copy()
        # season/week may be string in ff_opportunity — coerce to int
        ff_opp['season'] = ff_opp['season'].astype(int)
        ff_opp['week'] = ff_opp['week'].astype(int)
        keep = ['player_id', 'season', 'week',
                'total_fantasy_points_exp', 'total_fantasy_points_diff',
                'pass_fantasy_points_exp', 'rec_fantasy_points_exp',
                'rush_fantasy_points_exp']
        keep = [c for c in keep if c in ff_opp.columns]
        ff_opp = ff_opp[keep]
        ps = ps.merge(ff_opp, on=['player_id', 'season', 'week'],
                      how='left', suffixes=('', '_ff'))

    # 7. LEFT JOIN NGS (passing/rushing/receiving, via GSIS)
    for stat_type, cols in [
        ('passing', ['avg_time_to_throw',
                     'completion_percentage_above_expectation',
                     'aggressiveness']),
        ('rushing', ['efficiency', 'rush_yards_over_expected_per_att',
                     'percent_attempts_gte_eight_defenders']),
        ('receiving', ['avg_separation', 'avg_yac_above_expectation',
                       'avg_cushion', 'catch_percentage']),
    ]:
        ngs = _load_multi_season_parquets(
            data_dir, 'nextgen_stats',
            f'ngs_{stat_type}_{{season}}.parquet', seasons
        )
        if not ngs.empty:
            ngs = ngs[ngs['week'] > 0].copy()  # filter season aggregates
            keep = ['player_gsis_id', 'season', 'week'] + [
                c for c in cols if c in ngs.columns
            ]
            ngs = ngs[keep]
            rename_map = {c: f'ngs_{stat_type}_{c}' for c in cols if c in ngs.columns}
            ngs = ngs.rename(columns=rename_map)
            # Dedup guard against duplicate (player, season, week) rows
            ngs = ngs.drop_duplicates(
                subset=['player_gsis_id', 'season', 'week']
            )
            ps = ps.merge(ngs, left_on=['player_id', 'season', 'week'],
                          right_on=['player_gsis_id', 'season', 'week'],
                          how='left')
            ps = ps.drop(columns=['player_gsis_id'], errors='ignore')

    # 8. LEFT JOIN PFR advanced (via PFR ID)
    for stat_type, cols in [
        ('pass', ['times_pressured_pct', 'passing_bad_throw_pct',
                  'passing_drops']),
        ('rush', ['rushing_yards_after_contact_avg',
                  'rushing_broken_tackles']),
        ('rec', ['receiving_drop_pct', 'receiving_broken_tackles',
                 'receiving_rat']),
    ]:
        pfr = _load_multi_season_parquets(
            data_dir, 'pfr_advstats',
            f'pfr_{stat_type}_{{season}}.parquet', seasons
        )
        if not pfr.empty and 'pfr_id' in ps.columns:
            keep = ['pfr_player_id', 'season', 'week'] + [
                c for c in cols if c in pfr.columns
            ]
            pfr = pfr[keep]
            rename_map = {c: f'pfr_{stat_type}_{c}' for c in cols if c in pfr.columns}
            pfr = pfr.rename(columns=rename_map)
            # Dedup guard against duplicate (player, season, week) rows
            pfr = pfr.drop_duplicates(
                subset=['pfr_player_id', 'season', 'week']
            )
            ps = ps.merge(pfr, left_on=['pfr_id', 'season', 'week'],
                          right_on=['pfr_player_id', 'season', 'week'],
                          how='left')
            ps = ps.drop(columns=['pfr_player_id'], errors='ignore')

    # 9. LEFT JOIN depth_charts (via GSIS, only 2018-2024)
    dc = _load_multi_season_parquets(
        data_dir, 'depth_charts', 'depth_charts_{season}.parquet', seasons
    )
    if not dc.empty:
        dc = dc[dc['formation'] == 'Offense'].copy()
        dc_cols = ['gsis_id', 'season', 'week', 'depth_team', 'depth_position']
        keep = [c for c in dc_cols if c in dc.columns]
        dc = dc[keep].drop_duplicates(subset=['gsis_id', 'season', 'week'])
        ps = ps.merge(dc, left_on=['player_id', 'season', 'week'],
                      right_on=['gsis_id', 'season', 'week'],
                      how='left', suffixes=('', '_dc'))
        ps = ps.drop(columns=['gsis_id_dc'], errors='ignore')

    # Sort for downstream rolling operations
    ps = ps.sort_values(['player_id', 'season', 'week']).reset_index(drop=True)
    return ps
