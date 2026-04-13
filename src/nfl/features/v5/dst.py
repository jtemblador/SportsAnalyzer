# src/nfl/features/v5/dst.py
"""V5 DST feature engineering. Parallel team-week pipeline. Part 1: scoring + master table. Part 2 appends rolling/context/orchestrator below."""

from pathlib import Path
import pandas as pd

import numpy as np

from src.nfl.features.v5.config import (
    FANTASY_DST_WEIGHTS,
    CORE_DST_STATS_FOR_ROLLING,
    ROLLING_WINDOW,
    VERSION,
)
from src.nfl.features.v5.utils import (
    decay_weighted_avg as _decay_weighted_avg,
    rolling_decay_avg_series,
    rolling_variance_series,
    rolling_trend_series,
)


def _load_multi_season_parquets(data_dir, subdir, pattern, seasons):
    """Load and concatenate per-season Parquet files."""
    path = Path(data_dir) / subdir
    dfs = []
    for season in seasons:
        f = path / pattern.format(season=season)
        if f.exists():
            dfs.append(pd.read_parquet(f))
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def points_allowed_bonus(points_allowed: int) -> int:
    """Industry-standard ESPN/Yahoo tiered DST bonus.
    0 PA → +10, 1-6 → +7, 7-13 → +4, 14-20 → +1,
    21-27 → 0, 28-34 → -1, 35+ → -4."""
    pa = int(points_allowed)
    if pa <= 0:
        return 10
    if pa <= 6:
        return 7
    if pa <= 13:
        return 4
    if pa <= 20:
        return 1
    if pa <= 27:
        return 0
    if pa <= 34:
        return -1
    return -4


def compute_dst_fantasy_points(row) -> float:
    """Apply FANTASY_DST_WEIGHTS + points_allowed_bonus to a single team-week row.
    Used for hand-checking and downstream fantasy score derivation."""
    total = 0.0
    for stat, weight in FANTASY_DST_WEIGHTS.items():
        val = row.get(stat, 0) if hasattr(row, 'get') else row[stat] if stat in row else 0
        if val is None or (isinstance(val, float) and pd.isna(val)):
            val = 0
        total += float(val) * weight
    pa = row.get('points_allowed', 0) if hasattr(row, 'get') else (
        row['points_allowed'] if 'points_allowed' in row else 0
    )
    if pa is None or (isinstance(pa, float) and pd.isna(pa)):
        pa = 0
    total += points_allowed_bonus(pa)
    return float(total)


def build_master_dst_table(data_dir, seasons) -> pd.DataFrame:
    """Build team-week master table for DST features.
    Includes BOTH REG and POST rows (POST feeds rolling history; orchestrator filters before write).

    Returns DataFrame with columns:
      Keys: team, season, week, season_type, opponent_team, is_home
      Targets (6): sacks, interceptions, fumble_recoveries, defensive_tds, safeties, points_allowed
      Scoring-only stats: return_tds, blocked_kicks
      Schedules context (raw, used by Part 2): spread_line, total_line, team_implied_total,
        opponent_implied_total, team_rest, opponent_rest, temp, wind, roof, div_game
    """
    data_dir = Path(data_dir)

    # 1. Load team_stats (source of defensive/special-teams stats)
    ts = _load_multi_season_parquets(
        data_dir, 'team_stats', 'team_stats_{season}.parquet', seasons
    )
    if ts.empty:
        return pd.DataFrame()

    # Derive target + scoring-only columns from team_stats.
    # CRITICAL: fumble_recoveries = team_stats.fumble_recovery_opp
    # (def_fumbles counts defenders who themselves fumbled — e.g. after returning
    # an INT — not recoveries. League sums confirmed: def_fumbles=76 vs
    # fumble_recovery_opp=283 matches NFL's ~280 defensive recoveries/season.)
    dst = pd.DataFrame({
        'team': ts['team'],
        'season': ts['season'],
        'week': ts['week'],
        'season_type': ts['season_type'],
        'opponent_team': ts['opponent_team'],
        'sacks': ts['def_sacks'].fillna(0),
        'interceptions': ts['def_interceptions'].fillna(0),
        'fumble_recoveries': ts['fumble_recovery_opp'].fillna(0),
        'defensive_tds': ts['def_tds'].fillna(0),
        'safeties': ts['def_safeties'].fillna(0),
        'return_tds': ts['special_teams_tds'].fillna(0),
    })

    # 2. blocked_kicks: kicks THIS team's defense blocked = opponent's
    # (fg_blocked + pat_blocked) from the opponent's team_stats row for the
    # same (season, week). Self-join on opponent_team.
    opp_blocks = ts[['season', 'week', 'team', 'fg_blocked', 'pat_blocked']].copy()
    opp_blocks['opp_blocked_kicks'] = (
        opp_blocks['fg_blocked'].fillna(0) + opp_blocks['pat_blocked'].fillna(0)
    )
    opp_blocks = opp_blocks[['season', 'week', 'team', 'opp_blocked_kicks']].rename(
        columns={'team': 'opponent_team'}
    )
    dst = dst.merge(opp_blocks, on=['season', 'week', 'opponent_team'], how='left')
    dst['blocked_kicks'] = dst['opp_blocked_kicks'].fillna(0)
    dst = dst.drop(columns=['opp_blocked_kicks'])

    # 3. Schedules: points_allowed + context (Vegas/weather/rest).
    schedules = _load_multi_season_parquets(
        data_dir, 'schedules', 'schedules_{season}.parquet', seasons
    )
    if not schedules.empty:
        home_games = schedules[[
            'season', 'week', 'home_team', 'away_team', 'home_score', 'away_score',
            'spread_line', 'total_line', 'home_implied_total', 'away_implied_total',
            'home_rest', 'away_rest', 'temp', 'wind', 'roof', 'div_game',
        ]].copy()
        home_games['is_home'] = 1
        home_games = home_games.rename(columns={
            'home_team': 'team',
            'away_team': '_opp',
            'away_score': 'points_allowed',
            'home_score': '_team_score',
            'home_implied_total': 'team_implied_total',
            'away_implied_total': 'opponent_implied_total',
            'home_rest': 'team_rest',
            'away_rest': 'opponent_rest',
        })

        away_games = schedules[[
            'season', 'week', 'home_team', 'away_team', 'home_score', 'away_score',
            'spread_line', 'total_line', 'home_implied_total', 'away_implied_total',
            'home_rest', 'away_rest', 'temp', 'wind', 'roof', 'div_game',
        ]].copy()
        away_games['is_home'] = 0
        # Flip spread sign for away team (spread_line is home-team-relative).
        away_games['spread_line'] = -away_games['spread_line']
        away_games = away_games.rename(columns={
            'away_team': 'team',
            'home_team': '_opp',
            'home_score': 'points_allowed',
            'away_score': '_team_score',
            'away_implied_total': 'team_implied_total',
            'home_implied_total': 'opponent_implied_total',
            'away_rest': 'team_rest',
            'home_rest': 'opponent_rest',
        })

        all_games = pd.concat([home_games, away_games], ignore_index=True)
        all_games = all_games.drop(columns=['_opp', '_team_score'])

        dst = dst.merge(all_games, on=['season', 'week', 'team'], how='left')

    # Order columns for stability.
    col_order = [
        'team', 'season', 'week', 'season_type', 'opponent_team', 'is_home',
        # targets
        'sacks', 'interceptions', 'fumble_recoveries', 'defensive_tds',
        'safeties', 'points_allowed',
        # scoring-only
        'return_tds', 'blocked_kicks',
        # schedules context
        'spread_line', 'total_line', 'team_implied_total', 'opponent_implied_total',
        'team_rest', 'opponent_rest', 'temp', 'wind', 'roof', 'div_game',
    ]
    col_order = [c for c in col_order if c in dst.columns]
    dst = dst[col_order]

    dst = dst.sort_values(['team', 'season', 'week']).reset_index(drop=True)
    return dst


# ============================================================
# Part 2 (subagent 2): rolling, opponent-offense, context, orchestrator appended below
# ============================================================


def add_dst_rolling(df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """Add rolling_avg/variance/trend per CORE_DST_STATS_FOR_ROLLING.
    Decay-weighted (0.85), 6-game window, strictly prior weeks.
    Cross-season: groupby team only (warm-up convention from rolling.py).
    Also adds games_of_history per team (cumcount within team).
    """
    stats = [s for s in CORE_DST_STATS_FOR_ROLLING if s in df.columns]

    df = df.sort_values(['team', 'season', 'week']).reset_index(drop=True)

    # games_of_history: count of prior games per team across all seasons.
    df['games_of_history'] = df.groupby('team', sort=False).cumcount()

    # INTENTIONAL: group by team only (not [team, season]).
    # V5_ROADMAP specifies 2018-2019 as warm-up seasons so that Week 1 of
    # 2020 has a full rolling lookback window from 2019 tail games. This
    # cross-season history is desired throughout training (2020-2025) so
    # that Week 1 of each new season uses the prior season's tail, not
    # empty history. See docs/V5_ROADMAP.md "Season Range: 2018-2025".
    # Mirrors rolling.py (player pipeline).
    #
    # Position-safe: groupby().transform() preserves the original index
    # regardless of group iteration order — eliminates the silent-corruption
    # risk of the prior list-append + bulk-assign pattern.
    grouped = df.groupby('team', sort=False)
    for stat in stats:
        col = grouped[stat]
        df[f'rolling_avg_{stat}'] = col.transform(rolling_decay_avg_series, window=window)
        df[f'variance_{stat}'] = col.transform(rolling_variance_series, window=window)
        df[f'trend_{stat}'] = col.transform(rolling_trend_series, window=window)

    return df


def add_dst_opponent_offense(
    df: pd.DataFrame,
    team_stats: pd.DataFrame | None = None,
    data_dir=None,
    seasons=None,
    window: int = ROLLING_WINDOW,
) -> pd.DataFrame:
    """Add rolling features describing the opponent's offense quality.

    For each row, computes the opponent's prior-week rolling averages of:
      - opp_rolling_avg_off_yards   (passing_yards + rushing_yards)
      - opp_rolling_avg_off_tds     (passing_tds + rushing_tds)
      - opp_rolling_avg_off_turnovers (passing_interceptions
                                       + rushing_fumbles_lost
                                       + sack_fumbles_lost)

    Implementation: build a small offense-rolling table per team-week, then
    merge onto df via opponent_team. Uses the same cross-season groupby+shift
    pattern as rolling.py (warm-up convention).

    Pass team_stats DataFrame OR (data_dir, seasons) so the function can
    re-load it. team_stats must include season, week, team, season_type
    plus the offense raw columns.
    """
    if team_stats is None:
        if data_dir is None or seasons is None:
            raise ValueError(
                "Either team_stats or (data_dir, seasons) must be provided."
            )
        team_stats = _load_multi_season_parquets(
            Path(data_dir), 'team_stats', 'team_stats_{season}.parquet', seasons
        )

    if team_stats.empty:
        for c in ['opp_rolling_avg_off_yards',
                  'opp_rolling_avg_off_tds',
                  'opp_rolling_avg_off_turnovers']:
            df[c] = np.nan
        return df

    # Defensive column lookup: DataFrame.get(col, 0) returns the int 0 when
    # the column is missing, and int has no .fillna — so we can't chain the
    # original pattern. This helper returns a zero-filled Series instead.
    def _col(df, name):
        return df[name].fillna(0) if name in df.columns else pd.Series(0, index=df.index)

    off = pd.DataFrame({
        'team': team_stats['team'],
        'season': team_stats['season'],
        'week': team_stats['week'],
        'off_yards': _col(team_stats, 'passing_yards') + _col(team_stats, 'rushing_yards'),
        'off_tds': _col(team_stats, 'passing_tds') + _col(team_stats, 'rushing_tds'),
        'off_turnovers': (
            _col(team_stats, 'passing_interceptions')
            + _col(team_stats, 'rushing_fumbles_lost')
            + _col(team_stats, 'sack_fumbles_lost')
        ),
    })

    off = off.sort_values(['team', 'season', 'week']).reset_index(drop=True)

    # Rolling per team across seasons (cross-season warm-up).
    # Position-safe: transform preserves original index regardless of group order.
    grouped = off.groupby('team', sort=False)
    for stat in ['off_yards', 'off_tds', 'off_turnovers']:
        off[f'opp_rolling_avg_{stat}'] = grouped[stat].transform(
            rolling_decay_avg_series, window=window
        )

    # Merge onto df via opponent_team. Match on (season, week, opponent_team).
    merge_cols = ['season', 'week', 'team',
                  'opp_rolling_avg_off_yards',
                  'opp_rolling_avg_off_tds',
                  'opp_rolling_avg_off_turnovers']
    opp_lookup = off[merge_cols].rename(columns={'team': 'opponent_team'})
    df = df.merge(opp_lookup, on=['season', 'week', 'opponent_team'], how='left')

    return df


def add_dst_context(df: pd.DataFrame) -> pd.DataFrame:
    """Add Vegas/weather features:
      - game_script_index = -spread_line
      - is_dome (roof in {'dome', 'closed'})
      - is_high_wind (wind >= 15)
      - is_cold (temp <= 32)

    Note: thresholds match context.py's player-side semantics (>15 / <40 in
    player code is for general weather effect; here the spec asks for the
    DST-specific thresholds wind>=15 and temp<=32 — the harsher cutoffs that
    impact defensive scoring more directly).
    """
    if 'spread_line' in df.columns:
        df['game_script_index'] = -df['spread_line'].fillna(0)
    else:
        df['game_script_index'] = 0.0

    if 'roof' in df.columns:
        df['is_dome'] = df['roof'].isin(['dome', 'closed']).astype(int)
    else:
        df['is_dome'] = 0

    if 'wind' in df.columns:
        df['is_high_wind'] = (df['wind'].fillna(0) >= 15).astype(int)
    else:
        df['is_high_wind'] = 0

    if 'temp' in df.columns:
        # Dome games typically have NaN temp — treat as not cold.
        df['is_cold'] = (df['temp'].fillna(99) <= 32).astype(int)
    else:
        df['is_cold'] = 0

    return df


def build_dst_features(data_dir, seasons, output_dir=None, verbose=True) -> pd.DataFrame:
    """Orchestrate DST feature build for given seasons.

    Pipeline:
      1. build_master_dst_table (REG + POST rows)
      2. add_dst_rolling
      3. add_dst_opponent_offense (re-loads team_stats)
      4. add_dst_context
      5. Filter to season_type == 'REG' BEFORE writing
      6. If output_dir, save per-season Parquet to
         {output_dir}/v5/features_dst_{season}.parquet

    POST rows feed rolling history (so REG W1 of next season has lookback)
    but are filtered from the returned/output DataFrame.

    Returns the filtered (REG-only) DataFrame.
    """
    if verbose:
        print(f"V5 DST feature engineering: seasons "
              f"{min(seasons)}-{max(seasons)}")

    df = build_master_dst_table(data_dir, seasons)
    if verbose:
        print(f"  Master DST table: {len(df):,} rows "
              f"(includes REG+POST for rolling history)")

    if df.empty:
        return df

    df = add_dst_rolling(df)
    if verbose:
        print(f"  After rolling: {len(df.columns)} columns")

    df = add_dst_opponent_offense(df, data_dir=data_dir, seasons=seasons)
    if verbose:
        print(f"  After opp-offense: {len(df.columns)} columns")

    df = add_dst_context(df)
    if verbose:
        print(f"  After context: {len(df.columns)} columns")

    # Filter POST rows BEFORE writing.
    df = df[df['season_type'] == 'REG'].reset_index(drop=True)
    if verbose:
        print(f"  Filtered to REG: {len(df):,} rows, {len(df.columns)} columns")

    if output_dir:
        out = Path(output_dir) / VERSION
        out.mkdir(parents=True, exist_ok=True)
        for season, season_df in df.groupby('season'):
            path = out / f'features_dst_{season}.parquet'
            season_df.to_parquet(path)
            if verbose:
                print(f"  Saved {path} ({len(season_df):,} rows)")

    return df
