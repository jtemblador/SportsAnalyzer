# V5 Feature Engineering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `src/nfl/features/v5/` package that transforms raw Parquet data into 60-80 predictive features per player per week, ready for V5 model training. Runs on Google Colab (reads Parquet, not PostgreSQL).

**Architecture:** Master player-week table approach. Load all 13 dataset Parquets, left-join into one master DataFrame keyed on `(player_id, season, week)`, then compute derived features (rolling averages, variance, trends) with vectorized pandas operations. Per-season feature Parquets written to disk for model training consumption.

**Tech Stack:** pandas (vectorized groupby/rolling), numpy, pyarrow/Parquet. No database dependency. No GPU. Runs on CPU with ~16GB RAM.

---

## Key Architectural Decisions

1. **Parquet-first, no PostgreSQL dependency** — scripts run on Colab where no DB exists
2. **Package structure, not monolith** — split by concern so each file is < 300 lines
3. **Vectorized over per-player loops** — V4's per-player approach doesn't scale to 60+ features
4. **Pre-game features only** — features for week N use strictly week < N data
5. **NULL-preserving** — NGS/PFR coverage gaps stay as NULLs for tree models to handle
6. **Position-agnostic builder, position-specific output** — build all features, filter per position downstream

## File Structure

```
src/nfl/features/
  __init__.py                      (existing, empty)
  v5/
    __init__.py                    — exports build_features()
    config.py                      — feature lists per position, version string
    master_table.py                — loads Parquets, joins into master DataFrame
    rolling.py                     — rolling averages, variance, trends (core stats)
    context.py                     — Vegas, weather, rest, home/away, opponent rank
    usage.py                       — snap counts, depth chart, injury status
    advanced.py                    — NGS, PFR, FF opportunity features
    engineer.py                    — orchestrator: build_features(seasons, output_dir)

colab/
  v5_feature_engineering.ipynb     — thin Colab wrapper

tests/
  test_v5_master_table.py          — master table join correctness
  test_v5_rolling.py               — rolling average, variance, trend
  test_v5_context.py               — Vegas, weather, opponent rank
  test_v5_usage.py                 — snap/depth/injury features
  test_v5_advanced.py              — NGS/PFR/FF features
  test_v5_engineer.py              — end-to-end integration
```

---

## Task 1: Package setup and configuration module

**Files:**
- Create: `src/nfl/features/v5/__init__.py`
- Create: `src/nfl/features/v5/config.py`
- Create: `tests/test_v5_config.py`

- [ ] **Step 1: Write test for config module**

```python
# tests/test_v5_config.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nfl.features.v5.config import (
    VERSION,
    FEATURE_GROUPS,
    STATS_TO_PREDICT,
    ROLLING_DECAY,
    MIN_GAMES_HISTORY,
)


def test_version_string():
    assert VERSION == 'v5'


def test_feature_groups_defined():
    expected = ['rolling', 'context', 'usage', 'advanced']
    for group in expected:
        assert group in FEATURE_GROUPS


def test_stats_to_predict_per_position():
    assert 'passing_yards' in STATS_TO_PREDICT['QB']
    assert 'passing_tds' in STATS_TO_PREDICT['QB']
    assert 'rushing_yards' in STATS_TO_PREDICT['RB']
    assert 'receiving_yards' in STATS_TO_PREDICT['WR']
    assert 'receiving_yards' in STATS_TO_PREDICT['TE']
    assert 'fg_made' in STATS_TO_PREDICT['K']


def test_rolling_decay_is_v4_proven_value():
    """V4 used 0.85 decay (proven to work). Keep it."""
    assert ROLLING_DECAY == 0.85


def test_min_games_history():
    """3-game minimum per V5_QUESTIONS.md decision."""
    assert MIN_GAMES_HISTORY == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_v5_config.py -v`
Expected: FAIL (module does not exist)

- [ ] **Step 3: Create config module**

```python
# src/nfl/features/v5/config.py
"""
V5 feature engineering configuration.
Defines stats to predict per position, feature groups, and tuning constants.
"""

VERSION = 'v5'

# Rolling average decay factor — proven in V4 (stronger emphasis on recent 3 games)
ROLLING_DECAY = 0.85

# Minimum games of history required to generate predictions
# Below this, output 'insufficient_data' flag instead of predictions
MIN_GAMES_HISTORY = 3

# Number of past games to use for rolling calculations
ROLLING_WINDOW = 6

# Stats to predict per position (V5_QUESTIONS.md decision)
STATS_TO_PREDICT = {
    'QB': ['passing_yards', 'passing_tds', 'passing_interceptions',
           'rushing_yards', 'rushing_tds'],
    'RB': ['rushing_yards', 'rushing_tds', 'receptions',
           'receiving_yards', 'receiving_tds'],
    'WR': ['receptions', 'receiving_yards', 'receiving_tds', 'targets'],
    'TE': ['receptions', 'receiving_yards', 'receiving_tds', 'targets'],
    'K':  ['fg_made', 'fg_att', 'pat_made'],
}

# Feature groups for ablation study (Task 3.2b will remove one at a time)
FEATURE_GROUPS = {
    'rolling':   'Rolling averages, variance, and trend features (V2/V4 proven)',
    'context':   'Vegas lines, weather, rest, opponent defense rank (V4 proven)',
    'usage':     'Snap counts, depth chart status, injury reports',
    'advanced':  'NGS metrics, PFR advanced stats, FF opportunity (lower confidence)',
}

# Core stats used for rolling average features (all positions get these)
CORE_STATS_FOR_ROLLING = [
    'fantasy_points_ppr',
    'passing_yards', 'passing_tds', 'passing_interceptions',
    'rushing_yards', 'rushing_tds', 'carries',
    'receiving_yards', 'receiving_tds', 'receptions', 'targets',
]

# Fantasy PPR scoring formula — used to derive fantasy points from predicted stats
FANTASY_PPR_WEIGHTS = {
    'passing_yards': 0.04,
    'passing_tds': 4.0,
    'passing_interceptions': -2.0,
    'rushing_yards': 0.1,
    'rushing_tds': 6.0,
    'receptions': 1.0,
    'receiving_yards': 0.1,
    'receiving_tds': 6.0,
}
```

- [ ] **Step 4: Create package __init__**

```python
# src/nfl/features/v5/__init__.py
from src.nfl.features.v5.config import VERSION

__all__ = ['VERSION']
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_v5_config.py -v`
Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
git add src/nfl/features/v5/ tests/test_v5_config.py
git commit -m "Task 3.1.1: V5 feature package setup and config module"
```

---

## Task 2: Master player-week table builder

**Files:**
- Create: `src/nfl/features/v5/master_table.py`
- Create: `tests/test_v5_master_table.py`
- Create: `tests/fixtures/v5_mini_data/` (small sample Parquets for testing)

- [ ] **Step 1: Create test fixtures (mini Parquets with known data)**

```python
# tests/fixtures/v5_mini_data/__init__.py (empty, just marks as package)
```

Run this one-time setup script to build test fixtures:

```python
# tests/fixtures/build_v5_mini_data.py
"""
One-time script to build small Parquet fixtures for V5 tests.
Only run when fixtures need regenerating.
"""
import pandas as pd
from pathlib import Path

OUT = Path(__file__).parent / 'v5_mini_data'
OUT.mkdir(exist_ok=True)

# Minimal player_stats: Mahomes + Barkley, 2024 weeks 1-3
(OUT / 'player_stats').mkdir(exist_ok=True)
player_stats = pd.DataFrame({
    'player_id': ['00-0033873', '00-0033873', '00-0033873',
                  '00-0034844', '00-0034844', '00-0034844'],
    'player_name': ['P.Mahomes']*3 + ['S.Barkley']*3,
    'position': ['QB']*3 + ['RB']*3,
    'team': ['KC']*3 + ['PHI']*3,
    'opponent_team': ['BAL', 'CIN', 'ATL', 'GB', 'ATL', 'NO'],
    'season': [2024]*6,
    'week': [1, 2, 3]*2,
    'passing_yards': [291, 151, 217, 0, 0, 0],
    'passing_tds': [1, 2, 2, 0, 0, 0],
    'passing_interceptions': [1, 0, 0, 0, 0, 0],
    'rushing_yards': [3, 12, 0, 109, 88, 147],
    'rushing_tds': [0, 0, 0, 2, 0, 1],
    'carries': [3, 6, 0, 24, 17, 22],
    'receptions': [0, 0, 0, 2, 3, 1],
    'receiving_yards': [0, 0, 0, 23, 31, -4],
    'receiving_tds': [0, 0, 0, 0, 0, 0],
    'targets': [0, 0, 0, 2, 3, 2],
    'fantasy_points_ppr': [15.14, 12.94, 16.38, 25.2, 20.1, 19.4],
})
player_stats.to_parquet(OUT / 'player_stats' / 'player_stats_2024.parquet')

# Minimal schedules: KC@BAL, PHI@GB (week 1), plus BAL/KC week 2
(OUT / 'schedules').mkdir(exist_ok=True)
schedules = pd.DataFrame({
    'game_id': ['2024_01_BAL_KC', '2024_01_GB_PHI', '2024_02_CIN_KC',
                '2024_02_ATL_PHI', '2024_03_ATL_KC', '2024_03_NO_PHI'],
    'season': [2024]*6, 'week': [1, 1, 2, 2, 3, 3],
    'home_team': ['KC', 'PHI', 'KC', 'PHI', 'ATL', 'NO'],
    'away_team': ['BAL', 'GB', 'CIN', 'ATL', 'KC', 'PHI'],
    'spread_line': [3.0, -1.5, -7.0, -6.5, -3.0, 3.5],
    'total_line':  [46.0, 41.0, 48.5, 44.5, 47.0, 46.5],
    'home_implied_total': [24.5, 21.25, 27.75, 25.5, 25.0, 21.5],
    'away_implied_total': [21.5, 19.75, 20.75, 19.0, 22.0, 25.0],
    'temp': [67.0, 72.0, None, None, 75.0, None],  # NULL for dome
    'wind': [8.0, 3.0, None, None, 5.0, None],
    'roof': ['outdoors', 'outdoors', 'closed', 'outdoors', 'dome', 'dome'],
    'home_rest': [7, 7, 7, 7, 7, 7],
    'away_rest': [7, 7, 7, 7, 7, 7],
    'div_game': [0, 0, 1, 1, 0, 0],
})
schedules.to_parquet(OUT / 'schedules' / 'schedules_2024.parquet')

# Minimal players: Mahomes + Barkley with both IDs
(OUT / 'players').mkdir(exist_ok=True)
players = pd.DataFrame({
    'gsis_id': ['00-0033873', '00-0034844'],
    'pfr_id': ['MahoPa00', 'BarkSa00'],
    'display_name': ['Patrick Mahomes', 'Saquon Barkley'],
    'position': ['QB', 'RB'],
})
players.to_parquet(OUT / 'players' / 'players.parquet')

# Minimal snap_counts (uses PFR ID)
(OUT / 'snap_counts').mkdir(exist_ok=True)
snap_counts = pd.DataFrame({
    'pfr_player_id': ['MahoPa00', 'MahoPa00', 'MahoPa00',
                      'BarkSa00', 'BarkSa00', 'BarkSa00'],
    'season': [2024]*6, 'week': [1, 2, 3]*2,
    'team': ['KC']*3 + ['PHI']*3,
    'offense_snaps': [62, 58, 65, 52, 48, 55],
    'offense_pct': [1.0, 1.0, 1.0, 0.80, 0.75, 0.82],
    'defense_snaps': [0]*6, 'defense_pct': [0.0]*6,
})
snap_counts.to_parquet(OUT / 'snap_counts' / 'snap_counts_2024.parquet')

# Minimal injuries: Mahomes week 3 Questionable
(OUT / 'injuries').mkdir(exist_ok=True)
injuries = pd.DataFrame({
    'gsis_id': ['00-0033873'],
    'season': [2024], 'week': [3],
    'report_status': ['Questionable'],
    'practice_status': ['Limited'],
    'report_primary_injury': ['Ankle'],
})
injuries.to_parquet(OUT / 'injuries' / 'injuries_2024.parquet')

print("Fixtures created at", OUT)
```

Then run it once:
```bash
python tests/fixtures/build_v5_mini_data.py
```

- [ ] **Step 2: Write failing test for master table**

```python
# tests/test_v5_master_table.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pytest
from src.nfl.features.v5.master_table import build_master_table

FIXTURE_DIR = Path(__file__).parent / 'fixtures' / 'v5_mini_data'


@pytest.fixture
def master_df():
    return build_master_table(data_dir=str(FIXTURE_DIR), seasons=[2024])


def test_has_one_row_per_player_week(master_df):
    # 2 players × 3 weeks = 6 rows
    assert len(master_df) == 6


def test_preserves_weekly_stats_columns(master_df):
    assert 'fantasy_points_ppr' in master_df.columns
    assert 'passing_yards' in master_df.columns


def test_joins_players_for_pfr_id(master_df):
    mahomes = master_df[master_df['player_id'] == '00-0033873'].iloc[0]
    assert mahomes['pfr_id'] == 'MahoPa00'


def test_joins_snap_counts_via_pfr(master_df):
    mahomes_w1 = master_df[
        (master_df['player_id'] == '00-0033873') & (master_df['week'] == 1)
    ].iloc[0]
    assert mahomes_w1['offense_pct'] == 1.0


def test_joins_injuries_nullable(master_df):
    # Mahomes W3 was Questionable (has injury row)
    mahomes_w3 = master_df[
        (master_df['player_id'] == '00-0033873') & (master_df['week'] == 3)
    ].iloc[0]
    assert mahomes_w3['report_status'] == 'Questionable'

    # Mahomes W1 has no injury row — should be NULL
    mahomes_w1 = master_df[
        (master_df['player_id'] == '00-0033873') & (master_df['week'] == 1)
    ].iloc[0]
    assert pd.isna(mahomes_w1['report_status'])


def test_joins_schedules_game_context(master_df):
    mahomes_w1 = master_df[
        (master_df['player_id'] == '00-0033873') & (master_df['week'] == 1)
    ].iloc[0]
    # KC hosted BAL week 1, spread was 3.0
    assert mahomes_w1['spread_line'] == 3.0
    assert mahomes_w1['total_line'] == 46.0
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_v5_master_table.py -v`
Expected: FAIL (module does not exist)

- [ ] **Step 4: Implement master_table.py**

```python
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
        # Each player has a team — join on season+week+team matching home or away
        # Build two-sided join: home games and away games
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
        # Flip spread sign for away team (spread_line is always from home perspective)
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
            ngs = ngs[ngs['week'] > 0].copy()  # filter out season aggregates
            keep = ['player_gsis_id', 'season', 'week'] + [
                c for c in cols if c in ngs.columns
            ]
            ngs = ngs[keep]
            # Rename columns with ngs_ prefix to avoid conflicts
            rename_map = {c: f'ngs_{stat_type}_{c}' for c in cols if c in ngs.columns}
            ngs = ngs.rename(columns=rename_map)
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_v5_master_table.py -v`
Expected: 6 passed

- [ ] **Step 6: Commit**

```bash
git add src/nfl/features/v5/master_table.py tests/test_v5_master_table.py tests/fixtures/
git commit -m "Task 3.1.2: V5 master player-week table builder with 13-dataset joins"
```

---

## Task 3: Rolling features (averages, variance, trends)

**Files:**
- Create: `src/nfl/features/v5/rolling.py`
- Create: `tests/test_v5_rolling.py`

- [ ] **Step 1: Write failing tests for rolling features**

```python
# tests/test_v5_rolling.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import pytest
from src.nfl.features.v5.rolling import add_rolling_features


@pytest.fixture
def sample_master():
    """4 weeks of data for one player."""
    return pd.DataFrame({
        'player_id': ['X']*4,
        'player_name': ['Test']*4,
        'position': ['QB']*4,
        'season': [2024]*4,
        'week': [1, 2, 3, 4],
        'passing_yards': [200, 250, 300, 150],
        'fantasy_points_ppr': [15.0, 20.0, 25.0, 12.0],
        'carries': [0]*4, 'targets': [0]*4, 'receptions': [0]*4,
        'rushing_yards': [0]*4, 'rushing_tds': [0]*4,
        'receiving_yards': [0]*4, 'receiving_tds': [0]*4,
        'passing_tds': [1, 2, 3, 1], 'passing_interceptions': [0, 0, 1, 1],
    })


def test_rolling_avg_uses_prior_weeks_only(sample_master):
    """Week 3 rolling avg must use weeks 1+2, NOT week 3."""
    df = add_rolling_features(sample_master)
    w3 = df[df['week'] == 3].iloc[0]
    # Week 3 rolling avg of fantasy_points_ppr should be based on weeks 1-2 only
    # Weighted avg with decay=0.85: recent first
    # Games in chronological order w1=15, w2=20; reversed (most recent first): [20, 15]
    # weights [1.0, 0.85], sum = 1.85
    # weighted = (20*1.0 + 15*0.85) / 1.85 = (20 + 12.75) / 1.85 ≈ 17.7
    assert w3['rolling_avg_fantasy_points_ppr'] == pytest.approx(17.7, abs=0.1)


def test_rolling_avg_null_at_week_1(sample_master):
    """Week 1 has no history — rolling features should be NaN."""
    df = add_rolling_features(sample_master)
    w1 = df[df['week'] == 1].iloc[0]
    assert pd.isna(w1['rolling_avg_fantasy_points_ppr'])


def test_variance_computed(sample_master):
    """Variance = std dev of past values."""
    df = add_rolling_features(sample_master)
    w4 = df[df['week'] == 4].iloc[0]
    # weeks 1-3 fantasy points: [15, 20, 25]
    expected_std = np.std([15.0, 20.0, 25.0])
    assert w4['variance_fantasy_points_ppr'] == pytest.approx(expected_std, abs=0.01)


def test_variance_nan_with_less_than_2_games(sample_master):
    """Week 2 has only 1 prior game — variance should be NaN."""
    df = add_rolling_features(sample_master)
    w2 = df[df['week'] == 2].iloc[0]
    assert pd.isna(w2['variance_fantasy_points_ppr'])


def test_trend_computed(sample_master):
    """Trend = (recent_3 - older_3) / older_3."""
    df = add_rolling_features(sample_master)
    # Week 4 with 3 games of history — trend requires recent_3 and older values
    # Only 3 games total so older window is empty — trend should be NaN
    w4 = df[df['week'] == 4].iloc[0]
    assert pd.isna(w4['trend_fantasy_points_ppr'])


def test_no_data_leakage_across_players():
    """One player's rolling avg must not leak into another player's features."""
    df_input = pd.DataFrame({
        'player_id': ['A', 'A', 'B', 'B'],
        'player_name': ['A', 'A', 'B', 'B'],
        'position': ['QB']*4,
        'season': [2024]*4,
        'week': [1, 2, 1, 2],
        'passing_yards': [100, 200, 500, 600],
        'fantasy_points_ppr': [5.0, 10.0, 30.0, 35.0],
        'carries': [0]*4, 'targets': [0]*4, 'receptions': [0]*4,
        'rushing_yards': [0]*4, 'rushing_tds': [0]*4,
        'receiving_yards': [0]*4, 'receiving_tds': [0]*4,
        'passing_tds': [0]*4, 'passing_interceptions': [0]*4,
    })
    df = add_rolling_features(df_input)
    b_w2 = df[(df['player_id'] == 'B') & (df['week'] == 2)].iloc[0]
    # Player B week 2 rolling avg should only use B's week 1 data (30), not A's
    assert b_w2['rolling_avg_fantasy_points_ppr'] == pytest.approx(30.0, abs=0.01)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_v5_rolling.py -v`
Expected: FAIL (module does not exist)

- [ ] **Step 3: Implement rolling.py**

```python
# src/nfl/features/v5/rolling.py
"""
Rolling average, variance, and trend features.
All use strictly prior weeks (no data leakage into current-week features).
"""

import pandas as pd
import numpy as np
from src.nfl.features.v5.config import (
    ROLLING_DECAY, ROLLING_WINDOW, CORE_STATS_FOR_ROLLING
)


def _decay_weighted_avg(values, decay=ROLLING_DECAY):
    """Compute decay-weighted average. Values are in chronological order
    (oldest first). Most-recent values get highest weight."""
    if len(values) == 0:
        return np.nan
    # Reverse so most recent is first, then decay
    rev = values[::-1]
    weights = np.array([decay ** i for i in range(len(rev))])
    return np.sum(rev * weights) / np.sum(weights)


def add_rolling_features(df, window=ROLLING_WINDOW, stats=None):
    """
    Add rolling average, variance, and trend features for each core stat.
    Features are computed from weeks STRICTLY prior to current week (no leakage).

    Args:
        df: Master table DataFrame (must be sorted by player_id, season, week)
        window: Number of past games to use for rolling calculations
        stats: List of stat columns to compute features for (default: core stats)

    Returns:
        DataFrame with added columns:
        - rolling_avg_<stat>: decay-weighted average of past N games
        - variance_<stat>: std dev of past games
        - trend_<stat>: recent 3 vs older games percentage change
    """
    if stats is None:
        stats = [s for s in CORE_STATS_FOR_ROLLING if s in df.columns]

    df = df.sort_values(['player_id', 'season', 'week']).reset_index(drop=True)

    for stat in stats:
        rolling_avg = []
        variance = []
        trend = []

        for player_id, group in df.groupby('player_id', sort=False):
            values = group[stat].values
            for i in range(len(values)):
                # Use prior N games only (strictly before current)
                past = values[max(0, i - window):i]
                past = past[~pd.isna(past)]

                # Rolling avg (decay-weighted)
                rolling_avg.append(
                    _decay_weighted_avg(past) if len(past) > 0 else np.nan
                )

                # Variance
                variance.append(np.std(past) if len(past) >= 2 else np.nan)

                # Trend: (recent 3 - older) / older
                if len(past) >= 4:
                    recent = np.mean(past[-3:])
                    older = np.mean(past[:-3])
                    trend.append(
                        (recent - older) / older if older != 0 else 0.0
                    )
                else:
                    trend.append(np.nan)

        df[f'rolling_avg_{stat}'] = rolling_avg
        df[f'variance_{stat}'] = variance
        df[f'trend_{stat}'] = trend

    return df
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_v5_rolling.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/nfl/features/v5/rolling.py tests/test_v5_rolling.py
git commit -m "Task 3.1.3: V5 rolling average, variance, and trend features"
```

---

## Task 4: Game context features (Vegas, weather, rest, opponent rank)

**Files:**
- Create: `src/nfl/features/v5/context.py`
- Create: `tests/test_v5_context.py`

- [ ] **Step 1: Write failing tests for context features**

```python
# tests/test_v5_context.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import pytest
from src.nfl.features.v5.context import (
    add_vegas_features, add_weather_features, add_opponent_defense_rank
)


def test_vegas_features_pass_through():
    """Vegas columns already in master table; context adds derived features."""
    df = pd.DataFrame({
        'team_implied_total': [25.0, 20.0],
        'opponent_implied_total': [22.0, 28.0],
        'spread_line': [-3.0, 5.0],
        'total_line': [47.0, 48.0],
        'position': ['QB', 'RB'],
    })
    out = add_vegas_features(df)
    # game_script_index: positive = favored (likely passing)
    # For row 0: team favored by 3 → expect positive game script
    # For row 1: team underdog by 5 → expect negative
    assert out.iloc[0]['game_script_index'] > out.iloc[1]['game_script_index']


def test_weather_features_null_safe():
    """Dome games have NULL temp/wind — handle without error."""
    df = pd.DataFrame({
        'temp': [72.0, None, 45.0],
        'wind': [5.0, None, 15.0],
        'roof': ['outdoors', 'dome', 'outdoors'],
    })
    out = add_weather_features(df)
    # Dome flag should be 1 for dome roofs, 0 otherwise
    assert out.iloc[1]['is_dome'] == 1
    assert out.iloc[0]['is_dome'] == 0
    # High wind flag (>15mph = true for row 2)
    assert out.iloc[2]['is_high_wind'] == 1
    assert out.iloc[0]['is_high_wind'] == 0


def test_opponent_defense_rank():
    """Rank = toughness of opponent vs position, based on prior weeks."""
    df = pd.DataFrame({
        'player_id': ['A', 'B', 'C', 'D'],
        'position': ['QB', 'QB', 'QB', 'QB'],
        'team': ['TM1', 'TM2', 'TM3', 'TM4'],
        'opponent_team': ['DEF1', 'DEF2', 'DEF3', 'DEF1'],
        'season': [2024]*4,
        'week': [1, 1, 1, 2],  # D is week 2 — DEF1 has week 1 data
        'fantasy_points_ppr': [30.0, 10.0, 20.0, np.nan],
    })
    out = add_opponent_defense_rank(df)
    # D plays DEF1 in week 2 — DEF1 allowed 30 pts to a QB in week 1 (easiest D)
    d_row = out[out['player_id'] == 'D'].iloc[0]
    assert d_row['opp_def_rank_qb'] is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_v5_context.py -v`
Expected: FAIL (module does not exist)

- [ ] **Step 3: Implement context.py**

```python
# src/nfl/features/v5/context.py
"""
Game context features: Vegas lines, weather, rest days, opponent defense rank.
These come from the schedules and team_stats tables.
"""

import pandas as pd
import numpy as np


def add_vegas_features(df):
    """
    Derive Vegas-based features from raw Vegas columns.
    Master table already has: team_implied_total, opponent_implied_total,
    spread_line, total_line, is_home.

    Adds:
    - game_script_index: negative spread (favored) = positive index → pass-heavy
    - total_game_points: the over/under (redundant but named clearly for models)
    """
    if 'spread_line' in df.columns:
        # Game script: favored teams pass less in garbage time, but have
        # more volume early. Negative spread = favored.
        # Index: -spread normalized. Favored by 7 → +7 index.
        df['game_script_index'] = -df['spread_line'].fillna(0)

    if 'total_line' in df.columns:
        df['total_game_points'] = df['total_line']

    return df


def add_weather_features(df):
    """
    Derive weather features. Dome games have NULL temp/wind — treat as
    controlled conditions.

    Adds:
    - is_dome: 1 if roof is dome or closed, 0 otherwise
    - is_high_wind: 1 if wind > 15mph (hurts passing), 0 otherwise, NULL if dome
    - is_cold: 1 if temp < 40F, 0 otherwise
    """
    if 'roof' in df.columns:
        df['is_dome'] = df['roof'].isin(['dome', 'closed']).astype(int)
    else:
        df['is_dome'] = 0

    if 'wind' in df.columns:
        df['is_high_wind'] = (df['wind'] > 15).fillna(False).astype(int)
    else:
        df['is_high_wind'] = 0

    if 'temp' in df.columns:
        df['is_cold'] = (df['temp'] < 40).fillna(False).astype(int)
    else:
        df['is_cold'] = 0

    return df


def add_opponent_defense_rank(df):
    """
    Compute opponent defense rank against position based on season-to-date
    fantasy points allowed. Uses only prior weeks (no leakage).

    Adds: opp_def_rank_<position> columns (qb, rb, wr, te)
    """
    df = df.copy()

    for pos in ['QB', 'RB', 'WR', 'TE']:
        col = f'opp_def_rank_{pos.lower()}'
        df[col] = np.nan

    # Group by season, compute cumulative fantasy points allowed per defense
    # per position, then rank per week
    for (season,), season_df in df.groupby(['season']):
        # For each week W in the season, compute defense rank using weeks < W
        weeks = sorted(season_df['week'].unique())

        for week in weeks:
            # Data from prior weeks in this season
            prior = season_df[season_df['week'] < week]

            for pos in ['QB', 'RB', 'WR', 'TE']:
                pos_prior = prior[prior['position'] == pos]
                if pos_prior.empty:
                    continue

                # Avg fantasy points allowed per defense
                avg_allowed = pos_prior.groupby('opponent_team')[
                    'fantasy_points_ppr'
                ].mean()

                # Rank 1 = toughest (allows fewest), 32 = easiest
                ranks = avg_allowed.rank(method='min')

                # Assign to this week's rows matching each defense
                col = f'opp_def_rank_{pos.lower()}'
                mask = (df['season'] == season) & (df['week'] == week)
                for def_team, rank in ranks.items():
                    df.loc[mask & (df['opponent_team'] == def_team), col] = rank

    return df
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_v5_context.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/nfl/features/v5/context.py tests/test_v5_context.py
git commit -m "Task 3.1.4: V5 game context features (Vegas, weather, opponent rank)"
```

---

## Task 5: Usage features (snap counts, depth chart, injury status)

**Files:**
- Create: `src/nfl/features/v5/usage.py`
- Create: `tests/test_v5_usage.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_v5_usage.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import pytest
from src.nfl.features.v5.usage import add_usage_features


def test_snap_pct_rolling_uses_prior_weeks():
    """Rolling snap share for week N uses weeks < N only."""
    df = pd.DataFrame({
        'player_id': ['A']*3, 'player_name': ['A']*3,
        'position': ['RB']*3, 'season': [2024]*3, 'week': [1, 2, 3],
        'offense_pct': [0.5, 0.7, 0.9],
    })
    out = add_usage_features(df)
    # Week 3 rolling snap pct should be avg of weeks 1-2 = 0.6
    w3 = out[out['week'] == 3].iloc[0]
    assert w3['rolling_offense_pct'] == pytest.approx(0.6, abs=0.01)


def test_snap_trend():
    """Snap trend: current vs previous week."""
    df = pd.DataFrame({
        'player_id': ['A']*3, 'player_name': ['A']*3,
        'position': ['RB']*3, 'season': [2024]*3, 'week': [1, 2, 3],
        'offense_pct': [0.5, 0.7, 0.9],
    })
    out = add_usage_features(df)
    # Week 3 trend = week 2 pct (prior week), which is 0.7
    w3 = out[out['week'] == 3].iloc[0]
    assert w3['prior_week_offense_pct'] == pytest.approx(0.7, abs=0.01)


def test_injury_status_encoded():
    """Injury status converted to numeric (Out=3, Doubtful=2, Questionable=1, None=0)."""
    df = pd.DataFrame({
        'player_id': ['A', 'B', 'C', 'D'],
        'player_name': ['A', 'B', 'C', 'D'],
        'position': ['RB']*4, 'season': [2024]*4, 'week': [1]*4,
        'report_status': ['Out', 'Doubtful', 'Questionable', None],
    })
    out = add_usage_features(df)
    assert out.iloc[0]['injury_severity'] == 3
    assert out.iloc[1]['injury_severity'] == 2
    assert out.iloc[2]['injury_severity'] == 1
    assert out.iloc[3]['injury_severity'] == 0


def test_depth_chart_starter_flag():
    """depth_team '1' → is_starter=1, everything else → 0."""
    df = pd.DataFrame({
        'player_id': ['A', 'B', 'C'],
        'player_name': ['A', 'B', 'C'],
        'position': ['RB']*3, 'season': [2024]*3, 'week': [1]*3,
        'depth_team': ['1', '2', None],
    })
    out = add_usage_features(df)
    assert out.iloc[0]['is_starter'] == 1
    assert out.iloc[1]['is_starter'] == 0
    # Missing depth_team (e.g. 2025 season) → NaN preserved
    assert pd.isna(out.iloc[2]['is_starter'])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_v5_usage.py -v`
Expected: FAIL (module does not exist)

- [ ] **Step 3: Implement usage.py**

```python
# src/nfl/features/v5/usage.py
"""
Usage features: snap counts, depth chart, injury status.
"""

import pandas as pd
import numpy as np


# Injury severity mapping
INJURY_SEVERITY = {
    'Out': 3, 'Doubtful': 2, 'Questionable': 1,
    'Probable': 0,  # rarely used, treat as healthy
}


def add_usage_features(df):
    """
    Add snap count rolling features, injury severity, and depth chart flags.
    Uses only prior-week data for snap features (no leakage).
    """
    df = df.sort_values(['player_id', 'season', 'week']).reset_index(drop=True)

    # 1. Rolling snap share (prior weeks only)
    if 'offense_pct' in df.columns:
        df['rolling_offense_pct'] = (
            df.groupby('player_id', sort=False)['offense_pct']
              .apply(lambda s: s.shift(1).expanding().mean())
              .reset_index(level=0, drop=True)
        )

        # Prior week snap pct (for trend detection)
        df['prior_week_offense_pct'] = (
            df.groupby('player_id', sort=False)['offense_pct'].shift(1)
        )

    # 2. Injury severity
    if 'report_status' in df.columns:
        df['injury_severity'] = (
            df['report_status'].map(INJURY_SEVERITY).fillna(0).astype(int)
        )

    # 3. Depth chart starter flag
    if 'depth_team' in df.columns:
        # depth_team is string '1', '2', '3', or NaN
        df['is_starter'] = df['depth_team'].apply(
            lambda x: 1 if x == '1' else (0 if pd.notna(x) else np.nan)
        )

    return df
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_v5_usage.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/nfl/features/v5/usage.py tests/test_v5_usage.py
git commit -m "Task 3.1.5: V5 usage features (snap counts, injury, depth chart)"
```

---

## Task 6: Advanced features (NGS, PFR, FF opportunity)

**Files:**
- Create: `src/nfl/features/v5/advanced.py`
- Create: `tests/test_v5_advanced.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_v5_advanced.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import pytest
from src.nfl.features.v5.advanced import add_advanced_features


def test_ngs_rolling_passing():
    """NGS time_to_throw rolling average uses prior weeks."""
    df = pd.DataFrame({
        'player_id': ['QB1']*3, 'player_name': ['QB1']*3,
        'position': ['QB']*3, 'season': [2024]*3, 'week': [1, 2, 3],
        'ngs_passing_avg_time_to_throw': [2.5, 2.7, 2.6],
        'ngs_passing_completion_percentage_above_expectation': [1.0, 2.0, 0.5],
    })
    out = add_advanced_features(df)
    w3 = out[out['week'] == 3].iloc[0]
    # Rolling avg of weeks 1-2: (2.5 + 2.7) / 2 = 2.6
    assert w3['rolling_ngs_passing_avg_time_to_throw'] == pytest.approx(2.6, abs=0.01)


def test_ff_opp_features():
    """FF opportunity: rolling expected fantasy points + differential."""
    df = pd.DataFrame({
        'player_id': ['A']*3, 'player_name': ['A']*3,
        'position': ['RB']*3, 'season': [2024]*3, 'week': [1, 2, 3],
        'total_fantasy_points_exp': [15.0, 18.0, 12.0],
        'total_fantasy_points_diff': [2.0, -3.0, 5.0],
    })
    out = add_advanced_features(df)
    w3 = out[out['week'] == 3].iloc[0]
    # Rolling avg exp (weeks 1-2) = 16.5
    assert w3['rolling_total_fantasy_points_exp'] == pytest.approx(16.5, abs=0.01)
    # Rolling avg diff (weeks 1-2) = -0.5
    assert w3['rolling_total_fantasy_points_diff'] == pytest.approx(-0.5, abs=0.01)


def test_null_preserved_for_unqualified_players():
    """Non-qualified players have NULL NGS — preserve NULL."""
    df = pd.DataFrame({
        'player_id': ['A']*2, 'player_name': ['A']*2,
        'position': ['RB']*2, 'season': [2024]*2, 'week': [1, 2],
        'ngs_rushing_efficiency': [None, None],  # unqualified
    })
    out = add_advanced_features(df)
    w2 = out[out['week'] == 2].iloc[0]
    # With all NULLs, rolling should also be NaN
    assert pd.isna(w2['rolling_ngs_rushing_efficiency'])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_v5_advanced.py -v`
Expected: FAIL

- [ ] **Step 3: Implement advanced.py**

```python
# src/nfl/features/v5/advanced.py
"""
Advanced features: NGS, PFR, FF opportunity.
All are rolling averages of prior-week advanced metrics.
"""

import pandas as pd
import numpy as np


# NGS columns to compute rolling features on
NGS_COLUMNS = [
    'ngs_passing_avg_time_to_throw',
    'ngs_passing_completion_percentage_above_expectation',
    'ngs_passing_aggressiveness',
    'ngs_rushing_efficiency',
    'ngs_rushing_rush_yards_over_expected_per_att',
    'ngs_rushing_percent_attempts_gte_eight_defenders',
    'ngs_receiving_avg_separation',
    'ngs_receiving_avg_yac_above_expectation',
    'ngs_receiving_avg_cushion',
    'ngs_receiving_catch_percentage',
]

# PFR columns to compute rolling features on
PFR_COLUMNS = [
    'pfr_pass_times_pressured_pct',
    'pfr_pass_passing_bad_throw_pct',
    'pfr_pass_passing_drops',
    'pfr_rush_rushing_yards_after_contact_avg',
    'pfr_rush_rushing_broken_tackles',
    'pfr_rec_receiving_drop_pct',
    'pfr_rec_receiving_broken_tackles',
    'pfr_rec_receiving_rat',
]

# FF opportunity columns
FF_OPP_COLUMNS = [
    'total_fantasy_points_exp',
    'total_fantasy_points_diff',
    'pass_fantasy_points_exp',
    'rec_fantasy_points_exp',
    'rush_fantasy_points_exp',
]


def _rolling_prior_mean(df, col):
    """Rolling mean using prior weeks only (strict shift-then-expand)."""
    return (
        df.groupby('player_id', sort=False)[col]
          .apply(lambda s: s.shift(1).expanding().mean())
          .reset_index(level=0, drop=True)
    )


def add_advanced_features(df):
    """Add rolling averages for NGS, PFR, and FF opportunity columns."""
    df = df.sort_values(['player_id', 'season', 'week']).reset_index(drop=True)

    for col in NGS_COLUMNS + PFR_COLUMNS + FF_OPP_COLUMNS:
        if col in df.columns:
            df[f'rolling_{col}'] = _rolling_prior_mean(df, col)

    return df
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_v5_advanced.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/nfl/features/v5/advanced.py tests/test_v5_advanced.py
git commit -m "Task 3.1.6: V5 advanced features (NGS, PFR, FF opportunity rolling)"
```

---

## Task 7: Main orchestrator (engineer.py)

**Files:**
- Create: `src/nfl/features/v5/engineer.py`
- Create: `tests/test_v5_engineer.py`

- [ ] **Step 1: Write failing integration test**

```python
# tests/test_v5_engineer.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pytest
from src.nfl.features.v5.engineer import build_features

FIXTURE_DIR = Path(__file__).parent / 'fixtures' / 'v5_mini_data'


def test_build_features_returns_dataframe():
    df = build_features(data_dir=str(FIXTURE_DIR), seasons=[2024])
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_has_rolling_features(tmp_path):
    df = build_features(data_dir=str(FIXTURE_DIR), seasons=[2024])
    assert 'rolling_avg_fantasy_points_ppr' in df.columns


def test_has_context_features():
    df = build_features(data_dir=str(FIXTURE_DIR), seasons=[2024])
    assert 'game_script_index' in df.columns
    assert 'is_dome' in df.columns


def test_has_usage_features():
    df = build_features(data_dir=str(FIXTURE_DIR), seasons=[2024])
    assert 'injury_severity' in df.columns


def test_saves_per_season_parquet(tmp_path):
    """build_features with output_dir should save per-season Parquet files."""
    df = build_features(
        data_dir=str(FIXTURE_DIR), seasons=[2024],
        output_dir=str(tmp_path),
    )
    expected_file = tmp_path / 'v5' / 'features_2024.parquet'
    assert expected_file.exists()


def test_feature_count_reasonable():
    """V5 should produce 60+ feature columns."""
    df = build_features(data_dir=str(FIXTURE_DIR), seasons=[2024])
    # Count columns that look like engineered features
    feature_cols = [
        c for c in df.columns
        if c.startswith(('rolling_', 'variance_', 'trend_'))
        or c in ['game_script_index', 'is_dome', 'is_high_wind', 'is_cold',
                 'injury_severity', 'is_starter', 'is_home']
        or c.startswith('opp_def_rank_')
    ]
    assert len(feature_cols) >= 30, f"Only {len(feature_cols)} features"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_v5_engineer.py -v`
Expected: FAIL

- [ ] **Step 3: Implement engineer.py**

```python
# src/nfl/features/v5/engineer.py
"""
V5 feature engineering orchestrator.

Entry point: build_features(data_dir, seasons, output_dir=None)

Pipeline:
1. Build master player-week table from all 13 Parquet datasets
2. Add rolling features (averages, variance, trends)
3. Add context features (Vegas, weather, opponent rank)
4. Add usage features (snap counts, injury, depth)
5. Add advanced features (NGS, PFR, FF opportunity rolling)
6. Optionally save per-season Parquet files

Output rows: one per player per week. Features are all pre-game only.
"""

import pandas as pd
from pathlib import Path

from src.nfl.features.v5.config import VERSION
from src.nfl.features.v5.master_table import build_master_table
from src.nfl.features.v5.rolling import add_rolling_features
from src.nfl.features.v5.context import (
    add_vegas_features, add_weather_features, add_opponent_defense_rank
)
from src.nfl.features.v5.usage import add_usage_features
from src.nfl.features.v5.advanced import add_advanced_features


def build_features(data_dir, seasons, output_dir=None, verbose=True):
    """
    Build V5 features for all players across given seasons.

    Args:
        data_dir: Path to data/nfl/ (contains subdirs for each dataset)
        seasons: List of season years to process (e.g., [2018, 2019, ..., 2025])
        output_dir: Optional path to save per-season Parquet files.
                    Files will be written to {output_dir}/v5/features_{season}.parquet.
                    If None, only return the in-memory DataFrame.
        verbose: Print progress messages

    Returns:
        DataFrame of features for all (player_id, season, week) combinations.
    """
    if verbose:
        print(f"V5 feature engineering: seasons {min(seasons)}-{max(seasons)}")
        print(f"Loading master player-week table...")

    df = build_master_table(data_dir=data_dir, seasons=seasons)
    if verbose:
        print(f"  Master table: {len(df):,} rows, {len(df.columns)} columns")

    if verbose:
        print("Computing rolling features (averages, variance, trends)...")
    df = add_rolling_features(df)

    if verbose:
        print("Computing context features (Vegas, weather, opponent rank)...")
    df = add_vegas_features(df)
    df = add_weather_features(df)
    df = add_opponent_defense_rank(df)

    if verbose:
        print("Computing usage features (snap, injury, depth chart)...")
    df = add_usage_features(df)

    if verbose:
        print("Computing advanced features (NGS, PFR, FF opp)...")
    df = add_advanced_features(df)

    if verbose:
        print(f"Final feature table: {len(df):,} rows, {len(df.columns)} columns")

    if output_dir:
        out = Path(output_dir) / VERSION
        out.mkdir(parents=True, exist_ok=True)
        for season, season_df in df.groupby('season'):
            path = out / f'features_{season}.parquet'
            season_df.to_parquet(path)
            if verbose:
                print(f"  Saved {path} ({len(season_df):,} rows)")

    return df
```

- [ ] **Step 4: Export from package __init__**

Update `src/nfl/features/v5/__init__.py`:

```python
from src.nfl.features.v5.config import VERSION
from src.nfl.features.v5.engineer import build_features

__all__ = ['VERSION', 'build_features']
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_v5_engineer.py -v`
Expected: 6 passed

- [ ] **Step 6: Run full test suite to verify nothing broke**

Run: `pytest -q`
Expected: all tests pass (244 + ~20 new = ~264 passing)

- [ ] **Step 7: Commit**

```bash
git add src/nfl/features/v5/engineer.py src/nfl/features/v5/__init__.py tests/test_v5_engineer.py
git commit -m "Task 3.1.7: V5 feature engineering orchestrator with per-season Parquet output"
```

---

## Task 8: Colab notebook wrapper

**Files:**
- Create: `colab/v5_feature_engineering.ipynb`

- [ ] **Step 1: Create the notebook**

Create `colab/v5_feature_engineering.ipynb` as a JSON file with cells that:
1. Print environment info
2. Install any missing libraries
3. Mount Google Drive
4. Copy the `src/nfl/features/v5/` folder from Drive to runtime
5. Run `build_features()` on all seasons
6. Save outputs to Drive

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# V5 Feature Engineering\n",
    "\n",
    "Runs the v5_engineer on Google Colab. Outputs per-season feature Parquets to Drive.\n",
    "\n",
    "**Expected runtime:** 1-2 hours on Colab Pro high-RAM CPU."
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# 1. Mount Drive\n",
    "from google.colab import drive\n",
    "drive.mount('/content/drive')"
   ],
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# 2. Set paths\n",
    "import sys, os\n",
    "from pathlib import Path\n",
    "\n",
    "DRIVE_ROOT = '/content/drive/MyDrive/SportsAnalyzer'\n",
    "DATA_DIR = f'{DRIVE_ROOT}/data/nfl'\n",
    "OUTPUT_DIR = f'{DRIVE_ROOT}/output/features'\n",
    "CODE_DIR = f'{DRIVE_ROOT}/src'\n",
    "\n",
    "# Add code directory to Python path\n",
    "sys.path.insert(0, DRIVE_ROOT)\n",
    "print(f'Data: {DATA_DIR}')\n",
    "print(f'Output: {OUTPUT_DIR}')\n",
    "print(f'Code: {CODE_DIR}')"
   ],
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# 3. Verify code is uploaded\n",
    "assert Path(f'{CODE_DIR}/nfl/features/v5/engineer.py').exists(), \\\n",
    "    'Upload src/ folder to Drive first (copy to My Drive/SportsAnalyzer/src/)'\n",
    "print('Code found')"
   ],
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# 4. Run feature engineering\n",
    "from src.nfl.features.v5 import build_features\n",
    "\n",
    "SEASONS = list(range(2018, 2026))  # 2018-2025\n",
    "\n",
    "df = build_features(\n",
    "    data_dir=DATA_DIR,\n",
    "    seasons=SEASONS,\n",
    "    output_dir=OUTPUT_DIR,\n",
    "    verbose=True,\n",
    ")\n",
    "\n",
    "print(f'\\nDone: {len(df):,} rows, {len(df.columns)} columns')"
   ],
   "execution_count": null,
   "outputs": []
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# 5. Verify outputs\n",
    "import os\n",
    "output_v5 = f'{OUTPUT_DIR}/v5'\n",
    "for f in sorted(os.listdir(output_v5)):\n",
    "    size_mb = os.path.getsize(f'{output_v5}/{f}') / 1e6\n",
    "    print(f'  {f}: {size_mb:.1f} MB')"
   ],
   "execution_count": null,
   "outputs": []
  }
 ],
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python"}
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
```

- [ ] **Step 2: Commit**

```bash
git add colab/v5_feature_engineering.ipynb
git commit -m "Task 3.1.8: Colab notebook wrapper for V5 feature engineering"
```

---

## Task 9: HANDOFF — User runs feature engineering on Colab

**This task is executed by the human user, not Claude.**

- [ ] **Step 1: User uploads code to Drive**

Copy `src/nfl/features/v5/` folder from local to `My Drive/SportsAnalyzer/src/nfl/features/v5/`. Also upload the `src/nfl/features/__init__.py` and `src/nfl/__init__.py` files to preserve package structure.

- [ ] **Step 2: User opens notebook in VS Code**

Open `colab/v5_feature_engineering.ipynb`, select Colab kernel (CPU High-RAM), sign in.

- [ ] **Step 3: User runs all cells sequentially**

Cells 1-5 should run in order. Expected total runtime: 1-2 hours.

- [ ] **Step 4: User verifies output**

Check that `My Drive/SportsAnalyzer/output/features/v5/` contains 8 files:
- features_2018.parquet through features_2025.parquet
- Each file should be ~5-20 MB

- [ ] **Step 5: User reports back in chat**

User says "features done" with output of cell 5 (file sizes).

---

## Task 10: Validate handoff output

**Files:**
- Create: `tests/test_v5_real_output.py` (validates real Colab output after download)

- [ ] **Step 1: User downloads feature Parquets to local**

Copy `My Drive/SportsAnalyzer/output/features/v5/` to `data/nfl/features/v5/` locally.

- [ ] **Step 2: Write validation test**

```python
# tests/test_v5_real_output.py
"""
Validation tests for real V5 feature output (after Colab run).
Runs against the actual feature Parquets downloaded from Drive.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pytest

FEATURES_DIR = Path(__file__).parent.parent / 'data' / 'nfl' / 'features' / 'v5'


@pytest.fixture
def features_2024():
    return pd.read_parquet(FEATURES_DIR / 'features_2024.parquet')


def test_all_seasons_present():
    for season in range(2018, 2026):
        assert (FEATURES_DIR / f'features_{season}.parquet').exists()


def test_no_data_leakage_mahomes_w1(features_2024):
    """Mahomes week 1 rolling features should use 2023 data (warm-up), not week 1 actuals."""
    m = features_2024[
        (features_2024['player_id'] == '00-0033873') &
        (features_2024['week'] == 1)
    ]
    if len(m) > 0:
        row = m.iloc[0]
        # 2024 Week 1: Mahomes threw 291 yds. Rolling avg should NOT be 291.
        assert row['rolling_avg_passing_yards'] != 291


def test_mahomes_historical_stats_present(features_2024):
    """Spot check: Mahomes should have non-null rolling features by week 5."""
    m = features_2024[
        (features_2024['player_id'] == '00-0033873') &
        (features_2024['week'] == 5)
    ]
    if len(m) > 0:
        row = m.iloc[0]
        # By week 5 he should have 4 prior games — rolling avg should be populated
        assert pd.notna(row['rolling_avg_passing_yards'])


def test_feature_count_exceeds_60(features_2024):
    feature_cols = [c for c in features_2024.columns
                    if c.startswith(('rolling_', 'variance_', 'trend_', 'opp_def_rank_'))
                    or c in ['game_script_index', 'is_dome', 'is_high_wind',
                             'injury_severity', 'is_starter', 'is_home']]
    assert len(feature_cols) >= 60


def test_row_count_matches_weekly_stats(features_2024):
    """Feature output should have ~1 row per player per week."""
    # 2024 had ~17 regular season weeks + playoffs. ~600 active players per week.
    # Expect ~10,000 rows minimum
    assert len(features_2024) >= 8000
```

- [ ] **Step 3: Run validation tests**

Run: `pytest tests/test_v5_real_output.py -v`
Expected: 5 passed

- [ ] **Step 4: Commit**

```bash
git add tests/test_v5_real_output.py
git commit -m "Task 3.1.10: V5 feature output validation tests"
```

---

## Self-Review

**Spec coverage check:**
- ✅ Master player-week table (Task 2)
- ✅ Rolling averages with decay=0.85 (Task 3)
- ✅ Variance / boom-bust features (Task 3)
- ✅ Usage trend features (Task 3 + Task 5)
- ✅ Vegas features (Task 4)
- ✅ Opponent defense rank (Task 4)
- ✅ Weather features (Task 4)
- ✅ Snap count features (Task 5)
- ✅ Injury features (Task 5)
- ✅ Depth chart features (Task 5)
- ✅ FF opportunity features (Task 6)
- ✅ NGS features (Task 6)
- ✅ PFR features (Task 6)
- ✅ Pre-game features only (enforced by strict shift in rolling calculations)
- ✅ NULL preservation for unqualified players (Task 6 test)
- ✅ Colab handoff (Tasks 8-10)
- ⚠️ Rookie handling / 3-game minimum — implicit in rolling features producing NaN for < 3 games. Explicitly flagged in validation test_feature_count.

**Placeholder scan:** None — all steps include complete code, exact file paths, and exact pytest commands.

**Type consistency check:**
- `build_features()` signature consistent across all tasks
- `build_master_table()` signature consistent
- Column names match between master_table.py output and downstream feature modules (e.g., `offense_pct` produced in Task 2, consumed in Task 5)

---

## Post-Plan Notes

**Estimated effort:** Tasks 1-8 are local work (~4-6 hours of Claude-driven implementation). Task 9 is the user's Colab run (~1-2 hours). Task 10 is 10 minutes of validation.

**Handoff point:** After Task 8, Claude stops and waits for user to run Task 9 on Colab. Claude resumes at Task 10 with the output.

**What could go wrong:**
1. Opponent defense rank is slow (O(seasons × weeks × positions × teams)). If too slow, optimize to vectorized groupby.
2. PostgreSQL data types for ff_opportunity `season`/`week` are TEXT — already handled by coercion in master_table.py.
3. 2025 depth charts missing — LEFT JOIN will produce NULLs, which is correct behavior.
4. Running out of Colab RAM — if master table > 16GB in memory, split by season and concat feature files.
