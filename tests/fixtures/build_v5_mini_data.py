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

# Minimal schedules: KC@BAL, PHI@GB (week 1), plus games week 2-3
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
    'temp': [67.0, 72.0, None, None, 75.0, None],
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
