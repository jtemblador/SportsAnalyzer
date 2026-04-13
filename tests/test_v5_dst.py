# tests/test_v5_dst.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.nfl.features.v5.dst import (
    points_allowed_bonus,
    compute_dst_fantasy_points,
    build_master_dst_table,
)
from src.nfl.features.v5.config import (
    STATS_TO_PREDICT,
    FANTASY_DST_WEIGHTS,
    CORE_DST_STATS_FOR_ROLLING,
)

REAL_DATA_DIR = Path(__file__).parent.parent / 'data' / 'nfl'


# ---------- config extension ----------

def test_config_dst_stats_to_predict():
    assert STATS_TO_PREDICT['DST'] == [
        'sacks', 'interceptions', 'fumble_recoveries', 'defensive_tds',
        'safeties', 'points_allowed',
    ]


def test_config_fantasy_dst_weights():
    assert FANTASY_DST_WEIGHTS['sacks'] == 1.0
    assert FANTASY_DST_WEIGHTS['interceptions'] == 2.0
    assert FANTASY_DST_WEIGHTS['fumble_recoveries'] == 2.0
    assert FANTASY_DST_WEIGHTS['defensive_tds'] == 6.0
    assert FANTASY_DST_WEIGHTS['safeties'] == 2.0
    assert FANTASY_DST_WEIGHTS['blocked_kicks'] == 2.0
    assert FANTASY_DST_WEIGHTS['return_tds'] == 6.0


def test_config_core_dst_rolling_list():
    assert set(CORE_DST_STATS_FOR_ROLLING) == {
        'sacks', 'interceptions', 'fumble_recoveries', 'defensive_tds',
        'safeties', 'points_allowed', 'return_tds', 'blocked_kicks',
    }


# ---------- scoring ----------

def test_points_allowed_bonus():
    # exact values at boundaries
    assert points_allowed_bonus(0) == 10
    assert points_allowed_bonus(1) == 7
    assert points_allowed_bonus(6) == 7
    assert points_allowed_bonus(7) == 4
    assert points_allowed_bonus(13) == 4
    assert points_allowed_bonus(14) == 1
    assert points_allowed_bonus(20) == 1
    assert points_allowed_bonus(21) == 0
    assert points_allowed_bonus(27) == 0
    assert points_allowed_bonus(28) == -1
    assert points_allowed_bonus(34) == -1
    assert points_allowed_bonus(35) == -4
    assert points_allowed_bonus(50) == -4


def test_dst_scoring_formula():
    # 3 sacks + 1 INT + 2 FR + 1 def TD + 0 safeties + 0 blocked + 0 ret_td, 14 PA
    # = 3*1 + 1*2 + 2*2 + 1*6 + 0 + 0 + 0 + bonus(14)=+1  = 16
    row = pd.Series({
        'sacks': 3,
        'interceptions': 1,
        'fumble_recoveries': 2,
        'defensive_tds': 1,
        'safeties': 0,
        'blocked_kicks': 0,
        'return_tds': 0,
        'points_allowed': 14,
    })
    assert compute_dst_fantasy_points(row) == 16.0


# ---------- master table (synthetic fixtures) ----------

def _write_min_schedule(path, season, week, home, away, home_score, away_score):
    """Write a minimal schedules parquet with a single game."""
    sch = pd.DataFrame([{
        'game_id': f'{season}_{week:02d}_{away}_{home}',
        'season': season,
        'game_type': 'REG',
        'week': week,
        'gameday': '2024-09-01',
        'weekday': 'Sunday',
        'gametime': '13:00',
        'away_team': away,
        'away_score': away_score,
        'home_team': home,
        'home_score': home_score,
        'location': 'Home',
        'result': home_score - away_score,
        'total': home_score + away_score,
        'overtime': 0,
        'old_game_id': '',
        'gsis': '',
        'nfl_detail_id': '',
        'pfr': '',
        'pff': '',
        'espn': '',
        'ftn': '',
        'away_rest': 7,
        'home_rest': 7,
        'away_moneyline': -110,
        'home_moneyline': -110,
        'spread_line': 3.0,
        'away_spread_odds': -110,
        'home_spread_odds': -110,
        'total_line': 45.0,
        'under_odds': -110,
        'over_odds': -110,
        'div_game': 0,
        'roof': 'outdoors',
        'surface': 'grass',
        'temp': 70.0,
        'wind': 5.0,
        'away_qb_id': '',
        'home_qb_id': '',
        'away_qb_name': '',
        'home_qb_name': '',
        'away_coach': '',
        'home_coach': '',
        'referee': '',
        'stadium_id': '',
        'stadium': '',
        'home_implied_total': 24.0,
        'away_implied_total': 21.0,
    }])
    out = path / 'schedules'
    out.mkdir(parents=True, exist_ok=True)
    sch.to_parquet(out / f'schedules_{season}.parquet')


def _write_team_stats(path, season, rows):
    """Write a minimal team_stats parquet from dict rows; fills missing numeric cols with 0."""
    # Full set of columns we read/touch. Other cols can be absent.
    required = [
        'season', 'week', 'team', 'season_type', 'opponent_team',
        'def_sacks', 'def_interceptions', 'fumble_recovery_opp', 'def_tds',
        'def_safeties', 'def_fumbles', 'special_teams_tds',
        'fg_blocked', 'pat_blocked',
    ]
    df = pd.DataFrame(rows)
    for c in required:
        if c not in df.columns:
            df[c] = 0
    out = path / 'team_stats'
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / f'team_stats_{season}.parquet')


def test_fumble_recoveries_uses_recovery_opp(tmp_path):
    # REGRESSION GUARD: def_fumbles counts defenders' own fumbles (e.g. after
    # an INT return), NOT recoveries. Research confirmed league sums:
    # def_fumbles=76 vs fumble_recovery_opp=283. We must use fumble_recovery_opp.
    _write_team_stats(tmp_path, 2024, [
        {'season': 2024, 'week': 1, 'team': 'AAA', 'season_type': 'REG',
         'opponent_team': 'BBB',
         'def_fumbles': 99, 'fumble_recovery_opp': 2},
        {'season': 2024, 'week': 1, 'team': 'BBB', 'season_type': 'REG',
         'opponent_team': 'AAA',
         'def_fumbles': 99, 'fumble_recovery_opp': 0},
    ])
    _write_min_schedule(tmp_path, 2024, 1, 'AAA', 'BBB', 20, 17)

    out = build_master_dst_table(tmp_path, [2024])
    aaa = out[out['team'] == 'AAA'].iloc[0]
    assert aaa['fumble_recoveries'] == 2
    assert aaa['fumble_recoveries'] != 99


def test_blocked_kicks_from_opponent(tmp_path):
    # Team A has fg_blocked=1,pat_blocked=0. Team B has fg_blocked=0,pat_blocked=1.
    # A's DST row: blocked_kicks = B's (fg_blocked+pat_blocked) = 1
    # B's DST row: blocked_kicks = A's (fg_blocked+pat_blocked) = 1
    _write_team_stats(tmp_path, 2024, [
        {'season': 2024, 'week': 1, 'team': 'AAA', 'season_type': 'REG',
         'opponent_team': 'BBB',
         'fg_blocked': 1, 'pat_blocked': 0},
        {'season': 2024, 'week': 1, 'team': 'BBB', 'season_type': 'REG',
         'opponent_team': 'AAA',
         'fg_blocked': 0, 'pat_blocked': 1},
    ])
    _write_min_schedule(tmp_path, 2024, 1, 'AAA', 'BBB', 20, 17)

    out = build_master_dst_table(tmp_path, [2024])
    aaa = out[out['team'] == 'AAA'].iloc[0]
    bbb = out[out['team'] == 'BBB'].iloc[0]
    assert aaa['blocked_kicks'] == 1
    assert bbb['blocked_kicks'] == 1


# ---------- master table (real data) ----------

@pytest.fixture(scope='module')
def real_2024_dst():
    if not (REAL_DATA_DIR / 'team_stats' / 'team_stats_2024.parquet').exists():
        pytest.skip('real 2024 data not available')
    return build_master_dst_table(REAL_DATA_DIR, [2024])


def test_master_table_includes_post_rows(real_2024_dst):
    post = real_2024_dst[real_2024_dst['season_type'] == 'POST']
    assert len(post) > 0
    assert post['week'].min() >= 19
    assert post['week'].max() <= 22


def test_master_table_real_data_2024(real_2024_dst):
    reg_ct = (real_2024_dst['season_type'] == 'REG').sum()
    post_ct = (real_2024_dst['season_type'] == 'POST').sum()
    assert reg_ct == 544
    assert post_ct == 26
    assert real_2024_dst['team'].nunique() == 32
    for col in ['sacks', 'interceptions', 'fumble_recoveries',
                'defensive_tds', 'safeties', 'points_allowed']:
        assert real_2024_dst[col].isna().sum() == 0, f'{col} has NaN'


def test_points_allowed_real_game(real_2024_dst):
    # 2024 W1 BAL@KC was 27-20 (KC won 27-20, KC home).
    kc_w1 = real_2024_dst[
        (real_2024_dst['team'] == 'KC') & (real_2024_dst['week'] == 1)
        & (real_2024_dst['season_type'] == 'REG')
    ].iloc[0]
    bal_w1 = real_2024_dst[
        (real_2024_dst['team'] == 'BAL') & (real_2024_dst['week'] == 1)
        & (real_2024_dst['season_type'] == 'REG')
    ].iloc[0]
    assert kc_w1['points_allowed'] == 20
    assert bal_w1['points_allowed'] == 27


# ---------- Part 2: rolling, opponent-offense, context, orchestrator ----------

from src.nfl.features.v5.dst import (
    add_dst_rolling,
    add_dst_opponent_offense,
    add_dst_context,
    build_dst_features,
)
from src.nfl.features.v5.engineer import build_features


def test_no_leakage_rolling_dst():
    """W1 of first season has NaN rolling; current stat never equals its own
    rolling_avg (synthetic guarantee of strictly-prior-weeks slicing)."""
    df = pd.DataFrame({
        'team': ['AAA'] * 5 + ['BBB'] * 5,
        'season': [2024] * 10,
        'week': list(range(1, 6)) * 2,
        'season_type': ['REG'] * 10,
        'opponent_team': ['BBB'] * 5 + ['AAA'] * 5,
        'sacks': [3, 2, 5, 1, 4, 0, 1, 2, 3, 0],
        'interceptions': [1, 0, 2, 0, 1, 0, 1, 0, 1, 2],
        'fumble_recoveries': [0]*10, 'defensive_tds': [0]*10,
        'safeties': [0]*10, 'points_allowed': [20, 17, 24, 14, 31, 28, 21, 17, 10, 27],
        'return_tds': [0]*10, 'blocked_kicks': [0]*10,
    })
    out = add_dst_rolling(df)
    # W1 for both teams should have NaN rolling_avg_sacks.
    w1 = out[out['week'] == 1]
    assert w1['rolling_avg_sacks'].isna().all()

    # Pin the exact computation: AAA W2 should equal decay-weighted avg of
    # AAA's W1 sacks ([3]). With one value, the decay-weighted avg = that value.
    aaa_w2 = out[(out['team'] == 'AAA') & (out['week'] == 2)].iloc[0]
    assert aaa_w2['rolling_avg_sacks'] == 3.0

    # Pin a multi-value case: AAA W4 should equal decay-weighted avg of
    # AAA W1-W3 sacks ([3, 2, 5], oldest first), guarding against off-by-one
    # in the slicing window.
    from src.nfl.features.v5.utils import decay_weighted_avg
    aaa_w4 = out[(out['team'] == 'AAA') & (out['week'] == 4)].iloc[0]
    expected = decay_weighted_avg(np.array([3, 2, 5]))
    assert abs(aaa_w4['rolling_avg_sacks'] - expected) < 1e-9


def test_post_rows_in_rolling_but_not_output():
    """Real 2024+2025: POST rows feed history, are absent from output, AND
    2025 W1 has non-NaN rolling_avg_sacks (proving 2024 history fed it)."""
    if not (REAL_DATA_DIR / 'team_stats' / 'team_stats_2024.parquet').exists():
        pytest.skip('real 2024 data not available')
    if not (REAL_DATA_DIR / 'team_stats' / 'team_stats_2025.parquet').exists():
        pytest.skip('real 2025 data not available')
    df = build_dst_features(REAL_DATA_DIR, [2024, 2025], verbose=False)
    # (a) No POST rows.
    assert (df['season_type'] == 'REG').all()
    # (b) Some 2025 W1 row has non-NaN rolling_avg_sacks (2024 fed it).
    w1_2025 = df[(df['season'] == 2025) & (df['week'] == 1)]
    assert len(w1_2025) > 0
    assert w1_2025['rolling_avg_sacks'].notna().any()


def test_dst_context_features():
    """game_script_index sign for known game. KC home W1 2024 vs BAL,
    spread_line=3.0 (KC favored). Per formula game_script_index = -spread_line,
    KC gets -3.0 (favorite negative) and BAL gets +3.0 (underdog positive)."""
    if not (REAL_DATA_DIR / 'team_stats' / 'team_stats_2024.parquet').exists():
        pytest.skip('real 2024 data not available')
    df = build_dst_features(REAL_DATA_DIR, [2024], verbose=False)
    kc_w1 = df[(df['team'] == 'KC') & (df['week'] == 1)].iloc[0]
    bal_w1 = df[(df['team'] == 'BAL') & (df['week'] == 1)].iloc[0]
    assert kc_w1['game_script_index'] == pytest.approx(-3.0)
    assert bal_w1['game_script_index'] == pytest.approx(3.0)
    # is_dome/is_high_wind/is_cold are 0/1 ints.
    for c in ['is_dome', 'is_high_wind', 'is_cold']:
        assert df[c].dropna().isin([0, 1]).all()


def test_opponent_offense_handles_missing_columns():
    """Regression: add_dst_opponent_offense must not crash if optional offense
    columns are absent (e.g., schema drift in a future nflverse release).
    Synthetic team_stats lacks sack_fumbles_lost — function should treat as 0,
    not AttributeError."""
    df = pd.DataFrame({
        'team': ['AAA', 'BBB'] * 3,
        'season': [2024] * 6,
        'week': [1, 1, 2, 2, 3, 3],
        'opponent_team': ['BBB', 'AAA'] * 3,
    })
    ts = pd.DataFrame({
        'team': ['AAA', 'BBB'] * 3,
        'season': [2024] * 6,
        'week': [1, 1, 2, 2, 3, 3],
        'passing_yards': [200, 250, 180, 300, 220, 260],
        'rushing_yards': [100, 80, 120, 90, 110, 70],
        'passing_tds': [1, 2, 1, 3, 2, 1],
        'rushing_tds': [0, 1, 1, 0, 0, 1],
        'passing_interceptions': [1, 0, 0, 1, 1, 0],
        'rushing_fumbles_lost': [0, 0, 1, 0, 0, 0],
        # sack_fumbles_lost intentionally omitted
    })
    out = add_dst_opponent_offense(df, team_stats=ts)
    assert 'opp_rolling_avg_off_yards' in out.columns
    assert 'opp_rolling_avg_off_turnovers' in out.columns

    # Pin a value to guard against arithmetic regression in the missing-column
    # path. AAA W2 sees BBB's W1 as opponent. BBB's W1 turnovers should be
    # passing_interceptions + rushing_fumbles_lost + sack_fumbles_lost(absent=0)
    # = 0 + 0 + 0 = 0. With one prior value, decay-weighted avg = that value.
    aaa_w2 = out[(out['team'] == 'AAA') & (out['week'] == 2)].iloc[0]
    assert aaa_w2['opp_rolling_avg_off_turnovers'] == 0.0
    # BBB W2 sees AAA's W1 as opponent. AAA W1 turnovers = 1 + 0 + 0 = 1.
    bbb_w2 = out[(out['team'] == 'BBB') & (out['week'] == 2)].iloc[0]
    assert bbb_w2['opp_rolling_avg_off_turnovers'] == 1.0


def test_opponent_offense_features():
    """Real data sanity: opp_rolling_avg_off_yards in plausible NFL range."""
    if not (REAL_DATA_DIR / 'team_stats' / 'team_stats_2024.parquet').exists():
        pytest.skip('real 2024 data not available')
    df = build_dst_features(REAL_DATA_DIR, [2023, 2024], verbose=False)
    # Drop NaN early-season rows; check distribution of mature rolling values.
    yards = df['opp_rolling_avg_off_yards'].dropna()
    assert len(yards) > 0
    assert 200 <= yards.median() <= 500, (
        f"opp_rolling_avg_off_yards median {yards.median()} outside NFL range"
    )


def test_expected_dst_feature_count():
    """Lock in column count of build_dst_features output. Drift will surface."""
    if not (REAL_DATA_DIR / 'team_stats' / 'team_stats_2024.parquet').exists():
        pytest.skip('real 2024 data not available')
    df = build_dst_features(REAL_DATA_DIR, [2023, 2024], verbose=False)
    assert len(df.columns) == 56, (
        f"Expected 56 DST feature columns, got {len(df.columns)}: "
        f"{df.columns.tolist()}"
    )


def test_engineer_writes_dst_parquet(tmp_path):
    """build_features([2024], output_dir=tmp_path) writes both player AND
    DST parquets for the same season."""
    if not (REAL_DATA_DIR / 'team_stats' / 'team_stats_2024.parquet').exists():
        pytest.skip('real 2024 data not available')
    build_features(
        data_dir=REAL_DATA_DIR,
        seasons=[2024],
        output_dir=tmp_path,
        verbose=False,
    )
    assert (tmp_path / 'v5' / 'features_2024.parquet').exists()
    assert (tmp_path / 'v5' / 'features_dst_2024.parquet').exists()
