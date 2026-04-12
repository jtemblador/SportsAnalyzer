# tests/test_v5_real_data.py
"""
End-to-end validation tests for V5 feature engineering against REAL NFL data.

These tests run build_features() on the actual data/nfl/ Parquets for the 2024
season and verify output correctness on known players. Runs locally before
the full Colab production run (all 8 seasons).

Marked @pytest.mark.integration — skipped if real data is missing.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pytest
import time

from src.nfl.features.v5 import build_features

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / 'data' / 'nfl'

# Known player IDs for spot checks
MAHOMES_ID = '00-0033873'   # QB
BARKLEY_ID = '00-0034844'   # RB
KELCE_ID = '00-0030506'     # TE

# Skip if real data not available (e.g., in CI without data cloned)
pytestmark = pytest.mark.skipif(
    not (DATA_DIR / 'player_stats' / 'player_stats_2024.parquet').exists(),
    reason='Real NFL data not available — run locally from project root'
)


@pytest.fixture(scope='module')
def features_2024():
    """Build features for 2024 with 2023 as warm-up for cross-season rolling."""
    return build_features(
        data_dir=str(DATA_DIR),
        seasons=[2023, 2024],
        verbose=False,
    )


# ============================================================
# Smoke tests — shape and basic structure
# ============================================================

class TestShape:

    def test_returns_dataframe(self, features_2024):
        assert isinstance(features_2024, pd.DataFrame)

    def test_nonzero_rows(self, features_2024):
        assert len(features_2024) > 1000

    def test_only_predictable_positions(self, features_2024):
        positions = set(features_2024['position'].unique())
        allowed = {'QB', 'RB', 'WR', 'TE', 'K'}
        assert positions.issubset(allowed), f"Unexpected positions: {positions - allowed}"

    def test_has_all_feature_groups(self, features_2024):
        """All 4 feature groups produce at least some columns."""
        from src.nfl.features.v5.config import get_feature_columns_by_group
        for group in ['rolling', 'context', 'usage', 'advanced']:
            cols = get_feature_columns_by_group(features_2024.columns.tolist(), group)
            assert len(cols) > 0, f"Feature group '{group}' is empty"

    def test_feature_count_at_least_60(self, features_2024):
        """V5 should have at least 60 feature columns (plan target 60-80)."""
        from src.nfl.features.v5.config import get_feature_columns_by_group
        total = 0
        for group in ['rolling', 'context', 'usage', 'advanced']:
            total += len(get_feature_columns_by_group(
                features_2024.columns.tolist(), group
            ))
        assert total >= 60, f"Only {total} feature columns"


# ============================================================
# Happy path — spot check known players
# ============================================================

class TestMahomes:
    """Mahomes 2024 sanity checks."""

    def test_has_rows(self, features_2024):
        m = features_2024[
            (features_2024['player_id'] == MAHOMES_ID) &
            (features_2024['season'] == 2024)
        ]
        assert len(m) >= 15, f"Only {len(m)} Mahomes 2024 rows"

    def test_rolling_avg_populated_by_week_5(self, features_2024):
        m_w5 = features_2024[
            (features_2024['player_id'] == MAHOMES_ID) &
            (features_2024['season'] == 2024) &
            (features_2024['week'] == 5)
        ]
        if len(m_w5) == 0:
            pytest.skip("Mahomes did not have a W5 2024 record")
        row = m_w5.iloc[0]
        # By W5, Mahomes has W1-W4 of 2024 + entire 2023 season as history
        assert pd.notna(row['rolling_avg_passing_yards'])
        # His career passing avg is ~250-290 yds — validate bounds
        assert 150 < row['rolling_avg_passing_yards'] < 400

    def test_no_leakage_week_1(self, features_2024):
        """W1 2024 rolling avg should NOT equal W1 2024 actual stats."""
        m_w1 = features_2024[
            (features_2024['player_id'] == MAHOMES_ID) &
            (features_2024['season'] == 2024) &
            (features_2024['week'] == 1)
        ]
        if len(m_w1) == 0:
            pytest.skip()
        row = m_w1.iloc[0]
        # W1 2024 Mahomes threw 291 yards — rolling avg (from 2023 season) should be different
        assert row['rolling_avg_passing_yards'] != 291

    def test_games_of_history_correct(self, features_2024):
        """Mahomes played entire 2023 season (17 weeks) + some 2024 — should have many games of history."""
        m_w5 = features_2024[
            (features_2024['player_id'] == MAHOMES_ID) &
            (features_2024['season'] == 2024) &
            (features_2024['week'] == 5)
        ]
        if len(m_w5) == 0:
            pytest.skip()
        # 2023 season = 17 regular season weeks + playoffs; Mahomes played all
        # By 2024 W5: 2023 full season + W1-W4 of 2024 ≈ 21 prior games
        assert m_w5.iloc[0]['games_of_history'] >= 15


class TestBarkley:
    """Barkley 2024 sanity checks (RB with snap share data)."""

    def test_snap_pct_rolling_populated(self, features_2024):
        b_w5 = features_2024[
            (features_2024['player_id'] == BARKLEY_ID) &
            (features_2024['season'] == 2024) &
            (features_2024['week'] == 5)
        ]
        if len(b_w5) == 0:
            pytest.skip()
        row = b_w5.iloc[0]
        # Barkley is a starter, offense_pct typically 0.7-0.9
        if pd.notna(row['rolling_offense_pct']):
            assert 0.5 < row['rolling_offense_pct'] < 1.0


class TestKelce:
    """Kelce 2024 sanity checks (TE)."""

    def test_position_is_te(self, features_2024):
        k = features_2024[features_2024['player_id'] == KELCE_ID]
        if len(k) == 0:
            pytest.skip("Kelce not in 2024 data")
        assert (k['position'] == 'TE').all()


# ============================================================
# Data leakage tests on real data
# ============================================================

class TestDataLeakage:

    def test_rolling_avg_never_equals_current_stat(self, features_2024):
        """For all rows where rolling_avg_passing_yards is populated,
        it should NOT equal current-week passing_yards.
        (If it does for many rows, there's a leakage bug.)"""
        df = features_2024[features_2024['position'] == 'QB'].copy()
        df = df[
            df['rolling_avg_passing_yards'].notna() &
            df['passing_yards'].notna() &
            (df['passing_yards'] > 0)  # skip zero stats (trivially match if 0)
        ]
        # Tolerance for floating point
        exact_matches = (df['rolling_avg_passing_yards'] == df['passing_yards']).sum()
        # Should be 0 or very few (same-value coincidences)
        assert exact_matches < 5, (
            f"{exact_matches} exact matches between rolling avg and current week stat — "
            "possible data leakage"
        )

    def test_opp_def_rank_only_uses_prior_weeks(self, features_2024):
        """Week 1 of season should have NaN opp_def_rank (no prior weeks to rank from)."""
        qb_w1 = features_2024[
            (features_2024['position'] == 'QB') &
            (features_2024['week'] == 1)
        ]
        if len(qb_w1) == 0:
            pytest.skip()
        # Week 1 of first season has no prior. (2023 is the first season here.)
        w1_2023 = qb_w1[qb_w1['season'] == 2023]
        if len(w1_2023) > 0:
            # All opp_def_rank_qb for week 1 of the first season must be NaN
            assert w1_2023['opp_def_rank_qb'].isna().all()


# ============================================================
# Edge case tests
# ============================================================

class TestEdgeCases:

    def test_bye_weeks_handled(self, features_2024):
        """Players on bye don't have stat rows — this is fine. Just ensure
        no player has suspiciously many weeks (which would imply bye week
        rows were incorrectly generated)."""
        mahomes = features_2024[features_2024['player_id'] == MAHOMES_ID]
        # Max weeks a player can play: 18 regular season + 4 playoff = 22 per season
        # Plus 17-18 for 2023 warm-up ≈ max 40 rows across 2023+2024
        assert len(mahomes) < 50

    def test_null_preservation_for_unqualified_ngs(self, features_2024):
        """Most non-QB players won't have NGS passing stats — should be NaN, not 0."""
        non_qb = features_2024[features_2024['position'] != 'QB']
        if 'rolling_ngs_passing_avg_time_to_throw' not in non_qb.columns:
            pytest.skip("NGS passing rolling not present")
        # Some non-QB players may have NGS passing rolling (if they threw occasionally),
        # but the majority should be NaN
        null_pct = non_qb['rolling_ngs_passing_avg_time_to_throw'].isna().mean()
        assert null_pct > 0.9, f"Only {null_pct:.1%} of non-QBs have NULL NGS passing"


# ============================================================
# Performance tests
# ============================================================

class TestPerformance:

    def test_2024_season_runs_in_reasonable_time(self):
        """2024 season alone should build in < 5 minutes locally."""
        start = time.time()
        df = build_features(
            data_dir=str(DATA_DIR),
            seasons=[2024],
            verbose=False,
        )
        elapsed = time.time() - start
        assert elapsed < 300, f"2024 season took {elapsed:.1f}s (>5 min)"
        assert len(df) > 0
        print(f"\n2024 season built in {elapsed:.1f}s ({len(df):,} rows)")
