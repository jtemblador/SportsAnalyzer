"""V5 ablation study tests — group filter correctness, threshold logic, smoke."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.nfl.training.v5.ablation import (
    ABLATION_GROUPS,
    DEFAULT_DROP_THRESHOLD,
    aggregate_summary_csv,
    apply_drop_threshold,
    compute_position_aggregate_mae,
    get_ablation_exclude_columns,
    save_ablation_result,
)


def _synth_feature_df() -> pd.DataFrame:
    """Minimal feature DataFrame with one column from each group + identity."""
    rng = np.random.default_rng(0)
    n = 100
    return pd.DataFrame({
        # identity
        "season": [2023] * n,
        "week": list(range(1, n + 1)),
        "player_id": [f"P{i:03d}" for i in range(n)],
        "player_name": [f"Player_{i}" for i in range(n)],
        "position": ["WR"] * n,
        "team": ["KC"] * n,
        "games_of_history": [5] * n,
        # raw stat target (excluded by whitelist)
        "receiving_yards": rng.integers(0, 200, n).astype(float),
        # rolling group
        "rolling_avg_receiving_yards": rng.uniform(20, 100, n),
        "variance_receiving_yards": rng.uniform(0, 30, n),
        "trend_receiving_yards": rng.uniform(-10, 10, n),
        # context group
        "spread_line": rng.uniform(-7, 7, n),
        "total_line": rng.uniform(40, 55, n),
        "is_dome": [0] * n,
        # usage group
        "rolling_offense_pct": rng.uniform(0.4, 1.0, n),
        "injury_severity": [0] * n,
        # advanced group
        "rolling_ngs_receiving_avg_separation": rng.uniform(2, 5, n),
        "rolling_pfr_rec_receiving_drop_pct": rng.uniform(0, 0.15, n),
        # opp_offense group (only matters for DST but works for any df)
        "opp_rolling_avg_off_yards": rng.uniform(300, 400, n),
    })


class TestAblationConfig:
    def test_ablation_groups_cover_all_positions(self):
        assert set(ABLATION_GROUPS) == {"QB", "RB", "WR", "TE", "K", "DST"}

    def test_player_positions_have_4_groups(self):
        for pos in ["QB", "RB", "WR", "TE", "K"]:
            assert set(ABLATION_GROUPS[pos]) == {"rolling", "context", "usage", "advanced"}

    def test_dst_has_opp_offense_not_usage_advanced(self):
        assert "opp_offense" in ABLATION_GROUPS["DST"]
        assert "usage" not in ABLATION_GROUPS["DST"]
        assert "advanced" not in ABLATION_GROUPS["DST"]


class TestGroupFilter:
    def test_rolling_group_dropped_correctly(self):
        df = _synth_feature_df()
        excluded = get_ablation_exclude_columns(df, "WR", "rolling")
        assert "rolling_avg_receiving_yards" in excluded
        assert "variance_receiving_yards" in excluded
        assert "trend_receiving_yards" in excluded
        # Non-rolling should NOT be in exclude list
        assert "spread_line" not in excluded
        assert "rolling_ngs_receiving_avg_separation" not in excluded  # advanced, not rolling

    def test_context_group_dropped_correctly(self):
        df = _synth_feature_df()
        excluded = get_ablation_exclude_columns(df, "WR", "context")
        assert "spread_line" in excluded
        assert "total_line" in excluded
        assert "is_dome" in excluded
        assert "rolling_avg_receiving_yards" not in excluded

    def test_advanced_group_drops_ngs_and_pfr(self):
        df = _synth_feature_df()
        excluded = get_ablation_exclude_columns(df, "WR", "advanced")
        assert "rolling_ngs_receiving_avg_separation" in excluded
        assert "rolling_pfr_rec_receiving_drop_pct" in excluded
        # Rolling stats that AREN'T ngs/pfr stay
        assert "rolling_avg_receiving_yards" not in excluded

    def test_exclude_only_returns_whitelisted_cols(self):
        """Non-whitelisted raw stat columns should never appear in exclude list
        (they weren't features to begin with)."""
        df = _synth_feature_df()
        excluded = get_ablation_exclude_columns(df, "WR", "rolling")
        assert "receiving_yards" not in excluded  # raw target, not a feature
        assert "player_id" not in excluded  # identity, not a feature


class TestAggregateMae:
    def test_weighted_sum_matches_manual(self):
        stat_maes = {
            "receptions": {"mae": 1.0},         # weight 1.0
            "receiving_yards": {"mae": 20.0},   # weight 0.1
            "receiving_tds": {"mae": 0.3},      # weight 6.0
            "targets": {"mae": 2.0},            # weight 0 (not scored)
        }
        # Upper bound: 1*1 + 0.1*20 + 6*0.3 = 1 + 2 + 1.8 = 4.8
        # Realistic: 4.8 * 0.7 = 3.36
        agg = compute_position_aggregate_mae(stat_maes)
        assert abs(agg - 3.36) < 0.01

    def test_handles_missing_stats(self):
        """Unknown stats should be ignored, not crash."""
        stat_maes = {"mystery_stat": {"mae": 100.0}}
        assert compute_position_aggregate_mae(stat_maes) == 0.0

    def test_dst_stats_use_dst_weights_not_zero(self):
        """Regression: DST stats (sacks, interceptions, etc.) must use
        FANTASY_DST_WEIGHTS, not silently skip via PPR_WEIGHTS-miss."""
        stat_maes = {
            "sacks": {"mae": 1.0},               # weight 1.0
            "interceptions": {"mae": 0.5},       # weight 2.0
            "fumble_recoveries": {"mae": 0.5},   # weight 2.0
            "defensive_tds": {"mae": 0.2},       # weight 6.0
            "safeties": {"mae": 0.05},           # weight 2.0
            "points_allowed": {"mae": 7.0},      # NO weight (tiered scoring)
        }
        # Upper: 1*1 + 2*0.5 + 2*0.5 + 6*0.2 + 2*0.05 + 0*7
        #      = 1 + 1 + 1 + 1.2 + 0.1 + 0 = 4.3
        # Realistic: 4.3 * 0.7 = 3.01
        agg = compute_position_aggregate_mae(stat_maes)
        assert abs(agg - 3.01) < 0.01, f"Expected ~3.01, got {agg}"
        assert agg > 0, "DST aggregate must be nonzero"


class TestThresholdLogic:
    """Borderline band sits just BELOW the keep threshold (0.8*T to T),
    not around zero. See apply_drop_threshold docstring for reasoning."""

    def test_positive_delta_beyond_threshold_keeps_group(self):
        """Group clearly helped: delta ≥ threshold → KEEP."""
        assert apply_drop_threshold({"delta": 0.3}) == "keep"
        assert apply_drop_threshold({"delta": 0.05}) == "keep"  # exactly at threshold

    def test_delta_in_borderline_band(self):
        """Delta in (0.04, 0.05) is just below threshold → BORDERLINE.
        Note: 0.04 exactly can land either side due to FP rounding of
        0.05*0.8; we test safely-in-band values."""
        assert apply_drop_threshold({"delta": 0.045}) == "borderline"
        assert apply_drop_threshold({"delta": 0.049}) == "borderline"

    def test_zero_delta_drops_group(self):
        """Delta near zero = group had no effect → DROP."""
        assert apply_drop_threshold({"delta": 0.0}) == "drop"

    def test_modest_negative_delta_drops_group(self):
        """Ablated MAE better than baseline: group was noise → DROP."""
        assert apply_drop_threshold({"delta": -0.3}) == "drop"

    def test_delta_just_below_borderline_drops(self):
        """Delta = 0.03 is below borderline band (0.04) → DROP."""
        assert apply_drop_threshold({"delta": 0.03}) == "drop"

    def test_custom_threshold_scales_borderline_band(self):
        """Threshold 0.15 → borderline band [0.12, 0.15)."""
        assert apply_drop_threshold({"delta": 0.08}, threshold=0.05) == "keep"
        assert apply_drop_threshold({"delta": 0.08}, threshold=0.15) == "drop"
        assert apply_drop_threshold({"delta": 0.13}, threshold=0.15) == "borderline"


class TestSaveAndAggregate:
    def test_save_and_reaggregate_produces_summary(self, tmp_path):
        result = {
            "position": "WR", "group_removed": "advanced",
            "excluded_cols": ["rolling_ngs_x", "rolling_pfr_y"],
            "n_excluded": 2,
            "eval_seasons": [2023, 2024],
            "stats": {"receptions": {"mae": 1.5}},
        }
        comparison = {
            "position": "WR", "group_removed": "advanced",
            "baseline_agg_mae": 3.8, "ablated_agg_mae": 3.85,
            "delta": 0.05,
        }
        save_ablation_result(result, comparison, "keep", tmp_path)

        summary = aggregate_summary_csv(tmp_path)
        assert len(summary) == 1
        assert summary.iloc[0]["position"] == "WR"
        assert summary.iloc[0]["decision"] == "keep"
        assert abs(summary.iloc[0]["delta"] - 0.05) < 1e-6


class TestResumeCheck:
    """Resume check must detect eval_seasons mismatch so borderline re-runs
    on a wider window actually re-execute instead of silently reusing
    cached narrow-window results."""

    def test_no_file_returns_false(self, tmp_path):
        from src.nfl.training.v5.ablation import ablation_result_complete
        assert not ablation_result_complete("WR", "rolling", tmp_path)

    def test_same_eval_seasons_returns_true(self, tmp_path):
        from src.nfl.training.v5.ablation import ablation_result_complete
        result = {"position": "WR", "group_removed": "rolling",
                  "excluded_cols": [], "n_excluded": 0,
                  "eval_seasons": [2023, 2024], "stats": {}}
        comparison = {"position": "WR", "group_removed": "rolling",
                      "baseline_agg_mae": 3.8, "ablated_agg_mae": 3.9, "delta": 0.1}
        save_ablation_result(result, comparison, "keep", tmp_path)
        assert ablation_result_complete("WR", "rolling", tmp_path,
                                        expected_eval_seasons=[2023, 2024])

    def test_different_eval_seasons_returns_false(self, tmp_path):
        """Regression: borderline re-run with [2021-2024] must NOT reuse
        a cached [2023, 2024] result."""
        from src.nfl.training.v5.ablation import ablation_result_complete
        result = {"position": "WR", "group_removed": "rolling",
                  "excluded_cols": [], "n_excluded": 0,
                  "eval_seasons": [2023, 2024], "stats": {}}
        comparison = {"position": "WR", "group_removed": "rolling",
                      "baseline_agg_mae": 3.8, "ablated_agg_mae": 3.9, "delta": 0.1}
        save_ablation_result(result, comparison, "keep", tmp_path)
        # Caller wants [2021, 2022, 2023, 2024] → stale result → must return False
        assert not ablation_result_complete("WR", "rolling", tmp_path,
                                            expected_eval_seasons=[2021, 2022, 2023, 2024])

    def test_legacy_result_without_eval_seasons_returns_false(self, tmp_path):
        """Old JSON lacking eval_seasons field should trigger re-run."""
        from src.nfl.training.v5.ablation import ablation_result_complete
        legacy_path = tmp_path / "ablation_WR_remove_rolling.json"
        legacy_path.write_text(json.dumps({"position": "WR", "group_removed": "rolling"}))
        assert not ablation_result_complete("WR", "rolling", tmp_path,
                                            expected_eval_seasons=[2023, 2024])


class TestFeatureListOutput:
    """write_validated_feature_list produces the Task 3.2c handoff artifact."""

    def test_keeps_keep_and_borderline_drops_drop(self, tmp_path):
        from src.nfl.training.v5.ablation import write_validated_feature_list
        summary = pd.DataFrame([
            {"position": "QB", "group_removed": "rolling",  "decision": "keep"},
            {"position": "QB", "group_removed": "context",  "decision": "keep"},
            {"position": "QB", "group_removed": "usage",    "decision": "borderline"},
            {"position": "QB", "group_removed": "advanced", "decision": "drop"},
        ])
        out = tmp_path / "_validated_feature_groups.json"
        write_validated_feature_list(summary, out)
        loaded = json.loads(out.read_text())
        qb_keep = loaded["QB"]["keep"]
        assert "rolling" in qb_keep
        assert "context" in qb_keep
        assert "usage" in qb_keep       # borderline = keep
        assert "advanced" not in qb_keep
        assert loaded["QB"]["fully_ablated"] is True

    def test_partial_ablation_flags_not_fully_ablated(self, tmp_path):
        """If ablation didn't run for a (pos, group), keep it by default BUT
        flag fully_ablated=False so Task 3.2c knows to re-run."""
        from src.nfl.training.v5.ablation import write_validated_feature_list
        summary = pd.DataFrame([
            {"position": "QB", "group_removed": "rolling", "decision": "drop"},
            # context/usage/advanced not in summary
        ])
        out = tmp_path / "_validated_feature_groups.json"
        write_validated_feature_list(summary, out)
        loaded = json.loads(out.read_text())
        qb_keep = loaded["QB"]["keep"]
        assert "rolling" not in qb_keep   # ran and dropped
        assert "context" in qb_keep        # conservative default
        assert "usage" in qb_keep
        assert "advanced" in qb_keep
        assert loaded["QB"]["fully_ablated"] is False  # 3 groups unverified


class TestSmoke:
    """Integration smoke — only runs if real V5 feature parquets exist."""

    @pytest.mark.skipif(
        not Path("data/nfl/features/v5/features_2024.parquet").exists(),
        reason="V5 feature parquets not present locally",
    )
    def test_k_position_single_ablation_smoke(self, tmp_path):
        """Run ablation on K position removing 'advanced' group.
        K position has tiny data (~600 rows) + only 2 algorithms →
        should complete in <60s and return numeric MAE."""
        import time
        from src.nfl.training.v5.ablation import run_all_ablations
        t0 = time.time()
        summary = run_all_ablations(
            positions=["K"], groups=["advanced"],
            eval_seasons=[2024],
            output_dir=tmp_path,
        )
        elapsed = time.time() - t0
        # K / advanced group is empty (K has no NGS/PFR) — no columns to drop,
        # so run_all_ablations skips with "0 columns would be dropped".
        # Either empty summary (skipped) or a single row with delta ≈ 0.
        assert elapsed < 120, f"K ablation took {elapsed:.1f}s, expected <60s"
        # Empty = skipped cleanly (expected for K/advanced)
        # Non-empty = ran; delta should be ~0 since no cols actually dropped
        if not summary.empty:
            assert abs(summary.iloc[0]["delta"]) < 0.01
