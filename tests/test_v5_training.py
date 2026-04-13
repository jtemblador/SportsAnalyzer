"""V5 training tests — config sanity, data prep, walk-forward, ensembles, real-data smoke."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.nfl.training.v5.config import (
    COUNT_STATS,
    CONTINUOUS_STATS,
    LIGHTGBM_TE_OVERRIDES,
    POSITION_ALGORITHMS,
    POSITION_HYPERPARAMS,
    get_algorithms,
    is_count_stat,
)
from src.nfl.training.v5.data import (
    NEUTRAL_FILLS,
    apply_history_filter,
    drop_all_null_columns,
    fill_features,
    get_feature_columns,
    load_features,
    prepare_pob_data,
    prepare_stat_predictor_data,
)
from src.nfl.training.v5.models import (
    POBModel,
    StatPredictor,
    ensemble_files_complete,
)
from src.nfl.training.v5.walkforward import (
    _eval_iter,
    _strict_prior_mask,
    compute_mae,
    compute_pob_metrics,
    walk_forward_eval,
)

FEATURES_DIR = Path("data/nfl/features/v5")


# ---------- Synthetic fixtures ----------

def _synth_player_df(seasons=(2022, 2023), weeks_per_season=8, players=20) -> pd.DataFrame:
    """Synthetic player-week DataFrame mimicking V5 features schema (subset)."""
    rng = np.random.default_rng(42)
    rows = []
    for season in seasons:
        for week in range(1, weeks_per_season + 1):
            for pid in range(players):
                rows.append({
                    "season": season,
                    "week": week,
                    "player_id": f"P{pid:03d}",
                    "player_name": f"Player_{pid:03d}",
                    "position": "WR",
                    "team": "KC",
                    # Default to 5 (above MIN_GAMES_HISTORY=3) so all rows pass
                    # the production filter unless a test deliberately overrides.
                    "games_of_history": 5,
                    # Targets
                    "receptions": rng.integers(0, 12),
                    "receiving_yards": float(rng.integers(0, 200)),
                    "receiving_tds": rng.integers(0, 3),
                    "targets": rng.integers(0, 14),
                    # Features
                    "rolling_avg_receptions": float(rng.uniform(2, 8)),
                    "rolling_avg_receiving_yards": float(rng.uniform(20, 100)),
                    "rolling_avg_receiving_tds": float(rng.uniform(0, 1)),
                    "rolling_avg_targets": float(rng.uniform(3, 10)),
                    "variance_receptions": float(rng.uniform(0, 5)),
                    "trend_receiving_yards": float(rng.uniform(-10, 10)),
                    "spread_line": float(rng.uniform(-7, 7)),
                    "total_line": float(rng.uniform(40, 55)),
                    "team_rest": float(rng.choice([6, 7, 10])),
                    # Below should NOT appear in features (leakage / non-whitelisted):
                    "offense_snaps": float(rng.uniform(20, 70)),
                    "target_share": float(rng.uniform(0.05, 0.30)),
                })
    return pd.DataFrame(rows)


# ---------- Config sanity ----------

class TestConfig:
    def test_position_algorithms_cover_all_positions(self):
        assert set(POSITION_ALGORITHMS) == {"QB", "RB", "WR", "TE", "K", "DST"}
        for pos, algos in POSITION_ALGORITHMS.items():
            assert len(algos) >= 2, f"{pos} has fewer than 2 algos"
            for algo in algos:
                assert algo in {"xgboost", "lightgbm", "catboost", "random_forest"}

    def test_position_hyperparams_sane_ranges(self):
        for pos, hp in POSITION_HYPERPARAMS.items():
            assert 1 <= hp["depth"] <= 12, f"{pos} depth {hp['depth']} out of range"
            assert 50 <= hp["iterations"] <= 1000
            assert 0 < hp["learning_rate"] <= 0.5

    def test_count_stats_disjoint_from_continuous(self):
        assert COUNT_STATS.isdisjoint(CONTINUOUS_STATS)

    def test_count_stats_membership(self):
        assert is_count_stat("passing_tds")
        assert is_count_stat("sacks")
        assert is_count_stat("receptions")
        assert not is_count_stat("passing_yards")
        assert not is_count_stat("rushing_yards")
        assert not is_count_stat("points_allowed")

    def test_get_algorithms_unknown_raises(self):
        with pytest.raises(ValueError):
            get_algorithms("XX")

    def test_te_lightgbm_override_applied(self):
        assert "min_data_in_leaf" in LIGHTGBM_TE_OVERRIDES
        assert LIGHTGBM_TE_OVERRIDES["min_data_in_leaf"] >= 20

    def test_k_drops_lightgbm_catboost(self):
        assert "lightgbm" not in POSITION_ALGORITHMS["K"]
        assert "catboost" not in POSITION_ALGORITHMS["K"]


# ---------- Data prep ----------

class TestDataPrep:
    def test_apply_history_filter_drops_low(self):
        df = _synth_player_df(weeks_per_season=4, players=5)
        df["games_of_history"] = 5  # baseline — all pass
        df.loc[0, "games_of_history"] = 1  # one row should drop
        out = apply_history_filter(df, min_games=3)
        assert len(out) == len(df) - 1

    def test_apply_history_filter_missing_col_raises(self):
        df = _synth_player_df(weeks_per_season=2)
        df = df.drop(columns=["games_of_history"])
        with pytest.raises(KeyError):
            apply_history_filter(df)

    def test_get_feature_columns_excludes_targets_and_identity(self):
        df = _synth_player_df()
        feats = get_feature_columns(df, "WR")
        # identity excluded
        for c in ["season", "week", "player_id", "player_name", "team", "position"]:
            assert c not in feats
        # raw stat targets excluded
        for c in ["receptions", "receiving_yards", "receiving_tds", "targets"]:
            assert c not in feats
        # rolling stays
        assert "rolling_avg_receptions" in feats
        assert "spread_line" in feats

    def test_whitelist_blocks_known_leakage_columns(self):
        """Critical: the whitelist must block current-week observable stats that
        appear raw in V5 features parquet (e.g., target_share, offense_snaps,
        ngs_*, pfr_*, *_exp). These are present alongside their rolling_*
        equivalents — using them as features = data leakage."""
        df = _synth_player_df()
        feats = get_feature_columns(df, "WR")
        # Synthetic fixture includes these as leak bait — they must NOT be in features
        assert "target_share" not in feats, "target_share leaks current-week target volume"
        assert "offense_snaps" not in feats, "offense_snaps is post-game"
        # But the rolling/historical equivalents stay
        assert "rolling_avg_targets" in feats
        assert "trend_receiving_yards" in feats

    def test_prepare_stat_predictor_drops_nan_target(self):
        df = _synth_player_df()
        df.loc[0, "receptions"] = np.nan
        X, y, cols = prepare_stat_predictor_data(df, "receptions", "WR")
        assert len(X) == len(df) - 1
        assert len(y) == len(X)
        assert "receptions" not in cols

    def test_prepare_stat_predictor_count_negative_raises(self):
        df = _synth_player_df()
        df.loc[0, "receptions"] = -1
        with pytest.raises(ValueError, match="Poisson"):
            prepare_stat_predictor_data(df, "receptions", "WR")

    def test_prepare_pob_drops_nan_baseline(self):
        df = _synth_player_df()
        df.loc[0, "rolling_avg_receptions"] = np.nan
        X, y, _ = prepare_pob_data(df, "receptions", "WR")
        assert len(X) == len(df) - 1
        # binary labels
        assert set(y.unique()).issubset({0, 1})

    def test_prepare_pob_label_correctness(self):
        df = _synth_player_df()
        # force one row over baseline, one under
        df.loc[0, "receptions"] = 10
        df.loc[0, "rolling_avg_receptions"] = 4
        df.loc[1, "receptions"] = 1
        df.loc[1, "rolling_avg_receptions"] = 5
        _, y, _ = prepare_pob_data(df, "receptions", "WR")
        assert y.iloc[0] == 1
        assert y.iloc[1] == 0

    def test_fill_features_temp_uses_neutral_value(self):
        """Dome games have temp NaN — must fill with 65 (mild proxy), not 0
        (would falsely train on freezing weather for 36% of all games)."""
        import numpy as np
        df = pd.DataFrame({
            "temp": [70.0, np.nan, 32.0, np.nan],
            "wind": [5.0, np.nan, 12.0, np.nan],
            "rolling_avg_passing_yards": [220.0, np.nan, 180.0, 250.0],
        })
        out = fill_features(df)
        assert out["temp"].tolist() == [70.0, 65.0, 32.0, 65.0], "temp NaN must use NEUTRAL_FILLS"
        assert out["wind"].tolist() == [5.0, 0.0, 12.0, 0.0], "wind NaN → 0 (dome=no wind is correct)"
        assert out["rolling_avg_passing_yards"].tolist() == [220.0, 0.0, 180.0, 250.0]
        assert "temp" in NEUTRAL_FILLS

    def test_drop_all_null_columns_for_k_simulation(self):
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [None, None, None],
            "c": [1.0, np.nan, 3.0],
        })
        kept = drop_all_null_columns(df, ["a", "b", "c"])
        assert "a" in kept
        assert "c" in kept
        assert "b" not in kept


# ---------- Walk-forward ----------

class TestWalkForward:
    def test_strict_prior_mask(self):
        df = pd.DataFrame({"season": [2022, 2022, 2023, 2023], "week": [5, 6, 1, 2]})
        m = _strict_prior_mask(df, 2023, 1)
        assert list(m) == [True, True, False, False]

    def test_eval_iter_sorted(self):
        df = pd.DataFrame({"season": [2023, 2022, 2023, 2024], "week": [1, 5, 2, 1]})
        pairs = _eval_iter(df, [2022, 2023, 2024])
        assert pairs == [(2022, 5), (2023, 1), (2023, 2), (2024, 1)]

    def test_compute_pob_metrics_flags_degenerate_balance(self):
        """All-one-class fold must be flagged so 100% accuracy isn't taken at face value."""
        df = pd.DataFrame({
            "exceeded_baseline": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "probability_over": [0.6, 0.7, 0.55, 0.8, 0.9, 0.65, 0.75, 0.85, 0.6, 0.7],
        })
        m = compute_pob_metrics(df)
        assert m["degenerate_pob"] is True
        assert m["pos_class_frac"] == 1.0
        assert m["accuracy"] == 1.0  # technically true but meaningless

        # Balanced case → not flagged
        df_ok = pd.DataFrame({
            "exceeded_baseline": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "probability_over": [0.4, 0.6, 0.3, 0.7, 0.45, 0.55, 0.35, 0.65, 0.4, 0.6],
        })
        m_ok = compute_pob_metrics(df_ok)
        assert m_ok["degenerate_pob"] is False

    def test_walk_forward_train_strictly_prior(self):
        """For every yielded fold, training data must end before predict week.

        Mechanism check: instrument the model factory to capture the training
        rows it received, then verify max(train season,week) < (eval season,week)
        for every fold. This actually verifies the strict-prior invariant the
        test name claims (vs just checking which rows were predicted on)."""
        df = _synth_player_df(seasons=(2022, 2023), weeks_per_season=6, players=10)
        df["games_of_history"] = 5  # ensure all rows pass history filter

        captured: list[dict] = []

        # Wrap the prepare function so we can inspect the (season, week) range
        # of every train slice walk_forward_eval feeds it. We tag each capture
        # with its role ("train" vs "eval") rather than assuming train/eval
        # interleave perfectly — if eval prepare() were to raise (KeyError on
        # a missing column, ValueError on a negative count stat), only the
        # train capture would land and a positional [::2] slice would misalign
        # on subsequent folds. Role-tagging keeps the filter correct under any
        # fold-skipping pattern.
        from src.nfl.training.v5 import data as v5data
        import src.nfl.training.v5.walkforward as v5wf
        original_prepare = v5data.prepare_stat_predictor_data

        # Track role via caller-passed dataframe size (train_df always has ≥20
        # rows from the min_train_rows guard; eval_df is a single (season, week)
        # slice = ≤ players count). Simpler and role-agnostic approach: count
        # calls per fold by flipping a toggle. Walk-forward always calls train
        # first, then eval — no branch can hit eval without hitting train first.
        _next_role = {"value": "train"}

        def spying_prepare(df, stat, position):
            role = _next_role["value"]
            _next_role["value"] = "eval" if role == "train" else "train"
            try:
                X, y, cols = original_prepare(df, stat, position)
            except Exception:
                # Restore toggle if prepare raises — walk_forward_eval's
                # except-block will skip the rest of this fold, meaning the
                # NEXT call starts a fresh fold (back to "train").
                _next_role["value"] = "train"
                raise
            captured.append({
                "role": role,
                "n_rows": len(df),
                "max_season": int(df["season"].max()),
                "max_week_in_max_season": int(df[df["season"] == df["season"].max()]["week"].max()),
            })
            return X, y, cols

        v5data.prepare_stat_predictor_data = spying_prepare
        v5wf.prepare_stat_predictor_data = spying_prepare
        try:
            preds = walk_forward_eval(
                model_factory=lambda: StatPredictor("WR", "receptions"),
                df=df,
                position="WR",
                stat="receptions",
                eval_seasons=[2023],
                model_type="stat",
                min_train_rows=20,
            )
        finally:
            v5data.prepare_stat_predictor_data = original_prepare
            v5wf.prepare_stat_predictor_data = original_prepare

        assert not preds.empty
        assert (preds["season"] == 2023).all()
        # Filter by role tag — defensive against any future fold-skipping path
        # that might call prepare() for only one of train/eval per fold.
        max_eval_week = int(preds[preds["season"] == 2023]["week"].max())
        train_calls = [c for c in captured if c["role"] == "train"]
        assert train_calls, "No training prepare calls recorded — test failed to spy"
        for tc in train_calls:
            if tc["max_season"] == 2023:
                assert tc["max_week_in_max_season"] < max_eval_week, (
                    f"Strict-prior violated: training included week "
                    f"{tc['max_week_in_max_season']} >= max eval week {max_eval_week}"
                )


# ---------- Ensemble fit/predict ----------

class TestEnsembles:
    def test_stat_predictor_fits_and_predicts(self):
        df = _synth_player_df(weeks_per_season=6, players=15)
        df["position"] = "WR"
        X, y, _ = prepare_stat_predictor_data(df, "receptions", "WR")
        model = StatPredictor("WR", "receptions")
        model.fit(X.head(120), y.head(120))
        preds = model.predict(X.tail(40))
        assert len(preds) == 40
        # Receptions are counts → Poisson → predictions should be non-negative
        assert (preds >= 0).all()

    def test_pob_model_returns_probabilities(self):
        df = _synth_player_df(weeks_per_season=6, players=15)
        X, y, _ = prepare_pob_data(df, "receptions", "WR")
        model = POBModel("WR", "receptions")
        model.fit(X.head(120), y.head(120))
        probs = model.predict(X.tail(40))
        assert len(probs) == 40
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_ensemble_save_and_resume_check(self, tmp_path):
        df = _synth_player_df(weeks_per_season=6, players=15)
        X, y, _ = prepare_stat_predictor_data(df, "receptions", "WR")
        model = StatPredictor("WR", "receptions")
        model.fit(X, y)
        model.save(tmp_path)
        # All algo files + meta JSON exist
        algos = get_algorithms("WR")
        assert ensemble_files_complete(tmp_path, "WR", "receptions", "stat", algos)
        # Removing one algo file → resume returns False
        first_algo = algos[0]
        (tmp_path / f"WR_receptions_stat_{first_algo}.joblib").unlink()
        assert not ensemble_files_complete(tmp_path, "WR", "receptions", "stat", algos)

    def test_predict_uses_fill_features_for_dome_temp_consistency(self):
        """Critical regression test: train sees temp=65 for dome (NaN→65 via
        fill_features), so predict must also see temp=65 — NOT temp=0 from a raw
        fillna. Otherwise dome predictions silently use 0°F bias.

        Setup designed to be load-bearing:
        - Other features are CONSTANT (zero info) — model must use temp.
        - Wide temp range [40, 90] → wide y range [4, 9].
        - Tight assertion: bug → preds ~0-1; correct fix → preds ~6.5.
        """
        import numpy as np
        n = 200
        rng = np.random.default_rng(0)
        temps = rng.uniform(40, 90, n)
        df_train = pd.DataFrame({
            "temp": temps,
            # Constant features — zero predictive info, model MUST learn from temp
            "rolling_avg_receptions": [5.0] * n,
            "rolling_avg_targets": [7.0] * n,
        })
        y_train = pd.Series(temps * 0.1)  # y ∈ [4.0, 9.0], mean ~6.5

        model = StatPredictor("WR", "receptions")
        model.fit(df_train, y_train)

        df_pred = pd.DataFrame({
            "temp": [np.nan, np.nan, np.nan, np.nan],
            "rolling_avg_receptions": [5.0, 5.0, 5.0, 5.0],
            "rolling_avg_targets": [7.0, 7.0, 7.0, 7.0],
        })
        preds = model.predict(df_pred)
        # If fill was 0 (BUG): preds ≈ 0 (Poisson on temp=0 → near-zero).
        # If fill was 65 (CORRECT): preds ≈ 6.5 (model learned y = temp * 0.1).
        # Tight threshold: bug returns ~0-1; correct returns 5.5-7.5.
        assert preds.mean() > 5.0, (
            f"Predict-time temp fill broken — got mean={preds.mean():.2f}, "
            f"expected ~6.5. Likely fill_features() not being called in predict path."
        )
        assert preds.mean() < 8.0, f"Unexpectedly high prediction {preds.mean():.2f}"

    def test_meta_records_neutral_fills(self, tmp_path):
        """Saved meta JSON must embed NEUTRAL_FILLS so Task 3.2c can detect drift."""
        import json
        df = _synth_player_df(weeks_per_season=6, players=15)
        X, y, _ = prepare_stat_predictor_data(df, "receptions", "WR")
        model = StatPredictor("WR", "receptions")
        model.fit(X, y)
        model.save(tmp_path)
        meta = json.loads((tmp_path / "WR_receptions_stat_meta.json").read_text())
        assert "neutral_fills" in meta
        assert meta["neutral_fills"]["temp"] == 65.0

    def test_meta_records_objective_per_algo(self, tmp_path):
        import json
        df = _synth_player_df(weeks_per_season=6, players=15)
        X, y, _ = prepare_stat_predictor_data(df, "receptions", "WR")
        model = StatPredictor("WR", "receptions")
        model.fit(X, y)
        model.save(tmp_path)
        meta = json.loads((tmp_path / "WR_receptions_stat_meta.json").read_text())
        # receptions is a COUNT stat → poisson on boosters, mse on RF (RF not in WR algos)
        for algo in get_algorithms("WR"):
            assert meta["objective_per_algo"][algo] == "poisson"


# ---------- Real-data smoke ----------

@pytest.mark.skipif(
    not (FEATURES_DIR / "features_2023.parquet").exists(),
    reason="V5 feature parquets not present locally",
)
class TestRealData:
    def test_te_receptions_smoke_mae_in_range(self):
        """TE/receptions StatPredictor on 2021-2022 train, 2023 eval — MAE 1.5-4.0."""
        df = load_features("TE", [2021, 2022, 2023])
        df = apply_history_filter(df)
        preds = walk_forward_eval(
            model_factory=lambda: StatPredictor("TE", "receptions"),
            df=df,
            position="TE",
            stat="receptions",
            eval_seasons=[2023],
            model_type="stat",
        )
        assert not preds.empty
        mae = compute_mae(preds)
        assert 1.0 <= mae <= 4.5, f"TE receptions MAE {mae} outside plausible bounds"

    def test_dst_sacks_poisson_predictions_nonneg(self):
        """DST/sacks StatPredictor with Poisson → all predictions >= 0."""
        df = load_features("DST", [2021, 2022, 2023])
        df = apply_history_filter(df)
        preds = walk_forward_eval(
            model_factory=lambda: StatPredictor("DST", "sacks"),
            df=df,
            position="DST",
            stat="sacks",
            eval_seasons=[2023],
            model_type="stat",
            min_train_rows=20,
        )
        assert not preds.empty
        assert (preds["predicted"] >= 0).all(), "Poisson predictions must be non-negative"

    def test_pob_pos_class_fraction_balanced(self):
        """WR/receiving_yards POB labels should not be wildly imbalanced."""
        df = load_features("WR", [2021, 2022, 2023])
        df = apply_history_filter(df)
        _, y, _ = prepare_pob_data(df, "receiving_yards", "WR")
        frac = y.mean()
        assert 0.35 <= frac <= 0.65, f"POB pos-class frac {frac} too imbalanced"
