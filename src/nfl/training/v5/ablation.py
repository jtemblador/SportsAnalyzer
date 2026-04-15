"""V5 feature-group ablation study.

For each (position, feature_group) pair, retrain StatPredictors with that
group's columns DROPPED from the feature matrix, measure MAE vs baseline
(Task 3.2 output), decide keep/drop per the 0.05 MAE delta threshold.

Ablation runs do NOT write .joblib files — only meta JSON + aggregate CSV.
Actual production models get rebuilt in Task 3.2c after ablation decisions.

Usage:
    python -m src.nfl.training.v5.ablation                       # all 23 runs
    python -m src.nfl.training.v5.ablation --position QB --group advanced
    python -m src.nfl.training.v5.ablation --eval-seasons 2023 2024  # faster
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.nfl.features.v5.config import get_feature_columns_by_group
from src.nfl.training.v5.config import (
    STATS_TO_PREDICT,
    TRAIN_SEASONS,
    get_algorithms,
)
from src.nfl.training.v5.data import (
    apply_history_filter,
    get_feature_columns,
    load_features,
)
from src.nfl.training.v5.models import StatPredictor
from src.nfl.training.v5.walkforward import compute_mae, walk_forward_eval

REPO_ROOT = Path(__file__).resolve().parents[4]  # ablation.py is 5 levels deep

# Which groups apply to which positions. Player positions have all 4 standard
# groups (rolling/context/usage/advanced). DST only has rolling/context/
# opp_offense (usage + advanced are empty for DST rows).
ABLATION_GROUPS: dict[str, list[str]] = {
    "QB":  ["rolling", "context", "usage", "advanced"],
    "RB":  ["rolling", "context", "usage", "advanced"],
    "WR":  ["rolling", "context", "usage", "advanced"],
    "TE":  ["rolling", "context", "usage", "advanced"],
    "K":   ["rolling", "context", "usage", "advanced"],
    "DST": ["rolling", "context", "opp_offense"],
}
# Default keep/drop threshold. If MAE delta (ablated − baseline) is less than
# this, the group added no measurable signal → drop candidate.
DEFAULT_DROP_THRESHOLD = 0.05


def get_ablation_exclude_columns(df: pd.DataFrame, position: str,
                                 group_to_remove: str) -> list[str]:
    """Intersect the whitelist columns with the named group → list of columns
    to drop for this ablation run.

    We only drop columns that would have been features anyway (whitelist
    membership); dropping a column the model never saw is a no-op.
    """
    whitelist = set(get_feature_columns(df, position))
    group_cols = set(get_feature_columns_by_group(list(df.columns), group_to_remove))
    return sorted(whitelist & group_cols)


def run_position_ablation(
    position: str,
    group_to_remove: str,
    df: pd.DataFrame,
    eval_seasons: list[int],
    min_train_rows: int = 50,
) -> dict:
    """Train all stats for one position with group_to_remove's columns dropped.

    Returns:
        {stat: mae, ...} for every stat in STATS_TO_PREDICT[position], plus
        metadata about which columns were dropped AND the eval_seasons used
        (recorded in meta JSON so resume check can detect window changes).

    NOTE: not thread-safe — monkey-patches module-level
    v5data.get_feature_columns. Orchestrator is sequential.
    """
    exclude_cols = get_ablation_exclude_columns(df, position, group_to_remove)

    # Patch-module: the prepare function reads get_feature_columns live, so we
    # install a filter that drops exclude_cols from the returned feature list.
    import src.nfl.training.v5.data as v5data
    original_get_feature_cols = v5data.get_feature_columns

    def filtered_get_feature_columns(df_local, pos_local):
        cols = original_get_feature_cols(df_local, pos_local)
        return [c for c in cols if c not in exclude_cols]

    v5data.get_feature_columns = filtered_get_feature_columns

    results = {"position": position, "group_removed": group_to_remove,
               "excluded_cols": exclude_cols, "n_excluded": len(exclude_cols),
               "eval_seasons": list(eval_seasons),
               "stats": {}}

    try:
        for stat in STATS_TO_PREDICT[position]:
            preds_df = walk_forward_eval(
                model_factory=lambda p=position, s=stat: StatPredictor(p, s),
                df=df, position=position, stat=stat,
                eval_seasons=eval_seasons, model_type="stat",
                min_train_rows=min_train_rows,
            )
            mae = compute_mae(preds_df) if not preds_df.empty else float("nan")
            results["stats"][stat] = {
                "mae": mae,
                "n_eval_predictions": len(preds_df),
            }
            print(f"    {position}/{stat}: MAE={mae:.3f} "
                  f"({len(preds_df)} preds, dropped {len(exclude_cols)} cols)")
    finally:
        v5data.get_feature_columns = original_get_feature_cols

    return results


def compute_position_aggregate_mae(stat_maes: dict) -> float:
    """Weighted-sum fantasy MAE for a single position.

    For player stats: uses PPR_WEIGHTS (same as load_model_eval.py).
    For DST stats: falls back to FANTASY_DST_WEIGHTS from features/v5/config.py
    (sacks=1, interceptions=2, fumble_recoveries=2, defensive_tds=6, safeties=2).
    Without DST weights, DST ablations would yield delta=0 always (all stats
    skipped → 0 aggregate both baseline and ablated).
    """
    from src.nfl.db.load_model_eval import PPR_WEIGHTS
    from src.nfl.features.v5.config import FANTASY_DST_WEIGHTS
    total = 0.0
    for stat, r in stat_maes.items():
        # PPR for player stats; DST fantasy weights for DST stats
        w = PPR_WEIGHTS.get(stat)
        if w is None:
            w = FANTASY_DST_WEIGHTS.get(stat)
        if w is None:
            continue  # stat not in any scoring formula (e.g., points_allowed uses tiered bonus)
        mae = r["mae"] if isinstance(r, dict) else r
        if pd.notna(mae):
            total += w * mae
    return total * 0.7  # realistic 70% of upper bound


def compare_to_baseline(
    ablation_results: dict,
    baseline_csv_path: Path = REPO_ROOT / "data/nfl/models/v5/_mae_summary_consolidated.csv",
) -> dict:
    """Compute (baseline_agg_mae, ablated_agg_mae, delta) per (position, group)."""
    baseline_csv_path = Path(baseline_csv_path)
    if not baseline_csv_path.exists():
        raise FileNotFoundError(
            f"Baseline CSV not found: {baseline_csv_path}\n"
            f"  Task 3.2 training output is required before ablation. Either:\n"
            f"    1. Run colab/v5_training.ipynb first to produce it, OR\n"
            f"    2. Pass baseline_csv_path explicitly to compare_to_baseline()."
        )
    baseline = pd.read_csv(baseline_csv_path)
    baseline_stat = baseline[baseline["model_type"] == "stat"]

    pos = ablation_results["position"]
    group = ablation_results["group_removed"]
    baseline_sub = baseline_stat[baseline_stat["position"] == pos]

    # Build baseline dict of {stat: mae} for this position
    baseline_stat_maes = {}
    for _, row in baseline_sub.iterrows():
        baseline_stat_maes[row["stat"]] = {"mae": row["mae_v5"]}

    baseline_agg = compute_position_aggregate_mae(baseline_stat_maes)
    ablated_agg = compute_position_aggregate_mae(ablation_results["stats"])
    delta = ablated_agg - baseline_agg

    return {
        "position": pos,
        "group_removed": group,
        "baseline_agg_mae": baseline_agg,
        "ablated_agg_mae": ablated_agg,
        "delta": delta,
    }


def apply_drop_threshold(
    comparison: dict, threshold: float = DEFAULT_DROP_THRESHOLD,
) -> str:
    """Return 'keep' / 'drop' / 'borderline' for one (position, group).

    delta = ablated_mae - baseline_mae
    - delta >= threshold                       → group helped → KEEP
    - delta in [threshold*0.8, threshold)      → near-miss → BORDERLINE (re-eval full window)
    - delta in (-threshold*0.2, threshold*0.8) → group added no signal → DROP
    - delta <= -threshold*0.2                  → group was noise/harmful → DROP
    The borderline band sits just BELOW the keep threshold (not around zero)
    because that's where a genuinely ambiguous "did the group help?" signal
    lives. A small negative delta just means dropping the group was neutral
    or slightly helpful — no ambiguity.
    """
    delta = comparison["delta"]
    if delta >= threshold:
        return "keep"
    if threshold * 0.8 <= delta < threshold:
        return "borderline"
    return "drop"


def save_ablation_result(result: dict, comparison: dict, decision: str,
                         output_dir: Path) -> None:
    """Write per-(position, group) meta JSON. Atomic."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fname = f"ablation_{result['position']}_remove_{result['group_removed']}.json"
    path = output_dir / fname
    payload = {
        **result,
        **{k: v for k, v in comparison.items() if k not in result},
        "decision": decision,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    os.replace(tmp, path)


def ablation_result_complete(position: str, group: str, output_dir: Path,
                              expected_eval_seasons: list[int] | None = None) -> bool:
    """Return True if a valid ablation JSON exists for this (position, group).

    If expected_eval_seasons is provided, also verify the saved result was
    computed on the SAME eval window — otherwise the saved result is stale
    (e.g., cached [2023, 2024] result but caller wants [2021-2024] re-run).
    Returns False on stale window so caller retrains on the new window.
    """
    fname = f"ablation_{position}_remove_{group}.json"
    path = output_dir / fname
    if not path.exists():
        return False
    if expected_eval_seasons is None:
        return True
    try:
        saved = json.loads(path.read_text())
        saved_seasons = saved.get("eval_seasons")
        if saved_seasons is None:
            return False  # legacy result without seasons — retrain to record
        return sorted(saved_seasons) == sorted(expected_eval_seasons)
    except (json.JSONDecodeError, OSError):
        return False  # corrupted → retrain


def aggregate_summary_csv(output_dir: Path) -> pd.DataFrame:
    """Scan output_dir for ablation JSONs, produce the final summary table."""
    rows = []
    for jf in sorted(output_dir.glob("ablation_*_remove_*.json")):
        d = json.loads(jf.read_text())
        rows.append({
            "position": d["position"],
            "group_removed": d["group_removed"],
            "baseline_agg_mae": d["baseline_agg_mae"],
            "ablated_agg_mae": d["ablated_agg_mae"],
            "delta": d["delta"],
            "decision": d["decision"],
            "n_excluded": d["n_excluded"],
        })
    return pd.DataFrame(rows).sort_values(["position", "group_removed"])


def run_all_ablations(
    positions: list[str] | None = None,
    groups: list[str] | None = None,
    eval_seasons: list[int] | None = None,
    output_dir: Path | None = None,
    threshold: float = DEFAULT_DROP_THRESHOLD,
    force: bool = False,
) -> pd.DataFrame:
    """Orchestrator. Resumable per (position, group)."""
    positions = positions or list(ABLATION_GROUPS.keys())
    eval_seasons = eval_seasons or [2023, 2024]
    output_dir = output_dir or (REPO_ROOT / "data/nfl/models/v5_ablation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # .tmp orphan sweep (same pattern as train.py)
    for o in output_dir.glob("*.tmp"):
        o.unlink(missing_ok=True)

    for position in positions:
        applicable_groups = groups or ABLATION_GROUPS[position]
        # Skip positions the specified group doesn't apply to
        applicable_groups = [g for g in applicable_groups
                             if g in ABLATION_GROUPS[position]]
        if not applicable_groups:
            continue

        print(f"\n=== Loading features for {position} ===")
        try:
            df = load_features(position, TRAIN_SEASONS)
        except FileNotFoundError as e:
            print(f"  SKIP {position}: {e}")
            continue
        df = apply_history_filter(df)
        print(f"  {len(df)} rows after history filter")

        for group in applicable_groups:
            if not force and ablation_result_complete(position, group, output_dir,
                                                      expected_eval_seasons=eval_seasons):
                print(f"  SKIP {position}/remove_{group} — complete "
                      f"(matching eval_seasons)")
                continue
            exclude_preview = get_ablation_exclude_columns(df, position, group)
            if not exclude_preview:
                print(f"  SKIP {position}/remove_{group} — 0 columns would be dropped (empty group)")
                continue
            print(f"\n  ABLATE {position} remove {group} "
                  f"({len(exclude_preview)} cols, algos={get_algorithms(position)})")
            result = run_position_ablation(position, group, df, eval_seasons)
            comparison = compare_to_baseline(result)
            decision = apply_drop_threshold(comparison, threshold)
            save_ablation_result(result, comparison, decision, output_dir)
            print(f"  → baseline={comparison['baseline_agg_mae']:.3f}, "
                  f"ablated={comparison['ablated_agg_mae']:.3f}, "
                  f"delta={comparison['delta']:+.3f} → {decision.upper()}")

    summary = aggregate_summary_csv(output_dir)
    summary_csv = output_dir / "_ablation_summary.csv"
    # Atomic write — prevents corruption on mid-write crash
    tmp = summary_csv.with_suffix(".csv.tmp")
    summary.to_csv(tmp, index=False)
    os.replace(tmp, summary_csv)
    print(f"\nWrote summary: {summary_csv}")

    # Produce the Task 3.2c feature-list handoff artifact
    validated_path = output_dir / "_validated_feature_groups.json"
    write_validated_feature_list(summary, validated_path)
    print(f"Wrote validated feature groups: {validated_path}")

    return summary


def write_validated_feature_list(summary: pd.DataFrame, output_path: Path) -> None:
    """Produce the feature-group keep list per position for Task 3.2c.

    Output JSON shape:
        {
            "QB":  {"keep": ["rolling", "context", "usage", "advanced"],
                    "fully_ablated": true},
            "WR":  {"keep": ["rolling", "context", "usage"],
                    "fully_ablated": true},
            ...
        }

    - `keep` = groups to retain for production (decision ∈ {keep, borderline}).
    - `fully_ablated` = true iff every group in ABLATION_GROUPS[pos] has a row
      in the summary. If false, Task 3.2c MUST re-run the missing (pos, group)
      pairs before trusting this list (conservative defaults were used).

    borderline decisions are treated as KEEP (conservative — don't drop on
    ambiguous evidence).
    """
    result: dict[str, dict] = {}
    for position, groups in ABLATION_GROUPS.items():
        pos_rows = summary[summary["position"] == position]
        kept: list[str] = []
        all_covered = True
        for group in groups:
            matching = pos_rows[pos_rows["group_removed"] == group]
            if matching.empty:
                # No ablation ran — conservatively keep, but flag as not fully ablated
                kept.append(group)
                all_covered = False
                continue
            decision = matching.iloc[0]["decision"]
            if decision in ("keep", "borderline"):
                kept.append(group)
        result[position] = {"keep": kept, "fully_ablated": bool(all_covered)}
    tmp = output_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2))
    os.replace(tmp, output_path)


def ablation_jsons_to_db_csv(output_dir: Path, group: str,
                              csv_path: Path | None = None) -> Path:
    """Convert ablation meta JSONs for ONE group to the CSV schema
    load_model_eval.py expects (54 rows per version = 27 stat + 27 pob).

    Ablation is stat-only by design, but load_model_eval.py asserts 54 rows
    (its schema is generic). This converter emits the 27 real stat rows
    PLUS 27 stub POB rows (all POB metrics NULL) so the assertion passes
    and the version's DB entry reflects "ablation: no POB data".

    Run once per group:
        python src/nfl/db/load_model_eval.py \\
            --csv <output_dir>/_ablation_db_rows_rolling.csv \\
            --version v5_ablated_rolling
    """
    if csv_path is None:
        csv_path = output_dir / f"_ablation_db_rows_{group}.csv"

    # Load the meta JSONs FOR THIS GROUP ONLY
    jsons = sorted(output_dir.glob(f"ablation_*_remove_{group}.json"))
    if not jsons:
        raise FileNotFoundError(
            f"No ablation JSONs found for group '{group}' in {output_dir}"
        )

    version = f"v5_ablated_{group}"
    rows = []
    for jf in jsons:
        d = json.loads(jf.read_text())
        pos = d["position"]
        algos_str = ",".join(get_algorithms(pos))
        # Real stat rows from the ablation
        for stat, stat_result in d["stats"].items():
            rows.append({
                "version": version, "position": pos, "stat": stat,
                "model_type": "stat", "algorithms": algos_str,
                "n_features": None, "n_train_rows": None,
                "mae_v5": stat_result["mae"],
                "accuracy": None, "auc": None, "pos_class_frac": None,
                "degenerate_pob": 0,
                "n_eval_predictions": stat_result["n_eval_predictions"],
            })
            # Stub POB row so load_model_eval's 54-row assertion passes.
            # POB metrics NULL = "ablation did not evaluate POB classifiers."
            rows.append({
                "version": version, "position": pos, "stat": stat,
                "model_type": "pob", "algorithms": algos_str,
                "n_features": None, "n_train_rows": None,
                "mae_v5": None,
                "accuracy": None, "auc": None, "pos_class_frac": None,
                "degenerate_pob": 0,
                "n_eval_predictions": None,
            })

    df = pd.DataFrame(rows)
    tmp = csv_path.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, csv_path)
    return csv_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", nargs="*", default=None)
    parser.add_argument("--groups", nargs="*", default=None)
    parser.add_argument("--eval-seasons", nargs="*", type=int,
                        default=None, help="Default [2023, 2024] for speed")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--threshold", type=float, default=DEFAULT_DROP_THRESHOLD)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    summary = run_all_ablations(
        positions=args.positions,
        groups=args.groups,
        eval_seasons=args.eval_seasons,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        threshold=args.threshold,
        force=args.force,
    )
    print("\n" + "=" * 60)
    print("ABLATION SUMMARY")
    print("=" * 60)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
