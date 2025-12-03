#!/usr/bin/env python3
"""
File: testing/validate_accuracy.py

Validate prediction accuracy by comparing predictions to actual results.
This helps determine if models need retraining.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_predictions(season, week, version="v1_baseline_mae5.14"):
    """Load prediction file for a specific week and version"""
    pred_file = project_root / "data" / "nfl" / "predictions" / version / f"predictions_{season}_week_{week}.parquet"

    if not pred_file.exists():
        return None

    return pd.read_parquet(pred_file)


def load_actuals(season, week):
    """Load actual stats for a specific week"""
    actual_file = project_root / "data" / "nfl" / "raw" / f"player_stats_{season}_week_{week}.parquet"

    if not actual_file.exists():
        return None

    return pd.read_parquet(actual_file)


def calculate_metrics(y_true, y_pred):
    """Calculate accuracy metrics"""

    # Remove any NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return None

    # Mean Absolute Error
    mae = np.mean(np.abs(y_true - y_pred))

    # Root Mean Squared Error
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # Mean Absolute Percentage Error (avoid division by zero)
    mape_mask = y_true != 0
    if mape_mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mape_mask] - y_pred[mape_mask]) / y_true[mape_mask])) * 100
    else:
        mape = np.nan

    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

    return {
        'n': len(y_true),
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2
    }


def validate_week(season, week, version="v1_baseline_mae5.14"):
    """Validate predictions for a single week"""

    print(f"\n{'='*80}")
    print(f"  VALIDATING {season} WEEK {week} ({version})")
    print(f"{'='*80}\n")

    # Load data
    pred_df = load_predictions(season, week, version=version)
    actual_df = load_actuals(season, week)

    if pred_df is None:
        print(f"❌ No predictions found for Week {week}")
        return None

    if actual_df is None:
        print(f"❌ No actual stats found for Week {week}")
        return None

    print(f"✓ Loaded {len(pred_df)} predictions")
    print(f"✓ Loaded {len(actual_df)} actual player stats\n")

    # Filter for EVOB predictions (value predictions)
    evob_pred = pred_df[pred_df['model_type'] == 'evob'].copy()

    # Filter for POB predictions (probability predictions)
    pob_pred = pred_df[pred_df['model_type'] == 'pob'].copy()

    results = {
        'season': season,
        'week': week,
        'evob': {},
        'pob': {}
    }

    # Validate EVOB predictions (Fantasy Points)
    print("🎯 FANTASY POINTS (PPR) - EVOB Model")
    print("─" * 80)

    fantasy_pred = evob_pred[evob_pred['stat'] == 'fantasy_points_ppr']

    # Merge predictions with actuals
    merged = fantasy_pred.merge(
        actual_df[['player_id', 'fantasy_points_ppr']],
        on='player_id',
        suffixes=('_pred', '_actual')
    )

    if len(merged) > 0:
        metrics = calculate_metrics(
            merged['fantasy_points_ppr'].values,
            merged['predicted_value'].values
        )

        if metrics:
            results['evob']['fantasy_points_ppr'] = metrics

            print(f"  Players Matched: {metrics['n']}")
            print(f"  MAE:  {metrics['mae']:.2f} points")
            print(f"  RMSE: {metrics['rmse']:.2f} points")
            print(f"  MAPE: {metrics['mape']:.1f}%")
            print(f"  R²:   {metrics['r2']:.3f}")

            # Show best and worst predictions
            merged['error'] = np.abs(merged['fantasy_points_ppr'] - merged['predicted_value'])
            merged = merged.sort_values('error')

            print(f"\n  Best Predictions (lowest error):")
            for i, (_, row) in enumerate(merged.head(3).iterrows(), 1):
                print(f"    {i}. {row['player_name']:<20} - Predicted: {row['predicted_value']:5.1f}, " +
                      f"Actual: {row['fantasy_points_ppr']:5.1f}, Error: {row['error']:5.1f}")

            print(f"\n  Worst Predictions (highest error):")
            for i, (_, row) in enumerate(merged.tail(3).iterrows(), 1):
                print(f"    {i}. {row['player_name']:<20} - Predicted: {row['predicted_value']:5.1f}, " +
                      f"Actual: {row['fantasy_points_ppr']:5.1f}, Error: {row['error']:5.1f}")
    else:
        print("  ❌ No matching players found")

    # Validate POB predictions
    print(f"\n\n📊 PROBABILITY PREDICTIONS (POB) - Baseline Beat Rate")
    print("─" * 80)

    fantasy_pob = pob_pred[pob_pred['stat'] == 'fantasy_points_ppr']

    # Merge POB predictions with actuals
    pob_merged = fantasy_pob.merge(
        actual_df[['player_id', 'fantasy_points_ppr']],
        on='player_id',
        suffixes=('_pred', '_actual')
    )

    if len(pob_merged) > 0:
        # Calculate if player beat their baseline
        pob_merged['beat_baseline'] = pob_merged['fantasy_points_ppr'] > pob_merged['baseline']

        # Calculate calibration bins
        bins = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]
        pob_merged['prob_bin'] = pd.cut(pob_merged['probability_over'], bins=bins)

        overall_accuracy = pob_merged['beat_baseline'].mean() * 100

        print(f"  Players Matched: {len(pob_merged)}")
        print(f"  Overall Beat Rate: {overall_accuracy:.1f}%")
        print(f"\n  Probability Calibration:")
        print(f"  {'Predicted Prob':<20} {'Count':<8} {'Actual Beat %':<15}")
        print(f"  {'-'*43}")

        for bin_range in pob_merged['prob_bin'].cat.categories:
            bin_data = pob_merged[pob_merged['prob_bin'] == bin_range]
            if len(bin_data) > 0:
                actual_beat_pct = bin_data['beat_baseline'].mean() * 100
                avg_pred_prob = bin_data['probability_over'].mean() * 100
                print(f"  {str(bin_range):<20} {len(bin_data):<8} {actual_beat_pct:6.1f}% (pred: {avg_pred_prob:4.1f}%)")

        results['pob']['fantasy_points_ppr'] = {
            'n': len(pob_merged),
            'overall_accuracy': overall_accuracy
        }
    else:
        print("  ❌ No matching players found")

    # Validate by position
    print(f"\n\n📍 ACCURACY BY POSITION")
    print("─" * 80)
    print(f"{'Position':<10} {'Players':<10} {'MAE':<10} {'RMSE':<10} {'R²':<10}")
    print("─" * 80)

    for position in ['QB', 'RB', 'WR', 'TE', 'K']:
        pos_pred = evob_pred[(evob_pred['position'] == position) &
                              (evob_pred['stat'] == 'fantasy_points_ppr')]

        pos_merged = pos_pred.merge(
            actual_df[['player_id', 'fantasy_points_ppr']],
            on='player_id',
            suffixes=('_pred', '_actual')
        )

        if len(pos_merged) > 0:
            metrics = calculate_metrics(
                pos_merged['fantasy_points_ppr'].values,
                pos_merged['predicted_value'].values
            )

            if metrics:
                print(f"{position:<10} {metrics['n']:<10} {metrics['mae']:<10.2f} " +
                      f"{metrics['rmse']:<10.2f} {metrics['r2']:<10.3f}")

                results['evob'][f'{position}_fantasy'] = metrics

    return results


def validate_multiple_weeks(season, weeks, version="v1_baseline_mae5.14"):
    """Validate predictions across multiple weeks"""

    all_results = []

    for week in weeks:
        results = validate_week(season, week, version=version)
        if results:
            all_results.append(results)

    if not all_results:
        print("\n❌ No results to summarize")
        return

    # Summary across all weeks
    print(f"\n\n{'='*80}")
    print(f"  SUMMARY ACROSS WEEKS {min(weeks)}-{max(weeks)}")
    print(f"{'='*80}\n")

    # Aggregate EVOB metrics
    print("🎯 FANTASY POINTS (PPR) - Average Metrics")
    print("─" * 80)

    evob_metrics = [r['evob'].get('fantasy_points_ppr') for r in all_results
                     if 'fantasy_points_ppr' in r['evob']]

    if evob_metrics:
        avg_mae = np.mean([m['mae'] for m in evob_metrics])
        avg_rmse = np.mean([m['rmse'] for m in evob_metrics])
        avg_r2 = np.mean([m['r2'] for m in evob_metrics if not np.isnan(m['r2'])])
        total_n = sum([m['n'] for m in evob_metrics])

        print(f"  Total Predictions: {total_n}")
        print(f"  Average MAE:  {avg_mae:.2f} points")
        print(f"  Average RMSE: {avg_rmse:.2f} points")
        print(f"  Average R²:   {avg_r2:.3f}")

        # Interpretation
        print(f"\n  📋 INTERPRETATION:")
        if avg_mae < 5:
            print(f"  ✓ Excellent accuracy! MAE < 5 points")
        elif avg_mae < 7:
            print(f"  ✓ Good accuracy. MAE in acceptable range (5-7 points)")
        else:
            print(f"  ⚠ MAE > 7 points - consider retraining or feature engineering")

        if avg_r2 > 0.4:
            print(f"  ✓ Strong predictive power (R² > 0.4)")
        elif avg_r2 > 0.3:
            print(f"  ✓ Moderate predictive power (R² > 0.3)")
        else:
            print(f"  ⚠ Weak predictive power (R² < 0.3) - may need model improvements")

    # Aggregate POB metrics
    print(f"\n\n📊 POB MODEL - Average Beat Rate")
    print("─" * 80)

    pob_metrics = [r['pob'].get('fantasy_points_ppr') for r in all_results
                    if 'fantasy_points_ppr' in r['pob']]

    if pob_metrics:
        avg_accuracy = np.mean([m['overall_accuracy'] for m in pob_metrics])
        total_n = sum([m['n'] for m in pob_metrics])

        print(f"  Total Predictions: {total_n}")
        print(f"  Average Beat Rate: {avg_accuracy:.1f}%")

        print(f"\n  📋 INTERPRETATION:")
        if 48 <= avg_accuracy <= 52:
            print(f"  ✓ Well-calibrated (~50% is expected for balanced predictions)")
        elif avg_accuracy > 55:
            print(f"  ⚠ Predictions may be too optimistic")
        elif avg_accuracy < 45:
            print(f"  ⚠ Predictions may be too pessimistic")

    # Position-specific summary
    print(f"\n\n📍 AVERAGE ACCURACY BY POSITION")
    print("─" * 80)
    print(f"{'Position':<10} {'MAE':<15} {'RMSE':<15} {'R²':<15}")
    print("─" * 80)

    for position in ['QB', 'RB', 'WR', 'TE', 'K']:
        pos_metrics = []
        for r in all_results:
            if f'{position}_fantasy' in r['evob']:
                pos_metrics.append(r['evob'][f'{position}_fantasy'])

        if pos_metrics:
            avg_mae = np.mean([m['mae'] for m in pos_metrics])
            avg_rmse = np.mean([m['rmse'] for m in pos_metrics])
            avg_r2 = np.mean([m['r2'] for m in pos_metrics if not np.isnan(m['r2'])])

            print(f"{position:<10} {avg_mae:<15.2f} {avg_rmse:<15.2f} {avg_r2:<15.3f}")

    # Final recommendation
    print(f"\n\n{'='*80}")
    print(f"  RECOMMENDATION")
    print(f"{'='*80}\n")

    if evob_metrics:
        avg_mae = np.mean([m['mae'] for m in evob_metrics])
        avg_r2 = np.mean([m['r2'] for m in evob_metrics if not np.isnan(m['r2'])])

        if avg_mae < 7 and avg_r2 > 0.3:
            print("  ✅ Models are performing well! No retraining needed.")
            print("  → Proceed with generating predictions for remaining weeks")
            print("  → Integrate with dashboard (app.py)")
        elif avg_mae < 8:
            print("  ⚠️  Models are performing adequately but could be improved.")
            print("  → Consider feature engineering improvements")
            print("  → Monitor performance on upcoming weeks")
        else:
            print("  ❌ Models are underperforming. Recommend:")
            print("  → Review feature engineering (src/nfl/feature_engineer.py)")
            print("  → Check for data quality issues")
            print("  → Consider hyperparameter tuning")
            print("  → Retrain models with additional features")

    print()


def main():
    """Main function"""

    print("\n" + "="*80)
    print("  NFL PREDICTION ACCURACY VALIDATOR (VERSIONED)")
    print("="*80)

    # Default: validate weeks 10-12 for 2025
    season = 2025
    weeks = [10, 11, 12]
    version = "v1_baseline_mae5.14"

    # Allow command line override
    # Usage: python validate_accuracy.py [version] [week1] [week2] ...
    if len(sys.argv) > 1:
        # Check if first arg is a version string
        if sys.argv[1].startswith('v'):
            version = sys.argv[1]
            try:
                weeks = [int(w) for w in sys.argv[2:]] if len(sys.argv) > 2 else [10, 11, 12]
            except ValueError:
                print("Usage: python validate_accuracy.py [version] [week1] [week2] ...")
                print("Example: python validate_accuracy.py v2_variance_trends 10 11 12")
                return 1
        else:
            # Assume all args are weeks
            try:
                weeks = [int(w) for w in sys.argv[1:]]
            except ValueError:
                print("Usage: python validate_accuracy.py [version] [week1] [week2] ...")
                print("Example: python validate_accuracy.py v1_baseline_mae5.14 10 11 12")
                return 1

    print(f"  Version: {version}")
    print(f"  Validating predictions for {season} Weeks: {', '.join(map(str, weeks))}")
    print("="*80)

    validate_multiple_weeks(season, weeks, version=version)

    return 0


if __name__ == "__main__":
    sys.exit(main())
