#!/usr/bin/env python3
"""
Compare accuracy between different model versions side-by-side.
Usage: python compare_versions.py v1_baseline_mae5.14 v2_variance_trends 10 11 12
"""

import sys
import subprocess
from pathlib import Path

def run_validation(version, weeks):
    """Run validation for a specific version"""
    cmd = ["python3", "testing/validate_accuracy.py", version] + [str(w) for w in weeks]

    print(f"\n{'='*80}")
    print(f"Running validation for: {version}")
    print(f"{'='*80}\n")

    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode

def main():
    if len(sys.argv) < 4:
        print("Usage: python compare_versions.py <version1> <version2> <week1> [week2] ...")
        print("\nExample:")
        print("  python compare_versions.py v1_baseline_mae5.14 v2_variance_trends 10 11 12")
        return 1

    version1 = sys.argv[1]
    version2 = sys.argv[2]
    weeks = [int(w) for w in sys.argv[3:]]

    print("\n" + "="*80)
    print("  MODEL VERSION COMPARISON")
    print("="*80)
    print(f"  Version 1: {version1}")
    print(f"  Version 2: {version2}")
    print(f"  Weeks: {', '.join(map(str, weeks))}")
    print("="*80)

    # Run validation for version 1
    run_validation(version1, weeks)

    # Run validation for version 2
    run_validation(version2, weeks)

    print("\n" + "="*80)
    print("  COMPARISON COMPLETE")
    print("="*80)
    print("\nScroll up to compare the MAE, RMSE, and R² values between versions.")
    print("Look for improvements in:")
    print("  - Lower MAE (better accuracy)")
    print("  - Lower RMSE (better overall error)")
    print("  - Higher R² (better predictive power)")
    print("="*80 + "\n")

    return 0

if __name__ == "__main__":
    sys.exit(main())
