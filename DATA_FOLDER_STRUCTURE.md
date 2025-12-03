# NFL Data Folder Structure

## Overview
The `data/nfl/` folder contains 4 main directories that form the ML pipeline:

```
data/nfl/
├── raw/                    # Raw player statistics from nflverse
├── cleaned/                # Engineered features (versioned in future)
├── models/                 # Trained ML models (versioned)
└── predictions/            # Generated predictions (versioned)
```

---

## 1. RAW Data (`data/nfl/raw/`)

**Purpose:** Original player statistics fetched from nflverse API

**Files:** 103 files (one per week from 2020-2025)
- Format: `player_stats_{season}_week_{week}.parquet`
- Example: `player_stats_2025_week_12.parquet`

**Schema:** 114 columns including:
```python
{
    'player_id': '00-0019596',
    'player_name': 'J.Flacco',
    'position': 'QB',
    'team': 'CIN',
    'opponent_team': 'DAL',
    'season': 2025,
    'week': 12,

    # Actual game stats
    'fantasy_points_ppr': 9.62,
    'passing_yards': 183,
    'passing_tds': 1,
    'passing_interceptions': 0,
    'rushing_yards': 5,
    'receiving_yards': 0,
    'receptions': 0,
    'targets': 0,
    'fg_made': 0,
    'fg_att': 0,

    # Advanced stats (114 total columns)
    'passing_epa': 0.15,
    'rushing_epa': -0.05,
    'target_share': 0.0,
    'air_yards_share': 0.0,
    # ... and many more
}
```

**Size:** ~150KB per file, 15MB total

**Usage:**
- Source data for feature engineering
- Loaded by `FeatureEngineer` to calculate rolling averages
- Used for training targets (next week's actual stats)

---

## 2. CLEANED/Features Data (`data/nfl/cleaned/`)

**Purpose:** Engineered features used for ML training and prediction

**Files:** 102 files (one per week from 2020-2025)
- Format: `features_{season}_week_{week}.parquet`
- Example: `features_2025_week_12.parquet`

**Current Issue:** MIXED SCHEMAS
- Files from weeks 1-11: OLD schema (33 features, NO variance/trend)
- Files from weeks 12-13: NEW schema (42 features, WITH variance/trend)

### V1 Schema (OLD - 33 columns):
```python
{
    'player_id': '00-0019596',
    'player_name': 'J.Flacco',
    'position': 'QB',
    'team': 'CIN',
    'opponent_team': 'DAL',
    'week': 12,
    'season': 2025,

    # Rolling averages (exponentially weighted, decay=0.9)
    'rolling_avg_fantasy_ppr': 20.22,
    'rolling_avg_passing_yds': 289.04,
    'rolling_avg_passing_tds': 1.85,
    'rolling_avg_interceptions': 0.95,
    'rolling_avg_completions': 25.12,
    'rolling_avg_rushing_yds': 8.45,

    # Context
    'opponent_pass_defense_rank': 13.0,
    'games_in_history': 6,
    'has_sufficient_data': True,

    # Trends (old style)
    'pass_attempts_trend': 0.05,

    # Total: 33 columns
}
```

### V2 Schema (NEW - 42 columns):
```python
{
    # ... all v1 features PLUS:

    # NEW: Variance features (boom/bust indicators)
    'fantasy_pts_variance': 11.17,      # StdDev of last 6 games
    'passing_yds_variance': 87.35,
    'receiving_yds_variance': 15.22,
    'rushing_yds_variance': 5.10,

    # NEW: Trend features (hot/cold streaks)
    'fantasy_pts_trend': 0.15,          # Last 3 games vs previous 3
    'passing_yds_trend': 0.08,
    'targets_trend': -0.12,
    'carries_trend': 0.22,

    # Total: 42 columns
    # Decay factor: 0.85 (vs 0.9 in v1)
}
```

**Size:** ~50KB per file, 5MB total

**Usage:**
- Input features for ML model training
- Loaded by `NFLModelPipeline.load_training_data()`
- Used for prediction generation

---

## 3. MODELS (`data/nfl/models/`)

**Purpose:** Trained machine learning model files

**Structure:** VERSIONED folders
```
models/
└── v1_baseline_mae5.14/
    ├── QB_fantasy_points_ppr_evob.joblib
    ├── QB_fantasy_points_ppr_pob.joblib
    ├── QB_passing_yards_evob.joblib
    ├── ... (40 total models)
```

**Naming Convention:** `v{number}_{description}_mae{accuracy}`
- `v1_baseline_mae5.14` = Version 1, baseline features, 5.14 MAE
- `v2_variance_trends_mae4.82` = Version 2, with variance/trends, 4.82 MAE

**File Types:**
- `*_evob.joblib` = Expected Value Over Baseline (regression)
- `*_pob.joblib` = Probability Over Baseline (classification)
- `*_stat.joblib` = Individual stat predictor

**Model Coverage:** 40 models total
- QB: 7 models (fantasy_ppr, passing_yards, passing_tds, passing_int)
- RB: 7 models (fantasy_ppr, rushing_yards, rushing_tds, receptions)
- WR: 10 models (fantasy_ppr, receiving_yards, receiving_tds, receptions)
- TE: 7 models (fantasy_ppr, receiving_yards, receiving_tds, receptions)
- K: 4 models (fantasy_ppr, fg_made, fg_att)

**Size:** ~2-4MB per model, 91MB total

**Usage:**
- Loaded by `NFLModelPipeline` for generating predictions
- Each model is a serialized ensemble (XGBoost + LightGBM + CatBoost + RandomForest)

---

## 4. PREDICTIONS (`data/nfl/predictions/`)

**Purpose:** Generated predictions for future/past weeks

**Structure:** VERSIONED folders
```
predictions/
└── v1_baseline_mae5.14/
    ├── predictions_2025_week_2.parquet
    ├── predictions_2025_week_3.parquet
    ├── ... (11 files)
```

**Schema:** 15 columns
```python
{
    'player_id': '00-0019596',
    'player_name': 'A.Rodgers',
    'position': 'QB',
    'team': 'NYJ',
    'opponent': 'MIA',
    'season': 2025,
    'week': 10,

    'stat': 'fantasy_points_ppr',
    'model_type': 'evob',              # or 'pob'

    # EVOB predictions
    'predicted_value': 18.45,
    'predicted_diff': 2.58,
    'baseline': 15.87,
    'confidence_lower': 12.30,
    'confidence_upper': 24.60,

    # POB predictions
    'probability_over': 0.58,          # 58% chance to beat baseline
}
```

**Size:** ~100KB per file, 1-2MB total

**Usage:**
- Displayed in Streamlit dashboard
- Compared against actual results for validation
- Used by `validate_accuracy.py` to calculate MAE/RMSE/R²

---

## Data Flow Pipeline

```
1. FETCH RAW DATA
   nfl_pipeline.py → data/nfl/raw/

2. ENGINEER FEATURES
   raw/ + FeatureEngineer → cleaned/
   Calculates rolling averages, variance, trends

3. TRAIN MODELS
   cleaned/ + NFLModelPipeline → models/v{X}/
   Trains 40 models per version

4. GENERATE PREDICTIONS
   cleaned/ + models/v{X}/ → predictions/v{X}/
   Makes predictions for target week

5. VALIDATE ACCURACY
   predictions/v{X}/ + raw/ (actuals) → metrics
   Calculates MAE, RMSE, R², beat rate
```

---

## Versioning Strategy

### Current State (PROBLEM):
```
models/
├── v1_baseline_mae5.14/         ✅ Versioned
predictions/
├── v1_baseline_mae5.14/         ✅ Versioned
cleaned/
├── features_*.parquet           ❌ NOT versioned (MIXED schemas!)
```

### Desired State (SOLUTION):
```
models/
├── v1_baseline_mae5.14/         ✅ Versioned
predictions/
├── v1_baseline_mae5.14/         ✅ Versioned
features/
├── v1_baseline_mae5.14/         ✅ Versioned (33 columns, decay=0.9)
└── v2_variance_trends/          ✅ Versioned (42 columns, decay=0.85)
```

**Benefits:**
- Can train v1 models on v1 features
- Can train v2 models on v2 features
- No schema mismatches
- Can compare different feature engineering approaches
- Rollback capability

---

## File Size Summary

| Directory | Files | Size per File | Total Size | Versioned? |
|-----------|-------|---------------|------------|------------|
| raw/      | 103   | ~150 KB       | ~15 MB     | N/A        |
| cleaned/  | 102   | ~50 KB        | ~5 MB      | ❌ (TODO)  |
| models/   | 40    | ~2-4 MB       | ~91 MB     | ✅         |
| predictions/ | 11 | ~100 KB       | ~1 MB      | ✅         |

**Total:** ~112 MB (uncompressed parquet + joblib)

---

## Key Differences Between Schemas

### V1 Features (Baseline):
- **Columns:** 33
- **Decay Factor:** 0.9 (weaker, more historical weight)
- **Features:** Rolling averages, opponent rank, basic trends
- **No Variance:** Can't identify boom/bust players
- **No Trend Analysis:** Can't capture hot/cold streaks
- **MAE:** 5.14 points

### V2 Features (Improved):
- **Columns:** 42
- **Decay Factor:** 0.85 (stronger, emphasizes recent 3 games)
- **Features:** All v1 + variance + trend indicators
- **With Variance:** Identifies consistent vs boom/bust players
- **With Trends:** Captures player momentum (improving/declining)
- **MAE:** UNKNOWN (to be tested)

---

## Next Steps

1. ✅ Create versioned features structure
2. ✅ Move existing features to v1_baseline_mae5.14/
3. ✅ Update FeatureEngineer to accept version parameter
4. ✅ Update NFLModelPipeline to use versioned features
5. ✅ Update retrain_v2.py to generate v2 features
6. ✅ Regenerate all v2 features with consistent schema
7. ✅ Train v2 models
8. ✅ Compare v1 vs v2 accuracy
