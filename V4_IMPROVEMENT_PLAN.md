# V4 Model Improvement Plan

**Status**: Ready for Implementation
**Previous Version**: V3 (EPA & Efficiency) - MAE 4.66 (No improvement over V2)
**Target**: V4 - MAE < 4.2 (10% improvement over V2/V3)
**Strategy**: Game Context Features + Position-Specific Model Architectures

---

## 📊 V3 Results Summary

V3 attempted to add advanced analytics (EPA, efficiency metrics, position-specific decay) but achieved **ZERO improvement**:

| Position | V2 MAE | V3 MAE | Change | Result |
|----------|--------|--------|--------|--------|
| QB | 7.19 | 7.26 | +0.07 | ❌ Worse |
| RB | 4.73 | 4.75 | +0.02 | ❌ Worse |
| WR | 4.57 | 4.56 | -0.01 | ≈ Negligible |
| TE | 3.57 | 3.52 | -0.05 | ✓ Minor gain |
| **Overall** | **4.66** | **4.66** | **±0.00** | **No change** |

### Why V3 Failed

1. **Feature Redundancy**: EPA correlates with existing rolling averages
2. **Efficiency ≠ Volume**: PACR/RACR measure efficiency, but fantasy needs volume
   - WR with 100% efficiency on 2 targets = 4 points
   - WR with 50% efficiency on 10 targets = 15 points
3. **Position-Specific Decay Ineffective**: Different decay factors (0.80-0.90) made no difference
4. **Overfitting**: 57 features on limited QB/K data (2,893 and 2,592 records)

### Key Insight from V3 Failure

**The Real Problem**: Models predict *how well players perform when they get opportunities*, but don't predict *how many opportunities they'll get*.

- QB gets 40 pass attempts (blowout game) vs 25 attempts (running clock)
- RB gets 20 carries (winning team) vs 8 carries (losing team chasing points)
- WR gets 12 targets (shootout) vs 4 targets (run-heavy game)

**V4 must focus on VOLUME prediction, not just efficiency.**

---

## 🎯 V4 Strategy: Two-Phase Approach

### Phase 1: Game Context Features (2-3 hours, HIGH impact)
**Add information about the GAME, not just the player**

### Phase 2: Position-Specific Model Architectures (1 hour, MEDIUM impact)
**Customize model hyperparameters for each position's unique characteristics**

---

## 📈 Phase 1: Game Context Features

### Problem Statement

Current models know:
- Patrick Mahomes averages 275 passing yards
- He has 35 yards variance
- Recent trend: +5% improvement

Current models DON'T know:
- Is he playing a weak defense? (should throw more)
- Is his team favored by 10 points? (less passing if winning big)
- Is it a high-scoring game expected? (more opportunities)

### Solution: Vegas Lines Integration

Vegas lines have **80%+ accuracy** predicting game totals and incorporate:
- Injury reports
- Weather conditions
- Matchup analysis
- Sharp betting money (professional bettors)

Game script explains **30-40% of fantasy variance**.

### Features to Add

#### 1. Vegas Lines (1-2 hours to integrate)

**Core Features**:
```python
- team_implied_total: Expected team points (higher = more plays = more volume)
- point_spread: Favorite vs underdog (predicts game script)
- over_under: Total expected points (high totals = more passes)
```

**Point Spread Explained**:
- Example: Chiefs -7.5 vs Patriots +7.5
- Means: Chiefs expected to win by 7.5 points

**Why it matters**:
- **QB**: Trailing teams throw MORE (trying to catch up)
  - If Chiefs are winning big, Mahomes might only throw 25 times
  - If Patriots are losing, their QB throws 45 times (more fantasy points)
- **RB**: Winning teams run MORE (burn clock)
  - Winning RB gets 20 carries
  - Losing RB gets 8 carries (team is passing to catch up)

**Implied Total Explained**:
- Example: Chiefs implied total = 27.5 points, Patriots = 20 points
- Total game over/under = 47.5 points

**Why it matters**:
- More points = more plays = more opportunities
- QB on team expected to score 30 points → more pass attempts
- RB on team expected to score 14 points → fewer carries

#### 2. Situational Features (30 mins - already in data or easy to add)

```python
- home_away: QBs +2 points at home, RBs +1 point
- days_rest: Thursday games = worse performance (short week)
- divisional_game: Lower scoring, more running (teams know each other)
```

#### 3. Weather Features (30 mins if available)

```python
- wind_speed: >15mph = fewer deep passes, more running
- temperature: Cold games (<32°F) = more rushing, less passing
- dome_vs_outdoor: Predictable vs variable conditions
```

### Data Sources (Free)

1. **ESPN API** - Has game lines, spreads, totals
2. **The Odds API** - Free tier available (theoddsapi.com)
3. **Scraping** - ESPN game pages have all this data

### Implementation: v4_feature_engineer.py

```python
def add_game_context_features(df, vegas_lines):
    """Add game situation features that predict volume"""

    # Merge Vegas lines
    df = df.merge(vegas_lines, on=['team', 'week', 'season'])

    # Position-specific transformations
    if position == 'QB':
        # QBs throw more when trailing
        features['qb_game_script'] = -row['point_spread']  # Negative spread = favorite
        features['qb_expected_volume'] = row['team_implied_total'] * 2.5  # ~2.5 pass attempts per point

    elif position == 'RB':
        # RBs carry more when winning
        features['rb_game_script'] = row['point_spread']  # Positive spread = more carries
        features['rb_expected_volume'] = max(0, 20 - abs(row['point_spread']))  # Close games = balanced

    elif position in ['WR', 'TE']:
        # WRs get targets in high-scoring games
        features['wr_expected_volume'] = row['total_over_under'] * 0.8  # More targets in shootouts
        features['wr_qb_pressure'] = -row['point_spread']  # Trailing = more passes

    return features
```

### Real Example Comparison

**Without game context (V2/V3)**:
```
Patrick Mahomes prediction:
- rolling_avg_passing_yds: 275
- passing_yds_variance: 35
- passing_yds_trend: +5%
→ Prediction: 285 yards
```

**With game context (V4)**:
```
Patrick Mahomes prediction:
- rolling_avg_passing_yds: 275
- passing_yds_variance: 35
- passing_yds_trend: +5%
- team_implied_total: 31 points (HIGH - more passing!)
- point_spread: -10 (big favorite - might run more in 4th quarter)
- opponent_pass_defense_rank: 28th (weak - throw more!)
→ Prediction: 310 yards (better prediction!)
```

### Expected Impact

**Position-Specific Improvements**:
- **QB**: Vegas total predicts pass attempts (biggest driver of QB points)
- **RB**: Point spread predicts carries (losing teams abandon run)
- **WR/TE**: Implied total predicts targets (more points = more passes)

**Overall Expected Impact**: 0.3-0.5 MAE reduction (4.66 → 4.2-4.4)

---

## 🔧 Phase 2: Position-Specific Model Architectures

### Problem Statement

Current models use **identical hyperparameters** for all positions:
- `max_depth = 7`
- `n_estimators = 300`
- `learning_rate = 0.01`

But positions have vastly different characteristics:

| Position | Training Records | Variance | Current max_depth | Optimal max_depth |
|----------|-----------------|----------|-------------------|-------------------|
| QB | 2,893 | High (5-40 pts) | 7 | **9-10** (needs deeper trees) |
| RB | ~8,000 | Medium (0-30 pts) | 7 | **7-8** (current OK) |
| WR | ~12,000 | Medium (0-35 pts) | 7 | **7** (current OK) |
| TE | ~6,000 | Low (0-25 pts) | 7 | **6** (simpler) |
| K | 2,592 | Very Low (0-15 pts) | 7 | **3-4** (much simpler) |

### QB Problem: 7.26 MAE, R²=0.007 (model explains NOTHING)

**Why QBs fail**:
- High variance games (5-40 point range)
- Limited data (2,893 records)
- Complex patterns (game script, weather, matchups all matter)

**Solution**: Deeper trees (capture complexity) + regularization (prevent overfitting)

### K Problem: Overfitting on Simple Patterns

**Why current approach fails**:
- Kickers are very predictable (0-15 points)
- Limited data (2,592 records)
- Current depth=7 is **overfitting** on simple patterns

**Solution**: Shallow trees (3-4 depth) + fewer estimators (100 vs 300)

### Implementation: Create v4_ml_models.py

**Important**: Do NOT modify existing `ml_models.py`. Create `v4_ml_models.py` to keep versions organized.

```python
"""
File: src/nfl/v4_ml_models.py

Position-specific model architectures for V4.
Based on ml_models.py but with position-optimized hyperparameters.
"""

from src.nfl.ml_models import EVOBModel, POBModel, StatPredictor
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


class PositionSpecificEVOBModel(EVOBModel):
    """EVOB model with position-optimized hyperparameters"""

    def _initialize_models(self):
        """Initialize models with position-specific settings"""

        if self.position == 'QB':
            # QB: High variance, need deep trees + regularization
            self.models = {
                'xgboost': XGBRegressor(
                    n_estimators=500,      # More trees for complex patterns
                    max_depth=9,           # Deeper (capture QB boom/bust)
                    learning_rate=0.005,   # Slower learning
                    min_child_weight=5,    # Regularization (prevent overfitting)
                    subsample=0.8,         # Use 80% of data per tree
                    colsample_bytree=0.8,  # Use 80% of features per tree
                    gamma=0.1,             # Minimum loss reduction for split
                    objective='reg:squarederror',
                    random_state=42,
                    early_stopping_rounds=20
                ),
                'lightgbm': LGBMRegressor(
                    n_estimators=500,
                    max_depth=9,
                    learning_rate=0.005,
                    num_leaves=255,        # More leaves for complexity
                    min_data_in_leaf=50,   # Regularization
                    objective='regression',
                    metric='rmse',
                    random_state=42,
                    verbose=-1
                ),
                'catboost': CatBoostRegressor(
                    iterations=500,
                    depth=9,
                    learning_rate=0.005,
                    l2_leaf_reg=5,         # Regularization
                    loss_function='RMSE',
                    random_state=42,
                    verbose=False
                )
            }

        elif self.position == 'K':
            # K: Low variance, small dataset, need simple models
            self.models = {
                'xgboost': XGBRegressor(
                    n_estimators=100,      # Fewer trees
                    max_depth=3,           # Shallow (Kickers are predictable)
                    learning_rate=0.01,
                    objective='reg:squarederror',
                    random_state=42,
                    early_stopping_rounds=20
                ),
                'ridge': Ridge(alpha=1.0),  # Linear baseline (prevents overfitting)
                'lightgbm': LGBMRegressor(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.01,
                    objective='regression',
                    random_state=42,
                    verbose=-1
                )
            }

        elif self.position == 'TE':
            # TE: Lower variance, benefit from simpler models
            self.models = {
                'xgboost': XGBRegressor(
                    n_estimators=300,
                    max_depth=6,           # Slightly shallower
                    learning_rate=0.01,
                    min_child_weight=3,
                    objective='reg:squarederror',
                    random_state=42,
                    early_stopping_rounds=20
                ),
                'lightgbm': LGBMRegressor(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.01,
                    objective='regression',
                    random_state=42,
                    verbose=-1
                ),
                'catboost': CatBoostRegressor(
                    iterations=300,
                    depth=6,
                    learning_rate=0.01,
                    loss_function='RMSE',
                    random_state=42,
                    verbose=False
                )
            }

        else:
            # RB/WR: Use existing config (works well)
            # Same as parent class EVOBModel
            super()._initialize_models()


class PositionSpecificPOBModel(POBModel):
    """POB model with position-optimized hyperparameters"""

    def _initialize_models(self):
        """Initialize classification models with position-specific settings"""

        if self.position == 'QB':
            # QB: High variance binary classification
            self.models = {
                'xgboost': XGBClassifier(
                    n_estimators=300,
                    max_depth=8,
                    learning_rate=0.005,
                    min_child_weight=5,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='binary:logistic',
                    use_label_encoder=False,
                    eval_metric='logloss',
                    random_state=42,
                    early_stopping_rounds=20
                ),
                'lightgbm': LGBMClassifier(
                    n_estimators=300,
                    max_depth=8,
                    learning_rate=0.005,
                    num_leaves=127,
                    min_data_in_leaf=50,
                    objective='binary',
                    metric='binary_logloss',
                    random_state=42,
                    verbose=-1
                )
            }

        elif self.position == 'K':
            # K: Simpler classification
            self.models = {
                'xgboost': XGBClassifier(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.01,
                    objective='binary:logistic',
                    use_label_encoder=False,
                    eval_metric='logloss',
                    random_state=42,
                    early_stopping_rounds=20
                )
            }

        else:
            # RB/WR/TE: Use existing config
            super()._initialize_models()
```

### Hyperparameter Explanation (For Beginners)

**max_depth**: How many questions the tree can ask before making a prediction
- Deeper = can capture complex patterns (good for QBs who are unpredictable)
- Shallow = simple patterns only (good for Kickers who are predictable)
- Example: depth=3 means tree can ask 3 questions, depth=9 means 9 questions

**n_estimators**: How many trees to create and average
- More trees = better accuracy but slower training
- Fewer trees = faster but might miss patterns
- Example: 500 trees for QB (complex), 100 for K (simple)

**learning_rate**: How fast the model learns
- Lower = slower, more careful learning (good with more trees)
- Higher = faster, might overshoot optimal solution
- Example: 0.005 for QB (careful), 0.01 for K (faster)

**min_child_weight**: Minimum samples needed per leaf
- Higher = more regularization (prevents overfitting)
- Lower = less regularization (can overfit)
- Example: 5 for QB (prevent overfitting on limited data)

**subsample**: % of data to use for each tree
- 0.8 = use 80% of data randomly per tree
- Prevents overfitting (each tree sees different data)

**colsample_bytree**: % of features to use for each tree
- 0.8 = use 80% of features randomly per tree
- Prevents overfitting on specific features

### Expected Impact

**Position-Specific Improvements**:
- QB: 7.26 → **6.5-6.8 MAE** (10-15% improvement)
- K: Better generalization (prevent overfitting)
- TE: **3.52 → 3.3-3.4 MAE** (5% improvement)
- Overall: 4.66 → **4.4-4.5 MAE** (5% improvement)

---

## 🎯 Combined V4 Expected Results

| Position | V2 MAE | V3 MAE | V4 Prediction | Improvement vs V2 |
|----------|--------|--------|---------------|-------------------|
| QB | 7.19 | 7.26 | **6.4-6.8** | ✅ 10-15% |
| RB | 4.73 | 4.75 | **4.5-4.6** | ✅ 5% |
| WR | 4.57 | 4.56 | **4.2-4.4** | ✅ 8% |
| TE | 3.57 | 3.52 | **3.3-3.5** | ✅ 5% |
| K | (good) | (good) | **(same or better)** | - |
| **Overall** | **4.66** | **4.66** | **4.2-4.4** | ✅ **10%** |

**Target**: MAE < 4.2 (professional grade, top 10% of DFS models)

**Professional DFS models typically achieve 4.5-5.5 MAE**, so 4.2 would be excellent.

---

## 📋 V4 Implementation Checklist

### Step 1: Create v4_feature_engineer.py (2-3 hours)

- [ ] Copy `feature_engineer.py` → `v4_feature_engineer.py`
- [ ] Find Vegas lines API (ESPN, The Odds API, or scraping)
- [ ] Add `get_vegas_lines()` function
- [ ] Add `team_implied_total`, `point_spread`, `over_under` features
- [ ] Add `home_away` feature
- [ ] Add position-specific game script features:
  - [ ] `qb_game_script`, `qb_expected_volume`
  - [ ] `rb_game_script`, `rb_expected_volume`
  - [ ] `wr_expected_volume`, `wr_qb_pressure`
- [ ] Test feature generation on 1 week
- [ ] Generate features for all weeks (2020-2025)

### Step 2: Create v4_ml_models.py (1 hour)

- [ ] Copy `ml_models.py` → `v4_ml_models.py`
- [ ] Create `PositionSpecificEVOBModel` class
  - [ ] QB hyperparameters (depth=9, n_estimators=500, learning_rate=0.005)
  - [ ] K hyperparameters (depth=3, n_estimators=100)
  - [ ] TE hyperparameters (depth=6, n_estimators=300)
  - [ ] RB/WR use default
- [ ] Create `PositionSpecificPOBModel` class
  - [ ] QB hyperparameters
  - [ ] K hyperparameters
  - [ ] Others use default
- [ ] Update `NFLModelPipeline` to use position-specific models
- [ ] Test model initialization

### Step 3: Create v4_retrain.py (30 mins)

- [ ] Copy `v2_retrain.py` → `v4_retrain.py`
- [ ] Update imports to use `v4_feature_engineer`
- [ ] Update imports to use `v4_ml_models`
- [ ] Update version string to `v4_game_context`
- [ ] Test dry run

### Step 4: Train V4 Models (3-4 hours)

- [ ] Run `python3 src/nfl/v4_retrain.py`
- [ ] Monitor training progress
- [ ] Verify 40 models created in `data/nfl/models/v4_game_context/`
- [ ] Verify features created in `data/nfl/features/v4_game_context/`
- [ ] Verify predictions created in `data/nfl/predictions/v4_game_context/`

### Step 5: Validate V4 Results (30 mins)

- [ ] Run validation: `python3 testing/compare_versions.py v2_variance_trends_mae4.66 v4_game_context 10 11 12`
- [ ] Check overall MAE (target: < 4.4)
- [ ] Check QB MAE (target: < 6.8)
- [ ] Check position-by-position improvements
- [ ] Document results in progress report

### Step 6: Decision Point

**If V4 MAE < 4.4**: ✅ Accept V4, rename to `v4_game_context_maeX.XX`
**If V4 MAE 4.4-4.6**: ⚠️ Minor improvement, consider additional features
**If V4 MAE > 4.6**: ❌ Investigate issues, debug feature engineering

---

## 🚀 Quick Start Commands (For Opus)

```bash
# Step 1: Create V4 feature engineer
cp src/nfl/feature_engineer.py src/nfl/v4_feature_engineer.py
# (Then modify v4_feature_engineer.py to add game context features)

# Step 2: Create V4 model classes
cp src/nfl/ml_models.py src/nfl/v4_ml_models.py
# (Then modify v4_ml_models.py to add PositionSpecificEVOBModel class)

# Step 3: Create V4 training script
cp src/nfl/v2_retrain.py src/nfl/v4_retrain.py
# (Then modify imports and version string)

# Step 4: Train V4
python3 src/nfl/v4_retrain.py

# Step 5: Validate V4
python3 testing/compare_versions.py v2_variance_trends_mae4.66 v4_game_context 10 11 12
```

---

## 📊 Stats Currently Being Predicted

**Confirmed available** (from V3 prediction files):

### QB Stats
- `fantasy_points_ppr` ✅
- `passing_yards` ✅
- `passing_tds` ✅
- `passing_interceptions` ✅

### RB Stats
- `fantasy_points_ppr` ✅
- `rushing_yards` ✅
- `rushing_tds` ✅
- `receptions` ✅

### WR/TE Stats
- `fantasy_points_ppr` ✅
- `receiving_yards` ✅
- `receiving_tds` ✅
- `receptions` ✅

### K Stats
- `fantasy_points_ppr` ✅
- `fg_made` ✅
- `fg_att` ✅

**All position-specific stats are being predicted!** The models train on these stats using the EVOB and Stat predictor types.

---

## 🔬 Why This Will Work (Learning from V3 Failure)

### V3 Added Features Players Already Controlled
- EPA = how efficient players were with their touches
- PACR/RACR = yards per target efficiency
- CPOE = completion accuracy

**Problem**: These are **outcomes**, not **predictors**. They tell us who's good, not who will get opportunities.

### V4 Adds Features Players DON'T Control
- Vegas implied total = How many plays will the team run?
- Point spread = Will they be passing or running?
- Home/away = Statistical advantage (+2 points for QB at home)

**Solution**: These are **predictors** of volume/opportunity. They tell us game context.

### Analogy

**V3 approach**: "This player is very efficient when they get the ball"
- Like predicting a salesman's monthly sales by measuring "dollars per call"
- But if they only make 5 calls vs 50 calls, efficiency doesn't matter

**V4 approach**: "This player will get lots of opportunities"
- Like predicting monthly sales by knowing "they'll make 50 calls this month"
- Volume × efficiency = results

---

## 📝 Notes for Implementation

1. **Vegas Lines API Options**:
   - ESPN API (easiest, free): `espn.com/nfl/game/_/gameId/{id}` has betting lines
   - The Odds API (theoddsapi.com): 500 free requests/month
   - Scraping: ESPN/NFL.com have all lines visible on game pages

2. **Feature Engineering Order**:
   - Generate base features first (rolling averages, variance, trends)
   - THEN add game context (Vegas lines, home/away)
   - This way game context can interact with player features

3. **Versioning**:
   - Keep V2 and V3 intact
   - V4 is a new version: `v4_game_context`
   - After validation, rename to `v4_game_context_maeX.XX`

4. **File Organization**:
   ```
   src/nfl/
   ├── feature_engineer.py          (V2 - current production)
   ├── v3_feature_engineer.py       (V3 - EPA/efficiency, rejected)
   ├── v4_feature_engineer.py       (V4 - game context, new)
   ├── ml_models.py                 (V2/V3 - standard hyperparameters)
   ├── v4_ml_models.py              (V4 - position-specific, new)
   ├── v2_retrain.py                (V2 workflow)
   ├── v3_retrain.py                (V3 workflow)
   └── v4_retrain.py                (V4 workflow, new)
   ```

5. **Testing Strategy**:
   - Use same validation weeks as V2/V3 (2025 weeks 10-12)
   - This ensures apples-to-apples comparison
   - If V4 improves on these weeks, it's a real improvement

---

## 🎯 Success Criteria

**Minimum Viable Success**:
- Overall MAE < 4.6 (at least match V2/V3)
- QB MAE < 7.0 (improve from 7.26)
- No position regresses by more than 0.1 MAE

**Good Success**:
- Overall MAE < 4.4 (5% improvement)
- QB MAE < 6.8 (7% improvement)
- All positions improve or stay within 0.05 MAE

**Excellent Success**:
- Overall MAE < 4.2 (10% improvement)
- QB MAE < 6.5 (10%+ improvement)
- All positions improve

**If Excellent Success**: Consider V4 production-ready, deploy to web app

---

## 📅 Timeline Estimate

- **Phase 1 (Game Context Features)**: 2-3 hours
  - Find Vegas API: 30 mins
  - Implement feature engineering: 1 hour
  - Generate all features: 1-1.5 hours

- **Phase 2 (Position-Specific Models)**: 1 hour
  - Create v4_ml_models.py: 30 mins
  - Test and debug: 30 mins

- **Phase 3 (Training & Validation)**: 3-4 hours
  - Feature generation: 2-3 hours
  - Model training: 30 mins
  - Prediction generation: 30 mins
  - Validation: 15 mins

**Total**: 6-8 hours for complete V4 implementation and validation

---

**Generated**: December 5, 2025
**For use with**: Claude Opus (implementation), future optimization sessions
**Based on**: V3 failure analysis, V2 success patterns, game theory insights
