# V3 Model Improvement Plan

This document outlines potential improvements to increase prediction accuracy beyond V2. All features listed are available in the existing raw data or from free APIs.

---

## 🎯 HIGH PRIORITY (Quick Wins)

### 1. Vegas Lines Integration ⭐⭐⭐⭐⭐
**Impact**: HIGH | **Effort**: LOW | **Time**: ~30 mins
**Expected improvement**: 0.3-0.5 MAE reduction (target: ~4.6-4.8 MAE)

Vegas lines incorporate injury, weather, matchups, and sharp betting money. They're the gold standard for game outcomes.

**Features to add**:
- `vegas_implied_total` - Team's over/under line
- `vegas_opponent_total` - Opponent's implied total
- `vegas_spread` - Point spread (+/- for team)
- `home_away` - Home field advantage indicator

**Data sources** (free):
- ESPN API - has game lines
- TheSportsDB API - free tier
- Alternatively: scrape from ESPN game pages

**Implementation**:
```python
# In feature_engineer.py
def get_vegas_lines(team, opponent, season, week):
    # Fetch from ESPN or TheSportsDB
    return {
        'vegas_implied_total': float,
        'vegas_spread': float,
        'is_home_game': bool
    }
```

---

### 2. EPA (Expected Points Added) Features ⭐⭐⭐⭐⭐
**Impact**: HIGH | **Effort**: NONE (already in data!) | **Time**: ~20 mins
**Expected improvement**: 0.2-0.4 MAE reduction

EPA is the single best predictive stat in modern NFL analytics. It accounts for down, distance, field position, and game situation.

**Features ALREADY in raw data**:
- `passing_epa` - QB efficiency (how many expected points added per play)
- `rushing_epa` - RB efficiency
- `receiving_epa` - WR/TE efficiency

**Why it matters**:
- Separates lucky TD scorers from consistently good players
- Accounts for game context (garbage time vs clutch situations)
- Better predictor than raw yards

**Implementation**:
```python
# Just add to rolling average calculations!
'rolling_avg_passing_epa',
'rolling_avg_rushing_epa',
'rolling_avg_receiving_epa',
```

---

### 3. Advanced Efficiency Metrics ⭐⭐⭐⭐
**Impact**: MEDIUM-HIGH | **Effort**: NONE (already in data!) | **Time**: ~20 mins
**Expected improvement**: 0.1-0.2 MAE reduction

Modern analytics metrics that separate skill from volume.

**Features ALREADY in raw data**:
- `pacr` (Pass Air Conversion Ratio) - QB efficiency converting air yards to completions
- `racr` (Receiver Air Conversion Ratio) - WR/TE efficiency converting targets to yards
- `wopr` (Weighted Opportunity Rating) - WR/TE opportunity metric (targets + air yards share)
- `passing_cpoe` (Completion % Over Expected) - QB accuracy vs expected completion rate

**Why it matters**:
- PACR/RACR separate good route runners from volume receivers
- WOPR better predicts targets than raw target counts
- CPOE identifies accurate QBs vs checkdown QBs

**Implementation**:
```python
'rolling_avg_pacr',  # QB feature
'rolling_avg_racr',  # WR/TE feature
'rolling_avg_wopr',  # WR/TE feature
'rolling_avg_cpoe',  # QB feature
```

---

### 4. Target Share & Air Yards Share ⭐⭐⭐⭐
**Impact**: MEDIUM-HIGH | **Effort**: NONE (already in data!) | **Time**: ~15 mins
**Expected improvement**: 0.1-0.2 MAE reduction for WR/TE

These predict volume better than raw targets because they're team-context adjusted.

**Features ALREADY in raw data**:
- `target_share` - % of team targets (e.g., 25% means WR gets 1 in 4 targets)
- `air_yards_share` - % of team air yards (deep threat indicator)

**Why it matters**:
- Target share predicts future targets better than raw counts
- Air yards share identifies boom/bust WRs (deep threats vs possession receivers)
- Adjusts for team pass volume (high for pass-heavy offenses)

**Implementation**:
```python
'rolling_avg_target_share',      # WR/TE
'rolling_avg_air_yards_share',   # WR/TE
'target_share_variance',          # Consistency metric
```

---

### 5. Yards After Catch (YAC) ⭐⭐⭐⭐
**Impact**: MEDIUM | **Effort**: NONE (already in data!) | **Time**: ~15 mins
**Expected improvement**: 0.1-0.15 MAE reduction for WR/TE

YAC ability separates elusive playmakers from possession receivers.

**Features ALREADY in raw data**:
- `passing_yards_after_catch` - QB's receivers' YAC ability (QB feature)
- `receiving_yards_after_catch` - WR/TE elusiveness (WR/TE feature)

**Why it matters**:
- High YAC players are more consistent in PPR (short targets + run after catch)
- Low YAC players are boom/bust (depend on big plays downfield)
- QB YAC shows if they have playmaker receivers

**Implementation**:
```python
'rolling_avg_passing_yac',     # QB
'rolling_avg_receiving_yac',   # WR/TE
'yac_per_reception',           # Efficiency metric
```

---

## 📊 MEDIUM PRIORITY (More Effort, Still High ROI)

### 6. Red Zone Opportunity Features ⭐⭐⭐⭐
**Impact**: HIGH (for TD predictions) | **Effort**: MEDIUM | **Time**: ~1 hour
**Expected improvement**: 0.2-0.3 MAE reduction on TD predictions

TD scoring is more about red zone touches than total yards. Kickers who attempt more 40-49 yard FGs score differently than those attempting 20-29 yarders.

**Note**: Red zone data may not be in current raw data. Need to verify or source separately.

**Potential features**:
- `red_zone_carry_share` - % of RB carries inside 20-yard line
- `goal_line_usage` - RB usage inside 5-yard line (TD vulture indicator)
- `red_zone_target_share` - % of WR/TE targets inside 20
- `fg_attempt_distance_avg` - Average K field goal attempt distance

**For kickers** (ALREADY in raw data):
- `fg_made_40_49`, `fg_made_50_59` - Distance-specific success rates
- `fg_made_distance` - Average successful FG distance

**Why it matters**:
- RB with 60% of goal-line carries scores more TDs than workhorse with 80% snap share
- WRs targeted in red zone are TD-dependent vs PPR possession receivers
- Long FG kickers score more fantasy points but are less reliable

**Implementation**:
```python
# If red zone data available:
'rolling_avg_red_zone_carries',
'rolling_avg_red_zone_targets',
'goal_line_usage_rate',

# For kickers (already available):
'rolling_avg_fg_distance',
'fg_40plus_success_rate',
```

---

### 7. Snap Count & Route Participation ⭐⭐⭐⭐
**Impact**: MEDIUM | **Effort**: MEDIUM | **Time**: ~45 mins
**Expected improvement**: 0.1-0.2 MAE reduction

Playing time is the best predictor of opportunity. Snap counts and route participation show true role.

**Note**: Need to verify if snap data is in raw stats or requires separate source.

**Potential features**:
- `snap_share` - % of offensive snaps played
- `snap_share_trend` - Is player's role increasing or decreasing?
- `route_participation` - % of pass plays where WR/TE ran a route
- `pass_block_rate` - RB/TE staying in to block vs running routes

**Why it matters**:
- 30% snap RB won't score like 70% snap workhorse, regardless of talent
- WRs with increasing snap share are emerging (trend up)
- WRs who block frequently get fewer targets

**Implementation**:
```python
'rolling_avg_snap_share',
'snap_share_trend',
'route_participation_rate',  # WR/TE only
```

---

### 8. Ensemble Weight Optimization ⭐⭐⭐
**Impact**: MEDIUM | **Effort**: LOW | **Time**: ~20 mins
**Expected improvement**: 0.05-0.15 MAE reduction

Your ensemble currently averages 4 models equally (XGBoost, LightGBM, CatBoost, RandomForest). But they have different strengths - optimize their weights.

**Current approach**:
```python
prediction = (xgb + lgbm + cat + rf) / 4  # Equal weights
```

**Better approach**:
```python
# Use validation performance to weight models
weights = 1 / np.array([mae_xgb, mae_lgbm, mae_cat, mae_rf])
weights /= weights.sum()
prediction = weights[0]*xgb + weights[1]*lgbm + weights[2]*cat + weights[3]*rf
```

**Why it matters**:
- CatBoost might be best for QB but worst for K
- Inverse-error weighting gives more weight to better models
- Can be position-specific weights

**Implementation**:
```python
# In ml_models.py EVOB/POB ensemble prediction:
def optimize_ensemble_weights(self, X_val, y_val):
    # Get individual model predictions on validation set
    preds = {name: model.predict(X_val) for name, model in self.models.items()}
    # Calculate MAE for each
    maes = {name: mean_absolute_error(y_val, pred) for name, pred in preds.items()}
    # Inverse error weighting
    weights = {name: 1/mae for name, mae in maes.items()}
    weight_sum = sum(weights.values())
    weights = {name: w/weight_sum for name, w in weights.items()}
    return weights
```

---

## 🔬 ADVANCED (Longer Term / More Complex)

### 9. Position-Specific Hyperparameter Tuning ⭐⭐⭐
**Impact**: MEDIUM | **Effort**: HIGH | **Time**: ~2-3 hours
**Expected improvement**: 0.2-0.3 MAE reduction for QB/TE

Current models use same hyperparameters for all positions. But QB predictions need different tree depth than RB predictions.

**Observation from V1 results**:
- RB: R² = 0.473 (GOOD - models working well)
- QB: R² = -0.349 (BAD - worse than predicting mean!)
- TE: R² = -0.026 (BAD - barely better than random)

**Why QB/TE fail**:
- Higher variance in outcomes (QB has 5-40 point range)
- Fewer players per position (QB: ~33/week vs RB: ~80/week)
- Different feature importance (QB needs more emphasis on opponent defense)

**Solution**: Position-specific hyperparameters
```python
hyperparams = {
    'QB': {
        'max_depth': 8,  # Deeper trees for complex interactions
        'min_child_weight': 5,  # More samples per leaf
        'learning_rate': 0.005,  # Slower learning
    },
    'RB': {
        'max_depth': 6,  # Current (working well)
        'learning_rate': 0.01,
    },
    'TE': {
        'max_depth': 5,  # Shallower to avoid overfitting
        'min_child_weight': 10,  # More regularization
    }
}
```

---

### 10. Quantile Regression for Confidence Intervals ⭐⭐⭐
**Impact**: MEDIUM | **Effort**: MEDIUM | **Time**: ~1 hour
**Expected improvement**: Better calibrated confidence intervals

Current confidence intervals use ensemble variance (Z-score method). Quantile regression directly predicts 10th/50th/90th percentiles.

**Current**:
```python
mean = ensemble_average
std = ensemble_std
CI = (mean - 1.645*std, mean + 1.645*std)  # 90% CI
```

**Better**: Quantile regression
```python
from lightgbm import LGBMRegressor
model_10th = LGBMRegressor(objective='quantile', alpha=0.10)
model_50th = LGBMRegressor(objective='quantile', alpha=0.50)  # Median
model_90th = LGBMRegressor(objective='quantile', alpha=0.90)

# Predictions give actual percentiles
prediction_low = model_10th.predict(X)
prediction_mid = model_50th.predict(X)
prediction_high = model_90th.predict(X)
```

**Why it matters**:
- Actual coverage matches predicted coverage (90% CI truly contains 90% of outcomes)
- Better for high-variance players (boom/bust WRs)
- Useful for fantasy lineup optimization (risk-averse vs risk-seeking)

---

### 11. LSTM for Time-Series Patterns ⭐⭐
**Impact**: MEDIUM | **Effort**: VERY HIGH | **Time**: ~4-6 hours
**Expected improvement**: 0.1-0.3 MAE reduction (uncertain)

Current exponential decay is simple but may miss complex patterns. LSTM can learn sequential patterns in player performance.

**Current**: Exponential rolling average (0.85 decay)
**Alternative**: LSTM sequence model

**Why it might help**:
- Captures multi-week patterns (e.g., every other week boom/bust)
- Learns position-specific decay rates
- Accounts for schedule difficulty sequences

**Why it might not**:
- Much more complex
- Needs more data per player (many players lack long history)
- Can overfit to noise

**Verdict**: Test only after simpler improvements exhausted.

---

### 12. Injury Status Integration ⭐⭐⭐⭐⭐
**Impact**: VERY HIGH | **Effort**: HIGH | **Time**: Ongoing
**Expected improvement**: 0.3-0.6 MAE reduction (huge!)

Player injury status is the single biggest predictor of performance, but it's not currently used.

**Features needed**:
- `injury_status` - Healthy, Questionable, Doubtful, Out
- `injury_type` - Hamstring, ankle, concussion, etc.
- `games_missed_recent` - Has player missed games in last 3 weeks?
- `return_from_injury` - First game back from injury (often limited snaps)

**Data sources**:
- ESPN injury reports
- NFL official injury reports (Wednesday/Thursday/Friday each week)
- Sleeper app API (has injury data)

**Why it matters**:
- WR listed as Questionable scores 25% fewer points on average
- First game back from injury = 60% of normal production
- RBs with hamstring injuries see reduced workload for 2-3 weeks

**Implementation challenge**:
- Need to scrape injury reports weekly BEFORE predictions
- Injury status changes daily (Wednesday: Out → Friday: Questionable)
- Requires API integration or web scraping

**This is the highest impact improvement but requires infrastructure work.**

---

## 📅 RECOMMENDED IMPLEMENTATION ORDER

### Phase 1: Add Existing Raw Data Features (Total: ~2 hours)
1. EPA features (20 mins) - `passing_epa`, `rushing_epa`, `receiving_epa`
2. Efficiency metrics (20 mins) - `pacr`, `racr`, `wopr`, `cpoe`
3. Target share (15 mins) - `target_share`, `air_yards_share`
4. YAC metrics (15 mins) - `receiving_yards_after_catch`, `passing_yards_after_catch`
5. Ensemble weight optimization (20 mins)
6. Kicker distance features (15 mins) - `fg_made_distance`, distance brackets

**Expected combined improvement**: 0.5-0.8 MAE reduction
**Target**: MAE 4.4-4.7 (from current 5.2)

### Phase 2: Position-Specific Optimization (Total: ~3 hours)
1. Hyperparameter tuning for QB (1 hour)
2. Hyperparameter tuning for TE (1 hour)
3. Test and validate improvements (1 hour)

**Expected improvement**: 0.2-0.3 MAE reduction for QB/TE
**Target**: MAE 4.1-4.4

### Phase 3: External Data Integration (Total: ~4 hours)
1. Vegas lines integration (1 hour) - ESPN API or scraping
2. Snap count data (2 hours) - if available
3. Red zone data (1 hour) - if available

**Expected improvement**: 0.3-0.5 MAE reduction
**Target**: MAE 3.8-4.1

### Phase 4: Advanced Techniques (Future)
1. Injury status integration (ongoing project)
2. Quantile regression for confidence intervals
3. LSTM exploration (if simpler methods plateau)

---

## 🎯 REALISTIC TARGETS

| Version | MAE | Improvements |
|---------|-----|--------------|
| V1 | 5.14 | Baseline (0.9 decay, no variance/trends) |
| V2 | 5.10-5.20 | Stronger decay (0.85), variance, trends |
| V3 (Phase 1) | 4.4-4.7 | + EPA, efficiency, target share, YAC, ensemble weights |
| V3 (Phase 2) | 4.1-4.4 | + Position-specific hyperparams |
| V3 (Phase 3) | 3.8-4.1 | + Vegas lines, snap counts |
| V4 (Future) | 3.5-3.8 | + Injury status, advanced ML |

**Professional DFS models typically achieve 4.5-5.5 MAE**, so 3.8-4.1 would be excellent.

---

## 🔄 EXPERIMENTATION WORKFLOW

For each improvement:
1. Create new version folder (e.g., `v3_epa_efficiency_mae?.??`)
2. Test on same validation weeks (2025 weeks 10-12)
3. Compare to V2 baseline
4. If better: Keep and build on it
5. If worse: Revert and try different approach

**Use the versioning system!** That's why we built it.

---

## 📝 NOTES

- All "ALREADY in raw data" features require zero API calls or scraping
- Start with Phase 1 (quick wins using existing data)
- Most improvement will come from EPA + efficiency metrics (they're proven in analytics literature)
- Vegas lines are the "cheat code" but require API integration
- Injury status is the highest ceiling improvement but requires ongoing data pipeline

---

**Generated**: 2025-12-03
**For use with**: Claude Code, Claude Sonnet 4.5, future optimization sessions
