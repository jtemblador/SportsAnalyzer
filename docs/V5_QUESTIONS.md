# V5 Pre-Planning: Decisions and Architecture

Finalized decisions before building V5. Each section documents the question, the decision, and why.

---

## 1. What exactly are we predicting?

**Decision:** Predict individual stats per position, compute fantasy points from the sum.

The dashboard needs per-stat breakdowns (e.g., "expected 280 passing yards, 2.1 TDs") not just one fantasy number. We compute fantasy points using PPR scoring:
```
fantasy_ppr = passing_yards*0.04 + passing_tds*4 - interceptions*2
            + rushing_yards*0.1 + rushing_tds*6
            + receptions*1 + receiving_yards*0.1 + receiving_tds*6
```

**Stats to predict per position:**

| Position | Stats | Count |
|----------|-------|-------|
| QB | passing_yards, passing_tds, passing_interceptions, rushing_yards, rushing_tds | 5 |
| RB | rushing_yards, rushing_tds, receptions, receiving_yards, receiving_tds | 5 |
| WR | receptions, receiving_yards, receiving_tds, targets | 4 |
| TE | receptions, receiving_yards, receiving_tds, targets | 4 |
| K | fg_made, fg_att, pat_made | 3 |

---

## 2. Model architecture

**Decision:** 2 model types (StatPredictor + POB), 4-algorithm ensemble, position-specific hyperparameters.

From the Final Report, V4 used 40 total models (5 positions × 8 model types). V5 will use:
- **StatPredictor** — predicts the raw stat value (what the dashboard displays)
- **POB** — predicts probability of exceeding the player's rolling average baseline (the over/under confidence)

EVOB (expected value over baseline) is dropped — it's redundant when you have StatPredictor value + the baseline.

The 4-algorithm ensemble (XGBoost, LightGBM, CatBoost, RandomForest) stays — V4 showed this is robust. CatBoost was the strongest single algorithm per the Final Report.

Position-specific hyperparameters (proven in V4):
- QB: depth=9, iter=500, lr=0.005 (complex game scripts)
- RB/WR: depth=7, iter=300, lr=0.01 (standard)
- TE: depth=6, iter=300, lr=0.01 (moderate)
- K: depth=3, iter=100, lr=0.01 (simple, prevent overfitting)

**Estimated model count:** 5 positions × ~4 stats avg × 2 types = ~42 models (similar to V4's 40).

---

## 3. Feature strategy — what worked, what didn't

From the Final Report and V1-V4 progression:

**Proven features (carry forward to V5):**
- **Rolling averages with decay=0.85** (V2) — emphasizes recent 3 games = 60.7% weight. Improved MAE from 5.14 → 4.66.
- **Variance/boom-bust metrics** (V2) — TE improved 42% (6.15→3.57 MAE). Distinguishes consistent producers from TD-dependent players.
- **Usage trend features** (V2) — target share momentum, carry trend. Proven for WR especially.
- **Vegas features** (V4) — implied team total, spread, game script index. Strongest single feature group. QB improved 35% (7.19→4.67 MAE).
- **Opponent defense rank** (V1+) — proven across all versions.
- **Position-specific hyperparameters** (V4) — deeper trees for QB, simpler for K.

**Failed features (do NOT repeat):**
- **EPA and efficiency metrics** (V3) — zero MAE improvement despite 15 new features and 3-4 hours extra engineering time. Fantasy scoring depends on volume, not efficiency. 100 rushing yards scores the same on 25 carries (4.0 YPC) or 18 carries (5.6 YPC).

**New features to add in V5 (tiered by confidence):**

| Tier | Feature Source | What | Why |
|------|---------------|------|-----|
| High | Snap counts | Offense snap %, snap trend | Direct proxy for opportunity — can't score from sideline |
| High | Injuries | Player's own status (Out/Doubtful/Questionable) | Out = 0 points, Doubtful = likely out |
| High | Schedules | Rest days, home/away, dome vs outdoor | Well-established factors |
| Medium | FF opportunity | Expected fantasy points, actual vs expected differential | Model-based expected value, feeds POB directly |
| Medium | Injuries | Teammate injury impact (WR1 out → WR2 targets up) | Opportunity redistribution |
| Medium | Depth charts | Starter (1) vs backup (2/3) flag | Starters get the volume |
| Medium | Weather | Wind speed, temperature (for passing games) | Wind > 15mph reduces passing ~10% |
| Lower | NGS passing | Time to throw, CPOE, aggressiveness | Only ~32 QBs per week (qualified only) |
| Lower | NGS rushing/receiving | Rush yards over expected, separation, YAC | Limited coverage (~34 rushers, ~66 receivers) |
| Lower | PFR advanced | Pressure rate, drop rate, yards after contact | Only ~36 QBs, ~130 rushers, ~250 receivers |
| Avoid | Team-level EPA | Offensive/defensive EPA | V3 showed EPA doesn't help; team-level likely same |

**Strategy:** Build all High + Medium + Lower features. Run ablation study after training. Drop any feature group that improves MAE by < 0.05 when included.

---

## 4. Handling sparse data (NGS, PFR coverage gaps)

**Decision:** Let NULLs be NULLs. No fake imputation.

**What NGS/PFR are:**
- **NGS (Next Gen Stats)** — NFL's player tracking system using chips in shoulder pads. Measures time to throw, receiver separation, rush efficiency. Only tracks qualified players (starters with enough volume).
- **PFR (Pro Football Reference)** — Sports statistics website that computes advanced metrics like pressure rate, drop rate, yards after contact. Uses its own player ID system.

**Coverage per week (2024 W1):**
- NGS passing: 32 players (all starting QBs)
- NGS rushing: 34 players (top rushers only)
- NGS receiving: 66 players (top targets only)
- PFR pass: 36 QBs
- PFR rush: 133 players
- PFR rec: 249 players

Since we predict for ~39 QBs, ~89 RBs, ~142 WRs, ~67 TEs per week, many players will have NULL for NGS/PFR features. All 4 algorithms in our ensemble handle NULLs natively (CatBoost, XGBoost v1.5+, LightGBM natively; RandomForest via median imputation). A NULL means "no data available" — different from "the value is 0."

---

## 5. Training pipeline structure

**Decision:** Expanding-window walk-forward validation for evaluation. All available data for production model.

- **Training data:** 2020-2025 (6 seasons)
- **Warm-up data:** 2018-2019 (provides rolling average history so week 1 of 2020 has a lookback window)
- **Why not go further back:** NFL passing yards have increased ~15% since 2015. Stale data hurts accuracy. If we need more data, better to add more features from the same seasons, not older seasons.
- **Evaluation:** For each season 2021-2024, train on all prior data, predict each week, measure per-week MAE
- **Production model:** Train on all 2020-2025 data, predict 2026 season weekly
- **Data leakage prevention:** Features for week N only use data from weeks < N (enforced by query layer)

---

## 6. Database integration for feature engineering

**Decision:** Bulk SQL for batch feature engineering. Per-player queries for dashboard only.

V5 feature engineering loads entire tables per season via bulk SQL, then uses pandas groupby/rolling for vectorized computation. No per-player loops.

```
src/nfl/features/v5_engineer.py
  - build_features(season, week) → DataFrame of features for all players
  - Uses bulk SQL: "SELECT * FROM weekly_stats WHERE season BETWEEN ... AND ..."
  - Joins with other tables in pandas
  - Returns one DataFrame with all features, ready for training
```

The query layer from Task 2.1 (get_player_history, get_game_context, etc.) serves the dashboard — single-player lookups at query time.

---

## 7. Player ID mapping

**Decision:** Single master join at the start of feature engineering.

**What GSIS and PFR IDs are:**
- **GSIS ID** (e.g., `00-0033873` for Mahomes) — the NFL's official Game Statistics & Information System ID. Used by 6 of our datasets (player_stats, injuries, NGS, ff_opportunity, depth_charts).
- **PFR ID** (e.g., `MahoPa00` for Mahomes) — Pro Football Reference's ID system. Used by 4 datasets (snap_counts, pfr_advstats).
- The `players` table maps between them (99.7% coverage for active players).

Build a **master player-week table** at the start of feature engineering:
1. Start from weekly_stats (every player who played)
2. LEFT JOIN players to get PFR ID
3. LEFT JOIN snap_counts, pfr_advstats via PFR ID
4. LEFT JOIN injuries, NGS, ff_opportunity, depth_charts via GSIS ID
5. Result: one row per player per week with all available data

---

## 8. Rookies and players with no history

**Decision:** Position-average baselines, 3-game minimum for predictions.

- **Rookies:** Use average stats for their position as rolling average baseline
- **Team changes:** Track by player, not team. Rolling stats follow the player. Game context (opponent, Vegas) comes from the current game.
- **Minimum history:** 3 games required before generating predictions. Players with fewer games flagged as "insufficient data."

---

## 9. Prediction features — pre-game only

**Decision:** Train and predict using only pre-game features. No data leakage.

**What snap counts are:** Each individual play in a game (~60-70 per game). Snap count percentage = fraction of plays a player was on the field. Saquon Barkley at 80% offense snaps = he played ~50 of ~65 offensive plays. Direct measure of opportunity.

Pre-game features (available before kickoff):
- Rolling averages from prior weeks
- Vegas lines (spread, total, implied points)
- Injury status (from injury report, released before game)
- Snap count trend (from prior weeks, not current game)
- Opponent defense rank (from prior weeks)
- Weather, rest days, home/away, depth chart status

Post-game features (NOT used — would cause data leakage):
- Actual game stats, current-game snap counts, current-game NGS

---

## 10. MAE targets

**Decision:** < 4.0 overall as aspirational. Position-specific sub-targets:

| Position | Current (V4) | V5 Target | Strategy |
|----------|-------------|-----------|----------|
| TE | 3.62 | < 3.5 | Already strong. Snap count + injury features should help. |
| RB | 4.65 | < 4.5 | Snap counts are direct opportunity proxy for RBs. |
| WR | 4.66 | < 4.5 | Snap counts + target trends. V2 was best for WR (4.57), should beat that. |
| QB | 6.81 | < 6.5 | Most room to improve. Weather, pressure rate, Vegas passing volume. |

Professional sports betting models: 4-5 MAE typical. Getting to 4.0-4.2 is an excellent portfolio result.

---

## 11. Ablation study — validate what works

**Decision:** Dedicated task in the roadmap for feature ablation.

After initial V5 training:
1. Train with all features → measure MAE per position
2. Remove one feature group at a time → measure MAE change
3. Any group that improves MAE by < 0.05 when included → drop it
4. Document: "Snap counts improved RB MAE by X, NGS had no measurable impact"
5. Retrain final V5 with validated feature set only

This is standard ML practice and makes an excellent portfolio talking point ("I systematically validated which data sources contributed signal").

---

## 12. V5 build order

| Step | Task | Estimated Effort | Notes |
|------|------|-----------------|-------|
| 1 | Feature engineering (Task 3.1) | Heavy — building v5_engineer.py with batch SQL, 13 datasets | Largest single task in V5 |
| 2 | Feature validation (Task 3.1b) | Medium — verify no leakage, no NaN explosions, spot-check values | Quick but critical |
| 3 | Initial model training (Task 3.2) | Heavy — train all positions, walk-forward validation, ~20-30 min training time | Compute-intensive |
| 4 | Ablation study (Task 3.2b) | Heavy — retrain N times with feature groups removed, compare MAE | Most time-consuming step |
| 5 | Final retrain (Task 3.2c) | Medium — retrain with validated feature set, generate 2025 predictions | Faster since feature set is finalized |
| 6 | Load V5 predictions into DB | Quick — reuse load_predictions.py pattern from Task 2.3 | Already built |
| 7 | Accuracy dashboard (Task 3.3) | Medium — Streamlit tab comparing V1-V5, scatter plots, MAE charts | Visualization work |

**Time-intensive steps:** Feature engineering (#1), model training (#3), and ablation (#4) will take the most time. The ablation study alone requires retraining the full model ~8-10 times (once per feature group removal).

**Quick steps:** Loading predictions (#6) is already built. Feature validation (#2) is a spot-check pass.

---

## Summary of Key Decisions

| Decision | Choice |
|----------|--------|
| Prediction target | Individual stats per position, compute fantasy points from sum |
| Model types | StatPredictor + POB only (drop EVOB) |
| Algorithms | 4-algorithm ensemble (XGBoost, LightGBM, CatBoost, RF) |
| Feature strategy | Build all tiers, validate with ablation, drop noise |
| Sparse data | NULL-friendly models, no fake imputation |
| Training data | 2020-2025 train, 2018-2019 warm-up only |
| Validation | Expanding-window walk-forward |
| Data access | Bulk SQL for features, per-player queries for dashboard |
| ID mapping | Single master join, transparent GSIS↔PFR mapping |
| Rookies | Position-average baseline, 3-game minimum |
| Pre-game only | Train and predict using only pre-game features |
| MAE target | < 4.0 overall, position-specific sub-targets |
| Ablation | Dedicated task — prove which features helped, drop the rest |
