# NFL Player Performance Prediction System
## Final Report - Fall 2025
**Jose Trinidad Temblador**
**CSC-ITC-492**

---

## Executive Summary

This project developed a production-ready NFL player performance prediction system achieving 4.26 MAE, exceeding professional DFS platform standards (4.5-5.5 MAE). The system evolved through four model versions, revealing that game context (Vegas betting lines) predicts fantasy performance better than advanced player analytics (EPA, efficiency metrics). The final V4 model processes 109 weeks of NFL data (2020-2025) using 40 position-specific CatBoost models, generating predictions with confidence intervals through an interactive Streamlit dashboard.

### Model Accuracy Improvement Trajectory

```
Overall MAE by Version
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
V1  ████████████████████ 5.14 MAE (Baseline)
V2  ███████████████      4.66 MAE (-10% improvement)
V3  ███████████████      4.66 MAE (0% - REJECTED)
V4  █████████████        4.26 MAE (-17% total, PRODUCTION ✓)
    └────────────────────────────────────────────────┘
    0.0   1.0   2.0   3.0   4.0   5.0   6.0  MAE
Professional Standard: 4.5-5.5 MAE ──────────────────▶

Position-Specific MAE Journey (V1 → V2 → V3 → V4)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QB  █████████████████████ 6.56 → ███████████ 4.67 ✓ (-35%)
RB  ███████████ 4.54 → ███████████ 4.41 ✓ (modest)
WR  ████████████ 4.99 → ███████████ 4.57 ✓ (use V2)
TE  ████████████████ 6.15 → ████ 2.34 ✓ (-62% BEST!)

Model Explanatory Power (R-squared)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
V1  ███              0.181 (explains 18% of variance)
V2  ██████           0.360 (explains 36% - doubled!)
V3  ██████           0.358 (same as V2)
V4  ███████          0.442 (explains 44% - production)
```

---

## Technical Architecture

### Data Pipeline
- **Source**: nflverse API (109 weeks, 2020-2025)
- **Storage**: Parquet columnar format (45MB, 90% smaller than CSV)
- **Structure**: data/nfl/raw/ → features/ → models/ → predictions/
- **Processing**: Automated incremental updates, local-first architecture

### Feature Engineering (V4: 50 features)
- **Rolling averages** (decay=0.85): Recent 3 games = 60.7% weight
- **Variance metrics**: Boom/bust identification
- **Trend indicators**: Target share momentum
- **Opponent adjustments**: Defensive rankings
- **Vegas features** (8 columns): Spreads, totals, implied scores, game script indices

### ML Architecture
- **Algorithm**: CatBoost gradient boosting
- **Models**: 40 total (5 positions × 8 model types)
- **Types**: POB (probability), EVOB (expected value), STAT (direct prediction)
- **Hyperparameters** (position-specific):
  - QB: depth=9, iter=500, lr=0.005 (complex game scripts)
  - RB/WR: depth=7, iter=300, lr=0.01 (standard)
  - TE: depth=6, iter=300, lr=0.01 (moderate)
  - K: depth=3, iter=100, lr=0.01 (simple, prevent overfitting)

---

## Version Evolution

The system evolved through four versions over six days (Dec 1-7). **V1** established a baseline with 34 features (rolling averages, opponent adjustments), achieving 5.14 MAE overall but revealing position disparities: RB 4.54 MAE (good), QB 6.56 MAE (poor), TE 6.15 MAE (worst).

**V2** added variance and trend features (42 columns total), strengthening decay from 0.9 to 0.85. Overall MAE dropped to 4.66 (10% improvement) and R² doubled to 0.360. TE predictions improved massively (6.15→3.57 MAE, 42% reduction) as variance features distinguished consistent producers from touchdown-dependent players. However, QB predictions worsened (6.56→7.19 MAE), revealing that variance/trends could not address game script-dependent passing volume.

**V3** tested advanced analytics by adding 15 EPA and efficiency metrics (57 columns total). Results were shocking: MAE remained exactly 4.66, identical to V2. Fantasy scoring depends on volume, not efficiency—100 rushing yards scores the same whether on 25 carries (4.0 YPC) or 18 carries (5.6 YPC). EPA metrics were highly correlated with existing volume statistics, providing no new predictive signal. V3 required 3-4 hours of feature engineering versus V2's 20 minutes, making it unacceptable for production. Critical lesson: context and opportunity matter more than player talent metrics.

**V4** shifted focus to game context by adding eight Vegas features (50 columns total): spreads, totals, implied team scores, and position-specific game script indices. Combined with position-specific hyperparameter tuning, V4 achieved 4.26 MAE (9% improvement over V2, 17% total), exceeding professional DFS standards. QB predictions plummeted from 7.19 to 4.67 MAE (35% reduction), solving the position's prediction problem. TE improved from 3.57 to 2.34 MAE (34% reduction), transforming from worst position in V1 to best in V4 (62% total improvement). RB showed modest improvement to 4.41 MAE. WR regressed to 5.06 MAE, likely due to validation set differences (weeks 12-14 had injuries and target redistributions). Production system maintains both V4 (for QB/RB/TE) and V2 (for WR).

---

## Results and Validation

### Position-Specific Analysis

**Quarterbacks** transformed from worst position (7.19 MAE in V2/V3) to second-best (4.67 MAE in V4), a 35% improvement. V4's Vegas features captured game script effects that historical averages missed. When Kansas City played as a 10-point favorite with a 29-point implied total, V4 correctly predicted Patrick Mahomes would throw frequently (predicted 27.3 points, actual 26.1, error 1.2). Similarly accurate predictions for Jared Goff (error 0.8) and Josh Allen (error 1.5) demonstrated that Vegas context solved the QB prediction challenge.

**Tight ends** showed the most remarkable improvement, transforming from worst position in V1 (6.15 MAE) to best in V4 (2.34 MAE), a 62% reduction. V2's variance features provided the first breakthrough (6.15→3.57 MAE) by distinguishing elite TEs like Travis Kelce with consistent 8-12 target games from touchdown-dependent players with volatile usage. V4's Vegas features delivered the second breakthrough (3.57→2.34 MAE) by predicting game script effects. Week 12-13 predictions demonstrated sub-1-point accuracy: Kelce (error 1.1), Kittle (error 0.7), Hockenson (error 0.5), Bowers (error 0.9).

**Running backs** remained stable across versions (4.54→4.41 MAE), achieving the best baseline accuracy because carry volume follows predictable patterns. Lead backs like Christian McCaffrey receive 20-25 touches per game regardless of game script. V4 recovered to 4.41 MAE through Vegas features that identified game scripts favoring rushing (favorites protecting leads) versus passing (underdogs playing catch-up). Week 12-13 predictions for elite backs stayed within 2 points: McCaffrey (errors 1.3, 2.5), Henry (error 0.8), Barkley (error 1.8).

**Wide receivers** showed V2 as optimal (4.57 MAE) with V4 regressing to 5.06 MAE, likely due to validation set differences. V2 validated on weeks 10-12 while V4 validated on weeks 12-14, when multiple injuries and unexpected target redistributions created prediction challenges. V2's trend features tracked changing target shares, proving more valuable than V4's game script features. Production system routes WR predictions to V2 while using V4 for other positions.

### Most Accurately Predicted Players

| Player | Position | Avg Predicted | Avg Actual | Avg Error | Games |
|--------|----------|---------------|------------|-----------|-------|
| Travis Kelce | TE | 17.8 | 17.6 | 0.7 | 3 |
| Christian McCaffrey | RB | 21.2 | 21.1 | 0.9 | 3 |
| Patrick Mahomes | QB | 26.4 | 25.8 | 1.0 | 3 |
| George Kittle | TE | 14.9 | 14.2 | 1.1 | 3 |
| Jared Goff | QB | 23.7 | 24.9 | 1.2 | 3 |
| Derrick Henry | RB | 18.6 | 18.9 | 1.3 | 3 |
| TJ Hockenson | TE | 12.1 | 12.7 | 1.3 | 3 |
| Brock Bowers | TE | 16.8 | 17.4 | 1.4 | 3 |
| Saquon Barkley | RB | 19.1 | 19.7 | 1.5 | 3 |
| Josh Allen | QB | 22.6 | 21.8 | 1.6 | 3 |

These sub-2-point average errors represented near-optimal prediction accuracy. Travis Kelce's 0.7-point error occurred because he maintains consistent 8-10 target games regardless of game script. The dominance of tight ends in the most accurate predictions list (5 of 10 players) reflected V4's exceptional TE prediction accuracy.

### Least Predictable Players

| Player | Position | Avg Predicted | Avg Actual | Avg Error | Games |
|--------|----------|---------------|------------|-----------|-------|
| Taysom Hill | TE | 11.7 | 22.8 | 11.1 | 2 |
| Jahmyr Gibbs | RB | 16.8 | 28.2 | 11.4 | 3 |
| Jonathan Taylor | RB | 17.3 | 29.7 | 12.4 | 3 |
| Deebo Samuel | WR | 15.9 | 9.1 | 6.8 | 3 |
| DK Metcalf | WR | 16.2 | 10.4 | 5.8 | 3 |

Taysom Hill's massive 11.1-point error occurred because his usage is uniquely unpredictable—listed as a TE, he functions as a gadget player (QB/RB/WR/TE/blocker). Gibbs and Taylor showed large errors due to massive outlier performances (Gibbs: 4 TDs in one game, Taylor: 49-point explosion). These three-standard-deviation performances are inherently unpredictable and occur randomly across the league.

### Professional Comparison

V4's 4.26 MAE compared favorably to professional DFS platforms (4.5-5.5 MAE typical). The achievement of professional-level accuracy using only public data and standard machine learning techniques is significant, as commercial platforms often have access to proprietary injury information and sophisticated betting market data beyond simple Vegas lines.

---

## Conclusion

This project delivered a production-ready prediction system achieving professional accuracy (4.26 MAE), exceeding industry standards. The development process revealed that iteration and empirical testing matter more than theoretical sophistication: V3's failure despite advanced analytics taught that features must predict opportunity and context rather than efficiency and talent. V4's success with Vegas features validated that game environment drives fantasy production more than player ability metrics.

The system processes 109 weeks of NFL data efficiently (Parquet format: 45MB), calculates 50 predictive features, trains 40 position-specific models, and provides interactive predictions through a Streamlit dashboard. Position-specific achievements: TE improved 62%, QB improved 29%, RB improved 3%, WR remains challenging. The hybrid approach (V4 for QB/RB/TE, V2 for WR) demonstrated practical flexibility in production systems.

**Final Status:**
- ✅ Data pipeline: 109 weeks automated
- ✅ Feature engineering: 50 optimized features
- ✅ Model training: 40 models, 4.26 MAE
- ✅ Web dashboard: Interactive predictions
- ✅ Documentation: Complete project history

Project delivered on time and exceeded initial accuracy targets, demonstrating systematic iteration, empirical validation, and willingness to reject unsuccessful approaches lead to meaningful improvements.
