# Legacy Code — V1 through V4

This directory contains the ML pipeline code from model versions V1 through V4. It is preserved as reference for building V5, not as active code.

## MAE Progression

| Version | MAE | Key Change |
|---------|-----|-----------|
| V1 | 5.14 | Baseline — rolling averages, basic features |
| V2 | 4.66 | Added variance, usage trends, opponent context |
| V3 | 4.66 | EPA and efficiency metrics (no improvement — lesson learned) |
| V4 | 4.26 | Vegas odds integration, position-specific hyperparameters |

## What's Here

```
v1-v4/
  features/
    engineer.py         — V1/V2 feature engineer (rolling averages, trends)
    v3_engineer.py      — V3 (EPA, efficiency — did not improve MAE)
    v4_engineer.py      — V4 (Vegas integration, position-specific features)
  models/
    base.py             — POBModel, EVOBModel, StatPredictor, NFLModelPipeline
    v4_models.py        — PositionSpecificEVOBModel
  training/
    train.py            — Original training script
    generate_predictions.py — Prediction generation flow
    v1_retrain.py       — V1 hyperparameters and training config
    v2_retrain.py       — V2 hyperparameters
    v3_retrain.py       — V3 hyperparameters
    v4_retrain.py       — V4 hyperparameters (best reference for V5)
  data/
    column_mappings.py  — Display name mappings for old dashboard
```

## Key Takeaways for V5

- **V3 lesson:** Advanced analytics (EPA, efficiency) did not improve MAE. Simple features that predict context and opportunity (variance, trends, Vegas lines) outperformed complex player metrics.
- **V4 success:** Vegas implied totals were the strongest single feature for volume prediction. Position-specific hyperparameters (deeper trees for QB, simpler for K) improved all positions.
- **Three model types work well:** POB (probability over baseline), EVOB (expected value over baseline), and StatPredictor (direct regression). Ensemble averaging across 4 algorithms (XGBoost, LightGBM, CatBoost, RandomForest) is robust.
- **V4 hyperparameters in `v4_retrain.py`** are the best starting point for V5 tuning.

## Restoring Full V4 Codebase

The complete V4 codebase is tagged in git:

```bash
git checkout v4-final
```
