# Model Retrain Scripts - Summary

## All Retrain Scripts Updated

All version retrain scripts now have **consistent behavior**:

### ✅ What Changed:

**Before:**
- Hardcoded weeks 10-12 for predictions
- Manual week specification
- Only 3 prediction files generated

**After:**
- **Auto-detects** all available weeks in raw data
- Generates predictions for **ALL found weeks** (currently 1-14)
- Validates on the **most recent 3 weeks**
- Future-proof: automatically includes new weeks as data is added

### 📁 Updated Scripts:

1. **`src/nfl/v1_retrain.py`** - ✅ **CREATED** (didn't exist before)
   - V1 baseline model (decay=0.9, 34 features)
   - MAE: 5.14

2. **`src/nfl/v2_retrain.py`** - ✅ Updated
   - V2 improved model (decay=0.85, variance, trends, 42 features)
   - MAE: 4.66

3. **`src/nfl/v3_retrain.py`** - ✅ Updated
   - V3 EPA model (EPA, efficiency, 57 features)
   - MAE: 4.66 (no improvement)

4. **`src/nfl/v4_retrain.py`** - ✅ Updated
   - V4 position-specific (Phase 2 hyperparameters)
   - Target MAE: <4.4

## Usage

All scripts work the same way now:

```bash
# V1 - Baseline
python3 src/nfl/v1_retrain.py

# V2 - Current Best
python3 src/nfl/v2_retrain.py

# V3 - EPA (rejected)
python3 src/nfl/v3_retrain.py

# V4 - Position-specific
python3 src/nfl/v4_retrain.py
```

## What You Get

Each script now:

1. **Auto-detects available weeks**: Scans `data/nfl/raw/` for week files
2. **Generates predictions for ALL weeks**: Creates 14 prediction files (not just 3)
3. **Smart validation**: Uses last 3 weeks for comparison
4. **Consistent output**:
   ```
   Found raw data for 14 weeks: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
   Generating predictions for weeks 1-14...
   
   ✓ Week 1: XXX predictions
   ✓ Week 2: XXX predictions
   ...
   ✓ Week 14: XXX predictions
   
   ✅ Generated predictions for 14/14 weeks!
   
   Validating on weeks: [12, 13, 14]
   ```

## File Structure After Running

```
data/nfl/predictions/v2_variance_trends_mae4.66/
├── predictions_2025_week_1.parquet
├── predictions_2025_week_2.parquet
├── ...
└── predictions_2025_week_14.parquet
```

**Before**: 3 files (weeks 10-12)  
**After**: 14 files (weeks 1-14) ✓

## Benefits

✅ **Complete data**: All weeks have predictions  
✅ **Future-proof**: New weeks auto-included  
✅ **Consistent**: All scripts work the same  
✅ **Smart**: Only validates on recent data  
✅ **Flexible**: Adapts to available data  

## Next Steps

When new week data arrives (e.g., week 15):
1. Add raw data: `data/nfl/raw/player_stats_2025_week_15.parquet`
2. Run any retrain script
3. Predictions for week 15 automatically generated!

No code changes needed!
