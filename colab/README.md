# Colab Notebooks

Jupyter notebooks for running compute-intensive V5 workloads on Google Colab Pro.

## Why Colab

Feature engineering (~4-5 hours) and model training (~1-4 hours) are too slow on the local machine. Colab Pro provides high-RAM CPU instances that cut these times dramatically.

## Notebooks

| Notebook | Purpose | Status |
|----------|---------|--------|
| `colab_test.ipynb` | Verify Colab connection, Drive mount, data access, ML libraries | Ready |
| `v5_feature_engineering.ipynb` | Run V5 player + DST feature engineering on all seasons | Ready (HANDOFF #1.5) |
| `v5_training.ipynb` | Train V5 models (54 ensembles, 174 .joblib) with walk-forward eval | Ready (HANDOFF #2) |
| `v5_ablation.ipynb` | Feature ablation study | Pending Task 3.2b |
| `v5_final_retrain.ipynb` | Final V5 training with validated features + predictions | Pending Task 3.2c |

## Workflow

Each notebook follows the same pattern:
1. Mount Google Drive
2. Load Parquet data from `/content/drive/MyDrive/SportsAnalyzer/data/nfl/`
3. Run the pipeline (feature engineering, training, etc.)
4. Save output to `/content/drive/MyDrive/SportsAnalyzer/output/`

## Drive Structure

```
My Drive/SportsAnalyzer/
├── data/nfl/                 ← 13 dataset folders (uploaded once from local data/nfl/)
├── src/nfl/features/v5/      ← V5 features package
│   ├── __init__.py
│   ├── config.py
│   ├── master_table.py
│   ├── rolling.py
│   ├── context.py
│   ├── usage.py
│   ├── advanced.py
│   ├── dst.py        ← Task 3.1.5
│   ├── utils.py      ← shared rolling helpers
│   └── engineer.py
├── src/nfl/training/v5/      ← V5 training package (Task 3.2)
│   ├── __init__.py
│   ├── config.py     ← POSITION_ALGORITHMS, COUNT_STATS, hyperparams
│   ├── data.py       ← per-position loader, history filter, POB target
│   ├── models.py     ← StatPredictor + POBModel ensembles
│   ├── walkforward.py
│   └── train.py
└── output/
    ├── features/v5/  ← 16 parquets from HANDOFF #1.5
    ├── models/v5/    ← 174 .joblib + 54 .json + _mae_summary.csv (HANDOFF #2)
    └── predictions/  ← V5 predictions (Task 3.2c)
```

## Opening in VS Code

1. Click any `.ipynb` file
2. Click "Select Kernel" top right → Colab → CPU (High-RAM)
3. Sign in to Google
4. Run cells with `Shift+Enter`

## Opening in Browser

Upload the notebook to Drive, then double-click → "Open with Google Colaboratory."
