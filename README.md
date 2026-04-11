# NFL Sports Analyzer

A machine learning platform that predicts weekly NFL fantasy football player performance using ensemble models, engineered features, and Vegas odds integration.

## Results

| Version | MAE | Key Improvement |
|---------|-----|-----------------|
| V1 Baseline | 5.14 | Rolling averages, opponent defense ranks |
| V2 | 4.66 | Variance features, trend detection, stronger decay |
| V3 | 4.66 | EPA, efficiency metrics (PACR, RACR, WOPR, CPOE) |
| **V4 (Production)** | **4.26** | **Position-specific hyperparameters, Vegas odds** |

Professional fantasy analysts typically achieve 4-5 MAE. V4 exceeds this benchmark.

## How It Works

```
Raw Stats (nflreadpy) → Feature Engineering → ML Ensemble → Weekly Predictions → Dashboard
```

1. **Data Ingestion** — Fetches player stats, schedules (Vegas lines, weather), injuries, snap counts, and advanced metrics from nflreadpy
2. **Feature Engineering** — Builds 50+ predictive features per player: rolling averages, variance, trends, opponent adjustments, Vegas-derived game context
3. **ML Ensemble** — Trains 3 model types (POB, EVOB, StatPredictor) x 4 algorithms (CatBoost, XGBoost, LightGBM, RandomForest) per position
4. **Predictions** — Generates weekly fantasy point predictions with confidence intervals and probability-over-baseline scores
5. **Dashboard** — Streamlit app for exploring stats, predictions, and model performance

## Project Structure

```
Sports-Analyzer/
├── app.py                          # Streamlit dashboard
├── src/nfl/
│   ├── data/                       # Data ingestion (pipeline, fetchers)
│   ├── features/                   # Feature engineering (V1-V4)
│   ├── models/                     # ML model classes
│   ├── training/                   # Training & prediction scripts
│   └── odds/                       # Vegas odds integration
├── tests/                          # Test suite (35+ tests)
├── data/nfl/                       # Parquet data files
│   ├── raw/                        # Player stats (2018-2025)
│   ├── schedules/                  # Game schedules, odds, weather
│   ├── features/{v1,v2,v3,v4}/     # Engineered features by version
│   ├── models/{v1,v2,v3,v4}/       # Trained model files
│   └── predictions/{v1,v2,v3,v4}/  # Weekly predictions by version
└── docs/                           # Documentation, progress reports
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Fetch all data (player stats, schedules, etc.)
python src/nfl/data/pipeline.py

# Run the dashboard
streamlit run app.py
```

## Tech Stack

- **Data:** nflreadpy, pandas, Parquet
- **ML:** CatBoost, XGBoost, LightGBM, scikit-learn
- **Dashboard:** Streamlit, Plotly
- **Testing:** pytest

## Roadmap

See [docs/ROADMAP.md](docs/ROADMAP.md) for the full development plan:

- **Phase 0** (in progress) — Expand data collection to 12+ datasets from nflreadpy
- **Phase 1** — PostgreSQL database integration
- **Phase 2** — Pipeline reads from database
- **Phase 3** — V5 model with expanded features (target MAE < 4.0)
- **Phase 4** — FastAPI, automation, Docker
