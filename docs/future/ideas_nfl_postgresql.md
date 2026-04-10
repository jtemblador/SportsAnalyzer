# NFL Sports Prediction Platform: PostgreSQL Integration Ideas

## Current State

The Sports Analyzer project uses **flat Parquet/Joblib files** exclusively:
- **Raw stats:** `data/nfl/raw/player_stats_YEAR_week_N.parquet` (fetched via nflreadpy)
- **Engineered features:** `data/nfl/features/VERSION/features_YEAR_week_N.parquet` (V1–V4)
- **Trained models:** `data/nfl/models/VERSION/{POS}_{STAT}_{TYPE}.joblib`
- **Predictions:** `data/nfl/predictions/VERSION/predictions_YEAR_week_N.parquet`
- **Vegas odds:** `data/nfl/vegas_odds/team_lines/team_lines_week_N_*.parquet`
- **Dashboard:** `app.py` (Streamlit) reads directly from Parquet files

There is **no database** — all data flows through files on disk.

---

## Why Add PostgreSQL

The Parquet file approach works, but a database opens up capabilities that are hard to do with flat files:

1. **Cross-week queries without loading all files** — "show me Patrick Mahomes' trend over weeks 5–12" currently means loading 8 separate Parquet files
2. **Model run tracking** — compare accuracy across V1–V4 without digging through directories
3. **Prediction history** — query historical predictions vs actuals for any player/stat/week
4. **API-ready** — if you ever build a REST API on top (FastAPI/Flask), PostgreSQL is the natural backend
5. **Resume talking point** — "ML pipeline backed by a normalized PostgreSQL database" is stronger than "reads Parquet files from disk"

---

## Suggested Schema

```sql
-- Core reference tables
CREATE TABLE teams (
    id              SERIAL PRIMARY KEY,
    abbreviation    VARCHAR(5) UNIQUE NOT NULL,   -- e.g., 'KC', 'LAR'
    name            VARCHAR(100) NOT NULL,         -- e.g., 'Kansas City Chiefs'
    conference      VARCHAR(5),                    -- 'AFC', 'NFC'
    division        VARCHAR(20)                    -- 'West', 'North', etc.
);

CREATE TABLE players (
    id              SERIAL PRIMARY KEY,
    nfl_id          VARCHAR(50) UNIQUE,            -- nflreadpy player ID
    name            VARCHAR(200) NOT NULL,
    position        VARCHAR(10),                   -- QB, RB, WR, TE, K
    team_id         INTEGER REFERENCES teams(id),
    active          BOOLEAN DEFAULT TRUE
);

CREATE TABLE games (
    id              SERIAL PRIMARY KEY,
    season          INTEGER NOT NULL,
    week            INTEGER NOT NULL,
    game_date       DATE,
    home_team_id    INTEGER REFERENCES teams(id),
    away_team_id    INTEGER REFERENCES teams(id),
    home_score      INTEGER,
    away_score      INTEGER,
    UNIQUE(season, week, home_team_id)
);

-- Raw weekly stats (mirrors what's in Parquet now)
CREATE TABLE weekly_stats (
    id              SERIAL PRIMARY KEY,
    player_id       INTEGER REFERENCES players(id),
    game_id         INTEGER REFERENCES games(id),
    season          INTEGER NOT NULL,
    week            INTEGER NOT NULL,

    -- Passing
    passing_yards       NUMERIC,
    passing_tds         INTEGER,
    interceptions       INTEGER,
    completions         INTEGER,
    attempts            INTEGER,

    -- Rushing
    rushing_yards       NUMERIC,
    rushing_tds         INTEGER,
    carries             INTEGER,

    -- Receiving
    receiving_yards     NUMERIC,
    receiving_tds       INTEGER,
    receptions          INTEGER,
    targets             INTEGER,

    -- Kicking
    field_goals_made    INTEGER,
    field_goals_att     INTEGER,
    extra_points_made   INTEGER,

    -- Composite
    fantasy_points_ppr  NUMERIC,
    fantasy_points_half NUMERIC,

    UNIQUE(player_id, season, week)
);

-- Engineered features (flexible — version-aware)
CREATE TABLE engineered_features (
    id                  SERIAL PRIMARY KEY,
    player_id           INTEGER REFERENCES players(id),
    season              INTEGER NOT NULL,
    week                INTEGER NOT NULL,
    pipeline_version    VARCHAR(50) NOT NULL,      -- 'v1', 'v2', 'v3', 'v4'
    features            JSONB NOT NULL,             -- all 42–50 features as JSON

    UNIQUE(player_id, season, week, pipeline_version)
);

-- Vegas odds
CREATE TABLE vegas_odds (
    id                      SERIAL PRIMARY KEY,
    game_id                 INTEGER REFERENCES games(id),
    spread                  NUMERIC,                -- home team spread
    over_under              NUMERIC,
    home_implied_total      NUMERIC,
    away_implied_total      NUMERIC,
    fetched_at              TIMESTAMP DEFAULT NOW()
);

-- Model metadata and training runs
CREATE TABLE model_runs (
    id                  SERIAL PRIMARY KEY,
    pipeline_version    VARCHAR(50) NOT NULL,      -- 'v1', 'v2', 'v3', 'v4'
    model_name          VARCHAR(100) NOT NULL,     -- 'QB_fantasy_points_ppr_evob'
    model_type          VARCHAR(50) NOT NULL,      -- 'evob', 'pob', 'stat_predictor'
    algorithm           VARCHAR(50),               -- 'xgboost', 'lightgbm', 'catboost'
    position            VARCHAR(10),               -- 'QB', 'RB', etc.
    target_stat         VARCHAR(100),              -- 'fantasy_points_ppr'
    hyperparams         JSONB,                     -- full hyperparameter dict
    mae                 NUMERIC,
    rmse                NUMERIC,
    r_squared           NUMERIC,
    train_weeks         INTEGER,                   -- how many weeks of training data
    trained_at          TIMESTAMP DEFAULT NOW(),
    model_path          VARCHAR(500)               -- path to .joblib file
);

-- Predictions
CREATE TABLE predictions (
    id                  SERIAL PRIMARY KEY,
    player_id           INTEGER REFERENCES players(id),
    season              INTEGER NOT NULL,
    week                INTEGER NOT NULL,
    model_run_id        INTEGER REFERENCES model_runs(id),
    predicted_value     NUMERIC NOT NULL,
    actual_value        NUMERIC,                   -- filled in after the game
    confidence          NUMERIC,                   -- model confidence score
    pob_probability     NUMERIC,                   -- P(over baseline)
    created_at          TIMESTAMP DEFAULT NOW(),

    UNIQUE(player_id, season, week, model_run_id)
);

-- Indexes for common query patterns
CREATE INDEX idx_stats_player_season ON weekly_stats(player_id, season);
CREATE INDEX idx_predictions_player ON predictions(player_id, season, week);
CREATE INDEX idx_features_version ON engineered_features(pipeline_version);
CREATE INDEX idx_model_runs_version ON model_runs(pipeline_version);
CREATE INDEX idx_predictions_model ON predictions(model_run_id);
```

---

## Integration Ideas

### Idea 1: Prediction Accuracy Dashboard

Store predictions AND actuals in PostgreSQL, then query accuracy metrics directly:

```sql
-- Model accuracy by position (V4)
SELECT
    mr.position,
    mr.algorithm,
    COUNT(*) AS total_predictions,
    ROUND(AVG(ABS(p.predicted_value - p.actual_value)), 2) AS mae,
    ROUND(AVG(CASE WHEN p.actual_value > 0
              THEN ABS(p.predicted_value - p.actual_value) / p.actual_value * 100
              END), 1) AS mape_pct
FROM predictions p
JOIN model_runs mr ON p.model_run_id = mr.id
WHERE mr.pipeline_version = 'v4' AND p.actual_value IS NOT NULL
GROUP BY mr.position, mr.algorithm
ORDER BY mae;
```

This replaces manually comparing Parquet files across version directories.

### Idea 2: Player Trend Queries

```sql
-- Rolling 5-week performance for a player
SELECT
    week,
    fantasy_points_ppr,
    ROUND(AVG(fantasy_points_ppr) OVER (ORDER BY week ROWS 4 PRECEDING), 2) AS rolling_avg,
    ROUND(STDDEV(fantasy_points_ppr) OVER (ORDER BY week ROWS 4 PRECEDING), 2) AS rolling_stddev
FROM weekly_stats
WHERE player_id = (SELECT id FROM players WHERE name = 'Patrick Mahomes')
  AND season = 2025
ORDER BY week;

-- Boom/bust classification
SELECT
    p.name, ws.week, ws.fantasy_points_ppr,
    CASE
        WHEN ws.fantasy_points_ppr > AVG(ws.fantasy_points_ppr) OVER (PARTITION BY ws.player_id) * 1.5
            THEN 'BOOM'
        WHEN ws.fantasy_points_ppr < AVG(ws.fantasy_points_ppr) OVER (PARTITION BY ws.player_id) * 0.5
            THEN 'BUST'
        ELSE 'NORMAL'
    END AS game_type
FROM weekly_stats ws
JOIN players p ON ws.player_id = p.id
WHERE ws.season = 2025 AND p.position = 'WR'
ORDER BY ws.fantasy_points_ppr DESC;
```

### Idea 3: Model Version Comparison

Track every training run so you can compare V1 through V4 without looking at directory names:

```sql
-- Compare MAE across pipeline versions
SELECT
    pipeline_version,
    position,
    ROUND(AVG(mae), 2) AS avg_mae,
    ROUND(MIN(mae), 2) AS best_mae,
    COUNT(*) AS models_trained
FROM model_runs
GROUP BY pipeline_version, position
ORDER BY pipeline_version, position;

-- Which V4 models improved over V2?
SELECT
    v4.model_name,
    v2.mae AS v2_mae,
    v4.mae AS v4_mae,
    ROUND(v2.mae - v4.mae, 2) AS improvement
FROM model_runs v4
JOIN model_runs v2
    ON v4.model_name = v2.model_name
   AND v4.pipeline_version = 'v4'
   AND v2.pipeline_version = 'v2'
WHERE v4.mae < v2.mae
ORDER BY improvement DESC;
```

### Idea 4: Vegas Odds Correlation Analysis

```sql
-- Do favorites outperform predictions?
SELECT
    CASE WHEN vo.spread < 0 THEN 'Favorite' ELSE 'Underdog' END AS team_role,
    ROUND(AVG(p.actual_value - p.predicted_value), 2) AS avg_over_prediction,
    COUNT(*) AS n
FROM predictions p
JOIN model_runs mr ON p.model_run_id = mr.id
JOIN weekly_stats ws ON ws.player_id = p.player_id AND ws.season = p.season AND ws.week = p.week
JOIN games g ON ws.game_id = g.id
JOIN vegas_odds vo ON vo.game_id = g.id
WHERE mr.pipeline_version = 'v4' AND p.actual_value IS NOT NULL
GROUP BY team_role;
```

### Idea 5: Feature Importance Tracking

Store feature importance scores from each model run:

```sql
-- Add to model_runs or create a separate table
ALTER TABLE model_runs ADD COLUMN feature_importances JSONB;

-- Query: which features matter most for QB predictions?
SELECT
    key AS feature,
    ROUND(AVG(value::numeric), 4) AS avg_importance
FROM model_runs,
     jsonb_each_text(feature_importances)
WHERE position = 'QB' AND pipeline_version = 'v4'
GROUP BY key
ORDER BY avg_importance DESC
LIMIT 15;
```

### Idea 6: Streamlit Dashboard Enhancement

Instead of the dashboard loading multiple Parquet files, query PostgreSQL directly:

```python
# Before (Parquet):
import pandas as pd
dfs = []
for week in range(1, 19):
    path = f"data/nfl/predictions/v4_position_specific/predictions_2025_week_{week}.parquet"
    dfs.append(pd.read_parquet(path))
all_predictions = pd.concat(dfs)

# After (PostgreSQL):
import psycopg2
import pandas as pd

conn = psycopg2.connect(dbname="nfl_predictions", user="j0e")
all_predictions = pd.read_sql("""
    SELECT p.*, pl.name, pl.position, mr.pipeline_version
    FROM predictions p
    JOIN players pl ON p.player_id = pl.id
    JOIN model_runs mr ON p.model_run_id = mr.id
    WHERE p.season = 2025 AND mr.pipeline_version = 'v4'
    ORDER BY p.week, pl.position, p.predicted_value DESC
""", conn)
```

This is simpler, faster, and lets Streamlit filters translate directly into SQL `WHERE` clauses.

---

## What NOT to Put in PostgreSQL

Keep some things as files:

| Keep as Files | Why |
|---------------|-----|
| **Trained model weights** (`.joblib`) | Binary blobs, PostgreSQL doesn't help here |
| **Raw Parquet files from nflreadpy** | Keep as archival copies; load into DB via ETL |
| **Streamlit app code** | Obviously |

The database is for **queryable, relational data** — not binary artifacts.

---

## Implementation Approach

### Option A: Dual-write (recommended to start)

Keep the Parquet pipeline as-is. After each pipeline run, also insert the data into PostgreSQL. This lets you build the database layer incrementally without breaking anything.

```python
# In your pipeline code, after saving Parquet:
df.to_sql("weekly_stats", engine, if_exists="append", index=False)
```

### Option B: Database-first

Rewrite the pipeline to use PostgreSQL as the primary store. Parquet files become exports/backups. More work upfront but cleaner long-term.

### Option C: Read-only analytics database

Keep Parquet as the pipeline's data store. Periodically bulk-load everything into PostgreSQL just for querying and dashboard use. Lowest friction.

---

## Suggested Learning Path

| Step | What | SQL Skills |
|------|------|------------|
| 1 | Create the database and tables | `CREATE TABLE`, foreign keys, data types |
| 2 | Write a script to load `weekly_stats` from Parquet into PostgreSQL | `INSERT`, bulk loading, `pandas.to_sql()` |
| 3 | Write player trend queries in `psql` | `SELECT`, `JOIN`, `WHERE`, `ORDER BY` |
| 4 | Add window function queries (rolling avg, rank) | `OVER`, `PARTITION BY`, `ROWS PRECEDING` |
| 5 | Build the `model_runs` table and log training metadata | `JSONB`, `INSERT ... RETURNING` |
| 6 | Store predictions + backfill actuals | `UPDATE`, `JOIN` in updates |
| 7 | Connect Streamlit to PostgreSQL | `pd.read_sql()`, parameterized queries |
| 8 | Add `vegas_odds` and cross-table analysis | Multi-table JOINs, subqueries, CTEs |

Each step builds on the previous one and teaches a new set of SQL concepts. Steps 1–4 are the core — the rest are extensions you can tackle as interest dictates.
