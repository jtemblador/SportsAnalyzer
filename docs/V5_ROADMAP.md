# Sports Analyzer — Development Roadmap

## Current State
- V4 production model: 4.26 MAE (17% improvement over V1 baseline)
- **Phase 0 complete:** 13 datasets fetched, 248 Parquet files, 798,000+ records (2018-2025)
- **Phase 1 complete:** PostgreSQL database with 798,176 rows across 14 tables, verified against NFL.com. `--refresh-db` flag automates DB sync after fetch.
- **Phase 2 complete:** SQL query layer (7 functions), legacy code in `legacy/v1-v4/`, 65,921 predictions loaded with cross-version accuracy queries
- **Phase 3 Task 3.1 complete:** V5 feature engineer at `src/nfl/features/v5/` producing 90 feature columns on 2024 data in 16s (7 modules, 2 review passes, 51 V5 tests)
- 10 fetcher classes registered in unified pipeline (`fetch_all()`, `fetch_latest()`, `--refresh-db`)
- 295 tests passing, 0 failures
- `app.py` (Streamlit dashboard) is broken — will be rebuilt in Task 4.1
- Project restructured: active code in `src/nfl/` (data + db), V1-V4 ML code in `legacy/v1-v4/`
- Full V4 codebase tagged as `v4-final` for reproducibility
- Full dataset audit — see `docs/V5_AVAILABLE_DATASETS.md` and `docs/V5_DATA_CATALOG.md`
- **Database inventory (16 tables, 864K+ rows):**

| Table | Rows | V5 Feature Use |
|-------|------|----------------|
| weekly_stats | 147,050 | Rolling averages, variance, trends (proven V2/V4) |
| games | 2,227 | Vegas lines, spread, implied total, weather, rest, home/away (proven V4) |
| injuries | 45,337 | Player own status, teammate injury impact |
| snap_counts | 205,354 | Offense snap %, snap trend (direct opportunity proxy) |
| depth_charts | 258,942 | Starter/backup flag (2018-2024 only) |
| ff_opportunity | 47,282 | Expected fantasy points, actual vs expected differential |
| team_stats | 4,454 | Opponent defense rank against position (proven V4) |
| ngs_passing | 4,785 | Time to throw, CPOE, aggressiveness (~32 QBs/week) |
| ngs_rushing | 4,885 | Rush yards over expected, efficiency (~34 players/week) |
| ngs_receiving | 11,708 | Separation, YAC above expected (~66 players/week) |
| pfr_pass_advstats | 5,424 | Pressure rate, blitz rate (~36 QBs/week) |
| pfr_rush_advstats | 18,461 | Yards after contact, broken tackles (~130 players/week) |
| pfr_rec_advstats | 35,724 | Drop rate, passer rating when targeted (~250 players/week) |
| players | 6,543 | GSIS↔PFR ID mapping table |
| predictions | 65,921 | V1-V4 predictions with actuals backfilled |
| model_versions | 4 | V1-V4 metadata (MAE, description) |

## Goal
1. Expand data collection to all useful nflreadpy datasets (Tier 1 + Tier 2)
2. Add PostgreSQL as the data backbone
3. Build V5 model with all new data sources to push MAE below 4.0

## Data Architecture
```
nflreadpy (fetch all datasets)
    ├── Parquet files (disk backup, always written)
    └── PostgreSQL (primary query source, always written)

Pipeline reads from: PostgreSQL (with Parquet as fallback)
Dashboard reads from: PostgreSQL
Model training reads from: PostgreSQL
Parquet stays as: backup/rebuild source — never deleted
```

Two copies of data is intentional. Parquet = backup. PostgreSQL = query engine.

## Season Range: 2018-2025

- **Pull data from 2018-2025** (8 seasons) for all datasets
- **Train models on 2020-2025** (6 seasons) — modern NFL, 17-game era (2021+), avoids stale data
- **2018-2019 serve as warm-up** — provides rolling average history so Week 1 of 2020 has a full lookback window
- 2018-2019 are 16-game seasons (no Week 18) — handled by cross-season logic already in the feature engineer
- 2020 COVID season stays in training — only 1 of 6 seasons, weekly stats were still real football

## Unified Pipeline

Each fetcher is a class registered in `src/nfl/data/pipeline.py`. As tasks are completed, new fetchers are added to the pipeline's `fetch_all()` method. Running `pipeline.py` fetches ALL datasets, skipping anything already downloaded.

## Task Workflow

Every task follows the phase files in `.claude/instructions/`:
```
1plan.md → 2build.md → 3review.md → 4test.md → 5document.md → commit
```

---

## Phase 0: Data Expansion — Tier 1 (Core Datasets)
**Goal:** Pull all high-impact datasets into Parquet. No model changes yet — just get the data.

### Task 0.1 — Schedules (Vegas lines + weather + game context)
- [x] **Source:** `nfl.load_schedules(seasons)` — 46 columns
- [x] **Key data:** `spread_line`, `total_line`, `home_moneyline`, `away_moneyline`, `temp`, `wind`, `roof`, `surface`, `home_rest`, `away_rest`, `stadium`, `div_game`, `overtime`, `home_score`, `away_score`
- [x] Create `src/nfl/data/fetch_schedules.py`
- [x] Fetch seasons 2018-2025, save to `data/nfl/schedules/schedules_{season}.parquet`
- [x] Calculate derived fields: `home_implied_total`, `away_implied_total` from `total_line` and `spread_line`
- [x] Verify: spot-check a known game (e.g., 2024 Week 1 BAL@KC: spread 3.0, total 46.0)
- [x] **Deliverable:** All historical schedules with odds and weather stored locally
- [x] **Note:** This replaces the paid Odds API entirely
- [x] **Pipeline integration:** Register `ScheduleFetcher` in `NFLDataPipeline` so `fetch_all()` includes schedules
- [x] **Also:** Backfill raw player_stats for 2018-2019 (all weeks) and 2020 weeks 1-9 (currently missing)

### Task 0.2 — Injuries
- [x] **Source:** `nfl.load_injuries(seasons)` — 16 columns
- [x] **Key data:** `report_status` (Out/Doubtful/Questionable/Probable), `report_primary_injury`, `practice_status` (DNP/Limited/Full), per player per week
- [x] Create `src/nfl/data/fetch_injuries.py`
- [x] Fetch seasons 2018-2025, save to `data/nfl/injuries/injuries_{season}.parquet`
- [x] Verify: check a known injury (e.g., a star player who missed games in 2024)
- [x] **Deliverable:** Weekly injury reports for all players, all seasons

### Task 0.3 — Snap Counts
- [x] **Source:** `nfl.load_snap_counts(seasons)` — 16 columns
- [x] **Key data:** `offense_snaps`, `offense_pct`, `defense_snaps`, `defense_pct`, `st_snaps`, `st_pct`, per player per week
- [x] Create `src/nfl/data/fetch_snap_counts.py`
- [x] Fetch seasons 2018-2025, save to `data/nfl/snap_counts/snap_counts_{season}.parquet`
- [x] Verify: check a known starter has >70% offense_pct, a backup has <30%
- [x] **Deliverable:** Snap participation data for all players, all seasons

### Task 0.4 — Next Gen Stats (Passing)
- [x] **Source:** `nfl.load_nextgen_stats(seasons, stat_type='passing')` — 29 columns
- [x] **Key data:** `avg_time_to_throw`, `avg_completed_air_yards`, `avg_intended_air_yards`, `aggressiveness`, `completion_percentage_above_expectation`, `expected_completion_percentage`, `max_completed_air_distance`
- [x] Create `src/nfl/data/fetch_nextgen_stats.py` (handles all 3 stat types)
- [x] Fetch seasons 2018-2025, save to `data/nfl/nextgen_stats/ngs_passing_{season}.parquet`
- [x] Verify: check a known QB has reasonable values (Mahomes time_to_throw ~2.5-3.0s)
- [x] **Deliverable:** QB-level Next Gen passing metrics for all seasons

### Task 0.5 — Next Gen Stats (Rushing)
- [x] **Source:** `nfl.load_nextgen_stats(seasons, stat_type='rushing')` — 22 columns
- [x] **Key data:** `efficiency`, `avg_time_to_los`, `rush_yards_over_expected`, `rush_yards_over_expected_per_att`, `rush_pct_over_expected`, `percent_attempts_gte_eight_defenders`
- [x] Use the same `src/nfl/data/fetch_nextgen_stats.py` from Task 0.4
- [x] Fetch seasons 2018-2025, save to `data/nfl/nextgen_stats/ngs_rushing_{season}.parquet`
- [x] Verify: top RBs should have positive `rush_yards_over_expected`
- [x] **Deliverable:** RB-level Next Gen rushing metrics for all seasons

### Task 0.6 — Next Gen Stats (Receiving)
- [x] **Source:** `nfl.load_nextgen_stats(seasons, stat_type='receiving')` — 23 columns
- [x] **Key data:** `avg_cushion`, `avg_separation`, `avg_intended_air_yards`, `catch_percentage`, `avg_yac_above_expectation`, `avg_expected_yac`, `percent_share_of_intended_air_yards`
- [x] Use the same `src/nfl/data/fetch_nextgen_stats.py` from Task 0.4
- [x] Fetch seasons 2018-2025, save to `data/nfl/nextgen_stats/ngs_receiving_{season}.parquet`
- [x] Verify: elite WRs should have high separation (~3.0+) and positive YAC above expectation
- [x] **Deliverable:** WR/TE-level Next Gen receiving metrics for all seasons

### Task 0.7 — Fantasy Opportunity (Expected Fantasy Points)
- [x] **Source:** `nfl.load_ff_opportunity(seasons)` — 159 columns
- [x] **Key data:** `total_fantasy_points_exp`, `total_fantasy_points_diff` (actual minus expected), `pass_fantasy_points_exp`, `rec_fantasy_points_exp`, `rush_fantasy_points_exp`, plus team-level shares for all stats
- [x] Create `src/nfl/data/fetch_ff_opportunity.py`
- [x] Fetch seasons 2018-2025, save to `data/nfl/ff_opportunity/ff_opportunity_{season}.parquet`
- [x] Verify: `total_fantasy_points_exp` should correlate with actual fantasy points (r > 0.7)
- [x] **Deliverable:** Expected vs actual fantasy points for all players, all seasons
- [x] **Note:** This directly feeds our POB model — players consistently above expected are outperformers

---

## Phase 0B: Data Expansion — Tier 2 (Advanced Datasets)
**Goal:** Pull supplementary datasets that add meaningful signal.

### Task 0.8 — PFR Advanced Stats (Passing)
- [x] **Source:** `nfl.load_pfr_advstats(seasons, stat_type='pass')` — 24 columns
- [x] **Key data:** `passing_drops`, `passing_drop_pct`, `passing_bad_throws`, `passing_bad_throw_pct`, `times_pressured`, `times_pressured_pct`, `times_hurried`, `times_blitzed`, `times_hit`
- [x] Create `src/nfl/data/fetch_pfr_advstats.py` (handles all 3 stat types)
- [x] Fetch seasons 2018-2025, save to `data/nfl/pfr_advstats/pfr_pass_{season}.parquet`
- [x] Verify: QBs behind bad O-lines should have high `times_pressured_pct` (>30%)
- [x] **Deliverable:** QB pressure and accuracy metrics for all seasons

### Task 0.9 — PFR Advanced Stats (Rushing)
- [x] **Source:** `nfl.load_pfr_advstats(seasons, stat_type='rush')` — 16 columns
- [x] **Key data:** `rushing_yards_before_contact`, `rushing_yards_before_contact_avg`, `rushing_yards_after_contact`, `rushing_yards_after_contact_avg`, `rushing_broken_tackles`
- [x] Use the same `src/nfl/data/fetch_pfr_advstats.py` from Task 0.8
- [x] Fetch seasons 2018-2025, save to `data/nfl/pfr_advstats/pfr_rush_{season}.parquet`
- [x] Verify: elite RBs should have high yards_after_contact_avg (>2.5)
- [x] **Deliverable:** RB contact and elusiveness metrics for all seasons

### Task 0.10 — PFR Advanced Stats (Receiving)
- [x] **Source:** `nfl.load_pfr_advstats(seasons, stat_type='rec')` — 17 columns
- [x] **Key data:** `receiving_drop`, `receiving_drop_pct`, `receiving_broken_tackles`, `receiving_int` (INTs on targets), `receiving_rat` (passer rating when targeted)
- [x] Use the same `src/nfl/data/fetch_pfr_advstats.py` from Task 0.8
- [x] Fetch seasons 2018-2025, save to `data/nfl/pfr_advstats/pfr_rec_{season}.parquet`
- [x] Verify: reliable receivers should have drop_pct < 5%
- [x] **Deliverable:** WR/TE reliability and target quality metrics for all seasons

### Task 0.11 — Team Stats
- [x] **Source:** `nfl.load_team_stats(seasons)` — 102 columns
- [x] **Key data:** Team-level per-week totals — offensive/defensive EPA, yards, TDs, turnovers, all split by pass/rush/receive
- [x] Create `src/nfl/data/fetch_team_stats.py`
- [x] Fetch seasons 2018-2025, save to `data/nfl/team_stats/team_stats_{season}.parquet`
- [x] Verify: top offenses should have positive total EPA, bottom defenses should allow high yards
- [x] **Deliverable:** Team offensive and defensive quality metrics for all seasons
- [x] **Note:** Replaces our manual opponent defense rank calculation with direct data

### Task 0.12 — Depth Charts
- [x] **Source:** `nfl.load_depth_charts(seasons)` — 15 columns
- [x] **Key data:** `depth_team` (1=starter, 2=backup, 3=third string), `depth_position`, `formation` (Offense/Defense/Special Teams), weekly per player
- [x] Create `src/nfl/data/fetch_depth_charts.py`
- [x] Fetch seasons 2018-2024 (2025 schema incompatible), save to `data/nfl/depth_charts/depth_charts_{season}.parquet`
- [x] Verify: known starters should have `depth_team=1`, known backups `depth_team=2`
- [x] **Deliverable:** Weekly starter/backup status for all players, 2018-2024

### Task 0.13 — Unified data pipeline update
- [x] Update `NFLDataPipeline` in `src/nfl/data/pipeline.py` to orchestrate ALL fetch scripts
- [x] Add `fetch_all()` method that runs all fetchers for a given season range
- [x] Add `fetch_latest()` method that only fetches the most recent week across all datasets
- [x] Each fetcher skips if data already exists (same pattern as current raw stats)
- [x] Verify: running `fetch_all()` pulls all 12 datasets, skips already-downloaded data
- [x] **Deliverable:** Single entry point to fetch all data: `python src/nfl/data/pipeline.py`

### Task 0.14 — Player Stats Fetcher (per-season reorganization)
- [x] Create `src/nfl/data/fetch_player_stats.py` — PlayerStatsFetcher class
- [x] Re-fetch player stats as per-season files (8 files vs 144 per-week files)
- [x] Includes playoff weeks 19-22 (previously missing from raw/)
- [x] Data matching tests: regular season rows match old raw files exactly, columns identical
- [x] Register in pipeline fetch_all() and fetch_latest()
- [x] **Deliverable:** `data/nfl/player_stats/` — 8 per-season files, 147,223 records
- [x] **Note:** `data/nfl/raw/` kept as backup until PostgreSQL confirms data integrity

---

## Pre-Phase 1: Data Inventory & ID Mapping

### Findings: Player ID Formats
Two ID formats exist across our datasets:
- **GSIS** (`00-0033873`): Used by player_stats, injuries, NGS, ff_opportunity, depth_charts (6 datasets)
- **PFR** (`MahoPa00`): Used by snap_counts, pfr_advstats (4 datasets)

`nfl.load_players()` provides the mapping table (24,376 all-time players back to 1974). For our purposes, only ~3,300 players are relevant (last_season >= 2024). 22,214 have both GSIS + PFR IDs.

### Task 1.0 — Data catalog and player ID mapping table
- [x] Create `docs/DATA_CATALOG.md` — complete reference of all datasets, columns, join keys, quirks
- [x] Fetch `nfl.load_players()` and store as `data/nfl/players/players.parquet`
- [x] Build GSIS↔PFR ID mapping filtered to relevant players (last_season >= 2018) — 6,543 players
- [x] Verify mapping covers all players in snap_counts (99.7%) and pfr_advstats (100%)
- [x] Add PlayersFetcher to pipeline with `get_id_mapping()` convenience method
- [x] **Deliverable:** Data catalog doc + player ID lookup table for cross-dataset joins

---

## Phase 1: PostgreSQL Foundation
**Goal:** Get PostgreSQL running, schema designed around ALL datasets (see `docs/DATA_CATALOG.md`), data loaded.

### Task 1.1 — Database setup and schema
- [x] Install/configure PostgreSQL locally
- [x] Create `nfl_predictions` database
- [x] Design schema using DATA_CATALOG.md as the source of truth:
  - **Reference tables** (loaded first):
    - `teams` — 32 NFL teams with abbreviations, names, conference, division
    - `players` — player reference with both GSIS + PFR IDs (from `load_players()`)
  - **Game-level tables:**
    - `games` — from schedules: scores, Vegas lines, weather, rest days, coaches, referee (48 cols)
  - **Player-week tables** (join via player GSIS ID + season + week):
    - `weekly_stats` — from player_stats: box scores, fantasy points (114 cols)
    - `injuries` — from injuries: game-day status, practice status (16 cols)
    - `depth_charts` — from depth_charts: starter/backup status (15 cols, 2018-2024 only)
  - **Player-week tables** (join via player PFR ID + season + week, mapped through players table):
    - `snap_counts` — from snap_counts: offense/defense/ST snap pct (16 cols)
    - `pfr_pass_advstats` — from pfr_pass: pressure, drops, bad throws (24 cols)
    - `pfr_rush_advstats` — from pfr_rush: yards before/after contact (16 cols)
    - `pfr_rec_advstats` — from pfr_rec: drop rate, passer rating when targeted (17 cols)
  - **Player-week tables** (join via player GSIS ID + season + week, qualified players only):
    - `ngs_passing` — from ngs_passing: time to throw, CPOE, aggressiveness (29 cols)
    - `ngs_rushing` — from ngs_rushing: rush yards over expected, efficiency (22 cols)
    - `ngs_receiving` — from ngs_receiving: separation, YAC above expected (23 cols)
    - `ff_opportunity` — from ff_opportunity: expected fantasy points (159 cols)
  - **Team-week tables:**
    - `team_stats` — from team_stats: team-level EPA, yards, turnovers (102 cols)
- [x] Add indexes for common query patterns:
  - Player lookup: `(player_gsis_id, season, week)`, `(player_pfr_id, season, week)`
  - Team lookup: `(team, season, week)`
  - Game lookup: `(season, week, home_team)`
- [x] Write schema migration script: `src/nfl/db/schema.sql`
- [x] **Deliverable:** Empty database with all tables + indexes created

### Task 1.2 — Database connection layer
- [x] Create `src/nfl/db/connection.py` — get_connection() and get_engine()
- [x] Create `src/nfl/db/__init__.py` with convenience imports
- [x] Add `psycopg2-binary`, `sqlalchemy`, `python-dotenv` to `requirements.txt`
- [x] Add `.env.example` with database connection template
- [x] **Deliverable:** `get_engine()` and `get_connection()` functions that work

### Task 1.3 — Bulk load all Parquet data into PostgreSQL
- [x] Write `src/nfl/db/load_all.py` — reads every Parquet directory and loads into corresponding table
- [x] Load order: players → games → all player-week tables → team_stats → depth_charts
- [x] Filter out `player_id IS NULL` rows when loading weekly_stats (173 garbage rows filtered)
- [x] Handle duplicates gracefully (truncate + reload, idempotent)
- [x] Print summary: row counts per table, any skipped/failed records
- [x] Verify total row counts match Parquet files + third-party verified against NFL.com box scores
- [x] **Deliverable:** 798,176 rows across 14 tables, all queryable in PostgreSQL

### Task 1.4 — Database refresh after fetch
- [x] Add `refresh_db()` method to pipeline that calls `load_all()` after `fetch_all()` or `fetch_latest()`
- [x] Update `__main__` to support `--refresh-db` flag: fetch new data then reload DB
- [x] Test: run with `--latest --refresh-db` and verify new data appears in both Parquet and DB
- [x] **Deliverable:** `python src/nfl/data/pipeline.py --latest --refresh-db` keeps both in sync
- [x] **Note:** Simpler than true dual-write (modifying 10 fetcher classes). Full reload takes ~10 min but is idempotent and reliable.

### Task 1.5 — Legacy data cleanup
- [x] Delete `data/nfl/raw/` (144 per-week files, replaced by `data/nfl/player_stats/`)
- [x] Delete `data/nfl/cleaned/` (empty directory)
- [x] Delete `data/nfl/vegas_odds/` (3 files from paid Odds API, replaced by schedule data)
- [x] Update `.gitignore` if needed
- [x] Verify tests still pass after removal
- [x] **Deliverable:** Clean data directory with only current per-season files

---

## Phase 2: Query Layer, Feature Engineering, and Dashboard
**Goal:** Build SQL query functions, V5 feature engineer using all new data, and rebuild the dashboard.

### Task 2.1 — Database query layer
- [x] Create `src/nfl/db/queries.py` — reusable query functions:
  - `get_player_history(player_id, season, week, games_back)` — single SQL query replaces loading N Parquet files
  - `get_week_stats(season, week, position)` — replaces `pd.read_parquet()`
  - `get_player_injuries(player_id, season, week)` — injury status for a player
  - `get_snap_share(player_id, season, week)` — snap count percentage (handles PFR→GSIS ID mapping)
  - `get_game_context(season, week, team)` — Vegas lines, weather, rest days
  - `get_opponent_defense_rank(team, position, season, week)` — uses team_stats table
  - `get_nextgen_stats(player_id, season, stat_type)` — NGS metrics
- [x] **Deliverable:** All common data access patterns available as SQL queries (30 tests)

### Task 2.2 — Legacy code cleanup
- [x] Delete `src/nfl/odds/` — paid Odds API code, replaced by free schedule data in DB
- [x] Move `src/nfl/features/`, `models/`, `training/` to `legacy/v1-v4/` — preserves ML progression for portfolio
- [x] Tag `v4-final` — complete V4 codebase recoverable via `git checkout v4-final`
- [x] Remove broken tests (5 files) and ad-hoc scripts (`tests/scripts/`, 11 files)
- [x] Update smoke tests to reference active modules only
- [x] Create `legacy/README.md` with MAE progression and V5 takeaways
- [x] **Deliverable:** Clean `src/nfl/` (data + db only), legacy preserved in `legacy/v1-v4/`, 227 tests passing

### Task 2.3 — Load predictions and model runs into DB
- [x] Create `predictions` and `model_versions` tables in schema.sql
- [x] Write ETL (`src/nfl/db/load_predictions.py`) to load all `data/nfl/predictions/` (4 versions, 65,921 rows)
- [x] Store model metadata (version, description, MAE, prediction weeks, positions)
- [x] Backfill `actual_value` and `error` columns by joining predictions with weekly_stats (51,112 rows matched)
- [x] **Deliverable:** Cross-version accuracy queries work in SQL (V4 4.67 → V1 4.97 MAE, 17 tests)


---

## Phase 3: V5 Model — Feature Engineering, Training, and Validation
**Goal:** Use all 13 data sources to build V5 model. Target MAE < 4.0. See `docs/V5_QUESTIONS.md` for full architectural decisions.

**Key V5 decisions:**
- Predict individual stats per position (not just fantasy_points_ppr) — dashboard shows per-stat over/under
- 2 model types: StatPredictor (raw prediction) + POB (over/under probability). EVOB dropped.
- 4-algorithm ensemble: XGBoost, LightGBM, CatBoost, RandomForest
- Pre-game features only (no data leakage)
- **Scripts read from Parquet files (not PostgreSQL)** so they run on Google Colab
- Walk-forward validation, then ablation study to prove which features helped

### Google Colab Workflow for Heavy Compute

All compute-intensive tasks (feature engineering, training, ablation) run on Google Colab Pro to speed up development. Local machine is only used for code editing, PostgreSQL queries, and the dashboard.

**Drive structure** (`My Drive/SportsAnalyzer/`):
```
SportsAnalyzer/
├── data/nfl/         ← 13 dataset folders (~30MB, uploaded once)
│   ├── player_stats/, schedules/, injuries/, snap_counts/
│   ├── nextgen_stats/, ff_opportunity/, pfr_advstats/
│   ├── team_stats/, depth_charts/, players/
├── scripts/          ← Python scripts (uploaded per handoff)
│   ├── v5_engineer.py
│   ├── v5_train.py
│   └── v5_ablation.py
└── output/           ← Script outputs (downloaded after run)
    ├── features/     ← Per-season feature Parquet files
    ├── models/       ← Trained .joblib files
    └── predictions/  ← Final V5 predictions
```

**Handoff workflow at each compute-heavy step:**
1. Claude writes the script locally (reads from Parquet paths, not PostgreSQL)
2. User uploads the script to Drive `scripts/` folder
3. User opens Colab notebook in VS Code, connects to high-RAM runtime
4. User runs notebook cells: mount Drive → `%run scripts/<script>.py`
5. Script writes output to Drive `output/` folder
6. User downloads output to local machine (or just stays in Drive)
7. User confirms "done" in chat — Claude resumes with next step

**One-time setup (before first handoff):**
- [x] Upload `data/nfl/` folder to Drive (excluding `features/`, `models/`, `predictions/` — we're regenerating those)
- [x] Create `scripts/` and `output/` folders in Drive
- [x] Test Colab connection from VS Code, verify Drive mount works (`colab/colab_test.ipynb` passed)

### Task 3.0 — Colab notebooks (created alongside each Phase 3 task)

Notebooks are created per handoff (see each task below). Notebooks are **gitignored** (`colab/*.ipynb`) — they're throwaway runners that live on Drive. The `.py` scripts in `src/` contain all logic; notebooks are thin wrappers that mount Drive and invoke the script.

**Every notebook MUST include these cells in order:**
1. Mount Google Drive
2. Set paths (DRIVE_ROOT, DATA_DIR, OUTPUT_DIR, CODE_ROOT)
3. Verify code uploaded + create `__init__.py` files if missing
4. **High-RAM / CPU sanity check** — warn if `psutil.virtual_memory().total < 20GB`. This prevents wasted compute on free-tier runtimes.
5. Run the script (via `%run` or direct import)
6. Verify outputs (file sizes, row counts)
7. Spot-check known values (Mahomes W5 rolling avg, etc.)

- [x] `colab/colab_test.ipynb` — verify Drive mount + data access + ML libraries
- [x] `colab/v5_feature_engineering.ipynb` — runs feature engineering (paired with Task 3.1)
- [ ] `colab/v5_training.ipynb` — runs training script (paired with Task 3.2)
- [ ] `colab/v5_ablation.ipynb` — runs ablation study script (paired with Task 3.2b)
- [ ] `colab/v5_final_retrain.ipynb` — runs final retrain + prediction generation (paired with Task 3.2c)

### Task 3.1 — V5 feature engineering (**Heavy** — largest task in Phase 3) ✅ COMPLETE
- [x] Created `src/nfl/features/v5/` package (7 modules instead of monolith): config, master_table, rolling, context, usage, advanced, engineer
- [x] Master player-week table with 13 LEFT JOINs (GSIS↔PFR mapping at join level, position filter drops 65% of non-skill-position rows, drop_duplicates guards)
- [x] Carried forward proven V4 features (rolling decay=0.85, variance, trends, Vegas, opponent rank)
- [x] Added new features: snap counts, injury severity, depth chart, weather (dome/wind/cold), NGS (passing/rushing/receiving), PFR advanced (pass/rush/rec), FF opportunity
- [x] Pre-game features only (strict shift(1) + week < N enforcement, verified via 16 real-data tests)
- [x] NULL preservation for unqualified players (no fake imputation)
- [x] `games_of_history` column enables downstream MIN_GAMES_HISTORY filtering
- [x] `FEATURE_GROUP_PREFIXES` + `get_feature_columns_by_group()` helper for Task 3.2b ablation
- [x] Colab notebook `colab/v5_feature_engineering.ipynb` with Drive mount + high-RAM check + spot-check cell
- [x] **Delivered:** 90 feature columns on 2024 data in 16s, 295 tests passing
- [x] **HANDOFF POINT #1 COMPLETE:** Colab run successful. 8 per-season Parquets generated (52,207 total rows, 14.4 MB) and downloaded to `data/nfl/features/v5/`. Production validation: 90 feature columns, Mahomes W5 rolling_avg_passing_yards=240.4, all data leakage tests passed.

### Task 3.1b — Feature validation (**Quick** — spot-check pass) ✅ COMPLETE
- [x] No data leakage verified (W1 of first season has NaN opp_def_rank; Mahomes W1 rolling_avg ≠ current stat)
- [x] NGS/PFR nulls preserved as expected (90%+ non-QBs have NaN NGS passing)
- [x] Spot-checked Mahomes (240.35 rolling passing yds, 24 games_of_history), Barkley, Kelce
- [x] 90 feature columns total (rolling 43, context 20, usage 4, advanced 23) — exceeds 60-80 plan target
- [x] **Deliverable:** Real-data integration tests in `tests/test_v5_real_data.py` (16 tests)

### Task 3.2 — V5 model training (**Heavy** — compute-intensive, ~20-30 min per full run)
- [ ] Create `src/nfl/training/v5_train.py`
- [ ] Train per-position models: StatPredictor + POB for each stat
  - QB: passing_yards, passing_tds, passing_interceptions, rushing_yards, rushing_tds (5 stats × 2 types = 10 models)
  - RB: rushing_yards, rushing_tds, receptions, receiving_yards, receiving_tds (5 × 2 = 10)
  - WR: receptions, receiving_yards, receiving_tds, targets (4 × 2 = 8)
  - TE: receptions, receiving_yards, receiving_tds, targets (4 × 2 = 8)
  - K: fg_made, fg_att, pat_made (3 × 2 = 6)
  - **Total: ~42 models** (similar to V4's 40)
- [ ] Position-specific hyperparameters (from V4): QB depth=9, RB/WR depth=7, TE depth=6, K depth=3
- [ ] Expanding-window walk-forward validation: train on prior data, predict each week for 2021-2024
- [ ] Compute per-stat and per-position MAE, compare against V4
- [ ] Generate feature importance rankings per position
- [ ] Create `colab/v5_training.ipynb` with required cells: mount Drive, paths, **high-RAM/CPU check (`psutil.virtual_memory().total >= 20GB` assertion)**, install catboost/xgboost/lightgbm if needed, `%run v5_train.py`, verify output models, print MAE summary
- [ ] **Deliverable:** Trained V5 models in `data/nfl/models/v5/`, MAE results documented
- **>>> HANDOFF POINT #2:** User runs `colab/v5_training.ipynb` on Colab Pro (high-RAM CPU). Estimated 1-4 hours. Claude resumes to analyze results.

### Task 3.2b — Ablation study (**Heavy** — retrains model ~8-10 times)
- [ ] Remove one feature group at a time, retrain, measure MAE change:
  - Remove snap count features → measure impact
  - Remove injury features → measure impact
  - Remove NGS features → measure impact
  - Remove PFR features → measure impact
  - Remove FF opportunity features → measure impact
  - Remove weather features → measure impact
  - Remove depth chart features → measure impact
- [ ] Any feature group that improves MAE by < 0.05 when included → drop it
- [ ] Document results: "snap counts improved RB MAE by X, NGS had no measurable impact"
- [ ] Create `colab/v5_ablation.ipynb` with required cells: mount Drive, paths, **high-RAM/CPU check**, install ML libs, `%run v5_ablation.py`, summarize MAE deltas per feature group
- [ ] **Deliverable:** Validated feature set — only features that proved their value remain
- **>>> HANDOFF POINT #3:** User runs `colab/v5_ablation.ipynb` (retrains ~8-10 times, estimated 8-30 hours total — consider Colab Pro+ for background execution). Claude resumes to analyze results.

### Task 3.2c — Final V5 retrain (**Medium** — one training run with finalized features)
- [ ] Retrain V5 with ablation-validated feature set (noise features removed)
- [ ] Train production model on all 2020-2025 data
- [ ] Generate predictions for 2025 season (for comparison with V1-V4)
- [ ] Load V5 predictions into `predictions` table (reuse `load_predictions.py`)
- [ ] Compare V5 vs V4 MAE in database: `SELECT version, AVG(error) ... GROUP BY version`
- [ ] Create `colab/v5_final_retrain.ipynb` with required cells: mount Drive, paths, **high-RAM/CPU check**, install ML libs, `%run v5_final_retrain.py`, save models + predictions to Drive
- [ ] **Deliverable:** Final V5 model, predictions loaded, cross-version accuracy verified
- [ ] **Target MAE:** < 4.0 overall (TE < 3.5, RB < 4.5, WR < 4.5, QB < 6.5)
- **>>> HANDOFF POINT #4:** User runs `colab/v5_final_retrain.ipynb` (estimated 1-2 hours). Claude resumes to load results into DB and verify.

### Task 3.3 — Prediction accuracy dashboard (**Medium** — visualization work)
- [ ] New Streamlit page or tab: model accuracy comparison
- [ ] Powered by SQL queries on predictions table (already has actuals + errors)
- [ ] Charts:
  - MAE by version (bar chart: V1 → V5)
  - MAE by position by version (grouped bar chart)
  - Predicted vs actual scatter plot (by position, filterable by version)
  - Per-week MAE trend line (does accuracy degrade later in season?)
  - Feature importance bar chart (from ablation results)
- [ ] **Deliverable:** Interactive accuracy dashboard comparing V1-V5

---

## Phase 4: Dashboard & Production Hardening (Future)
**Goal:** Rebuild the dashboard with all data sources and make the project production-grade.

### Task 4.1 — Dashboard rebuild (new app.py)
- [ ] Rebuild `app.py` from scratch — old version uses legacy per-week raw files and is broken
- [ ] Use old `app.py` as a reference for layout and features, not as a starting point
- [ ] Read data from PostgreSQL via query layer (Task 2.1)
- [ ] Tabs: Player Explorer, Performance Trends, Predictions, Model Accuracy
- [ ] Use all new datasets: show injuries, snap counts, NGS metrics, expected fantasy points
- [ ] Streamlit filters translate to SQL WHERE clauses
- [ ] **Deliverable:** Fully rebuilt dashboard backed by PostgreSQL and new data pipeline

### Task 4.2 — FastAPI REST endpoints
- [ ] API layer on top of PostgreSQL (player stats, predictions, model accuracy)
- [ ] Proper request/response schemas with Pydantic

### Task 4.3 — Automated weekly pipeline
- [ ] Cron job or scheduled task: fetch new data → features → predictions → DB
- [ ] End-to-end automation

### Task 4.4 — Docker containerization
- [ ] Dockerfile for app + PostgreSQL via docker-compose
- [ ] One-command setup for anyone cloning the repo

---

## Working Agreement

- **One task at a time.** Complete, test, and commit before starting the next.
- **Follow the phase files.** Each task goes through: plan → build → review → test → document.
- **Dual-write always.** Every fetch writes to both Parquet and PostgreSQL (after Phase 1).
- **Parquet stays as fallback.** Don't delete Parquet files — they're the rebuild source.
- **No scope creep within tasks.** If something feels like it belongs in a later task, defer it.
- **Commit after each task.** Small, descriptive commits. No mega-commits.
- **No Co-Authored-By lines** in commit messages.
