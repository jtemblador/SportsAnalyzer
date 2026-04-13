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
- [x] `colab/v5_training.ipynb` — runs training script (paired with Task 3.2) — READY, pending user HANDOFF #2 execution
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
- ⚠️ **2026-04-13 — STALE:** Player feature parquets from this run were deleted after Task 3.1.5 review uncovered a latent silent-corruption hazard in the rolling computation (see Task 3.1.5 → "Code-quality refactor"). Math is unchanged but the implementation pattern was hardened. Re-run Handoff #1 jointly with Task 3.1.5 — see HANDOFF POINT #1.5 below.

### Task 3.1b — Feature validation (**Quick** — spot-check pass) ✅ COMPLETE
- [x] No data leakage verified (W1 of first season has NaN opp_def_rank; Mahomes W1 rolling_avg ≠ current stat)
- [x] NGS/PFR nulls preserved as expected (90%+ non-QBs have NaN NGS passing)
- [x] Spot-checked Mahomes (240.35 rolling passing yds, 24 games_of_history), Barkley, Kelce
- [x] 90 feature columns total (rolling 43, context 20, usage 4, advanced 23) — exceeds 60-80 plan target
- [x] **Deliverable:** Real-data integration tests in `tests/test_v5_real_data.py` (16 tests)

### Task 3.1.5 — DST (Defense/Special Teams) feature engineering (**Medium** — parallel pipeline)

**Goal:** Add the 6th fantasy position (DST) as a parallel team-week pipeline. Adds DEF/DST predictions to V5 alongside the existing 5 player positions. Uses standard ESPN/Yahoo/NFL.com scoring formula.

**Stats to predict per team-week:**
- `sacks` (from `team_stats.def_sacks`)
- `interceptions` (from `team_stats.def_interceptions`)
- `fumble_recoveries` (from `team_stats.fumble_recovery_opp`) ⚠ NOT `def_fumbles` — researched: that column counts defenders' own fumbles (e.g., on INT returns), not recoveries. League sums confirm: def_fumbles=76, fumble_recovery_opp=283 (matches NFL ~280 defensive recoveries/season). Regression test guards this.
- `defensive_tds` (from `team_stats.def_tds`)
- `safeties` (from `team_stats.def_safeties`)
- `points_allowed` (computed: opponent's score from `schedules`)

**DST scoring formula** (industry standard — ESPN/Yahoo/NFL.com/Sleeper):
```
fantasy_points_dst =
    sacks * 1
    + interceptions * 2
    + fumble_recoveries * 2
    + defensive_tds * 6
    + safeties * 2
    + blocked_kicks * 2
    + return_tds * 6
    + points_allowed_bonus(points_allowed)

where points_allowed_bonus:
    0 PA      → +10
    1-6 PA    → +7
    7-13 PA   → +4
    14-20 PA  → +1
    21-27 PA  → 0
    28-34 PA  → -1
    35+ PA    → -4
```

**Implementation plan:**
- [x] Create `src/nfl/features/v5/dst.py` — `build_dst_features(data_dir, seasons)` produces team-week DataFrame
- [x] Master DST table: one row per (team, season, week, season_type). REG + POST rows kept; POST feeds rolling history but is filtered before write.
- [x] DST features (56 columns total on 2-season smoke):
  - **Rolling defensive stats** — rolling_avg / variance / trend per stat in `CORE_DST_STATS_FOR_ROLLING` (8 stats × 3 = 24 cols)
  - **`opp_rolling_avg_off_yards|tds|turnovers`** — opponent offense quality (3 cols)
  - **Vegas/weather context** — `game_script_index`, `is_dome`, `is_high_wind`, `is_cold`, plus passthrough `spread_line`/`total_line`/`team_implied_total`/`opponent_implied_total`/`team_rest`/`opponent_rest`/`temp`/`wind`/`roof`/`div_game`
  - **`blocked_kicks`** — derived via self-join: opponent's `(fg_blocked + pat_blocked)` for the same (season, week)
- [x] Added `FANTASY_DST_WEIGHTS`, `STATS_TO_PREDICT['DST']`, `CORE_DST_STATS_FOR_ROLLING` to config.py; `points_allowed_bonus()` and `compute_dst_fantasy_points()` in dst.py
- [x] Wired `build_dst_features` into `build_features` in engineer.py — when `output_dir` is set, both `features_{season}.parquet` and `features_dst_{season}.parquet` are written. (Docstring documents the side-effect contract; in-memory callers should call `build_dst_features` directly.)
- [x] Tests in `tests/test_v5_dst.py` — 17 tests, including scoring formula, points_allowed_bonus boundaries, fumble_recoveries source regression guard, blocked_kicks self-join, no-leakage with pinned decay-weighted value, POST-rows-in-rolling-but-not-output, missing-column path with pinned arithmetic value, expected column count.
- **Code-quality refactor (2026-04-13, after 3 review passes):**
  - Extracted `decay_weighted_avg` and rolling helpers (`rolling_decay_avg_series`, `rolling_variance_series`, `rolling_trend_series`) to `src/nfl/features/v5/utils.py` — single source of truth for both player and DST pipelines.
  - Refactored both `rolling.py` and `dst.py` to use `groupby(...).transform(...)` instead of the prior list-append + bulk-assign pattern. Math is byte-for-byte identical (all 68 v5 tests pass without value drift); transform makes the position-safety contract structural rather than implicit, eliminating a latent silent-corruption hazard if the input DataFrame ordering ever changed.
  - Fixed `add_dst_opponent_offense` brittleness — `DataFrame.get(col, 0).fillna(0)` would have crashed if any optional column ever went missing in a future nflverse release. Replaced with a defensive `_col` helper returning a zero-filled Series. Regression test now pins arithmetic values.
  - Strengthened rolling leakage test with explicit decay-weighted-average value pin (catches off-by-one in window slicing).
- [x] **Deliverable:** 8 per-season DST feature Parquets (~576 rows × 8 seasons ≈ 4,500 rows total) + 8 re-generated player feature Parquets ready for training.
- **Local re-run is fast (~3 min for both pipelines, 8 seasons).** Colab handoff is optional but recommended to keep the production-output path consistent with Task 3.1.
- **>>> HANDOFF POINT #1.5:** Re-run `colab/v5_feature_engineering.ipynb` (or local equivalent) — `build_features` now produces both `features_{season}.parquet` AND `features_dst_{season}.parquet` per season. Stale player parquets from Handoff #1 were deleted (rolling math is identical post-refactor, but re-running keeps the production pipeline in sync with the code). Estimated 30-60 min on Colab Pro for all 8 seasons. Output: 16 Parquets total (8 player + 8 DST) in `data/nfl/features/v5/`.

### Task 3.2 — V5 model training (**Heavy** — compute-intensive, Colab handoff)

**Goal:** Train and validate the V5 stat-prediction models. Two model types per (position, stat) — `StatPredictor` (regression on raw stat) + `POB` (binary classifier: P(stat > player's rolling baseline)). Each is a 4-algorithm ensemble (XGBoost, LightGBM, CatBoost, RandomForest). Total: 27 stat-keys × 2 types × 4 algos = **216 trained model files**, organized as 54 ensembles. Generates per-position/per-stat MAE on a walk-forward holdout (2021-2024) for V4 comparison.

**Models trained per (position × stat × type) — 27 stat-keys, 54 ensembles:**

| Position | Stats predicted | Stat-keys | × 2 types | Notes |
|----------|-----------------|-----------|-----------|-------|
| QB | passing_yards, passing_tds, passing_interceptions, rushing_yards, rushing_tds | 5 | 10 | depth=9 (V4-proven) |
| RB | rushing_yards, rushing_tds, receptions, receiving_yards, receiving_tds | 5 | 10 | depth=7 |
| WR | receptions, receiving_yards, receiving_tds, targets | 4 | 8 | depth=7 |
| TE | receptions, receiving_yards, receiving_tds, targets | 4 | 8 | depth=6 |
| K | fg_made, fg_att, pat_made | 3 | 6 | depth=3 (sparse outcome) |
| **DST** | sacks, interceptions, fumble_recoveries, defensive_tds, safeties, points_allowed | 6 | 12 | **depth=5, iter=200** (smaller dataset) |
| **Total** | | **27** | **54** | × 4 algos = 216 model files |

**Key architecture decisions (lock these in before build):**

1. **Train/eval/production splits.**
   - Warm-up data: 2018-2019 (rolling history only — never trained on directly; already filtered out by `games_of_history >= MIN_GAMES_HISTORY=3` in features)
   - Eval split: walk-forward on **2021-2024** (4 seasons of holdout). For each evaluation week W of season S: train on all rows with `(season, week) < (S, W)`, predict week W. Rolling expanding window, NOT a fixed 6-season chunk. Justification: matches how the model is actually used in production (predict next week given everything before).
   - **2020 is intentionally excluded from eval** — COVID season, weird game scripts, but kept in training data for everything afterward.
   - Production model (Task 3.2c): retrained on all 2020-2025 data after ablation.

2. **POB baseline definition.** POB target = `1 if actual_value > rolling_avg_<stat> else 0`. Baseline column already exists in features (e.g., `rolling_avg_passing_yards`). Rows where `rolling_avg_<stat>` is NaN (insufficient history) are dropped from POB training AND POB evaluation — model only learns from players with history. POB output column in predictions table: `probability_over` (FLOAT 0-1).

3. **Insufficient-history filter.** Apply `games_of_history >= MIN_GAMES_HISTORY=3` (config.py constant). Drop rows below threshold from BOTH training and evaluation. Side effect: rookies' first 3 games never appear in metrics. Acceptable for portfolio (matches how betting models work — wait for sample size).

4. **Model file naming.** Convention: `data/nfl/models/v5/{POS}_{stat}_{type}_{algo}.joblib`
   - Example: `QB_passing_yards_stat_xgboost.joblib`, `QB_passing_yards_pob_catboost.joblib`
   - Plus per-ensemble metadata: `{POS}_{stat}_{type}_meta.json` (feature columns, training rows, MAE, weights for ensemble averaging if non-uniform)
   - DST naming: `DST_sacks_stat_xgboost.joblib` etc. (uses 'DST' as the position token).

5. **Ensemble averaging.** Simple mean across 4 algorithms for V5 baseline (V4-proven). Feature engineering already gives each algo NULL-safe inputs (CatBoost, XGBoost ≥1.5, LightGBM all native; RandomForest gets median-imputed via sklearn pipeline at training time).

6. **DST hyperparameters reduced from V4 player defaults.** With only ~4,254 rows (vs player ~52K), default depth=7 risks overfitting. Use depth=5, iter=200 for DST. Validate with cross-position MAE comparison after first run — if DST overfits (train MAE << eval MAE), reduce further.

7. **Predictions table integration (carries forward to 3.2c).** DST predictions reuse the existing `predictions` table schema with `player_id = team_abbr` (e.g., 'KC'), `player_name = team full name`, `position = 'DST'`, `team = team_abbr`, `opponent = opponent_team`. Permissive schema accommodates both per-player and per-team rows. No new table.

**Implementation plan:**
- [x] Created `src/nfl/training/v5/` package (6 files, ~1,600 lines): config.py (POSITION_ALGORITHMS per-position subsets + POSITION_HYPERPARAMS + COUNT_STATS for Poisson), data.py (whitelist feature selection + fill_features with NEUTRAL_FILLS for dome temp), models.py (StatPredictor + POBModel with atomic save + meta JSON recording neutral_fills policy), walkforward.py (strict-prior mask + eval_df alignment), train.py (orchestrator with .tmp sweep + schema-drift CSV rotation), __init__.py.
- [x] Tests in `tests/test_v5_training.py` (27 synthetic + 3 real-data, 30 total): hyperparams sanity, apply_history_filter, compute_pob_target, walk-forward strict-prior invariant (monkey-patched prepare verifies actual invariant), ensemble predict returns mean, TE/receptions real-data smoke + DST Poisson non-negative + WR POB balance. Plus load-bearing regressions: whitelist blocks known leakage columns, fill_features temp=65 consistency across train+predict, meta JSON records neutral_fills, degenerate_pob flag.
- [x] Created `colab/v5_training.ipynb` (8 cells): mount Drive, verify paths + DST parquet preflight, high-RAM check, install ML libs, resume check, feature shape inspection, training loop with monkey-patched path helpers, ensemble count verification, MAE summary with degenerate_pob warning.
- [x] **Deliverable planned:** 174 .joblib files (not 216 — per-position algo subsets) + 54 metadata JSON in `output/models/v5/`, `_mae_summary.csv` with columns: version, position, stat, model_type, algorithms, n_train_rows, n_features, n_eval_predictions, status, trained_at, mae_v5 (or accuracy/auc/pos_class_frac/degenerate_pob for POB).
- [x] **Estimated runtime:** 1-3 hours on Colab Pro high-RAM CPU. Local baseline: DST/sacks/stat = 125s end-to-end. Naive 54x extrapolation = 112 min; realistic accounting for player-position row counts = 1-3 hours.
- **>>> HANDOFF POINT #2:** User runs `colab/v5_training.ipynb` on Colab Pro. Claude resumes to analyze MAE results before triggering Task 3.2b. **READY** — see `docs/progress/2026-04-13_task_3.2_v5_training.txt` HANDOFF section.

**Code-quality enhancements beyond original plan (from 4 review rounds, 23 fixes):**
- [x] Whitelist-based feature selection (prefix-match only, prevents current-week leakage via ngs_*, pfr_*, target_share, *_exp that the V5 parquet contains alongside rolling equivalents).
- [x] Poisson loss for count stats (xgboost count:poisson, lightgbm/catboost native Poisson); RF falls back to MSE.
- [x] NEUTRAL_FILLS for dome games (temp=65.0; 36% of rows are domes where filling with 0 would train "0°F weather").
- [x] Walk-forward eval_df dropna alignment + length assertion (prevents index misalignment bug).
- [x] Atomic .joblib writes + .tmp orphan sweep + schema-drift CSV rotation (survives Colab disconnects).
- [x] Meta JSON embeds neutral_fills + feature_columns + objective_per_algo so Task 3.2c detects policy drift.
- [x] degenerate_pob flag (pos_class_frac < 0.05 or > 0.95) surfaces misleading-accuracy folds.

**Risks / pre-flight checks:**
- DST model count is small (~4,254 rows for training; walk-forward may leave only ~500-1000 rows for early-eval seasons). Possible solution: pool DST with a longer history window OR reduce eval to 2023-2024 only. Decide after first MAE pass.
- POB labels are imbalanced if rolling_avg is biased — verify `compute_pob_target` produces ~50/50 split per (position, stat). If skewed (e.g., 70/30), POB models will underperform. Stratified sampling may be needed.
- V4 MAE baseline numbers (5.14/4.66/4.66/4.26) come from a different feature set + train range. For honest comparison, also retrain V4 on the V5 train/eval split — OR accept that V5-vs-V4 is approximate. Recommendation: report V5 numbers standalone with V4 numbers cited as reference (not apples-to-apples).
- Colab disconnects mid-training: notebook must be resumable per-ensemble (skip if .joblib + meta JSON exist). Built into Step 5.
- Feature column drift: training must read feature columns from each Parquet's actual schema, not a hardcoded list. Use `get_feature_columns_by_group(df.columns, group)` for player; for DST, build a similar helper or compile the DST feature list dynamically.

### Task 3.2b — Ablation study (**Heavy** — retrains StatPredictor model N times)

**Goal:** Validate which feature groups carry signal. Drop noise features so the production model (Task 3.2c) is leaner and faster. Produces a portfolio talking point ("I systematically validated which data sources contributed signal — snap counts improved RB MAE by X, NGS contributed Y, FF opportunity Z").

**Key architecture decisions:**

1. **Ablate StatPredictor only, not POB.** POB is a derived classifier; ablation results would be noise on top of noise. StatPredictor is the headline metric. Halves the compute budget.

2. **Ablate by feature group, not individual columns.** Use `FEATURE_GROUP_PREFIXES` from `config.py` (already built for this purpose). Groups for player pipeline: `rolling`, `context`, `usage`, `advanced`. Each group dropped independently per run.

3. **DST feature groups are different from player.** Player pipeline ablates {rolling, context, usage, advanced}; DST ablates {rolling_def, opp_offense, context_dst}. Keep the two ablation sweeps separate so rules-of-thumb stay position-coherent.

4. **Threshold for dropping.** Ablation removes the group → MAE goes UP if the group helped. If removing group G makes MAE worse by < 0.05 (overall) → group G provided no measurable signal → drop from production. Threshold of 0.05 is V4-era convention; revisit if early ablation runs show stat-level deltas dominate the average.

5. **Fixed eval window.** Use the same walk-forward 2021-2024 split as Task 3.2 — never re-tune the split mid-ablation (would invalidate comparisons).

6. **Don't ablate per-position-per-stat — ablate at position level.** Aggregate MAE per position is the decision unit. Per-stat results are reported but not gated on. Avoids chasing noise on rare-event stats (rushing_tds, fg_made).

**Implementation plan:**
- [ ] Create `src/nfl/training/v5/ablation.py`:
  - `ABLATION_GROUPS = {'player': ['rolling', 'context', 'usage', 'advanced'], 'dst': ['rolling_def', 'opp_offense', 'context_dst']}`
  - `run_ablation(group_to_remove, positions, eval_seasons)` — calls walk-forward training with that group's columns dropped from feature matrix; returns per-stat MAE
  - `compare_to_baseline(baseline_mae, ablation_results)` — computes delta per position/stat
  - `apply_drop_threshold(deltas, threshold=0.05)` — returns groups to keep + groups to drop
- [ ] Add DST FEATURE_GROUP_PREFIXES to config.py if not yet (`'rolling_def'`, `'opp_offense'`, `'context_dst'`) and `get_dst_feature_columns_by_group()` helper.
- [ ] Tests in `tests/test_v5_ablation.py`:
  - Group dropping actually removes the right columns (synthetic feature df + assert column lists)
  - Threshold logic correctly identifies drop candidates
  - Smoke: run ablation on TE/receptions removing 'usage' group, completes in <30s, returns numeric MAE
- [ ] Create `colab/v5_ablation.ipynb`:
  - Mount Drive, paths, **high-RAM/CPU check**, install ML libs
  - Step: load baseline MAE summary (from Task 3.2 output)
  - Resumable per (group_removed, position) — check for existing `_ablation_{position}_remove_{group}.csv` artifacts
  - Run loop: 4 player groups × 5 player positions + 3 DST groups × 1 DST position = **23 ablation runs**. Each is one full walk-forward training run for 1 position (not all 6) — much cheaper than 3.2's 54-ensemble run.
  - Estimated runtime per ablation run: ~10-30 min for player position; ~1 min for DST. Total: 4-12 hours on Colab Pro. Pro+ recommended for background execution.
  - Step: aggregate results into `_ablation_summary.csv` (columns: position, group_removed, mae_with_group, mae_without_group, delta, drop_decision)
- [ ] **Deliverable:** `_ablation_summary.csv` + decision document `docs/progress/{date}_ablation_results.md` listing groups kept/dropped per position with rationale + final feature list for Task 3.2c.
- **>>> HANDOFF POINT #3:** User runs `colab/v5_ablation.ipynb`. Estimated 4-12 hours on Colab Pro (consider Pro+ for background). Claude resumes to analyze deltas and write the decision document.

**Risks / pre-flight checks:**
- 0.05 threshold is per-position-aggregate; some stats may have larger deltas (e.g., NGS may help passing_yards but not rushing). Sub-group keep/drop is acceptable if results clearly show it. Document case-by-case.
- If a feature group's contribution is mixed across stats, the safer call is to keep it (small bloat) — only drop on clear "no signal" verdicts.
- DST has small dataset → ablation MAE deltas may be noisy. Consider reporting DST ablation as exploratory only; default to keeping all DST groups unless one is clearly hurting.
- Task 3.2 must finish first to provide baseline MAE. Don't start until 3.2 results are in DB-ready form.
- Heaviest task in Phase 3 by wall-clock. Plan accordingly — user may want to run overnight.

### Task 3.2c — Final V5 retrain + DB load (**Medium** — production model + predictions integration)

**Goal:** Retrain V5 with ablation-validated feature set on all 2020-2025 data. Generate 2025 weekly predictions for V1-V5 cross-version comparison in the database. Load both player and DST predictions into the existing `predictions` table.

**Key architecture decisions:**

1. **Production model = full data, no eval split.** Eval was Task 3.2's job. Production uses every available training row (2020-2025). Justification: walk-forward eval validated the model class; production retrain just maximizes data for next season's predictions.

2. **2025 predictions for portfolio comparison.** Generate per-week predictions for the 2025 season (already played, actuals available). Load into `predictions` table alongside V1-V4 → enables V1→V5 MAE bar chart in the dashboard. Production predictions for 2026 season come AFTER 2026 W1 raw data lands (separate ongoing pipeline, not part of this task).

3. **DST predictions reuse `predictions` schema.** No new table. DST rows stored with `player_id = team_abbr` (e.g., 'KC'), `player_name = team full name` (lookup via team_stats or hardcoded map), `position = 'DST'`, `team = team_abbr`, `opponent = opponent_team`. Stat names: `sacks`, `interceptions`, etc. (same as STATS_TO_PREDICT['DST']). Plus a synthetic `fantasy_points_dst` row per team-week so dashboard can compare DST fantasy scoring across versions.

4. **Backfill `actual_value` for 2025 predictions.** Reuse `load_predictions.py` pattern: after inserting predictions, UPDATE `actual_value` from `weekly_stats` (player) or compute from `team_stats` (DST). Compute `error = predicted_value - actual_value` for accuracy queries.

5. **Model versioning in DB.** Insert into `model_versions` table: `version='v5'`, `description='V5 with ablation-validated features + DST'`, `n_features=<count>`, `n_models=54`, `train_seasons='2020-2025'`. Required by FK from predictions table.

6. **MAE verification step.** Before declaring task done, run `SELECT version, position, AVG(ABS(error)) FROM predictions WHERE actual_value IS NOT NULL AND season=2025 GROUP BY version, position` and assert V5 numbers are within target bands. Fail loudly if a position regressed badly vs V4 (>0.5 MAE worse) — likely a bug, not a model failure.

**Implementation plan:**
- [ ] Create `src/nfl/training/v5/final_retrain.py`:
  - Reads ablation-validated feature list from Task 3.2b output
  - Trains 54 ensembles on full 2020-2025 data (no eval holdout)
  - Generates 2025 weekly predictions (one row per (player_id, season, week, stat, model_type))
  - Saves models to `data/nfl/models/v5_final/` (separate dir from 3.2's eval models — these are the ones used in production)
  - Saves predictions to `data/nfl/predictions/v5/predictions_{season}_week_{week}.parquet`
- [ ] Extend `src/nfl/db/load_predictions.py` to handle V5:
  - Add V5 to the version registry
  - Add DST handling — `position == 'DST'` rows skip the players-table FK-style lookup; use team_abbr as player_id directly
  - For DST `actual_value` backfill, query `team_stats` via (season, week, team) and pull the relevant defensive stat (sacks, interceptions, etc.). For `fantasy_points_dst` actuals, compute via `compute_dst_fantasy_points` from team_stats raw columns.
  - Insert V5 row into `model_versions` table before predictions insert (FK ordering — see Task 2.3 lessons)
- [ ] Tests in `tests/test_v5_final_retrain.py`:
  - Final model loads + predicts on a single feature row without error
  - Predictions DataFrame has all required columns (matches load_predictions.py expectations)
  - DST prediction rows: player_id is a team abbr, position is 'DST', stat is in DST stat list
  - 2025 predictions row count ≈ expected (5 player positions × ~weekly active players × stats + DST 32 teams × stats)
- [ ] Tests in `tests/test_load_predictions.py` (extend existing):
  - V5 player predictions load correctly
  - V5 DST predictions load with synthetic team_abbr player_id
  - actual_value backfill works for both player and DST
  - Re-run does not duplicate (TRUNCATE+INSERT pattern from Task 2.3)
- [ ] Create `colab/v5_final_retrain.ipynb`:
  - Mount Drive, paths, **high-RAM/CPU check**, install ML libs
  - Step: load ablation-validated feature list (from Task 3.2b artifact uploaded to Drive)
  - Step: train 54 final ensembles on full data (no eval) — resumable per ensemble
  - Step: generate 2025 week-by-week predictions
  - Step: download predictions Parquets back to local `data/nfl/predictions/v5/`
  - Step: spot-check (Mahomes 2025 W5 predicted_passing_yards in plausible range; KC 2025 W1 DST predicted_sacks > 0)
- [ ] After Colab run completes, run locally: `python -m src.nfl.db.load_predictions --version v5` to load into PostgreSQL
- [ ] Run verification SQL: `SELECT version, position, AVG(ABS(error)) AS mae FROM predictions WHERE actual_value IS NOT NULL AND season=2025 GROUP BY version, position ORDER BY version, position;` — produces the V1→V5 comparison table
- [ ] **Deliverable:** Final V5 model in `data/nfl/models/v5_final/`, 2025 predictions in `data/nfl/predictions/v5/` AND in PostgreSQL `predictions` table with `actual_value` backfilled
- [ ] **Target MAE (carried from V5_QUESTIONS.md):** < 4.0 overall. Position-specific: TE < 3.5, RB < 4.5, WR < 4.5, QB < 6.5. K and DST: targets TBD after first run (no V4 baseline).
- **>>> HANDOFF POINT #4:** User runs `colab/v5_final_retrain.ipynb` (estimated 1-2 hours). Claude resumes to load results into DB and verify MAE bands.

**Risks / pre-flight checks:**
- DST team-name lookup needs a stable source. Easiest: hardcode team abbr → full name map in load_predictions.py (32 entries, doesn't change). Add a test that asserts all 32 NFL team abbrs are mapped.
- `predictions` table has `player_id TEXT` — accepts team abbrs. But existing indexes may not be optimized for DST lookup patterns. If dashboard queries get slow on DST rows, add `idx_predictions_dst ON predictions (position) WHERE position = 'DST'`.
- 2025 partial season risk — if running mid-season, week count is incomplete. Filter predictions to `actual_value IS NOT NULL` for accuracy comparisons.
- Production 2026 inference is OUT OF SCOPE for this task. Task 4.x will build the live inference pipeline (separate from training).
- If V5 MAE misses targets, do NOT silently ship. Document the gap, decide whether to iterate (more features, hyperparameter tuning) or accept (portfolio talking point: "V5 missed target X by Y; here's what we learned"). User-driven decision.

### Task 3.3 — Prediction accuracy dashboard (**Medium** — Streamlit visualization)

**Goal:** Visualize V1→V5 accuracy progression in a Streamlit dashboard backed entirely by PostgreSQL queries on the `predictions` table. Portfolio-grade: someone reviewing the project should grasp model evolution, current accuracy, and per-position strengths/weaknesses in under 60 seconds. Ships as a NEW page (not a rewrite of `app.py` — that's Task 4.1).

**Key architecture decisions:**

1. **PostgreSQL-only data source.** No Parquet reads. All charts are powered by SQL on the `predictions` table (which has `actual_value` and `error` already backfilled by Task 3.2c). One file (`pages/accuracy.py`) + one query module (`src/nfl/db/queries_accuracy.py`).

2. **DST shown alongside player positions.** The dashboard treats DST as a 6th position with its own MAE row. Stat-level breakdown (sacks MAE, INTs MAE, etc.) shown in a drill-down table, not the headline chart.

3. **Per-version-per-stat MAE table is the foundation.** All 5 charts derive from one core query: `SELECT version, position, stat, COUNT(*), AVG(ABS(error)) AS mae FROM predictions WHERE actual_value IS NOT NULL GROUP BY version, position, stat`. Cache it with Streamlit's `@st.cache_data` (TTL=1 hour or invalidate on prediction-table refresh).

4. **Charts (all interactive Plotly):**
   - **Headline:** MAE by version (V1→V5) — single bar chart, overall fantasy_points MAE
   - **Per-position grouped bars:** version × position MAE matrix (6 positions × 5 versions = 30 bars)
   - **Per-stat table:** filterable per (version, position) → drill down to stat-level MAE
   - **Predicted vs actual scatter:** per-position, per-version filter, regression line + R² overlay (~7,000 points per filter combo, downsample if needed)
   - **Per-week MAE trend line:** does accuracy degrade in late season? Useful insight if true (suggests model is stale-data-biased).
   - **Feature importance bar chart:** loaded from Task 3.2's `_ablation_summary.csv` (kept/dropped groups + their MAE contribution). Static — refreshes only when ablation re-runs.

5. **Coexistence with the broken app.py.** Old `app.py` still uses legacy per-week raw files and is broken. Add the accuracy page as a new entry point: `streamlit run pages/accuracy.py` — does NOT touch `app.py`. Task 4.1 will rebuild `app.py` and integrate this page properly.

6. **Filter UX.** Sidebar filters: version multiselect (default = all V1-V5), position multiselect (default = all), season filter (default = 2025 since that's the cross-version comparison year). Filters update all charts in real time via Streamlit reactivity.

**Implementation plan:**
- [ ] Create `src/nfl/db/queries_accuracy.py`:
  - `mae_by_version(versions, seasons)` → DataFrame: version, mae
  - `mae_by_version_position(versions, seasons)` → DataFrame: version, position, mae
  - `mae_by_version_position_stat(versions, seasons)` → DataFrame: version, position, stat, mae, n_predictions
  - `predictions_vs_actuals(version, position, season)` → DataFrame: predicted, actual (for scatter)
  - `mae_by_week(version, season)` → DataFrame: week, mae (for trend line)
- [ ] Create `pages/accuracy.py` (Streamlit page):
  - Sidebar filters (version, position, season)
  - 5 chart sections, top to bottom
  - Use `st.cache_data(ttl=3600)` on each query function
- [ ] Tests in `tests/test_queries_accuracy.py`:
  - Each query function returns expected schema (column names + types)
  - mae_by_version returns 5 rows (V1-V5) when all are loaded
  - DST rows surface in mae_by_version_position with position='DST'
  - Empty-result handling (e.g., position with no predictions yet) returns empty DataFrame, not error
- [ ] Manual UX test: launch dashboard, click through filter combinations, verify no errors, verify numeric ranges plausible
- [ ] **Deliverable:** Working `pages/accuracy.py` + tested query module. Run with `streamlit run pages/accuracy.py`.

**Risks / pre-flight checks:**
- Depends entirely on Task 3.2c loading V5 + DST predictions correctly. If 3.2c is incomplete, dashboard will show only V1-V4. Verify `SELECT DISTINCT version FROM predictions;` shows v1, v2, v3, v4, v5 before starting build.
- Scatter plot at full resolution (~7K points per position×version) may be slow to render. Plotly `px.scatter` handles 10K+ fine; if not, switch to `px.density_heatmap` or downsample to 2K random samples per filter.
- Per-week MAE trend may be confounded by sample-size variance (W18 has fewer games due to late-season rest). Show n_predictions on hover so user can sanity-check thin weeks.
- Streamlit `pages/` convention requires `app.py` exists at root with a sidebar nav — if `app.py` is broken, may need a stub `app.py` or a separate Streamlit entry. Decide during build phase.
- Feature importance chart depends on Task 3.2b output. If ablation hasn't run, hide that section with a "Run Task 3.2b first" placeholder.

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
