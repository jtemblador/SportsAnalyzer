# Sports Analyzer — Development Roadmap

## Current State
- V4 production model: 4.26 MAE (17% improvement over V1 baseline)
- **Phase 0 complete:** 12 datasets fetched, 247 Parquet files, 791,000+ records (2018-2025)
- All data stored as per-season Parquet files on disk (consistent format across all datasets)
- 9 fetcher classes registered in unified pipeline (`fetch_all()` and `fetch_latest()`)
- 179 tests across project
- Streamlit dashboard reads directly from Parquet
- Project restructured into modular `src/nfl/` sub-packages
- Full dataset audit — see `docs/AVAILABLE_DATASETS.md`

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
- [ ] Install/configure PostgreSQL locally
- [ ] Create `nfl_predictions` database
- [ ] Design schema using DATA_CATALOG.md as the source of truth:
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
- [ ] Add indexes for common query patterns:
  - Player lookup: `(player_gsis_id, season, week)`, `(player_pfr_id, season, week)`
  - Team lookup: `(team, season, week)`
  - Game lookup: `(season, week, home_team)`
- [ ] Write schema migration script: `src/nfl/db/schema.sql`
- [ ] **Deliverable:** Empty database with all tables + indexes created

### Task 1.2 — Database connection layer
- [ ] Create `src/nfl/db/connection.py` — connection pooling, config from env vars
- [ ] Create `src/nfl/db/__init__.py` with convenience imports
- [ ] Add `psycopg2` and `sqlalchemy` to `requirements.txt`
- [ ] Add `.env.example` with database connection template
- [ ] **Deliverable:** `get_engine()` and `get_connection()` functions that work

### Task 1.3 — Bulk load all Parquet data into PostgreSQL
- [ ] Write `src/nfl/db/load_all.py` — reads every Parquet directory and loads into corresponding table
- [ ] Load order: teams → players → games → all player-week tables → team_stats
- [ ] Handle duplicates gracefully (upsert or skip)
- [ ] Print summary: row counts per table, any skipped/failed records
- [ ] Verify total row counts match Parquet files (see DATA_CATALOG.md for expected counts)
- [ ] **Deliverable:** All historical data (2018-2025, 791K+ records) queryable in PostgreSQL

### Task 1.4 — Dual-write data pipeline
- [ ] Modify data pipeline so every fetch writes to both Parquet AND PostgreSQL
- [ ] All fetcher classes insert into their respective DB tables after saving Parquet
- [ ] New players and teams auto-inserted on first encounter
- [ ] **Deliverable:** `python src/nfl/data/pipeline.py` writes to both destinations

### Task 1.5 — Database smoke tests
- [ ] Test connection, table existence, row counts per table
- [ ] Test basic queries: player by name, stats by week, team lookup, injury by week
- [ ] Test cross-ID join: player_stats (GSIS) → players → snap_counts (PFR) for same player
- [ ] Test that loaded data matches sample Parquet files exactly
- [ ] Test dual-write: mock a fetch → verify data in both Parquet and DB
- [ ] **Deliverable:** `tests/test_database.py` passes

---

## Phase 2: Pipeline Reads from DB
**Goal:** Feature engineering and dashboard read from PostgreSQL instead of Parquet files.

### Task 2.1 — Database query layer
- [ ] Create `src/nfl/db/queries.py` — reusable query functions:
  - `get_player_history(player_id, season, week, games_back)` — single SQL query replaces loading N Parquet files
  - `get_week_stats(season, week, position)` — replaces `pd.read_parquet()`
  - `get_player_injuries(player_id, season, week)` — injury status for a player
  - `get_snap_share(player_id, season, week)` — snap count percentage
  - `get_game_context(season, week, team)` — Vegas lines, weather, rest days
  - `get_opponent_defense_rank(team, position, season, week)` — uses team_stats table
  - `get_nextgen_stats(player_id, season, stat_type)` — NGS metrics
- [ ] **Deliverable:** All common data access patterns available as SQL queries

### Task 2.2 — Feature engineer reads from DB
- [ ] Create DB-backed version of `load_player_history()` (single SQL query vs loading N Parquet files)
- [ ] Add `source='db'` or `source='parquet'` parameter to FeatureEngineer
- [ ] Benchmark: DB reads should be faster for cross-week queries
- [ ] **Deliverable:** Feature engineering can run against PostgreSQL

### Task 2.3 — Dashboard reads from DB
- [ ] Modify `app.py` to query PostgreSQL instead of loading Parquet files
- [ ] Streamlit filters translate to SQL WHERE clauses
- [ ] Fallback to Parquet if DB connection fails
- [ ] **Deliverable:** Dashboard works identically but backed by PostgreSQL

### Task 2.4 — Load predictions and model runs into DB
- [ ] Create `predictions` and `model_runs` tables
- [ ] Write ETL to load all `data/nfl/predictions/` (all versions) into database
- [ ] Store model metadata (version, position, algorithm, MAE, hyperparams)
- [ ] Backfill `actual_value` column by joining predictions with weekly_stats
- [ ] **Deliverable:** Cross-version accuracy queries work in SQL

---

## Phase 3: V5 Model Improvements
**Goal:** Use all new data sources + database to improve MAE below 4.0.

### Task 3.1 — Replace paid Odds API with schedule data
- [ ] Rewrite V4 Vegas features to pull from `games` table (nflreadpy schedule data)
- [ ] Remove dependency on `src/nfl/odds/api_client.py` and The Odds API
- [ ] Verify V4 MAE is preserved or improved with the new odds source
- [ ] **Deliverable:** V4 model works without paid API, using free schedule data

### Task 3.2 — V5 feature engineering
- [ ] Build new features from the expanded datasets:
  - **From schedules:** implied team total, spread, weather features (wind, temp, dome), rest days, divisional game flag
  - **From injuries:** teammate injury impact (is WR1 out? is starting QB out?), player's own injury status
  - **From snap counts:** snap share trend, snap share vs team average
  - **From NGS passing:** time to throw, CPOE, aggressiveness, air yards differential
  - **From NGS rushing:** rush yards over expected, efficiency, stacked box %
  - **From NGS receiving:** separation, YAC above expected, cushion, air yards share
  - **From FF opportunity:** expected fantasy points, actual vs expected differential, team fantasy share
  - **From PFR pass:** pressure rate, blitz rate, bad throw rate
  - **From PFR rush:** yards after contact avg, broken tackles per attempt
  - **From PFR rec:** drop rate, passer rating when targeted
  - **From team stats:** opponent offensive/defensive EPA, opponent pass/rush EPA
  - **From depth charts:** starter/backup flag, depth chart changes week-over-week
- [ ] Target: 70-80 features (up from V4's 50)
- [ ] **Deliverable:** V5 feature engineer in `src/nfl/features/v5_engineer.py`

### Task 3.3 — V5 model training and evaluation
- [ ] Train V5 models with expanded feature set
- [ ] Compare MAE against V4 (4.26) per position
- [ ] Feature importance analysis — which new datasets helped most?
- [ ] Position-specific tuning
- [ ] **Target MAE:** < 4.0

### Task 3.4 — Prediction accuracy dashboard
- [ ] New Streamlit tab: model accuracy over time, by position, by version
- [ ] Powered by SQL queries joining predictions with actuals
- [ ] Visual: predicted vs actual scatter plots, MAE trend charts
- [ ] **Deliverable:** Interactive accuracy comparison across V1-V5

---

## Phase 4: Production Hardening (Future)
**Goal:** Make the project production-grade and impressive on a resume.

### Task 4.1 — FastAPI REST endpoints
- [ ] API layer on top of PostgreSQL (player stats, predictions, model accuracy)
- [ ] Proper request/response schemas with Pydantic

### Task 4.2 — Automated weekly pipeline
- [ ] Cron job or scheduled task: fetch new data → features → predictions → DB
- [ ] End-to-end automation

### Task 4.3 — Docker containerization
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
