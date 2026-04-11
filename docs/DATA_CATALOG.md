# Data Catalog — NFL Sports Analyzer

Quick reference for all datasets, their columns, join keys, and quirks. Use this to know what data is available without querying files or the database.

**Last updated:** 2026-04-11
**Season range:** 2018-2025 (8 seasons)
**Total records:** ~791,000+
**Total Parquet files:** 247

---

## Join Key Summary

All datasets join through `season` + `week`. Player-level joins use two ID formats:

| ID Format | Column Name | Example | Used By |
|-----------|-------------|---------|---------|
| **GSIS** (primary) | `player_id`, `gsis_id`, `player_gsis_id` | `00-0033873` | player_stats, injuries, NGS (all 3), ff_opportunity, depth_charts |
| **PFR** | `pfr_player_id` | `MahoPa00` | snap_counts, pfr_advstats (all 3) |

**ID mapping:** `nfl.load_players()` has both IDs for ~3,300 recent players (last_season >= 2024). The full table has 24,376 players (all-time back to 1974) — filter to relevant seasons before use.

**Team abbreviations** are consistent across datasets (`KC`, `BAL`, etc.) except depth_charts which uses `club_code` instead of `team`.

---

## Datasets

### 1. player_stats — Weekly Player Box Scores
- **Path:** `data/nfl/player_stats/player_stats_{season}.parquet`
- **Seasons:** 2018-2025 (8 files)
- **Rows:** 147,223 total (~18,000/season)
- **Columns:** 114-115
- **Join key:** `player_id` (GSIS) + `season` + `week`
- **Includes:** Regular season + playoffs (weeks 1-22)

| Column Group | Key Columns | Notes |
|-------------|-------------|-------|
| Identity | `player_id`, `player_name`, `position`, `team`, `opponent_team` | `player_id` ~0.1% null |
| Passing | `passing_yards`, `passing_tds`, `passing_interceptions`, `completions`, `attempts`, `passing_epa`, `passing_cpoe`, `pacr` | EPA/CPOE 96% null (non-QBs) |
| Rushing | `rushing_yards`, `rushing_tds`, `carries`, `rushing_epa` | EPA 88% null (non-rushers) |
| Receiving | `receiving_yards`, `receiving_tds`, `receptions`, `targets`, `target_share`, `air_yards_share`, `wopr`, `racr`, `receiving_epa` | EPA 76% null |
| Kicking | `fg_made`, `fg_att`, `fg_pct`, `pat_made`, `fg_made_*` (by distance) | 97%+ null (non-kickers) |
| Fantasy | `fantasy_points`, `fantasy_points_ppr` | **Target variable** |

### 2. schedules — Game Context, Vegas Lines, Weather
- **Path:** `data/nfl/schedules/schedules_{season}.parquet`
- **Seasons:** 2018-2025 (8 files)
- **Rows:** 2,227 total (~280/season)
- **Columns:** 48 (46 original + 2 derived)
- **Join key:** `season` + `week` + `home_team`/`away_team`
- **No player IDs** — this is game-level data

| Column Group | Key Columns | Notes |
|-------------|-------------|-------|
| Game | `game_id`, `season`, `week`, `home_team`, `away_team`, `gameday`, `gametime` | |
| Scores | `home_score`, `away_score`, `total`, `overtime` | |
| Vegas | `spread_line`, `total_line`, `home_moneyline`, `away_moneyline` | 0% null — 100% coverage |
| Derived | `home_implied_total`, `away_implied_total` | Calculated: `(total ± spread) / 2` |
| Weather | `temp`, `wind`, `roof`, `surface` | temp/wind 36% null (dome games) |
| Context | `home_rest`, `away_rest`, `div_game`, `referee`, `stadium` | |
| Coaches/QBs | `home_coach`, `away_coach`, `home_qb_name`, `away_qb_name` | |

### 3. injuries — Weekly Injury Reports
- **Path:** `data/nfl/injuries/injuries_{season}.parquet`
- **Seasons:** 2018-2025 (8 files)
- **Rows:** 45,337 total (~5,700/season)
- **Columns:** 16
- **Join key:** `gsis_id` (GSIS) + `season` + `week`
- **Important:** Only players ON the injury report appear. No row = healthy.

| Column | Notes |
|--------|-------|
| `report_status` | Out / Doubtful / Questionable. **54% null** = not on game-day report (practice only). |
| `practice_status` | Full / Limited / DNP. **99% populated** — most reliable signal. |
| `report_primary_injury` | Ankle, Knee, etc. 55% null (same as report_status). |

### 4. snap_counts — Weekly Snap Participation
- **Path:** `data/nfl/snap_counts/snap_counts_{season}.parquet`
- **Seasons:** 2018-2025 (8 files)
- **Rows:** 205,354 total (~26,000/season)
- **Columns:** 16
- **Join key:** `pfr_player_id` (PFR — needs mapping) + `season` + `week`
- **0% nulls** across all columns

| Column | Notes |
|--------|-------|
| `offense_snaps`, `offense_pct` | Direct opportunity measure. Pct is 0.0-1.0 scale. |
| `defense_snaps`, `defense_pct` | Defensive players. |
| `st_snaps`, `st_pct` | Special teams. |

### 5-7. nextgen_stats — NFL Next Gen Tracking Metrics
- **Path:** `data/nfl/nextgen_stats/ngs_{type}_{season}.parquet`
- **Seasons:** 2018-2025 (24 files total, 8 per type)
- **Join key:** `player_gsis_id` (GSIS) + `season` + `week`
- **Week 0 = season aggregate.** Filter to `week > 0` for per-game data.
- **Only qualified players** (minimum attempts/targets). Low-volume players missing.

| Type | Columns | Rows | Key Metrics |
|------|---------|------|-------------|
| **passing** | 29 | 4,785 | `avg_time_to_throw`, `aggressiveness`, `completion_percentage_above_expectation`, `expected_completion_percentage` |
| **rushing** | 22 | 4,885 | `efficiency`, `rush_yards_over_expected`, `rush_yards_over_expected_per_att`, `percent_attempts_gte_eight_defenders` |
| **receiving** | 23 | 11,708 | `avg_cushion`, `avg_separation`, `avg_yac_above_expectation`, `catch_percentage` |

### 8. ff_opportunity — Expected Fantasy Points
- **Path:** `data/nfl/ff_opportunity/ff_opportunity_{season}.parquet`
- **Seasons:** 2018-2025 (8 files)
- **Rows:** 47,282 total (~6,000/season)
- **Columns:** 159
- **Join key:** `player_id` (GSIS) + `season` + `week`
- **`player_id` is 7% null** — filter `notna()` when joining

| Column Group | Key Columns | Notes |
|-------------|-------------|-------|
| Expected | `total_fantasy_points_exp`, `pass_fantasy_points_exp`, `rec_fantasy_points_exp`, `rush_fantasy_points_exp` | Model-predicted expected points |
| Actual | `total_fantasy_points`, `pass_fantasy_points`, `rec_fantasy_points`, `rush_fantasy_points` | 0% null |
| Differential | `total_fantasy_points_diff` (= actual - expected) | Over/under performance. **Feeds POB model.** |
| Team shares | `*_team` columns (84 columns) | Player's share of team totals |

### 9-11. pfr_advstats — PFR Advanced Stats
- **Path:** `data/nfl/pfr_advstats/pfr_{type}_{season}.parquet`
- **Seasons:** 2018-2025 (24 files total, 8 per type)
- **Join key:** `pfr_player_id` (PFR — needs mapping) + `season` + `week`

| Type | Columns | Rows | Key Metrics |
|------|---------|------|-------------|
| **pass** | 24 | 5,424 | `times_pressured`, `times_pressured_pct`, `passing_drops`, `passing_bad_throws`, `times_blitzed` |
| **rush** | 16 | 18,461 | `rushing_yards_before_contact`, `rushing_yards_after_contact`, `rushing_broken_tackles` |
| **rec** | 17 | 35,724 | `receiving_drop`, `receiving_drop_pct`, `receiving_broken_tackles`, `receiving_rat` (passer rating when targeted) |

**Note:** Some columns are 100% null in certain types (e.g., `receiving_drop` in pass type, `rushing_broken_tackles` in rec type). These are cross-type columns that only populate for the relevant stat type.

### 12. team_stats — Team-Level Weekly Stats
- **Path:** `data/nfl/team_stats/team_stats_{season}.parquet`
- **Seasons:** 2018-2025 (8 files)
- **Rows:** 4,454 total (~560/season, 32 teams x ~18 weeks)
- **Columns:** 102-103
- **Join key:** `team` + `season` + `week`
- **No player IDs** — team-level aggregates
- **Use for:** Opponent defense strength (join on `opponent_team`)

Same column structure as player_stats but aggregated at team level. Includes `passing_epa`, `rushing_epa`, `receiving_epa` at team level.

### 13. depth_charts — Starter/Backup Status
- **Path:** `data/nfl/depth_charts/depth_charts_{season}.parquet`
- **Seasons:** 2018-2024 (7 files — **2025 excluded**, incompatible schema)
- **Rows:** 258,942 total (~37,000/season)
- **Columns:** 15
- **Join key:** `gsis_id` (GSIS) + `season` + `week`
- **Team column:** `club_code` (not `team`)

| Column | Notes |
|--------|-------|
| `depth_team` | `'1'` = starter, `'2'` = backup, `'3'` = third string. **String type, not int.** |
| `formation` | `Offense` / `Defense` / `Special Teams`. Filter to Offense for fantasy. |
| `depth_position` | Specific position on depth chart (e.g., `RG`, `WR1`). |

---

## Known Quirks & Gotchas

1. **Two ID formats:** GSIS (`00-0033873`) vs PFR (`MahoPa00`). Snap counts and PFR advstats use PFR. Everything else uses GSIS. Map via `nfl.load_players()`.
2. **EPA columns are mostly null in player_stats:** `passing_epa` is 96% null because only QBs have it. Same for rushing/receiving EPA. This is position-specific sparsity, not missing data.
3. **Injury report nulls = healthy:** No `report_status` means the player wasn't on the injury report. Treat as available.
4. **NGS week 0 = season aggregate:** Has high attempt counts (500+ for QBs). Filter `week > 0` for per-game features.
5. **FF opportunity player_id 7% null:** Rows with no player identification. Filter `player_id.notna()` when joining.
6. **Depth chart 2025 missing:** nflverse changed the schema entirely. Only 2018-2024 available.
7. **Weather temp/wind 36% null in schedules:** Dome/closed roof games have no outdoor readings. Fill with neutral defaults (72°F, 0 mph) during feature engineering.
8. **PFR advstats cross-type nulls:** `pfr_pass` has `receiving_drop` (100% null), `pfr_rec` has `rushing_broken_tackles` (100% null). These columns exist but are irrelevant for that stat type.
9. **Kicking stats are 97%+ null in player_stats:** Only kickers have these. Position-specific sparsity.
10. **depth_team is string type** (`'1'`, `'2'`, `'3'`), not integer. Convert during feature engineering if needed.
