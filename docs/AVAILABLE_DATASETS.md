# nflreadpy — Complete Dataset Audit

## What We Currently Pull
**1 dataset:** `load_player_stats()` (114 columns) — individual player box scores per week.

That's it. We're leaving a massive amount of free data on the table.

---

## Complete Dataset Inventory

### TIER 1 — Must Have (directly impacts prediction accuracy)

These datasets contain signals that directly predict player fantasy performance.

#### 1. Player Stats (ALREADY PULLING)
- **Function:** `load_player_stats(seasons)`
- **Columns:** 114 | **Rows:** ~1,000/week
- **Contains:** Individual box scores — passing, rushing, receiving, kicking, defense, fantasy points
- **Status:** Already pulling and storing in Parquet

#### 2. Schedules (Game Context + Vegas Lines + Weather)
- **Function:** `load_schedules(seasons)`
- **Columns:** 46 | **Rows:** ~285/season
- **Contains:**
  - **Vegas lines:** `spread_line`, `total_line`, `home_moneyline`, `away_moneyline`, `over_odds`, `under_odds`
  - **Weather:** `temp`, `wind`, `roof`, `surface`
  - **Rest:** `home_rest`, `away_rest` (days since last game)
  - **Game info:** `stadium`, `home_coach`, `away_coach`, `referee`, `div_game`, `gameday`, `gametime`
  - **Scores:** `home_score`, `away_score`, `total`, `overtime`
- **Why it matters:** Vegas lines are the #1 external predictor of game script. Weather affects passing. Rest days affect performance. This replaces the paid Odds API entirely.

#### 3. Injuries
- **Function:** `load_injuries(seasons)`
- **Columns:** 16 | **Rows:** ~6,200/season
- **Contains:** `report_status` (Out/Doubtful/Questionable), `report_primary_injury`, `practice_status`, weekly per player
- **Why it matters:** A QB playing with a hand injury throws fewer yards. A WR1 being Out shifts targets to WR2/WR3. Injury data is one of the biggest gaps in our current model.

#### 4. Snap Counts
- **Function:** `load_snap_counts(seasons)`
- **Columns:** 16 | **Rows:** ~26,600/season
- **Contains:** `offense_snaps`, `offense_pct`, `defense_snaps`, `defense_pct`, `st_snaps`, per player per week
- **Why it matters:** Direct measure of opportunity. A RB with 80% snap share will score more than one with 40%. We're currently inferring usage from stats — snap counts are the direct signal.

#### 5. Next Gen Stats — Passing
- **Function:** `load_nextgen_stats(seasons, stat_type='passing')`
- **Columns:** 29 | **Rows:** ~600/season
- **Contains:** `avg_time_to_throw`, `avg_completed_air_yards`, `avg_intended_air_yards`, `aggressiveness`, `completion_percentage_above_expectation`, `expected_completion_percentage`
- **Why it matters:** Separates QB quality from volume. A QB with high CPOE is performing better than his stats suggest — predictive of future improvement.

#### 6. Next Gen Stats — Rushing
- **Function:** `load_nextgen_stats(seasons, stat_type='rushing')`
- **Columns:** 22 | **Rows:** ~600/season
- **Contains:** `efficiency`, `avg_time_to_los`, `rush_yards_over_expected`, `rush_yards_over_expected_per_att`, `rush_pct_over_expected`, `percent_attempts_gte_eight_defenders`
- **Why it matters:** Rush yards over expected separates talent from offensive line quality. A RB getting 2.0 RYOE/att is elite regardless of raw yards.

#### 7. Next Gen Stats — Receiving
- **Function:** `load_nextgen_stats(seasons, stat_type='receiving')`
- **Columns:** 23 | **Rows:** ~1,400/season
- **Contains:** `avg_cushion`, `avg_separation`, `avg_expected_yac`, `avg_yac_above_expectation`, `catch_percentage`, `percent_share_of_intended_air_yards`
- **Why it matters:** Separation and YAC above expectation identify receivers who create their own production vs. those who depend on scheme.

#### 8. Fantasy Opportunity (Expected Fantasy Points)
- **Function:** `load_ff_opportunity(seasons)`
- **Columns:** 159 | **Rows:** ~6,000/season
- **Contains:**
  - **Expected stats:** `pass_yards_gained_exp`, `rush_yards_gained_exp`, `rec_yards_gained_exp`, `pass_touchdown_exp`, etc.
  - **Expected fantasy points:** `total_fantasy_points_exp`, `pass_fantasy_points_exp`, `rec_fantasy_points_exp`
  - **Over/under expected:** `total_fantasy_points_diff` (actual minus expected)
  - **Team shares:** every stat also available as team totals for calculating market share
- **Why it matters:** This is essentially a pre-built "expected vs actual" dataset. A player consistently scoring above expected is performing at an elite level. A player scoring below expected is due for regression. This is directly aligned with our POB (Probability Over Baseline) model.

---

### TIER 2 — High Value (adds meaningful predictive signal)

#### 9. PFR Advanced Stats — Passing
- **Function:** `load_pfr_advstats(seasons, stat_type='pass')`
- **Columns:** 24 | **Rows:** ~700/season
- **Contains:** `passing_drops`, `passing_bad_throws`, `times_pressured`, `times_hurried`, `times_blitzed`, `times_pressured_pct`
- **Why it matters:** Pressure rate is one of the strongest predictors of QB performance. A QB under pressure 40% of the time will underperform his talent level.

#### 10. PFR Advanced Stats — Rushing
- **Function:** `load_pfr_advstats(seasons, stat_type='rush')`
- **Columns:** 16 | **Rows:** ~2,400/season
- **Contains:** `rushing_yards_before_contact`, `rushing_yards_before_contact_avg`, `rushing_yards_after_contact`, `rushing_broken_tackles`
- **Why it matters:** Yards after contact measures RB talent independent of blocking. Broken tackles indicate elusiveness.

#### 11. PFR Advanced Stats — Receiving
- **Function:** `load_pfr_advstats(seasons, stat_type='rec')`
- **Columns:** 17 | **Rows:** ~4,500/season
- **Contains:** `receiving_drop`, `receiving_drop_pct`, `receiving_broken_tackles`, `receiving_int` (interceptions on targets), `receiving_rat` (passer rating when targeted)
- **Why it matters:** Drop rate identifies unreliable receivers. Passer rating when targeted measures how well a receiver creates for his QB.

#### 12. Team Stats
- **Function:** `load_team_stats(seasons)`
- **Columns:** 102 | **Rows:** ~570/season
- **Contains:** Team-level offensive/defensive totals per week — EPA, yards, TDs, turnovers, all split by pass/rush/receive
- **Why it matters:** Opponent defense strength. We currently calculate this manually from player stats — this gives it directly at the team level.

#### 13. Depth Charts
- **Function:** `load_depth_charts(seasons)`
- **Columns:** 15 | **Rows:** ~37,000/season
- **Contains:** `depth_team` (1=starter, 2=backup, 3=third string), `depth_position`, weekly per player
- **Why it matters:** Starter vs backup status directly predicts snap count and opportunity. A RB2 being promoted to RB1 mid-season is a strong signal.

---

### TIER 3 — Supplementary (useful for specific features or edge cases)

#### 14. Rosters
- **Function:** `load_rosters(seasons)`
- **Columns:** 36 | **Rows:** ~3,200/season
- **Contains:** `birth_date`, `height`, `weight`, `years_exp`, `draft_club`, `draft_number`, `college`
- **Use case:** Rookie projections (draft capital correlates with early opportunity), age curves

#### 15. Players (Static)
- **Function:** `load_players()`
- **Columns:** 39 | **Rows:** ~24,400 total
- **Contains:** All-time player database with IDs across platforms (gsis, espn, pfr, pff, sleeper, yahoo)
- **Use case:** Player ID mapping across data sources, career lookup

#### 16. Participation
- **Function:** `load_participation(seasons)`
- **Columns:** 26 | **Rows:** ~46,000/season
- **Contains:** Play-level participation — which players were on field for each play, offensive formation, personnel groupings, coverage type
- **Use case:** Advanced route analysis, personnel tendency features (future)

#### 17. FTN Charting
- **Function:** `load_ftn_charting(seasons)`
- **Columns:** 29 | **Rows:** ~48,000/season
- **Contains:** Play-level charting — `is_play_action`, `is_screen_pass`, `is_rpo`, `is_no_huddle`, `n_blitzers`, `is_drop`, `is_contested_ball`
- **Use case:** Play-calling tendency features (future)

#### 18. Officials
- **Function:** `load_officials(seasons)`
- **Columns:** 9 | **Rows:** ~2,000/season
- **Contains:** Which referee crew officiated each game
- **Use case:** Some refs call more penalties, affecting game pace and scoring

---

### NOT NEEDED (for our prediction goals)

| Dataset | Function | Why Skip |
|---------|----------|----------|
| Draft Picks | `load_draft_picks()` | Historical draft data, not weekly — covered by rosters |
| Combine | `load_combine()` | Pre-draft athletic testing — marginal for weekly predictions |
| Contracts | `load_contracts()` | Salary data — doesn't predict weekly performance |
| Trades | `load_trades()` | Transaction records — covered by roster changes |
| FF Player IDs | `load_ff_playerids()` | ID mapping only, no stats |
| FF Rankings | `load_ff_rankings()` | Expert consensus rankings — interesting but not a model input |
| Play-by-Play | `load_pbp()` | 372 columns x 49,000 rows/season — too granular for weekly player predictions. The aggregated stats in other datasets already capture what we need. Could use in future for advanced play-calling features. |

---

## Summary: What to Pull

| Priority | Dataset | Function | Key Signals |
|----------|---------|----------|-------------|
| **NOW** | Schedules | `load_schedules()` | Vegas lines, weather, rest days, stadium |
| **NOW** | Injuries | `load_injuries()` | Player availability, practice status |
| **NOW** | Snap Counts | `load_snap_counts()` | Opportunity share |
| **NOW** | NGS Passing | `load_nextgen_stats('passing')` | Time to throw, CPOE, aggressiveness |
| **NOW** | NGS Rushing | `load_nextgen_stats('rushing')` | Rush yards over expected, efficiency |
| **NOW** | NGS Receiving | `load_nextgen_stats('receiving')` | Separation, YAC over expected |
| **NOW** | FF Opportunity | `load_ff_opportunity()` | Expected fantasy points, over/under expected |
| **SOON** | PFR Pass | `load_pfr_advstats('pass')` | Pressure rate, drops, bad throws |
| **SOON** | PFR Rush | `load_pfr_advstats('rush')` | Yards before/after contact |
| **SOON** | PFR Rec | `load_pfr_advstats('rec')` | Drop rate, passer rating when targeted |
| **SOON** | Team Stats | `load_team_stats()` | Team-level offense/defense EPA |
| **SOON** | Depth Charts | `load_depth_charts()` | Starter vs backup status |
| **LATER** | Rosters | `load_rosters()` | Age, experience, draft capital |
| **LATER** | Players | `load_players()` | ID mapping, career info |
