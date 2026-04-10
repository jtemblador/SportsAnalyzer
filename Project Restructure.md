# Project Restructure: NFL Sports Analyzer — Industry-Standard Organization

## Context

This is my **NFL Fantasy Sports Prediction Platform** — a machine learning pipeline that predicts weekly fantasy football scores using CatBoost, engineered features, and Vegas odds integration. It's a key project on my professional portfolio (josetemblador.com) and I plan to continue expanding it. Right now the folder structure is cluttered and disorganized — files that belong together are scattered, there's no clear separation of concerns, and it would be difficult to walk a recruiter through the project or onboard a collaborator.

## Objective

Reorganize the entire project directory to follow **industry-standard Python/ML project conventions** so that:

1. The structure is immediately intuitive to any developer or recruiter reviewing the repo
2. Related files are logically grouped (data pipeline, feature engineering, model training, prediction, evaluation, presentation, etc.)
3. The project is easy to extend — I want to continue adding features, lowering MAE scores, and eventually integrating a PostgreSQL database
4. The codebase tells a clear story: raw data → feature engineering → model training → predictions → evaluation → dashboard

## What I Need You To Do

### Phase 1 — Understand the Pipeline

Before moving anything, map out the entire project:
- Identify every file and its purpose
- Trace the data flow: what feeds into what, in what order, and why
- Document which files depend on each other (imports, file path references, config references)
- Identify files that can be deleted (e.g., `claude_prompt.txt`, any scratch/temp files that aren't part of the pipeline)
- Identify files that are future work and should be clearly marked as such (e.g., `ideas_nfl_postgresql.md` — I want to add PostgreSQL eventually, but not in this phase)

### Phase 2 — Design the New Structure

Propose a reorganized directory layout that follows conventions seen in production ML projects. Think along the lines of:
- `src/` or a named package for all source code, with submodules for each pipeline stage (data ingestion, feature engineering, training, prediction, evaluation)
- `data/` with clear subdirectories for raw, processed, features, models, predictions, etc.
- `notebooks/` if any exploratory work exists
- `tests/` for all test files
- `configs/` for any configuration or hyperparameter files
- `docs/` or `presentations/` for documentation, reports, and presentation materials
- A clean root directory with only standard files: `README.md`, `requirements.txt` / `pyproject.toml`, `.gitignore`, `Makefile` or similar entry points
- `app.py` (Streamlit dashboard) in a logical location

Present the proposed structure for my approval before executing.

### Phase 3 — Write Tests Before Reorganizing

Before moving any files:
- Write integration/smoke tests that verify the current pipeline works end-to-end (or at minimum, that key modules import correctly and core functions produce expected outputs)
- These tests will serve as a safety net to confirm nothing breaks after the move

### Phase 4 — Execute the Reorganization

- Move files to their new locations
- Update ALL internal references: imports, file paths (especially the hardcoded Parquet paths), config references, and any path logic in the Streamlit dashboard
- Update `.gitignore` if needed
- Delete files confirmed as unnecessary
- Ensure the Streamlit app, model training pipeline, and prediction pipeline all still function correctly

### Phase 5 — Verify Nothing Broke

- Run the tests from Phase 3 to confirm the pipeline still works
- Verify the Streamlit dashboard launches and displays data correctly
- Confirm model files are still loadable from their new paths

## Important Constraints

- **Do NOT implement the PostgreSQL migration** — that's a future phase. The `ideas_nfl_postgresql.md` file should be preserved in a `docs/` or `future/` directory as reference material for later
- **Do NOT modify model logic, feature engineering, or training code** — this is strictly a reorganization effort. The only code changes should be updating file paths and imports
- **Preserve git history as much as possible** — use `git mv` where appropriate so file history is trackable
- **This is a portfolio/resume project** — the structure should demonstrate that I understand software engineering best practices: separation of concerns, modular design, clear naming conventions, and maintainable architecture

## Recruiter Walkthrough Perspective

After this reorganization, I want to be able to walk a technical recruiter or hiring manager through this project and have them immediately understand:
- What the project does
- How the ML pipeline flows
- Where to find any given component
- That the project is built to be extended, not just hacked together

### Technical Interview Questions & How the Structure Helps You Answer

**1. "Walk me through the architecture of this project."**
Point to the `src/nfl/` sub-packages: `data/` (ingestion), `features/` (engineering), `models/` (ML classes), `training/` (orchestration), `odds/` (external API integration). Each maps to a pipeline stage. The data flows: raw stats -> engineered features -> trained models -> predictions -> Streamlit dashboard. The directory structure literally mirrors the pipeline.

**2. "How did you iterate on model accuracy?"**
Point to the versioned approach in `features/` (engineer.py, v3_engineer.py, v4_engineer.py) and `models/` (base.py, v4_models.py). Each version added specific improvements: V2 added variance/trend features, V3 added EPA/efficiency metrics, V4 added position-specific hyperparameters + Vegas odds. The `data/` directory preserves all 4 versions of features/models/predictions side-by-side for comparison. The `docs/progress/` folder has detailed notes from each iteration.

**3. "How do you test this? What happens when you change something?"**
Point to `tests/test_smoke.py` — 17 smoke tests covering imports, data file integrity, and model loading. These run in under 3 seconds and catch any broken path or import. The `tests/` directory separates proper tests from ad-hoc analysis scripts (`tests/scripts/`).

**4. "Why did you choose CatBoost/XGBoost/LightGBM? How do they compare?"**
The `models/base.py` file shows the ensemble approach — each model type (POBModel, EVOBModel, StatPredictor) trains 4 algorithms and combines them. This is visible in a single file rather than scattered across the codebase. Position-specific tuning lives in `models/v4_models.py`.

**5. "How would you extend this to add a new data source or feature?"**
The modular structure makes this clear: add a new fetcher in `data/` or `odds/`, create a new feature engineer version in `features/`, and the training pipeline picks it up. The PostgreSQL migration plan in `docs/future/` shows forward-thinking. The separation of concerns means adding a new data source doesn't touch the model layer.

**6. "How does the Vegas odds integration work?"**
Point to `odds/api_client.py` (API client) and `odds/fetcher.py` (orchestrator). These feed into `features/v4_engineer.py` which adds 8 Vegas-derived features (implied totals, spread, game script index). The separation keeps the external API concern isolated from core ML logic.

**7. "What's your MAE and how does it compare to industry benchmarks?"**
4.26 MAE on fantasy points PPR — a 17% improvement from V1 (5.14) to V4 (4.26). Professional fantasy analysts typically achieve 4-5 MAE. The `docs/progress/2025-12-07_comprehensive_version_analysis.md` has the full comparison breakdown. Validation scripts in `tests/scripts/validate_accuracy.py` make results reproducible.
