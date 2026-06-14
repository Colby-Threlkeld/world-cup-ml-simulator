# CLAUDE.md — world-cup-ml-simulator

**Read this fully at the start of every session before writing code.** Keep it
current: when a slice lands, update "Current status".

## What this is
A portfolio ML project that predicts 2026 FIFA World Cup match probabilities from
historical international football data and simulates the 48-team tournament with
Monte Carlo to estimate title odds. Built one vertical slice at a time, judged
against an honest baseline. Quality bar: good enough to show on GitHub and defend
in an interview.

## Current status
- ✅ **Slice 0** — scaffold + tooling. Data audited; V1 schema agreed (see below). `data/raw/` holds martj42 CSVs (gitignored).
- ✅ **Slice 1** — ingestion: `load_raw_matches` + `clean_matches` → `data/interim/matches.parquet` (49,409 played rows; 68 unplayed 2026 fixtures split out). `scripts/build_matches.py` CLI. Team-name normalization with alias map + unknown detection.
- ⬜ **Slice 2 (next)** — temporal Elo baseline + walk-forward backtest (Brier / log loss), reading `matches.parquet`.
- All work uncommitted on branch `main`. Suite: **66 tests** green (`python -m pytest -q`).
- Build order: 0 scaffold · 1 ingest · 2 Elo baseline+backtest · 3 features · 4 Poisson model · 5 calibration · 6 Monte Carlo sim · 7 reporting/app.

## Operating rules (non-negotiable)
1. Don't dump giant untested code. Work **module by module**, smallest useful slice.
2. **Actually implement files**, don't just explain. Then add/adjust tests.
3. Type hints, Google-style docstrings, small functions, explicit errors (custom exceptions, not bare asserts).
4. **Add or update tests for every module**; keep the suite green.
5. After each change, **run the tests** (or give the exact command) and report real output.
6. **Never fake** predictions, metrics, datasets, or results. No numbers without real data behind them.
7. **No leakage** — see doctrine below. This is the single most important constraint.
8. Keep it **runnable on a small Azure CPU VM** (no GPU; ~2 vCPU / 4 GB; tiny data; deterministic).
9. Prefer **simple models first**, then improve with evidence (beat the baseline's score).
10. Reproducible: seed everything from `worldcup.config.RANDOM_SEED` (42); deterministic outputs.

## Review lenses (apply before and after writing code)
The named role-agents below are **not configured as real subagents** in this repo
(available agent types: `Explore`, `Plan`, `general-purpose`, …). So **simulate
these roles internally** as explicit review passes by default. Spawn a *real*
agent only when it genuinely pays off — e.g. `Explore` for a broad codebase
search, `Plan` for a large design — not for small edits. (If real agents get
added under `.claude/agents/`, prefer delegating to the matching one.)
- **Architect** — design coherence, module boundaries, no circular deps, right altitude.
- **Data engineer** — schema fidelity, ingestion, validation, dtypes, dedup, encoding.
- **ML engineer** — modeling soundness, **leakage**, proper scoring rules, temporal eval.
- **Simulation engineer** — tournament format correctness, tiebreakers, knockout/penalty logic.
- **QA/testing** — meaningful tests (incl. a leakage test per feature), edge cases, determinism.
- **Docs** — README + this file + docstrings stay truthful and current.

## No-leakage doctrine (the crux — rule 7)
- Every feature is computed **as-of kickoff** from prior matches only.
- **Drop the 68 unplayed rows** (null scores = 2026 WC fixtures) before training; they feed the simulation, never the model.
- Elo: features use **`elo_pre`**, never `elo_post`. Compute ratings strictly in date order.
- Rolling features: `shift(1)` so a match never sees itself.
- FIFA ranking (when added): join on **`rank_date ≤ kickoff`**, never the latest; mind the 2018 scale change.
- `result`, `home_score`, `away_score`, `total_goals`, `goal_diff` are **labels, not features**.
- Splits are **temporal only** — never random K-fold. `date` may be a split key but never an X column.
- Backtest predictions: `predicted_at ≤ kickoff`.

## Architecture & key decisions
- **Match model:** Poisson goals (λ_home, λ_away → scoreline grid → W/D/L + knockout resolution).
- **Baseline:** temporal Elo (slice 2). Every later model must beat its Brier / log loss.
- **Data:** martj42 `international_results` (`results.csv` spine + `shootouts`/`goalscorers`). **Self-computed Elo** is the primary strength signal. FIFA ranking = V1.1 enhancement. Betting odds = V2 **benchmark only, never a feature**.

## Agreed V1 schema (build 6 of 8)
`matches` (played, wide) · `elo_ratings` (computed) · `match_features` (Elo + rolling form + rest_days + neutral; leakage-safe) · `fixtures_2026` (the 68 unplayed rows + group/stage from config) · `predictions` (λ + p_home/draw/away) · `simulation_results` (per-team advance/title probs).
**Deferred to V1.1:** `fifa_rankings` and the unified `team_ratings` (an Elo-only V1 needs no unification). `team_matches` (long) is the feature substrate behind `match_features`.
Keys: `match_id` assigned over all sorted results rows, then split. Canonical team = name string; `team_id` optional later.

## Naming normalization
`results.csv` is already canonical English (United States, South Korea, Iran, Czech Republic, Turkey). `configs/team_name_map.yaml` maps **FIFA → results** spellings. Keep defunct states (Yugoslavia, Czechoslovakia, German DR) **distinct** — never merge into successors.

## Commands
Local dev is Windows + Anaconda **Python 3.13** (has pandas/numpy/pytest/yaml). Repo pins 3.12 via `.python-version`; `requires-python = ">=3.11"`.
```bash
python -m pytest -q                 # fast, VERIFIED green (pythonpath=src; no install needed)
python scripts/audit_data.py        # profile every CSV in data/raw/
make install                        # uv venv + editable install + dev tools (VM/mac/linux; needs uv)
make test | make lint | make format # uv run pytest | ruff+black --check | ruff --fix + black
make app                            # streamlit run app/streamlit_app.py
```
Use `python -m pytest` for quick local checks; use `make`/`uv` targets on the VM or once uv is installed.

## Environment & VM-friendliness (rule 8)
CPU-only, no GPU. Data is a few MB → fits in RAM trivially. Don't add heavy deps without justification (XGBoost/LightGBM only if they beat the baseline — they live in the `boost` extra). Keep Monte Carlo vectorized/seeded so 10k sims run in seconds on 2 vCPUs.

## Git
This repo is its **own** git root at `…/fifa_project/world-cup-ml-simulator/`, branch `main`. The parent home folder `C:\Users\colby` is a *separate, unrelated* git repo — **never run git from there**. `data/` (except `.gitkeep`) and `.env` are gitignored; never commit datasets or secrets.

## Definition of done (every module)
- [ ] Small, typed, documented functions; clear custom errors.
- [ ] Tests added/updated — including a **leakage test** for any feature.
- [ ] `python -m pytest -q` green; lint clean.
- [ ] No fabricated data/metrics; no post-kickoff info in features.
- [ ] End-of-task summary (rule 10): **changed files · assumptions · TODOs · tests run + result · next best step.**

## Hard "do not"s
No fake predictions/metrics/datasets · no giant untested dumps · no future data in features · no committing data or secrets · no real logic in notebooks (notebooks are throwaway exploration; reusable code graduates to `src/worldcup/`).
