# ⚽ World Cup ML Simulator

> **Time-aware machine learning that forecasts international football matches and
> simulates the 48-team 2026 World Cup — leakage-safe by construction, calibrated,
> and judged against honest baselines.**

<!-- Badge placeholders — replace OWNER/REPO once the repo is on GitHub. -->
[![CI](https://github.com/OWNER/REPO/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-251%20passing-brightgreen)](tests/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000)](https://github.com/psf/black)
[![Lint: ruff](https://img.shields.io/badge/lint-ruff-261230)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A portfolio project that takes ~150 years of international results, learns to
predict the 3-way outcome of a match (**team A win / draw / team B win**) without
ever peeking at the future, and then plays the 2026 World Cup tens of thousands of
times with Monte Carlo. Everything is judged against honest baselines, backtested
on past tournaments, and wrapped in a small Streamlit dashboard. It runs on a cheap
CPU-only VM in seconds.

It is deliberately **not** a black box that claims to "predict the World Cup."
Football is high-variance and a single tournament is a tiny sample; the goal is a
*well-calibrated, leakage-free* forecasting system that is honest about what it
knows and what it doesn't.

> **For recruiters:** the [Highlights](#highlights) and [Evaluation](#evaluation)
> sections are the 60-second version. **For engineers:** start at
> [How it works](#how-it-works-architecture) and the
> [no-leakage doctrine](#no-leakage-the-core-engineering-constraint).

---

## Highlights

- **Leakage-safe by construction.** Every feature is computed *as of kickoff* from
  prior matches only; all train/validation/test splits are by **date**, never
  random. A unit test fails if a future result changes a past feature.
- **Calibrated probabilities, not vibes.** A multinomial-logistic model over a
  self-computed **Elo** difference plus rolling form. On a temporal hold-out it
  scores **0.867 log loss / 60.8% accuracy** with **ECE ≈ 0.016** — a small but
  real edge over the strongest baseline (Elo-only, 0.877) and well clear of an
  uninformed model (1.099).
- **Honest backtest.** Trained only on data available *before* each tournament,
  the pre-tournament Elo favourite ordering placed **every** eventual champion
  (Germany '14, France '18, Argentina '22) in its **top 5**. (That's a proxy for a
  full winner-probability sim — see the caveats.)
- **A real 48-team Monte Carlo engine** for the new format (12 groups, eight best
  third-placed teams, knockout bracket), seeded and reproducible — 1,000
  tournaments in ~3.5 s on 2 vCPUs.
- **Reproducible and cheap.** One seed (`42`) drives everything; no GPU, a few MB
  of data, designed to deallocate an Azure VM the moment you're done.

---

## How it works (architecture)

```
 data/raw/results.csv ──▶ clean + validate ──▶ data/interim/matches.parquet
 (martj42, 1872–present)   · canonical team names      (49,409 played matches;
                           · type/score checks          68 unplayed 2026 fixtures
                           · drop unplayed fixtures      split out, never trained on)
                                       │
                                       ▼
              build_feature_matrix  (leakage-safe, "as of kickoff")
              ├─ rolling form: last-5/10 points, goals for/against, GD  (shift(1))
              ├─ rest days + matches in last 365d
              └─ walk-forward Elo  (elo_pre only — a match never sees its result)
                                       │
                      temporal split:  train  <  val  <  test   (by date)
                                       │
            multinomial logistic  ──▶  isotonic / Platt calibration (on val)
                                       │
        ┌──────────────────────────────┼───────────────────────────────┐
        ▼                              ▼                                ▼
   evaluation                     backtesting                     Monte Carlo sim
  log loss · Brier            2014 / 2018 / 2022,              48-team format:
  · ECE · reliability         train only on pre-              groups → best thirds
  · vs baselines              tournament matches               → knockout bracket
        └──────────────────────────────┴───────────────────────────────┘
                                       │
                                       ▼
                          Streamlit dashboard (read-only)
```

**Design choices worth defending in an interview:**

- **Modules, not a monolith.** `data/` (load, clean, validate, team-name
  canonicalization) → `features/` (leakage-safe as-of features) → `models/`
  (baselines, calibrated model, train/eval/calibrate) → `simulation/` (groups,
  tiebreakers, knockout, Monte Carlo) → `backtesting.py` → `visualization/`.
- **Config as data.** The tournament format, team-name aliases, and model
  hyperparameters live in `configs/*.yaml`, validated on load — not hardcoded.
- **Custom exceptions, not bare asserts** (`DataValidationError`, `LeakageError`,
  `TournamentConfigError`, …); validators *accumulate* problems and report them
  together.

### No-leakage: the core engineering constraint

Sports time-series is where models quietly cheat. The whole pipeline is built
leakage-first:

- Rolling features use `shift(1)` and look strictly backward; Elo records
  **`elo_pre`** and only then applies the result.
- Rating joins are as-of (`merge_asof`, `direction="backward"`) and re-validated
  after the join — a rating dated even one day after a match can never be used.
- Splits are **temporal only**; `date` may be a split key but is never a feature.
- The 68 unplayed 2026 fixtures (null scores) are dropped before training and feed
  *only* the simulation.

---

## Data

- **Source:** [martj42 / international_results](https://github.com/martj42/international_results)
  — `results.csv` (every men's international, 1872–present) plus `shootouts.csv`
  and `goalscorers.csv`. **~49,409 played matches** after cleaning.
- **Not committed.** `data/` is gitignored; pull the CSVs into `data/raw/` and run
  the pipeline. `scripts/audit_data.py` profiles every CSV before anything is built
  on it.
- **Team-name canonicalization.** `results.csv` is the canonical spelling; a YAML
  alias map folds accent/case/apostrophe variants (`Côte d'Ivoire` → `Ivory
  Coast`, `Korea Republic` → `South Korea`). Defunct sides (Yugoslavia,
  Czechoslovakia, Soviet Union, …) are deliberately **kept distinct** from their
  successors.
- **Self-computed Elo is the primary strength signal.** The FIFA world ranking is a
  planned V1.1 enhancement (as-of join, mind the 2018 scale change); betting odds,
  if ever used, are a benchmark — **never** a feature.

---

## Modeling

- **Target:** the 3-class match outcome `team_a_win / draw / team_b_win` (team A is
  the home/listed side; the format is symmetric so it also serves neutral-venue
  simulation).
- **Model:** **multinomial logistic regression** in a small, deterministic sklearn
  pipeline (constant-impute → standardize → logistic). With ~10–20 leakage-safe
  difference features and tens of thousands of rows, a regularized linear model is
  the honest first choice: fast on 2 vCPUs, hard to overfit, and a clean thing to
  beat. A capped `HistGradientBoosting` is available behind a config flag — used
  only if it earns its keep.
- **Features (9):** `elo_diff`, `form_5_diff`, `form_10_diff`, `goals_for_5_diff`,
  `goals_against_5_diff`, `goal_diff_10_diff`, `rest_days_diff`, `is_team_a_home`,
  `is_neutral`.
- **Calibration:** the base model is fit on **train**, a per-class isotonic (or
  Platt) calibrator is fit on a separate **validation** window, and **test** is
  scored once. Calibration matters here because these probabilities feed a Monte
  Carlo downstream — a biased 70% compounds over a tournament.

> Note: the simulation samples a *cosmetic* Poisson scoreline only to populate
> goal-difference tiebreakers. The originally-planned Poisson λ-grid goals model
> was not built; outcomes come from the logistic model above. The README and code
> say so rather than implying a model that doesn't exist.

---

## Evaluation

Probabilistic forecasts are scored with **proper scoring rules** on a **temporal
hold-out** — the most recent 15% of matches, which the model never saw during
training or calibration (test cutoff `2026-06-12`, **n = 7,411**).

| Model | Log loss ↓ | Brier ↓ | Accuracy ↑ |
|---|---|---|---|
| **Main model (calibrated)** | **0.867** | **0.506** | **60.8 %** |
| Elo-only logistic *(best baseline)* | 0.877 | 0.516 | 60.1 % |
| Recent-form logistic | 0.964 | 0.572 | 54.6 % |
| Class-frequency prior | 1.050 | 0.633 | 47.9 % |
| Uniform (⅓ each) | 1.099 | 0.667 | 47.9 % |

Calibration error (ECE) on the test set improves from 0.022 → **0.016** after
isotonic scaling. **How to read this:** log loss of ~1.099 is what a model that
knows nothing scores (`ln 3`); the main model lands at 0.867. That's a *small,
real* edge over an Elo-only baseline — not state of the art, and not presented as
such. Elo is the dominant signal; rolling form and home advantage add a little on
top. (`reports/evaluation_report.md` explains every metric in plain English and
ships a reliability diagram + confusion matrix.)

---

## Backtesting

The question a backtest answers: *standing before a past World Cup with only the
data available then, how would this system have done?* So leakage is the whole
ballgame — the model is trained only on matches before the tournament's start, and
every match uses a walk-forward Elo computed from prior matches only.

| Tournament | Champion | Pre-tournament Elo rank | Match log loss |
|---|---|---|---|
| 2014 | Germany | **4** / 32 | 0.923 |
| 2018 | France | **5** / 32 | 0.952 |
| 2022 | Argentina | **2** / 32 | 1.028 |

Mean match log loss **0.97** vs **1.10** for an uninformed model; the champion
landed in the pre-tournament top 5 all three times.

**Caveats (this is the honest part):**

- The champion **rank** is the pre-tournament Elo favourite ordering — a *proxy*
  for a full Monte-Carlo winner-probability run, which would need each year's
  official group draw and bracket encoded. Those historical brackets aren't shipped
  (only the 2026 placeholder exists), so that path is wired but unused — I don't
  fabricate a draw.
- Three tournaments, 64 matches each, is a **small sample**. Knockouts are short
  series; a strong model can still rank the eventual winner outside the top few.
- No hyperparameters were tuned on the tournament being scored.

---

## Tournament simulation

A seeded Monte Carlo of the **expanded 48-team format**: 12 groups of 4 → top two
of each group plus the **8 best third-placed teams** (32 qualifiers) → a
single-elimination bracket to a champion. Per simulation it:

1. Plays each group's round-robin: the match outcome is drawn from the model's
   `(win, draw, loss)` probabilities, then a plausible scoreline is sampled for the
   standings.
2. Ranks groups by FIFA's **primary** tiebreakers — points → goal difference →
   goals scored → a seeded random draw. (Head-to-head and fair-play are declared in
   the config but **deferred**; a tie still level after goals scored falls through
   to the draw.)
3. Selects the 8 best thirds, seeds the round of 32, and plays the bracket.
   Knockout matches can't draw, so the draw probability is redistributed to the two
   sides in proportion to their win probabilities (a documented stand-in for
   explicit extra-time/penalties).

Counts are accumulated (no per-run state), so 10,000 simulations stay tiny in
memory. One numpy `Generator` from seed `42` makes a whole run reproducible.

> **⚠️ Honest status:** the official 2026 group draw isn't encoded yet, so the
> shipped simulation runs over **placeholder slots** (`A1…L4`) with a *uniform*
> predictor. Its output therefore reflects **bracket structure only — not a
> team-specific forecast**, and I do not publish "Brazil 14%"-style odds from it.
> To produce a real forecast you supply a per-team strength table and the drawn
> groups (`run_simulation.py --strengths …`); the engine, tiebreakers, and bracket
> logic are built and tested today. The match model, Elo ratings, and backtest
> above are real now.

---

## Streamlit dashboard

`make app` launches a six-page, **read-only** dashboard. It never trains a model —
it loads saved pipeline artifacts and computes only a cheap cached Elo table; any
page whose artifact is missing shows the exact command to generate it (no
fabricated numbers).

| Page | What it shows |
|------|----------------|
| **Overview** | Project summary, which artifacts are ready, top contenders (with the placeholder caveat surfaced) |
| **Team Explorer** | Current self-computed Elo strength + world rank per team |
| **Match Predictor** | Win/draw/loss probabilities for any Team A vs Team B (neutral-venue toggle) |
| **Tournament Simulator** | Monte-Carlo advancement/title probabilities from a saved run |
| **Backtesting** | Match metrics + champion-rank for 2014/2018/2022 |
| **Methodology** | Features, model, calibration curve, the no-leakage doctrine |

### Screenshots

_Add images to `docs/screenshots/` and they'll appear here._

| Overview | Match Predictor | Tournament Simulator |
|----------|-----------------|----------------------|
| _`docs/screenshots/overview.png`_ | _`docs/screenshots/match_predictor.png`_ | _`docs/screenshots/simulator.png`_ |

<!-- When the PNGs exist, uncomment:
| ![Overview](docs/screenshots/overview.png) | ![Match Predictor](docs/screenshots/match_predictor.png) | ![Simulator](docs/screenshots/simulator.png) |
-->

---

## Setup

Requires **Python 3.11+** (the dev interpreter is pinned to 3.12) and, ideally,
[`uv`](https://docs.astral.sh/uv/).

```bash
# With uv (recommended):
make install       # full dev env: pipeline + tests + lint + the Streamlit app
make install-min   # lean compute-only env (no Streamlit/dev tools) for a VM

# Without uv (any existing conda/system Python ≥ 3.11):
python -m pip install -e ".[dev]"   # or `-e .` for the lean install, `.[app]` for the dashboard
python -m pytest                    # pythonpath=src is preconfigured — no extra setup
```

Then drop the martj42 CSVs into `data/raw/` (`results.csv`, `shootouts.csv`,
`goalscorers.csv`).

## Example commands

The pipeline runs in stages; each logs its runtime, and `build-features` caches its
output (skips the rebuild when nothing upstream changed; `--force` to override).

```bash
make build-data        # clean raw results        -> data/interim/matches.parquet
make build-features    # leakage-safe feature table (attaches walk-forward Elo)
make train-baselines   # fit + score the baselines -> reports/baseline_metrics.json
make train-model       # calibrated match model    -> data/processed/model/ + metrics
make evaluate          # evaluation report + figures -> reports/
make backtest          # backtest 2014 / 2018 / 2022 -> reports/backtesting/
make simulate-quick    # Monte Carlo, 1,000 sims (~3.5s) -> reports/simulation/
make simulate-full     # Monte Carlo, 10,000 sims
make app               # launch the Streamlit dashboard

# Shortcuts:
make pipeline-quick    # whole chain on small samples (fast end-to-end smoke)
make pipeline-full     # whole chain at full size
make check             # the CI gate: ruff + black --check + pytest
```

No `make`? Every target is a thin wrapper — run the underlying `python
scripts/<name>.py` shown in each script's docstring (e.g.
`python scripts/run_simulation.py --quick`).

---

## Running on Azure (cost-controlled)

The whole project is CPU-only, single-machine, and seeded — **no GPU, no cluster**.
A cheap burstable VM is plenty; the priority is not leaving it running.

```bash
# 1. Provision a small VM (2 vCPU / 4 GB is enough; data is a few MB).
az group create -n worldcup-rg -l eastus
az vm create -g worldcup-rg -n worldcup-vm \
  --image Ubuntu2404 --size Standard_B2s \
  --admin-username azureuser --generate-ssh-keys

# 2. Lean setup + run the pipeline.
ssh azureuser@<vm-ip>
git clone <repo> && cd world-cup-ml-simulator
make install-min
make build-data build-features train-model evaluate simulate-quick
```

**Cost control — the part that actually matters:**

- **Deallocate when idle.** *Stopping* from inside the OS still bills for compute;
  `az vm deallocate -g worldcup-rg -n worldcup-vm` stops the meter (you keep only
  the small disk cost). `az vm start …` to resume.
- **Auto-shutdown** as a safety net: `az vm auto-shutdown -g worldcup-rg -n worldcup-vm --time 0200`.
- **Prefer quick mode** on the VM; run `make simulate-full` only for final figures.
- **Tear it all down** when done: `az group delete -n worldcup-rg`.

**View the dashboard without exposing it.** Keep the NSG closed (no inbound 8501),
bind Streamlit to localhost, and reach it over an SSH tunnel:

```bash
# On the VM:
streamlit run app/streamlit_app.py --server.headless true \
  --server.address 127.0.0.1 --server.port 8501
# On your laptop, then open http://localhost:8501 :
ssh -N -L 8501:127.0.0.1:8501 azureuser@<vm-ip>
```

---

## Limitations

I'd rather under-claim than oversell. What this system **does not** do:

- **It does not "predict the World Cup."** It estimates *calibrated probabilities*.
  A 20% title shot is genuinely uncertain, and a single tournament is one noisy
  sample.
- **The shipped simulation is structure-only.** Until the official 2026 draw is
  encoded, it runs over placeholder slots with a uniform predictor — so it does not
  output real per-team title odds (and the README/app say so loudly).
- **The model is deliberately simple.** Its edge over an Elo-only baseline is small.
  No player/squad data, injuries, lineups, or fatigue beyond rest days.
- **No FIFA ranking yet**, and the goal model is cosmetic — group goal-difference is
  sampled, not learned, so it shouldn't be read as a scoreline forecast.
- **The backtest champion-rank is a proxy** (Elo ordering, not a Monte-Carlo
  winner-probability run) over a small 3-tournament sample.

## Future work

- **Encode the official 2026 draw + Round-of-32 bracket lookup** → turn the
  simulation into a genuine title-odds forecast (the engine is already built).
- **Add the FIFA world ranking** as an as-of feature (V1.1) and re-evaluate.
- **A real goals model** (Poisson / bivariate-Poisson with a Dixon–Coles
  correction) for honest scorelines and goal-based tiebreakers.
- **Full FIFA tiebreakers** (head-to-head, fair-play) before the random draw.
- **Vectorize the Monte Carlo** to push 100k+ simulations and tighten interval
  estimates.
- **Richer features** (rolling Elo momentum, competition weighting, confederation
  effects) — added only if they beat the baseline on the temporal hold-out.

---

## Project structure

```
src/worldcup/
  config.py            # paths, seeds, YAML config loaders
  data/                # load, clean, validate, team-name canonicalization
  features/            # leakage-safe as-of features (rolling form, Elo diff)
  models/              # baselines, calibrated match model, train/eval/calibrate
  simulation/          # group stage, tiebreakers, knockout, Monte Carlo
  backtesting.py       # walk-forward Elo + past-tournament backtests
  visualization/       # plots for reports and the app
configs/               # tournament format, team-name map, model hyperparameters
scripts/               # CLI entry points (build, train, simulate, backtest)
app/streamlit_app.py   # the dashboard (loads saved outputs; no fabricated numbers)
reports/               # generated metrics JSON, figures, and Markdown reports
tests/                 # 251-test pytest suite (incl. a leakage test per feature)
data/{raw,interim,processed}/   # gitignored; datasets are downloaded locally
```
