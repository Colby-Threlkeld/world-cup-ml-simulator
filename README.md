# World Cup ML Simulator

<!-- Badge placeholders — replace OWNER/REPO once the repo is on GitHub. -->
[![CI](https://github.com/OWNER/REPO/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](pyproject.toml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000)](https://github.com/psf/black)
[![Lint: ruff](https://img.shields.io/badge/lint-ruff-261230)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Forecast international football match outcomes from 150+ years of results, then
simulate the **2026 FIFA World Cup** (48 teams, 12 groups) with Monte Carlo to
estimate each nation's title odds — all judged against honest baselines and
**backtested** on the 2014, 2018, and 2022 tournaments.

An interactive **Streamlit dashboard** ties it together: explore team strengths,
predict any match-up, browse simulated title odds, and read the backtest.

## Highlights

- **Leakage-safe by construction** — every feature is computed *as of kickoff*
  from prior matches only; all evaluation splits are temporal, never random.
- **Calibrated probabilities, not vibes** — a multinomial-logistic match model
  with isotonic/Platt calibration and reliability diagnostics.
- **Honest backtest** — trained only on pre-tournament data, the system ranked
  every eventual champion (Germany '14, France '18, Argentina '22) in its
  pre-tournament **top 5**, with mean log loss **0.97** vs **1.10** for an
  uninformed model.
- **Reproducible** — everything is seeded; 1,000 tournament simulations run in
  ~3s on 2 CPU cores. No GPU, tiny data.
- **No fabricated numbers** — if an output hasn't been generated, the app and
  reports say so and show the command to produce it.

## Dashboard

`make app` launches a six-page dashboard:

| Page | What it shows |
|------|----------------|
| **Overview** | Project summary, which artifacts are ready, top title contenders |
| **Team Explorer** | Current self-computed Elo strength + world rank per team |
| **Match Predictor** | Win/draw/loss probabilities for any Team A vs Team B (neutral toggle) |
| **Tournament Simulator** | Monte-Carlo title and advancement probabilities by team |
| **Backtesting** | Match metrics + champion-rank for 2014/2018/2022 |
| **Methodology** | Features, model, calibration curve, no-leakage doctrine |

### Screenshots

> _Placeholder — add images under `docs/screenshots/` and link them here._

| Overview | Match Predictor | Tournament Simulator |
|----------|-----------------|----------------------|
| _`docs/screenshots/overview.png`_ | _`docs/screenshots/match_predictor.png`_ | _`docs/screenshots/simulator.png`_ |

## Quickstart

Requires Python 3.11+ (pinned dev interpreter 3.12) and
[`uv`](https://docs.astral.sh/uv/).

```bash
make install     # full dev env: pipeline + tests + lint + the Streamlit app
make install-min # lean compute-only env (no Streamlit, no dev tools) for a VM
make test        # run the test suite (240+ tests)
make app         # launch the Streamlit dashboard
```

Without `uv` (e.g. an existing conda/system Python ≥ 3.11):

```bash
python -m pip install -e ".[dev]"     # or `-e .` for the lean compute-only install
python -m pytest                      # pythonpath=src is preconfigured
```

## Generate the data and outputs

The dashboard is **read-only**: it loads saved artifacts and never retrains a
model. Produce those artifacts with the pipeline (datasets live in the gitignored
`data/`). Every stage logs its runtime, and `build-features` **caches** its output
(it skips the rebuild when nothing upstream changed; `--force` to override).

```bash
make build-data        # clean raw results -> data/interim/matches.parquet
make build-features    # leakage-safe feature table (cached)
make train-baselines   # baseline models + metrics
make train-model       # calibrated match model -> data/processed/model/ artifacts
make evaluate          # evaluation report + figures
make simulate-quick    # Monte-Carlo title odds (1,000 sims, ~3s)
make simulate-full     # full Monte-Carlo run (10,000 sims)
make backtest          # backtest 2014 / 2018 / 2022
make app               # launch the dashboard
```

Shortcuts: **`make pipeline-quick`** runs the whole chain on small samples (a fast
end-to-end smoke), and **`make pipeline-full`** runs it at full size. Each app page
that lacks its artifact shows the exact command to generate it.

### Quick vs full mode

| | Quick | Full |
|--|-------|------|
| Simulation | `make simulate-quick` (1,000 sims, ~3s) | `make simulate-full` (10,000 sims) |
| Training | `... --sample 5000` (seconds) | `make train-model` (~all 49k matches) |
| Whole pipeline | `make pipeline-quick` | `make pipeline-full` |

Use **quick** mode while iterating or on a small/idle VM; switch to **full** only
for the final numbers. Monte-Carlo estimates tighten with more simulations, so
quote full-mode figures in any report.

## Running on Azure (cost-controlled)

The whole project is CPU-only, single-machine, and seeded — there is **no GPU and
no cluster**. A cheap burstable VM is plenty; the priority is not leaving it
running.

**1. Provision a small VM**

```bash
az group create -n worldcup-rg -l eastus
az vm create -g worldcup-rg -n worldcup-vm \
  --image Ubuntu2404 --size Standard_B2s \
  --admin-username azureuser --generate-ssh-keys
```

`Standard_B2s` (2 vCPU / 4 GB) handles the full pipeline; the data is a few MB and
fits in RAM. Use a **Standard SSD**, not Premium.

**2. Set up the lean environment**

```bash
ssh azureuser@<vm-ip>
git clone <repo> && cd world-cup-ml-simulator
make install-min          # core pipeline only (no Streamlit) — smaller, faster
make build-data build-features train-model evaluate simulate-quick
```

Add `.[app]` (or `make install`) only on the box that actually hosts the dashboard.

**3. Cost control — the important part**

- **Auto-shutdown** (set it once; the VM powers off daily even if you forget):
  ```bash
  az vm auto-shutdown -g worldcup-rg -n worldcup-vm --time 0200
  ```
- **Deallocate when idle** — *stopping* from inside the OS still bills for compute;
  **deallocate** from Azure to stop the meter (you keep only the small disk cost):
  ```bash
  az vm deallocate -g worldcup-rg -n worldcup-vm   # stops compute billing
  az vm start      -g worldcup-rg -n worldcup-vm   # resume later
  ```
- **Prefer quick mode** on the VM; run `make simulate-full` only when you need the
  final figures. The cached feature table avoids paying to rebuild it each run.
- **Tear down** the whole thing when done: `az group delete -n worldcup-rg`.

**4. View the dashboard safely (no public exposure)**

Keep the network security group closed (no inbound 8501). Bind Streamlit to
localhost on the VM and reach it through an SSH tunnel:

```bash
# On the VM:
streamlit run app/streamlit_app.py \
  --server.headless true --server.address 127.0.0.1 --server.port 8501

# On your laptop, forward the port, then open http://localhost:8501:
ssh -N -L 8501:127.0.0.1:8501 azureuser@<vm-ip>
```

## Project structure

```
src/worldcup/
  config.py            # paths, seeds, YAML config loaders
  data/                # load, clean, validate, team-name canonicalization
  features/            # leakage-safe as-of-date features (rolling form, Elo diff)
  models/              # baselines, calibrated match model, train/eval/calibrate
  simulation/          # group stage, tiebreakers, knockout, Monte Carlo
  backtesting.py       # walk-forward Elo + past-tournament backtests
  visualization/       # plots for reports and the app
configs/               # tournament format, team-name map, model hyperparameters
scripts/               # CLI entry points (build, train, simulate, backtest)
app/streamlit_app.py   # the dashboard (loads saved outputs; no fabricated numbers)
reports/               # generated metrics JSON, figures, and Markdown reports
tests/                 # pytest suite
data/{raw,interim,processed}/   # gitignored; datasets are downloaded locally
```

## Data

Datasets are **not** committed (`data/` is gitignored). The spine is martj42's
`international_results` (match results, 1872–present), pulled into `data/raw/`.
`scripts/audit_data.py` profiles every CSV before anything is built on it. The
FIFA men's ranking is a deferred V1.1 enhancement (needs Kaggle credentials; see
`.env.example`) and is not required.

## License

MIT.
