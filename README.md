# World Cup ML Simulator

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
make install   # create .venv and install the package + dev tools
make test      # run the test suite (240+ tests)
make app       # launch the Streamlit dashboard
```

Without `uv` (e.g. an existing conda/system Python ≥ 3.11):

```bash
python -m pip install -e ".[dev]"
python -m pytest                              # pythonpath=src is preconfigured
streamlit run app/streamlit_app.py            # launch the app
```

## Generate the data and outputs

The dashboard is **read-only**: it loads saved artifacts and never retrains a
model. Produce those artifacts with the pipeline scripts (datasets live in the
gitignored `data/`):

```bash
python scripts/build_matches.py              # clean raw results -> data/interim/matches.parquet
python scripts/build_features.py             # leakage-safe feature matrix
python scripts/train_model.py                # calibrated match model + metrics
python scripts/generate_evaluation_report.py # evaluation report + figures
python scripts/run_simulation.py --quick     # Monte-Carlo title odds (1,000 sims, ~3s)
python scripts/run_backtest.py               # backtest 2014/2018/2022
```

Each app page that lacks its artifact shows the exact command above.

## Run the app

**Locally**

```bash
make app
# or:
streamlit run app/streamlit_app.py
# then open http://localhost:8501
```

**On a headless Azure VM (safely)**

Don't expose Streamlit to the public internet. Bind it to localhost on the VM and
reach it through an SSH tunnel:

```bash
# On the VM (a cheap Burstable B2s, Ubuntu 24.04 is plenty):
streamlit run app/streamlit_app.py \
  --server.headless true --server.address 127.0.0.1 --server.port 8501

# On your laptop, forward the port over SSH, then open http://localhost:8501:
ssh -N -L 8501:127.0.0.1:8501 <user>@<vm-ip>
```

VM hygiene: enable **auto-shutdown**, **deallocate** when idle, and keep the NSG
closed (no inbound 8501) — the SSH tunnel is the only path in. The app is
CPU-only and needs no GPU.

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
