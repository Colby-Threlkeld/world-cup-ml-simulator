# World Cup ML Simulator

Predict 2026 FIFA World Cup match-outcome probabilities from historical
international football data, then simulate the full 48-team tournament with Monte
Carlo to estimate each nation's title odds.

> **Status:** under active construction, built one vertical slice at a time.
> No predictions are published until real data has been processed end-to-end.
> The current build is **slice 0** (scaffold + tooling).

## Why this project

International football forecasting is a clean showcase for a real ML workflow:
leakage-safe feature engineering, probability calibration, proper scoring rules,
and Monte Carlo simulation — judged against an honest baseline rather than vibes.

## Approach

- **Match model:** a Poisson goals model (expected goals per side) so simulated
  scorelines resolve knockout extra-time and penalties naturally.
- **Baseline first:** an Elo model is the slice-2 baseline; every later model
  must beat its Brier / log-loss score on a *temporal* hold-out.
- **No leakage, enforced structurally:** every feature is computed as-of kickoff
  from prior matches only, and all evaluation splits are by time — never random
  K-fold.
- **Probabilities, not certainties:** calibrated outputs with reliability
  diagnostics.

## Build order (vertical slices)

| # | Slice | Status |
|---|-------|--------|
| 0 | Scaffold + tooling | ✅ done |
| 1 | Data ingestion (international results + FIFA rankings) | ⬜ next |
| 2 | Elo baseline + temporal backtest | ⬜ |
| 3 | Leakage-safe feature engineering | ⬜ |
| 4 | Poisson goals match model | ⬜ |
| 5 | Probability calibration + evaluation | ⬜ |
| 6 | 2026 Monte Carlo tournament simulation | ⬜ |
| 7 | Reporting + Streamlit app | ⬜ |

## Project structure

```
src/worldcup/
  config.py            # paths, seeds, YAML config loaders
  data/                # load, clean, validate, team-name canonicalization
  features/            # leakage-safe as-of-date features (rolling form, etc.)
  models/              # Elo baseline, Poisson model, train/eval/calibrate
  simulation/          # group stage, tiebreakers, knockout, Monte Carlo
  visualization/       # plots for reports and the app
configs/               # tournament format, team-name map, model hyperparameters
notebooks/             # throwaway exploration only (never imported)
app/streamlit_app.py   # front-end (no fabricated numbers)
tests/                 # pytest suite
data/{raw,interim,processed}/   # gitignored; datasets are downloaded locally
```

## Quickstart (local)

Requires Python 3.11+ and [`uv`](https://docs.astral.sh/uv/). The pinned dev
interpreter is 3.12 (see `.python-version`).

```bash
make install   # create .venv and install the package + dev tools
make test      # run the test suite
make lint      # ruff + black --check
make format    # auto-fix + format
make app       # launch the Streamlit app
```

### Windows / without `uv`

With `uv` installed, run the underlying commands directly:

```powershell
uv venv
uv pip install -e ".[dev]"
uv run pytest
```

Without `uv` (e.g. an existing conda/system Python ≥ 3.11) — this installs the
same dev tools from the `dev` extra:

```powershell
python -m pip install -e ".[dev]"
python -m pytest            # then: ruff check src tests  ·  mypy src
```

For a quick test run with nothing installed, `python -m pytest` works on its own
(`pythonpath = src` is set in `pyproject.toml`).

## Azure VM (optional, deferred)

The early slices are cloud-agnostic and run on a laptop. A cheap Azure
**Burstable B2s** Ubuntu 24.04 VM is the planned remote option once compute
justifies it (large Monte Carlo runs, or hosting the app). Rules of thumb:
**deallocate** when idle, set **auto-shutdown**, use a **Standard SSD**, and
**tunnel** Streamlit over SSH instead of exposing it publicly.

## Data

Datasets are **not** committed (`data/` is gitignored except `.gitkeep`). The V1
spine is martj42's `international_results`, pulled from GitHub raw into
`data/raw/` (no credentials needed):

- `results.csv` — international match results, 1872–present (the spine)
- `shootouts.csv`, `goalscorers.csv` — companions (V2 features)

The FIFA men's ranking (a **V1.1** enhancement) is the only source needing Kaggle
credentials — see `.env.example`. It is not required for V1.

### Audit the raw data

Before building anything on a dataset, profile it. `scripts/audit_data.py` reads
every CSV in `data/raw/` and prints row counts, columns and dtypes, missing-value
counts, duplicate rows, date ranges, the top teams, and sample rows — so you
understand the real data instead of assuming its shape.

```bash
python scripts/audit_data.py            # audits every CSV in data/raw/
python scripts/audit_data.py --dir x/   # audit CSVs in another directory
```

It only needs `pandas` and is intentionally decoupled from the `worldcup`
package, so it runs before any install. It reads only — it never writes or
cleans data.

## License

MIT.
