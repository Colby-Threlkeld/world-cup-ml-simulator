# World Cup ML Simulator — common developer + pipeline commands.
# Uses `uv` (https://docs.astral.sh/uv/). On Windows without `make`, run the
# underlying `python scripts/...` commands shown in the README instead.
#
# Pipeline order:  build-data -> build-features -> train-baselines -> train-model
#                  -> evaluate -> simulate-quick|simulate-full -> backtest -> app
# `make pipeline-quick` runs the whole chain fast; `make pipeline-full` runs it
# at full size. All stages log their runtime; build-features caches its output.

.PHONY: install install-min test lint format typecheck clean \
        build-data build-features train-baselines train-model evaluate \
        simulate-quick simulate-full backtest app \
        pipeline-quick pipeline-full

PY := uv run python

## install: full dev environment (pipeline + tests + lint + the Streamlit app).
install:
	uv venv
	uv pip install -e ".[dev]"

## install-min: lean compute-only install (no Streamlit, no dev tools) for a VM.
install-min:
	uv venv
	uv pip install -e .

## test: run the test suite.
test:
	uv run pytest

## lint: check style and imports without modifying files.
lint:
	uv run ruff check src tests
	uv run black --check src tests

## format: auto-fix lint issues and format the code.
format:
	uv run ruff check --fix src tests
	uv run black src tests

## typecheck: run static type checking.
typecheck:
	uv run mypy src

# --- data + model pipeline --------------------------------------------------

## build-data: clean raw results into data/interim/matches.parquet.
build-data:
	$(PY) scripts/build_matches.py

## build-features: build the leakage-safe feature table (cached; --force to rebuild).
build-features:
	$(PY) scripts/build_features.py --matches data/interim/matches.parquet

## train-baselines: fit and score the baseline models.
train-baselines:
	$(PY) scripts/train_baselines.py

## train-model: train + calibrate the main model, saving artifacts.
train-model:
	$(PY) scripts/train_model.py

## evaluate: generate the evaluation report (metrics JSON, figures, markdown).
evaluate:
	$(PY) scripts/generate_evaluation_report.py

## simulate-quick: fast Monte-Carlo run (1,000 simulations, ~3s).
simulate-quick:
	$(PY) scripts/run_simulation.py --quick

## simulate-full: full Monte-Carlo run (10,000 simulations).
simulate-full:
	$(PY) scripts/run_simulation.py --simulations 10000

## backtest: backtest the model against the 2014/2018/2022 World Cups.
backtest:
	$(PY) scripts/run_backtest.py

## app: launch the Streamlit dashboard (read-only; never retrains).
app:
	uv run streamlit run app/streamlit_app.py

# --- convenience chains -----------------------------------------------------

## pipeline-quick: end-to-end on small samples (smoke test the whole flow).
pipeline-quick: build-data
	$(PY) scripts/build_features.py --matches data/interim/matches.parquet
	$(PY) scripts/train_baselines.py --sample 5000
	$(PY) scripts/train_model.py --sample 5000
	$(PY) scripts/generate_evaluation_report.py
	$(PY) scripts/run_simulation.py --quick

## pipeline-full: end-to-end at full size (then run `make app` to view).
pipeline-full: build-data build-features train-baselines train-model evaluate simulate-full backtest

## clean: remove caches and build artifacts (not data or reports).
clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov build dist *.egg-info
