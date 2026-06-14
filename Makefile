# World Cup ML Simulator — common developer commands.
# Uses `uv` (https://docs.astral.sh/uv/). On Windows without `make`, run the
# underlying commands shown in the README instead.

.PHONY: install test lint format typecheck app clean

## install: create the venv and install the package (editable) with dev tools.
install:
	uv venv
	uv pip install -e ".[dev]"

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

## app: launch the Streamlit front-end.
app:
	uv run streamlit run app/streamlit_app.py

## clean: remove caches and build artifacts.
clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov build dist *.egg-info
