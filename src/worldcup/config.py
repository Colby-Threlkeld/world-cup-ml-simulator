"""Project paths and configuration loading.

Centralizes the filesystem layout and YAML config access so no other module
hardcodes paths. Import these helpers instead of building paths by hand.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Repo layout: this file is src/worldcup/config.py, so the project root is two
# parents up from the package directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
CONFIGS_DIR = PROJECT_ROOT / "configs"

# Single source of truth for reproducibility. Pass this into every RNG.
RANDOM_SEED = 42


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed contents, or an empty dict if the file is empty.
    """
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def tournament_config() -> dict[str, Any]:
    """Return the 2026 tournament format configuration."""
    return load_yaml(CONFIGS_DIR / "tournament_2026.yaml")


def model_config() -> dict[str, Any]:
    """Return the model hyperparameter configuration."""
    return load_yaml(CONFIGS_DIR / "model_config.yaml")


def team_name_map_path() -> Path:
    """Return the path to the team-name alias map config."""
    return CONFIGS_DIR / "team_name_map.yaml"
