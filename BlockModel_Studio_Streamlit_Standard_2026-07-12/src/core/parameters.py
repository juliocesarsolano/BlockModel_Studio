from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PARAMETERS_DIR = ROOT / "parameters"


@lru_cache(maxsize=None)
def load_parameter_file(filename: str) -> dict[str, Any]:
    """Load a JSON parameter file from the project parameters folder.

    Missing or malformed parameter files return an empty dictionary so the app
    can still run with code-level fallbacks.
    """
    path = PARAMETERS_DIR / filename
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


@lru_cache(maxsize=1)
def app_defaults() -> dict[str, Any]:
    return load_parameter_file("app_defaults.json")


@lru_cache(maxsize=1)
def column_aliases() -> dict[str, Any]:
    return load_parameter_file("column_aliases.json")


@lru_cache(maxsize=1)
def grade_defaults() -> dict[str, Any]:
    return load_parameter_file("grade_defaults.json")


def display_defaults() -> dict[str, Any]:
    return app_defaults().get("display", {})


def cleaning_defaults() -> dict[str, Any]:
    return app_defaults().get("cleaning", {})


def validation_defaults() -> dict[str, Any]:
    return app_defaults().get("validation", {})


def master_filter_defaults() -> dict[str, Any]:
    return app_defaults().get("master_filters", {})


def period_definitions() -> list[dict[str, Any]]:
    periods = app_defaults().get("periods", [])
    if not periods:
        return [
            {"label": "2Y", "years": 2},
            {"label": "5Y", "years": 5},
            {"label": "10Y", "years": 10},
            {"label": "LOM", "years": None},
        ]
    return periods


def period_labels() -> list[str]:
    return [str(item.get("label")) for item in period_definitions() if item.get("label")]


def period_year_count(label: str) -> int | None:
    for item in period_definitions():
        if item.get("label") == label:
            years = item.get("years")
            return None if years is None else int(years)
    if label == "LOM":
        return None
    try:
        return int(label.upper().removesuffix("Y"))
    except ValueError:
        return None
