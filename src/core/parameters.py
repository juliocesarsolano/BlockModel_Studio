"""
PV BlockModel Studio
====================

Centralized parameter-loading utilities for the Streamlit block-model
evaluation, comparison and reporting application.

Project
-------
PV BlockModel Studio

Module
------
src/core/parameters.py

Purpose
-------
This module provides one controlled access point for the JSON configuration
files stored in the project ``parameters`` directory. It exposes stable helper
functions used by ``pages.py`` and the rest of the application to retrieve:

- Application display defaults.
- Data-cleaning defaults.
- Validation rules.
- Master-filter definitions.
- Column aliases.
- Grade-variable defaults.
- Planning-period definitions.

Author
------
Julio Solano
Geological Engineer | Mineral Resource Evaluation | Geostatistics |
Data Science, Data Analytics, GIS and Mining Applications

Organization / Context
----------------------
Mineral Resource Management – Pueblo Viejo

Version
-------
1.1

Year
----
2026

Documentation and logic review
------------------------------
2026-07-16

Copyright
---------
© 2026 Julio Solano. All rights reserved.

Compatibility principles
------------------------
- Public function names and return types are preserved for compatibility with
  ``pages.py`` and the complete Streamlit application.
- Missing, unreadable or malformed parameter files continue to return empty
  dictionaries so code-level defaults remain operational.
- Valid existing JSON configuration files retain the same behavior.
- Defensive checks prevent malformed JSON structures from propagating errors
  into the user interface.
- Parameter files are loaded only from the project ``parameters`` directory.

Maintenance notes
-----------------
- Keep business defaults in the JSON files whenever possible.
- Keep this module focused on loading, validating and exposing configuration.
- Call ``clear_parameter_caches`` after changing JSON files during a running
  Python process; a normal Streamlit/application restart also refreshes them.
"""

from __future__ import annotations

# =============================================================================
# Python standard-library imports
# =============================================================================
# ``json`` decodes the external parameter files.
# ``lru_cache`` avoids reading the same small configuration files on every
# Streamlit rerun.
# ``Path`` provides platform-independent filesystem path handling.
# ``Any`` is used because JSON values can contain mixed scalar, list and mapping
# types.
import json
from functools import lru_cache
from pathlib import Path
from typing import Any


# =============================================================================
# Project paths
# =============================================================================
# This module is expected at ``src/core/parameters.py``. Moving two parent
# levels upward therefore resolves the project root:
#
#     project_root/
#         parameters/
#         src/
#             core/
#                 parameters.py
ROOT = Path(__file__).resolve().parents[2]

# All external configuration files are read from this directory. Keeping one
# canonical path avoids duplicated path-building logic throughout the app.
PARAMETERS_DIR = ROOT / "parameters"


# =============================================================================
# Code-level fallbacks
# =============================================================================
# These definitions preserve the application behavior when ``app_defaults.json``
# is absent, malformed or does not contain a valid ``periods`` collection.
#
# A tuple is used internally so the module-level fallback cannot be mutated
# accidentally. ``period_definitions`` returns fresh dictionaries to callers.
_DEFAULT_PERIODS: tuple[tuple[str, int | None], ...] = (
    ("2Y", 2),
    ("5Y", 5),
    ("10Y", 10),
    ("LOM", None),
)


# =============================================================================
# Internal validation helpers
# =============================================================================
def _parameter_path(filename: str) -> Path | None:
    """Return a safe resolved path inside ``PARAMETERS_DIR``.

    Normal application calls use fixed filenames such as
    ``app_defaults.json``. The containment check is a defensive safeguard that
    prevents absolute paths or ``..`` path traversal from escaping the project
    parameter directory.

    Nested paths inside ``parameters`` remain supported.
    """

    try:
        parameter_root = PARAMETERS_DIR.resolve()
        candidate = (parameter_root / str(filename)).resolve()
        candidate.relative_to(parameter_root)
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    return candidate


def _mapping_section(section_name: str) -> dict[str, Any]:
    """Return one dictionary section from ``app_defaults.json``.

    A malformed section, for example a list where a JSON object is expected,
    is treated as unavailable. This preserves the public dictionary return type
    expected by ``pages.py``.
    """

    section = app_defaults().get(section_name, {})
    return section if isinstance(section, dict) else {}


def _default_period_definitions() -> list[dict[str, Any]]:
    """Build a fresh mutable copy of the standard period definitions."""

    return [
        {"label": label, "years": years}
        for label, years in _DEFAULT_PERIODS
    ]


# =============================================================================
# Generic JSON parameter loader
# =============================================================================
@lru_cache(maxsize=None)
def load_parameter_file(filename: str) -> dict[str, Any]:
    """Load one JSON object from the project ``parameters`` directory.

    Parameters
    ----------
    filename:
        Parameter filename, or a nested relative path, located under the
        project ``parameters`` directory.

    Returns
    -------
    dict[str, Any]
        Parsed JSON object. An empty dictionary is returned when the path is
        invalid, the file does not exist, the file cannot be read, the JSON is
        malformed, or the JSON root is not an object.

    Notes
    -----
    Returning ``{}`` instead of raising an exception is intentional. The
    application already provides code-level fallbacks and must remain usable
    when an optional parameter file is unavailable.
    """

    path = _parameter_path(filename)
    if path is None or not path.is_file():
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        # Preserve the existing fail-soft behavior for unavailable or malformed
        # external configuration.
        return {}

    # The public contract of this module is dictionary-based. Rejecting a valid
    # JSON list/scalar here prevents later ``.get(...)`` attribute errors.
    return payload if isinstance(payload, dict) else {}


# =============================================================================
# Cached top-level configuration files
# =============================================================================
@lru_cache(maxsize=1)
def app_defaults() -> dict[str, Any]:
    """Return the main application defaults from ``app_defaults.json``."""

    return load_parameter_file("app_defaults.json")


@lru_cache(maxsize=1)
def column_aliases() -> dict[str, Any]:
    """Return variable-name aliases and role-detection configuration."""

    return load_parameter_file("column_aliases.json")


@lru_cache(maxsize=1)
def grade_defaults() -> dict[str, Any]:
    """Return grade-variable candidates, labels and default unit rules."""

    return load_parameter_file("grade_defaults.json")


# =============================================================================
# Typed sections from app_defaults.json
# =============================================================================
def display_defaults() -> dict[str, Any]:
    """Return display, formatting and interface defaults."""

    return _mapping_section("display")


def cleaning_defaults() -> dict[str, Any]:
    """Return data-cleaning and exclusion defaults."""

    return _mapping_section("cleaning")


def validation_defaults() -> dict[str, Any]:
    """Return validation ranges, policies and accepted-value defaults."""

    return _mapping_section("validation")


def master_filter_defaults() -> dict[str, Any]:
    """Return defaults used by transversal application filters."""

    return _mapping_section("master_filters")


# =============================================================================
# Planning-period configuration
# =============================================================================
def period_definitions() -> list[dict[str, Any]]:
    """Return configured planning periods or the standard application fallback.

    Only dictionary entries are returned. This defensive filtering prevents one
    malformed item in the external JSON list from breaking period selectors in
    the application. Extra keys on valid period dictionaries are preserved.
    """

    periods = app_defaults().get("periods", [])
    if not isinstance(periods, list):
        return _default_period_definitions()

    valid_periods = [
        dict(item)
        for item in periods
        if isinstance(item, dict)
    ]
    return valid_periods or _default_period_definitions()


def period_labels() -> list[str]:
    """Return non-empty period labels in configured display order."""

    labels: list[str] = []
    for item in period_definitions():
        label = item.get("label")
        if label is None:
            continue
        text = str(label)
        if text:
            labels.append(text)
    return labels


def period_year_count(label: str) -> int | None:
    """Return the number of years represented by a planning-period label.

    ``None`` represents Life of Mine (LOM) or an unrecognized/malformed label.
    Configured definitions are checked first. Labels such as ``2Y`` or ``10Y``
    are then parsed as a backward-compatible fallback.
    """

    for item in period_definitions():
        if item.get("label") == label:
            years = item.get("years")
            if years is None:
                return None
            try:
                return int(years)
            except (TypeError, ValueError):
                return None

    normalized_label = str(label).strip().upper()
    if normalized_label == "LOM":
        return None

    try:
        return int(normalized_label.removesuffix("Y"))
    except (TypeError, ValueError):
        return None


# =============================================================================
# Cache maintenance
# =============================================================================
def clear_parameter_caches() -> None:
    """Clear all in-process parameter caches.

    This helper is primarily useful in development, automated tests or custom
    refresh workflows after editing JSON files without restarting Python.
    Normal application execution does not need to call it.
    """

    load_parameter_file.cache_clear()
    app_defaults.cache_clear()
    column_aliases.cache_clear()
    grade_defaults.cache_clear()
