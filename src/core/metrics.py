"""
PV BlockModel Studio
====================

Core calculation and formatting utilities for block-model tonnage, volume,
grade, contained-metal and planning-period summaries.

Project
-------
PV BlockModel Studio

Module
------
src/core/metrics.py

Purpose
-------
This module centralizes the numerical operations used throughout the
Streamlit application, including:

- Tonnage and volume totals.
- Tonnage-unit conversion.
- Tonnage-weighted grade calculations.
- Contained-metal calculations.
- Grouped resource summaries.
- Basic descriptive statistics.
- Negative, null and zero diagnostics.
- Categorical and planning-period filters.
- Display formatting for tonnage and grade values.

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
- Public constants, function names, arguments and return types are preserved.
- The module remains compatible with ``pages.py`` and the complete Streamlit
  application.
- Normal calculations on already-cleaned model data retain the same results.
- Defensive numeric conversion is applied only to prevent malformed values
  from crashing summaries.
- Missing configured columns continue to produce empty tables, ``NaN`` values
  or ``None`` results, depending on the existing public function contract.

Calculation conventions
-----------------------
- Tonnage-weighted mean:
  sum(value × tonnes) / sum(tonnes), using strictly positive tonnes.
- Au and Ag contained metal for g/t or ppm:
  sum(grade × tonnes) / 31.1034768, reported in troy ounces.
- Percent-grade contained metal:
  sum(grade × tonnes) / 100, reported in metric tonnes.
- Negative mass and volume values are excluded from totals by clipping them to
  zero, consistent with the application's validation and reporting workflow.

Maintenance notes
-----------------
- Keep shared calculations here rather than duplicating formulas in UI pages.
- Preserve the troy-ounce conversion factor unless the application standard is
  formally changed.
- Keep ``PERIODS`` available at module level because ``pages.py`` imports it.
- Business definitions for periods remain controlled by ``parameters.py`` and
  the corresponding JSON parameter files.
"""

from __future__ import annotations

# =============================================================================
# Python standard-library imports
# =============================================================================
# ``Iterable`` and ``Sequence`` describe flexible filter and grouping inputs.
# ``Any`` is required because category values may be strings, numbers or mixed.
from collections.abc import Iterable, Sequence
from typing import Any

# =============================================================================
# Third-party numerical libraries
# =============================================================================
# NumPy provides weighted averaging and NaN constants.
# pandas provides tabular filtering, grouping and summary operations.
import numpy as np
import pandas as pd

# =============================================================================
# Internal project dependencies
# =============================================================================
# ``GradeSpec`` and ``ModelConfig`` define the configured model schema.
# Period labels and year counts are read through the centralized parameter
# layer to keep UI and calculation logic synchronized.
from .models import GradeSpec, ModelConfig
from .parameters import period_labels, period_year_count


# =============================================================================
# Display-unit and planning-period constants
# =============================================================================
# Divisors convert raw metric tonnes into the display unit selected by the user.
# Both ``Kt`` and ``kt`` are accepted for backward compatibility.
TONNAGE_DIVISORS = {
    "t": 1.0,
    "Kt": 1_000.0,
    "kt": 1_000.0,
    "Mt": 1_000_000.0,
}

# ``PERIODS`` is intentionally evaluated at import time because other modules,
# including ``pages.py``, import this public constant directly.
PERIODS = period_labels()


# =============================================================================
# Internal numeric helpers
# =============================================================================
def _numeric_series(series: pd.Series) -> pd.Series:
    """Return a numeric representation of a series with invalid values as NaN.

    The application normally supplies cleaned numeric columns. This helper adds
    a defensive layer for malformed strings without changing valid calculations.
    """

    return pd.to_numeric(series, errors="coerce")


def _aligned_numeric_pair(
    values: pd.Series,
    weights: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Align two series by index and convert both to numeric values."""

    aligned = pd.concat(
        [
            _numeric_series(values).rename("__value__"),
            _numeric_series(weights).rename("__weight__"),
        ],
        axis=1,
        join="inner",
    )
    return aligned["__value__"], aligned["__weight__"]


# =============================================================================
# Unit conversion and display labels
# =============================================================================
def tonnage_divisor(unit: str) -> float:
    """Return the divisor used to convert raw tonnes into a display unit.

    Unknown units retain the historical application fallback of one million,
    equivalent to displaying values in Mt.
    """

    return TONNAGE_DIVISORS.get(unit, 1_000_000.0)


def display_tonnage(tonnes: float, config: ModelConfig) -> float:
    """Convert a raw-tonnage value into the unit configured for the model."""

    return float(tonnes) / tonnage_divisor(config.tonnage_unit)


def tonnage_column_name(config: ModelConfig) -> str:
    """Return the standardized tonnage column heading used in app tables."""

    return f"Tonnage ({config.tonnage_unit})"


# =============================================================================
# Weighted grade and total calculations
# =============================================================================
def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    """Calculate a tonnage-weighted mean using strictly positive weights.

    Null, non-numeric and non-positive weight records are excluded. ``NaN`` is
    returned when no valid weighted observations remain.
    """

    numeric_values, numeric_weights = _aligned_numeric_pair(values, weights)
    valid = (
        numeric_values.notna()
        & numeric_weights.notna()
        & numeric_weights.gt(0)
    )
    if not valid.any():
        return float("nan")

    return float(
        np.average(
            numeric_values.loc[valid].astype(float),
            weights=numeric_weights.loc[valid].astype(float),
        )
    )


def total_tonnage(data: pd.DataFrame, config: ModelConfig) -> float:
    """Return non-negative total model tonnage in raw metric tonnes."""

    if config.mass_column not in data.columns:
        return float("nan")

    mass = _numeric_series(data[config.mass_column]).fillna(0.0).clip(lower=0)
    return float(mass.sum())


def total_volume(data: pd.DataFrame, config: ModelConfig) -> float:
    """Return non-negative total model volume in the source volume unit."""

    if not config.volume_column or config.volume_column not in data.columns:
        return float("nan")

    volume = _numeric_series(data[config.volume_column]).fillna(0.0).clip(lower=0)
    return float(volume.sum())


def contained_metal(
    values: pd.Series,
    tonnes: pd.Series,
    unit: str,
) -> tuple[float, str] | None:
    """Calculate contained metal from grade and tonnage.

    Parameters
    ----------
    values:
        Grade values.
    tonnes:
        Corresponding block or record tonnes.
    unit:
        Configured grade unit.

    Returns
    -------
    tuple[float, str] | None
        ``(ounces, "oz")`` for g/t or ppm grades;
        ``(metric_tonnes, "t")`` for percent grades;
        ``None`` for unsupported units or when no valid records exist.
    """

    numeric_values, numeric_tonnes = _aligned_numeric_pair(values, tonnes)
    valid = (
        numeric_values.notna()
        & numeric_tonnes.notna()
        & numeric_tonnes.gt(0)
    )
    if not valid.any():
        return None

    product = float(
        (
            numeric_values.loc[valid].astype(float)
            * numeric_tonnes.loc[valid].astype(float)
        ).sum()
    )

    if unit in {"g/t", "ppm"}:
        return product / 31.1034768, "oz"
    if unit == "%":
        return product / 100.0, "t"
    return None


# =============================================================================
# Model-level summaries
# =============================================================================
def model_summary(
    data: pd.DataFrame,
    config: ModelConfig,
) -> dict[str, float]:
    """Build one model-level summary dictionary.

    Every configured grade retains its expected summary key. When a configured
    grade or mass column is unavailable, the corresponding result is ``NaN``
    rather than an application-stopping ``KeyError``.
    """

    summary: dict[str, float] = {
        "Rows": float(len(data)),
        "Tonnage": total_tonnage(data, config),
    }

    if config.volume_column:
        summary["Volume"] = total_volume(data, config)

    mass_available = config.mass_column in data.columns
    weights = (
        data[config.mass_column]
        if mass_available
        else pd.Series(index=data.index, dtype=float)
    )

    for spec in config.grade_specs:
        grade_key = f"{spec.label} ({spec.unit})"

        if spec.column not in data.columns or not mass_available:
            summary[grade_key] = float("nan")
            continue

        summary[grade_key] = weighted_mean(data[spec.column], weights)

        metal = contained_metal(data[spec.column], weights, spec.unit)
        if metal:
            value, metal_unit = metal
            summary[f"{spec.label} contained ({metal_unit})"] = value

    return summary


def grouped_summary(
    data: pd.DataFrame,
    config: ModelConfig,
    group_columns: str | Sequence[str],
    include_volume: bool = True,
) -> pd.DataFrame:
    """Summarize tonnage, volume and weighted grades by one or more categories.

    Group labels are rendered through ``ModelConfig.category_label`` so tables
    remain consistent with user-configured display names.
    """

    if isinstance(group_columns, str):
        group_columns = [group_columns]

    valid_group_columns = [
        column
        for column in group_columns
        if column in data.columns
    ]
    if not valid_group_columns or data.empty:
        return pd.DataFrame()

    # Tonnage is essential to all grouped resource calculations.
    if config.mass_column not in data.columns:
        return pd.DataFrame()

    source = data.dropna(subset=valid_group_columns).copy()
    if source.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    grouped = source.groupby(
        valid_group_columns,
        sort=True,
        dropna=False,
        observed=True,
    )

    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)

        row: dict[str, Any] = {}

        for column, value in zip(valid_group_columns, keys, strict=True):
            label = config.category_label(column)

            # Integer-valued floating categories such as 2026.0 are displayed
            # as 2026 to avoid unwanted decimals in app tables.
            if (
                pd.notna(value)
                and isinstance(value, (float, np.floating))
                and float(value).is_integer()
            ):
                value = int(value)

            row[label] = value

        mass = _numeric_series(group[config.mass_column]).fillna(0.0).clip(lower=0)
        row[tonnage_column_name(config)] = (
            float(mass.sum()) / tonnage_divisor(config.tonnage_unit)
        )

        if (
            include_volume
            and config.volume_column
            and config.volume_column in group.columns
        ):
            volume = (
                _numeric_series(group[config.volume_column])
                .fillna(0.0)
                .clip(lower=0)
            )
            row[f"Volume ({config.volume_unit})"] = float(volume.sum())

        for spec in config.grade_specs:
            grade_key = f"{spec.label} ({spec.unit})"
            row[grade_key] = (
                weighted_mean(group[spec.column], group[config.mass_column])
                if spec.column in group.columns
                else float("nan")
            )

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# Descriptive statistics and diagnostics
# =============================================================================
def basic_stats(
    data: pd.DataFrame,
    config: ModelConfig,
) -> pd.DataFrame:
    """Return descriptive statistics for configured numeric model variables."""

    columns = [config.mass_column]

    if config.volume_column:
        columns.append(config.volume_column)

    columns.extend(config.grade_columns)
    columns = [column for column in columns if column in data.columns]

    if not columns:
        return pd.DataFrame()

    # Defensive numeric conversion keeps the expected statistical structure
    # even when an imported column contains occasional malformed text.
    numeric = data[columns].apply(pd.to_numeric, errors="coerce")
    stats = (
        numeric
        .describe(percentiles=[0.05, 0.1, 0.5, 0.9, 0.95])
        .T
        .reset_index()
    )
    return stats.rename(columns={"index": "Variable", "50%": "median"})


def negative_counts(
    data: pd.DataFrame,
    config: ModelConfig,
) -> pd.DataFrame:
    """Count negative, null and zero values for core numeric variables."""

    columns = [config.mass_column]

    if config.volume_column:
        columns.append(config.volume_column)

    columns.extend(config.grade_columns)

    rows: list[dict[str, Any]] = []

    for column in columns:
        if column not in data.columns:
            continue

        numeric = _numeric_series(data[column])
        rows.append(
            {
                "Variable": column,
                "Negative count": int(numeric.lt(0).sum()),
                "Null count": int(data[column].isna().sum()),
                "Zero count": int(numeric.eq(0).sum()),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# General categorical filtering
# =============================================================================
def apply_categorical_filters(
    data: pd.DataFrame,
    selections: dict[str, Iterable[Any]],
) -> pd.DataFrame:
    """Apply selected categorical values sequentially to a dataframe.

    Empty selections intentionally leave a column unfiltered, matching the
    existing application behavior.
    """

    filtered = data

    for column, values in selections.items():
        selected = list(values)
        if selected and column in filtered.columns:
            filtered = filtered[filtered[column].isin(selected)]

    return filtered


# =============================================================================
# Planning-period selection and summaries
# =============================================================================
def years_for_period(
    year_values: pd.Series,
    start_year: int,
    period: str,
) -> list[int]:
    """Return available integer years included in a planning period."""

    numeric_years = pd.to_numeric(year_values, errors="coerce").dropna()
    available = sorted(
        {
            int(value)
            for value in numeric_years
            if float(value).is_integer()
        }
    )

    years_count = period_year_count(period)

    if years_count is None:
        return [year for year in available if year >= int(start_year)]

    return [
        year
        for year in available
        if int(start_year) <= year < int(start_year) + years_count
    ]


def apply_period_filter(
    data: pd.DataFrame,
    year_column: str | None,
    start_year: int | None,
    period: str,
) -> tuple[pd.DataFrame, list[int]]:
    """Filter a dataframe to the years represented by a planning period."""

    if (
        not year_column
        or year_column not in data.columns
        or start_year is None
    ):
        return data, []

    selected_years = years_for_period(
        data[year_column],
        int(start_year),
        period,
    )

    # Compare through a numeric representation so integer years stored as text
    # remain compatible with the selected integer period values.
    numeric_years = pd.to_numeric(data[year_column], errors="coerce")
    return data[numeric_years.isin(selected_years)], selected_years


def period_summary(
    data: pd.DataFrame,
    config: ModelConfig,
    start_year: int,
    group_column: str | None = None,
) -> pd.DataFrame:
    """Build 2Y, 5Y, 10Y and LOM summaries from a selected start year."""

    year_column = config.column_for_role("Year")
    if not year_column or year_column not in data.columns:
        return pd.DataFrame()

    rows: list[pd.DataFrame] = []

    for period in PERIODS:
        subset, selected_years = apply_period_filter(
            data,
            year_column,
            start_year,
            period,
        )
        if subset.empty:
            continue

        if group_column and group_column in subset.columns:
            table = grouped_summary(
                subset,
                config,
                group_column,
            )
        else:
            summary = model_summary(subset, config)
            row: dict[str, Any] = {
                "Rows": int(summary.get("Rows", 0)),
                tonnage_column_name(config): (
                    summary.get("Tonnage", np.nan)
                    / tonnage_divisor(config.tonnage_unit)
                ),
            }

            if "Volume" in summary:
                row[f"Volume ({config.volume_unit})"] = summary["Volume"]

            for spec in config.grade_specs:
                key = f"{spec.label} ({spec.unit})"
                row[key] = summary.get(key, np.nan)

            table = pd.DataFrame([row])

        table.insert(0, "Years", ", ".join(map(str, selected_years)))
        table.insert(0, "Period", period)
        rows.append(table)

    return (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame()
    )


# =============================================================================
# Display formatting
# =============================================================================
def format_tonnage(
    value_tonnes: float,
    unit: str,
    decimals: int,
) -> str:
    """Format a raw-tonnage value using the requested unit and decimals."""

    if pd.isna(value_tonnes):
        return "N/A"

    converted = value_tonnes / tonnage_divisor(unit)
    return f"{converted:,.{decimals}f} {unit}"


def format_grade(
    value: float,
    spec: GradeSpec,
    default_decimals: int,
) -> str:
    """Format a grade using its configured unit and decimal precision."""

    if pd.isna(value):
        return "N/A"

    decimals = (
        spec.decimals
        if spec.decimals is not None
        else default_decimals
    )
    return f"{value:,.{decimals}f} {spec.unit}"
