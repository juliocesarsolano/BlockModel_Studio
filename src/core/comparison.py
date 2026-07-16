"""
PV BlockModel Studio
====================

Core comparison utilities for block-model volume validation, summary
comparison, unit compatibility and category-based reconciliation.

Project
-------
PV BlockModel Studio

Module
------
src/core/comparison.py

Purpose
-------
This module provides the model-comparison calculations consumed by
``src/ui/pages.py``. Its responsibilities include:

- Mandatory global-volume comparison.
- Volume-tolerance gate evaluation.
- Pairwise volume-difference matrices.
- Detection of common category labels, category roles and grade labels.
- Identification of incompatible grade units.
- Side-by-side model summaries.
- Category comparisons based on configured semantic roles.

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
- Normal comparisons using valid configured models retain the same output
  structure and numerical results.
- Improvements are limited to defensive validation and consistency with the
  shared calculation utilities in ``metrics.py``.
- The first model with a valid global volume remains the reference used by the
  global-volume table.

Comparison conventions
----------------------
- Relative difference:
  (comparison value - reference value) / reference value × 100.
- The global-volume validation is mandatory before analytical comparison,
  unless the user explicitly accepts the override exposed in the UI.
- A missing, zero or non-calculable reference volume cannot pass the volume
  gate.
- Pairwise matrices use the model on each row as the reference model.

Maintenance notes
-----------------
- Keep calculation logic independent from Streamlit UI code.
- Preserve the ``Variance vs first (%)`` column name because ``pages.py`` and
  the volume-gate workflow depend on it.
- Use the shared ``tonnage_divisor`` helper rather than duplicating unit rules.
- Add new comparison outputs through new functions instead of changing the
  existing public table schemas.
"""

from __future__ import annotations

# =============================================================================
# Third-party numerical libraries
# =============================================================================
# NumPy supplies NaN values and finite-number checks.
# pandas supplies model-comparison tables and grouped concatenation.
import numpy as np
import pandas as pd

# =============================================================================
# Internal project dependencies
# =============================================================================
# Shared metrics guarantee that comparison calculations use the same tonnage,
# volume and weighted-grade conventions as Model Evaluation and Report Builder.
from .metrics import (
    grouped_summary,
    model_summary,
    tonnage_divisor,
    total_volume,
)

# ``ModelBundle`` packages the configured model, cleaned data and validation
# result used by all comparison functions.
from .models import ModelBundle


# =============================================================================
# Internal helpers
# =============================================================================
def _valid_reference_volume(volumes: list[float]) -> float | None:
    """Return the first finite, non-null volume in model insertion order.

    The original comparison workflow used the first available valid volume as
    its global reference. This helper makes that behavior explicit and reusable.
    A zero value is returned as zero so the gate can reject it with a clear
    non-calculable-variance result.
    """

    for value in volumes:
        if pd.notna(value) and np.isfinite(float(value)):
            return float(value)
    return None


def _common_configured_values(
    bundles: dict[str, ModelBundle],
    attribute_name: str,
) -> list[str]:
    """Return values configured in every selected model.

    ``attribute_name`` must identify a string attribute on category or grade
    specification objects, for example ``label`` or ``role``.
    """

    value_sets: list[set[str]] = []

    for bundle in bundles.values():
        specs = (
            bundle.config.grade_specs
            if attribute_name == "grade_label"
            else bundle.config.category_specs
        )

        if attribute_name == "grade_label":
            values = {
                str(spec.label)
                for spec in specs
            }
        else:
            values = {
                str(getattr(spec, attribute_name))
                for spec in specs
            }

        value_sets.append(values)

    if not value_sets:
        return []

    return sorted(set.intersection(*value_sets))


# =============================================================================
# Mandatory global-volume validation
# =============================================================================
def compare_global_volumes(
    bundles: dict[str, ModelBundle],
) -> pd.DataFrame:
    """Build the global-volume comparison table for selected models.

    The output keeps the historical visible columns used by ``pages.py``:
    ``Model``, ``Rows``, ``Volume column``, one or more unit-labelled volume
    columns, and ``Variance vs first (%)`` when a valid reference exists.

    Volume variance is calculated directly from the ordered volume values
    rather than from the first unit-labelled dataframe column. This avoids
    sparse-column errors when selected models use different display-unit labels.
    """

    rows: list[dict[str, object]] = []
    volumes: list[float] = []

    for name, bundle in bundles.items():
        volume = total_volume(bundle.data, bundle.config)
        volumes.append(volume)

        rows.append(
            {
                "Model": name,
                "Rows": len(bundle.data),
                "Volume column": (
                    bundle.config.volume_column
                    or "Not configured"
                ),
                f"Volume ({bundle.config.volume_unit})": volume,
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    reference = _valid_reference_volume(volumes)
    if reference is None:
        return table

    numeric_volumes = pd.Series(
        volumes,
        index=table.index,
        dtype=float,
    )

    if reference != 0:
        table["Variance vs first (%)"] = (
            (numeric_volumes - reference)
            / reference
            * 100.0
        )
    else:
        # A zero reference cannot support a relative-difference calculation.
        table["Variance vs first (%)"] = np.nan

    return table


def volume_gate_status(
    volume_table: pd.DataFrame,
    tolerance_pct: float,
) -> tuple[bool, str]:
    """Evaluate whether all selected model volumes pass the tolerance gate.

    The gate fails when:

    - Fewer than two models are available.
    - The variance column cannot be calculated.
    - Any selected model has a missing/non-calculable variance.
    - The tolerance is invalid.
    - Maximum absolute variance exceeds the selected tolerance.

    These checks prevent a comparison from passing merely because pandas
    ignored a missing model volume while calculating the maximum.
    """

    if volume_table.empty or len(volume_table) < 2:
        return (
            False,
            "At least two selected models with configured volume data are required.",
        )

    variance_column = "Variance vs first (%)"
    if variance_column not in volume_table.columns:
        return (
            False,
            "At least one selected model has no configured or calculable volume.",
        )

    try:
        tolerance = float(tolerance_pct)
    except (TypeError, ValueError):
        return False, "Global-volume tolerance is not a valid number."

    if not np.isfinite(tolerance) or tolerance < 0:
        return False, "Global-volume tolerance must be a finite non-negative value."

    variances = pd.to_numeric(
        volume_table[variance_column],
        errors="coerce",
    )

    if variances.isna().any():
        return (
            False,
            "At least one selected model has a missing, zero-reference or non-calculable global volume.",
        )

    max_abs = float(variances.abs().max())

    if max_abs <= tolerance:
        return (
            True,
            f"Global volumes are within ±{tolerance:.4f}%. "
            f"Maximum variance: {max_abs:.4f}%.",
        )

    return (
        False,
        f"Global volumes exceed ±{tolerance:.4f}%. "
        f"Maximum variance: {max_abs:.4f}%.",
    )


def pairwise_volume_matrix(
    bundles: dict[str, ModelBundle],
) -> pd.DataFrame:
    """Return pairwise volume differences using each row model as reference.

    A cell is ``NaN`` when either volume is unavailable or the row reference
    volume is zero.
    """

    names = list(bundles)
    volumes = {
        name: total_volume(bundle.data, bundle.config)
        for name, bundle in bundles.items()
    }

    matrix = pd.DataFrame(
        index=names,
        columns=names,
        dtype=float,
    )

    for left in names:
        for right in names:
            reference = volumes[left]
            comparison = volumes[right]

            if (
                pd.isna(reference)
                or pd.isna(comparison)
                or not np.isfinite(float(reference))
                or not np.isfinite(float(comparison))
                or float(reference) == 0.0
            ):
                matrix.loc[left, right] = np.nan
                continue

            matrix.loc[left, right] = (
                (float(comparison) - float(reference))
                / float(reference)
                * 100.0
            )

    return (
        matrix
        .reset_index()
        .rename(columns={"index": "Reference model"})
    )


# =============================================================================
# Shared schema discovery
# =============================================================================
def common_category_labels(
    bundles: dict[str, ModelBundle],
) -> list[str]:
    """Return category display labels configured in every selected model."""

    return _common_configured_values(
        bundles,
        "label",
    )


def common_roles(
    bundles: dict[str, ModelBundle],
) -> list[str]:
    """Return semantic category roles configured in every selected model."""

    return _common_configured_values(
        bundles,
        "role",
    )


def common_grade_labels(
    bundles: dict[str, ModelBundle],
) -> list[str]:
    """Return grade display labels configured in every selected model."""

    return _common_configured_values(
        bundles,
        "grade_label",
    )


def incompatible_grade_units(
    bundles: dict[str, ModelBundle],
) -> pd.DataFrame:
    """Identify common grade labels configured with different units.

    Only grades present in every selected model are evaluated. The returned
    dataframe is empty when all common grade units are compatible.
    """

    rows: list[dict[str, str]] = []

    for label in common_grade_labels(bundles):
        units: dict[str, str] = {}

        for name, bundle in bundles.items():
            spec = next(
                (
                    item
                    for item in bundle.config.grade_specs
                    if item.label == label
                ),
                None,
            )
            if spec is not None:
                units[name] = spec.unit

        if len(set(units.values())) > 1:
            rows.append(
                {
                    "Grade": label,
                    **units,
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# Model summary comparison
# =============================================================================
def compare_model_summaries(
    bundles: dict[str, ModelBundle],
) -> pd.DataFrame:
    """Return one side-by-side summary row for each selected model.

    Raw tonnage from ``model_summary`` is converted with the same shared unit
    helper used across the rest of the application.
    """

    rows: list[dict[str, object]] = []

    for name, bundle in bundles.items():
        summary = model_summary(
            bundle.data,
            bundle.config,
        )

        row: dict[str, object] = {
            "Model": name,
            "Rows": int(
                summary.pop(
                    "Rows",
                    len(bundle.data),
                )
            ),
        }

        for key, value in summary.items():
            if key == "Tonnage":
                row[
                    f"Tonnage ({bundle.config.tonnage_unit})"
                ] = (
                    value
                    / tonnage_divisor(bundle.config.tonnage_unit)
                )
            elif key == "Volume":
                row[
                    f"Volume ({bundle.config.volume_unit})"
                ] = value
            else:
                row[key] = value

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# Category comparison
# =============================================================================
def category_comparison_by_role(
    bundles: dict[str, ModelBundle],
    role: str,
) -> pd.DataFrame:
    """Compare model summaries grouped by a shared semantic category role.

    Models that do not configure the requested role, whose mapped column is not
    present, or whose grouped result is empty are skipped. Remaining tables are
    concatenated with a leading ``Model`` column.
    """

    rows: list[pd.DataFrame] = []

    for name, bundle in bundles.items():
        column = bundle.config.column_for_role(role)

        if (
            not column
            or column not in bundle.data.columns
        ):
            continue

        table = grouped_summary(
            bundle.data,
            bundle.config,
            column,
        )
        if table.empty:
            continue

        table.insert(0, "Model", name)
        rows.append(table)

    return (
        pd.concat(
            rows,
            ignore_index=True,
            sort=False,
        )
        if rows
        else pd.DataFrame()
    )
