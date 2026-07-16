"""
PV BlockModel Studio
====================

Core model-validation utilities for the Streamlit block-model evaluation,
comparison and reporting application.

Project
-------
PV BlockModel Studio

Module
------
src/core/validation.py

Purpose
-------
This module validates the structural and numerical integrity of a configured
block model before the data are used in Model Evaluation, Model Comparison or
Report Builder.

The validation workflow covers:

- Required-column availability.
- Null, negative and zero tonnage.
- Null and negative volume.
- Null, negative and non-positive grades.
- Null categorical variables.
- Configured Year ranges.
- Representative row samples for each detected issue.

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
- The public ``validate_model`` function name, arguments and return type are
  preserved for compatibility with ``pages.py`` and the complete application.
- Existing rule names and severity levels remain unchanged.
- Valid numeric model data produce the same validation results as before.
- Defensive numeric conversion prevents malformed text values from causing
  unhandled comparison errors.
- Missing required columns still create ``Critical`` issues and stop the
  remaining numerical validation checks.
- Zero tonnage remains a warning; zero volume remains accepted.
- Negative-grade severity continues to respect
  ``config.reject_negative_grades``.
- ``config.require_positive_grades`` continues to report every grade value
  less than or equal to zero as an error.

Validation severity
-------------------
- Critical: required structural information is missing and validation stops.
- Error: the model violates a mandatory numerical rule.
- Warning: the value may be usable but requires technical review.

Maintenance notes
-----------------
- Keep validation rules independent from Streamlit interface code.
- Preserve rule names where possible because the UI and exported reports may
  display them directly.
- Add new validation rules through ``_add_issue`` so issue construction remains
  consistent.
- Keep row samples small to avoid unnecessarily large session-state objects and
  report payloads.
"""

from __future__ import annotations

# =============================================================================
# Third-party tabular library
# =============================================================================
# pandas supplies the dataframe, series, null detection and vectorized masks
# used by every validation rule.
import pandas as pd

# =============================================================================
# Internal project models
# =============================================================================
# ``ModelConfig`` defines the configured model schema and validation controls.
# ``ValidationIssue`` stores one detected issue.
# ``ValidationReport`` aggregates all issues for the current model.
from .models import (
    ModelConfig,
    ValidationIssue,
    ValidationReport,
)


# =============================================================================
# Internal validation helpers
# =============================================================================
def _numeric_series(series: pd.Series) -> pd.Series:
    """Return a numeric representation of a series.

    The cleaning pipeline normally supplies numeric mass, volume, grade and
    year columns. This defensive conversion prevents occasional malformed text
    values from raising ``TypeError`` during vectorized comparisons.

    Non-numeric values become ``NaN`` and therefore do not satisfy negative,
    zero or range masks. They remain visible to the dedicated validation-detail
    tables elsewhere in the application.
    """

    return pd.to_numeric(series, errors="coerce")


def _sample_index(
    mask: pd.Series,
    limit: int = 10,
) -> list[int]:
    """Return up to ``limit`` representative row identifiers.

    Integer-compatible dataframe indices are preserved. When an index label
    cannot be converted to an integer, its zero-based row position is used so
    validation never fails solely because a dataframe has a textual index.
    """

    if limit <= 0 or mask.empty:
        return []

    normalized_mask = mask.fillna(False).astype(bool)
    matching_labels = normalized_mask[normalized_mask].index[:limit]

    samples: list[int] = []
    for label in matching_labels:
        try:
            samples.append(int(label))
        except (TypeError, ValueError, OverflowError):
            # Resolve the first matching positional location for non-integer
            # labels. This fallback is used only for diagnostic display.
            location = normalized_mask.index.get_indexer_for([label])
            if len(location) > 0 and int(location[0]) >= 0:
                samples.append(int(location[0]))

    return samples


def _add_issue(
    report: ValidationReport,
    severity: str,
    rule: str,
    column: str,
    count: int,
    message: str,
    sample_rows: list[int] | None = None,
) -> None:
    """Append a validation issue only when at least one record is affected."""

    if count <= 0:
        return

    report.issues.append(
        ValidationIssue(
            severity=severity,
            rule=rule,
            column=column,
            count=int(count),
            message=message,
            sample_rows=sample_rows or [],
        )
    )


def _required_columns(config: ModelConfig) -> list[str]:
    """Return configured required columns without duplicates.

    The ordering follows the application workflow: mass, grades, categories
    and finally volume when configured.
    """

    columns = [
        config.mass_column,
        *config.grade_columns,
        *config.category_columns,
    ]

    if config.volume_column:
        columns.append(config.volume_column)

    # ``dict.fromkeys`` preserves insertion order while removing duplicates.
    return list(dict.fromkeys(columns))


# =============================================================================
# Public model-validation workflow
# =============================================================================
def validate_model(
    data: pd.DataFrame,
    config: ModelConfig,
) -> ValidationReport:
    """Validate one cleaned block-model dataframe against its configuration.

    Parameters
    ----------
    data:
        Cleaned model dataframe to validate.
    config:
        Model schema, grade controls, category roles and accepted Year range.

    Returns
    -------
    ValidationReport
        Aggregated validation issues and total row count.

    Processing order
    ----------------
    1. Confirm all configured columns exist.
    2. Stop immediately when a required column is missing.
    3. Validate tonnage.
    4. Validate volume when configured.
    5. Validate every configured grade variable.
    6. Validate every configured categorical variable.
    """

    report = ValidationReport(total_rows=len(data))

    # -------------------------------------------------------------------------
    # Required-column validation
    # -------------------------------------------------------------------------
    # Structural issues are evaluated first because subsequent rules rely on
    # direct access to these columns.
    for column in _required_columns(config):
        if column not in data.columns:
            _add_issue(
                report=report,
                severity="Critical",
                rule="Missing column",
                column=column,
                count=len(data),
                message="Required column is not present.",
            )

    # ``ValidationReport.is_blocked`` is expected to become True when Critical
    # issues exist. Returning here avoids secondary KeyError exceptions.
    if report.is_blocked:
        return report

    # -------------------------------------------------------------------------
    # Tonnage validation
    # -------------------------------------------------------------------------
    raw_mass = data[config.mass_column]
    mass = _numeric_series(raw_mass)

    null_mass = raw_mass.isna()
    _add_issue(
        report=report,
        severity="Error",
        rule="Null tonnage",
        column=config.mass_column,
        count=int(null_mass.sum()),
        message="Tonnage is null.",
        sample_rows=_sample_index(null_mass),
    )

    negative_mass = mass.lt(0).fillna(False)
    _add_issue(
        report=report,
        severity="Error",
        rule="Negative tonnage",
        column=config.mass_column,
        count=int(negative_mass.sum()),
        message="Tonnage must not be negative.",
        sample_rows=_sample_index(negative_mass),
    )

    zero_mass = mass.eq(0).fillna(False)
    _add_issue(
        report=report,
        severity="Warning",
        rule="Zero tonnage",
        column=config.mass_column,
        count=int(zero_mass.sum()),
        message=(
            "Rows with zero tonnage do not contribute to weighted averages."
        ),
        sample_rows=_sample_index(zero_mass),
    )

    # -------------------------------------------------------------------------
    # Volume validation
    # -------------------------------------------------------------------------
    # Zero volume is intentionally accepted. Only null and negative values are
    # reported, matching the application's model rules.
    if config.volume_column:
        raw_volume = data[config.volume_column]
        volume = _numeric_series(raw_volume)

        null_volume = raw_volume.isna()
        _add_issue(
            report=report,
            severity="Warning",
            rule="Null volume",
            column=config.volume_column,
            count=int(null_volume.sum()),
            message="Volume is null.",
            sample_rows=_sample_index(null_volume),
        )

        negative_volume = volume.lt(0).fillna(False)
        _add_issue(
            report=report,
            severity="Error",
            rule="Negative volume",
            column=config.volume_column,
            count=int(negative_volume.sum()),
            message="Volume must not be negative.",
            sample_rows=_sample_index(negative_volume),
        )

    # -------------------------------------------------------------------------
    # Grade-variable validation
    # -------------------------------------------------------------------------
    for spec in config.grade_specs:
        raw_grade = data[spec.column]
        grade = _numeric_series(raw_grade)

        null_grade = raw_grade.isna()
        _add_issue(
            report=report,
            severity="Warning",
            rule="Null grade",
            column=spec.column,
            count=int(null_grade.sum()),
            message=f"{spec.label} has null values.",
            sample_rows=_sample_index(null_grade),
        )

        negative_grade = grade.lt(0).fillna(False)
        negative_severity = (
            "Error"
            if config.reject_negative_grades
            else "Warning"
        )
        negative_message = (
            f"{spec.label} has negative values. "
            "Negative grades are not accepted by the current controls."
            if config.reject_negative_grades
            else f"{spec.label} has negative values."
        )
        _add_issue(
            report=report,
            severity=negative_severity,
            rule="Negative grade",
            column=spec.column,
            count=int(negative_grade.sum()),
            message=negative_message,
            sample_rows=_sample_index(negative_grade),
        )

        # Preserve the current application rule: when positive-only grades are
        # required, both zero and negative grades are included in this issue.
        if config.require_positive_grades:
            zero_or_negative = (
                grade.notna()
                & grade.le(0)
            ).fillna(False)

            _add_issue(
                report=report,
                severity="Error",
                rule="Non-positive grade",
                column=spec.column,
                count=int(zero_or_negative.sum()),
                message=(
                    f"{spec.label} must be greater than zero "
                    "under the current controls."
                ),
                sample_rows=_sample_index(zero_or_negative),
            )

    # -------------------------------------------------------------------------
    # Categorical-variable validation
    # -------------------------------------------------------------------------
    for spec in config.category_specs:
        category = data[spec.column]

        null_category = category.isna()
        _add_issue(
            report=report,
            severity="Warning",
            rule="Null category",
            column=spec.column,
            count=int(null_category.sum()),
            message=f"{spec.label} has null categories.",
            sample_rows=_sample_index(null_category),
        )

        # Year is the only categorical role with a configured numeric range in
        # this core validation module.
        if spec.role == "Year":
            year = _numeric_series(category)
            out_of_range = (
                year.notna()
                & (
                    year.lt(config.year_min)
                    | year.gt(config.year_max)
                )
            ).fillna(False)

            _add_issue(
                report=report,
                severity="Warning",
                rule="Year outside range",
                column=spec.column,
                count=int(out_of_range.sum()),
                message=(
                    f"Year should be between "
                    f"{config.year_min} and {config.year_max}."
                ),
                sample_rows=_sample_index(out_of_range),
            )

    return report
