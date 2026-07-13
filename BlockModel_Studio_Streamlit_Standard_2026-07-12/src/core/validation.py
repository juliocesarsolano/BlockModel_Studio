from __future__ import annotations

import pandas as pd

from .models import ModelConfig, ValidationIssue, ValidationReport


def _sample_index(mask: pd.Series, limit: int = 10) -> list[int]:
    return [int(index) for index in mask[mask].index[:limit]]


def _add_issue(
    report: ValidationReport,
    severity: str,
    rule: str,
    column: str,
    count: int,
    message: str,
    sample_rows: list[int] | None = None,
) -> None:
    if count > 0:
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


def validate_model(data: pd.DataFrame, config: ModelConfig) -> ValidationReport:
    report = ValidationReport(total_rows=len(data))

    required_columns = [config.mass_column, *config.grade_columns, *config.category_columns]
    if config.volume_column:
        required_columns.append(config.volume_column)

    for column in dict.fromkeys(required_columns):
        if column not in data.columns:
            _add_issue(report, "Critical", "Missing column", column, len(data), "Required column is not present.")

    if report.is_blocked:
        return report

    mass = data[config.mass_column]
    null_mass = mass.isna()
    _add_issue(report, "Error", "Null tonnage", config.mass_column, int(null_mass.sum()), "Tonnage is null.", _sample_index(null_mass))

    negative_mass = mass < 0
    _add_issue(report, "Error", "Negative tonnage", config.mass_column, int(negative_mass.sum()), "Tonnage must not be negative.", _sample_index(negative_mass))

    zero_mass = mass == 0
    _add_issue(report, "Warning", "Zero tonnage", config.mass_column, int(zero_mass.sum()), "Rows with zero tonnage do not contribute to weighted averages.", _sample_index(zero_mass))

    if config.volume_column:
        volume = data[config.volume_column]
        null_volume = volume.isna()
        _add_issue(report, "Warning", "Null volume", config.volume_column, int(null_volume.sum()), "Volume is null.", _sample_index(null_volume))
        negative_volume = volume < 0
        _add_issue(report, "Error", "Negative volume", config.volume_column, int(negative_volume.sum()), "Volume must not be negative.", _sample_index(negative_volume))

    for spec in config.grade_specs:
        grade = data[spec.column]
        null_grade = grade.isna()
        _add_issue(report, "Warning", "Null grade", spec.column, int(null_grade.sum()), f"{spec.label} has null values.", _sample_index(null_grade))

        negative_grade = grade < 0
        negative_severity = "Error" if config.reject_negative_grades else "Warning"
        negative_message = (
            f"{spec.label} has negative values. Negative grades are not accepted by the current controls."
            if config.reject_negative_grades
            else f"{spec.label} has negative values."
        )
        _add_issue(report, negative_severity, "Negative grade", spec.column, int(negative_grade.sum()), negative_message, _sample_index(negative_grade))

        if config.require_positive_grades:
            zero_or_negative = grade.notna() & (grade <= 0)
            _add_issue(
                report,
                "Error",
                "Non-positive grade",
                spec.column,
                int(zero_or_negative.sum()),
                f"{spec.label} must be greater than zero under the current controls.",
                _sample_index(zero_or_negative),
            )

    for spec in config.category_specs:
        null_category = data[spec.column].isna()
        _add_issue(report, "Warning", "Null category", spec.column, int(null_category.sum()), f"{spec.label} has null categories.", _sample_index(null_category))

        if spec.role == "Year":
            year = data[spec.column]
            out_of_range = year.notna() & ((year < config.year_min) | (year > config.year_max))
            _add_issue(report, "Warning", "Year outside range", spec.column, int(out_of_range.sum()), f"Year should be between {config.year_min} and {config.year_max}.", _sample_index(out_of_range))

    return report
