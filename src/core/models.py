from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd


MODEL_TYPES = [
    "Resource Model",
    "Reserve Model",
    "Grade Control",
    "Stockpiles",
    "Quarry",
    "Geometallurgical",
    "Custom",
]

CATEGORY_ROLES = [
    "Domain",
    "Category",
    "Bench",
    "Mettype",
    "Destination",
    "Lithology",
    "Alteration",
    "Weathering",
    "Year",
    "Month",
    "Phase",
    "Pit_Phase",
    "Pit",
    "Block Model",
    "Source",
    "Other",
]

GRADE_UNITS = ["g/t", "ppm", "%", "other"]
TONNAGE_UNITS = ["t", "Kt", "Mt"]
VOLUME_UNITS = ["Mm3", "km3", "m3", "ft3"]


@dataclass(slots=True)
class GradeSpec:
    column: str
    label: str
    unit: str = "g/t"
    decimals: int = 2


@dataclass(slots=True)
class CategorySpec:
    column: str
    label: str
    role: str = "Other"


@dataclass(slots=True)
class ModelConfig:
    model_name: str
    model_type: str
    report_title: str
    mass_column: str
    volume_column: str | None = None
    volume_unit: str = "Mm3"
    grade_specs: list[GradeSpec] = field(default_factory=list)
    category_specs: list[CategorySpec] = field(default_factory=list)
    null_tokens: list[str] = field(
        default_factory=lambda: ["", "NA", "N/A", "NULL", "None", "-99", "-999", "-9999"]
    )
    decimal_separator: str = "."
    thousands_separator: str = ","
    tonnage_unit: str = "Mt"
    tonnage_decimals: int = 2
    grade_decimals: int = 2
    year_min: int = 2022
    year_max: int = 2050
    reject_negative_grades: bool = True
    require_positive_grades: bool = False

    @property
    def grade_columns(self) -> list[str]:
        return [spec.column for spec in self.grade_specs]

    @property
    def category_columns(self) -> list[str]:
        return [spec.column for spec in self.category_specs]

    def column_for_role(self, role: str) -> str | None:
        wanted = role.casefold()
        return next((spec.column for spec in self.category_specs if spec.role.casefold() == wanted), None)

    def category_label(self, column: str) -> str:
        return next((spec.label for spec in self.category_specs if spec.column == column), column)

    def grade_label(self, column: str) -> str:
        return next((spec.label for spec in self.grade_specs if spec.column == column), column)

    def grade_spec_for_label(self, label: str) -> GradeSpec | None:
        return next((spec for spec in self.grade_specs if spec.label == label or spec.column == label), None)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModelConfig":
        data = dict(payload)
        data["grade_specs"] = [GradeSpec(**item) for item in data.get("grade_specs", [])]
        data["category_specs"] = [CategorySpec(**item) for item in data.get("category_specs", [])]
        data.setdefault("reject_negative_grades", True)
        data.setdefault("require_positive_grades", False)
        return cls(**data)


@dataclass(slots=True)
class ValidationIssue:
    severity: str
    rule: str
    column: str
    count: int
    message: str
    sample_rows: list[int] = field(default_factory=list)


@dataclass(slots=True)
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)
    total_rows: int = 0

    @property
    def critical_count(self) -> int:
        return sum(issue.count for issue in self.issues if issue.severity == "Critical")

    @property
    def error_count(self) -> int:
        return sum(issue.count for issue in self.issues if issue.severity == "Error")

    @property
    def warning_count(self) -> int:
        return sum(issue.count for issue in self.issues if issue.severity == "Warning")

    @property
    def is_blocked(self) -> bool:
        return self.critical_count > 0

    def as_frame(self) -> pd.DataFrame:
        columns = ["Severity", "Rule", "Column", "Count", "Message", "Sample rows"]
        rows = [
            {
                "Severity": issue.severity,
                "Rule": issue.rule,
                "Column": issue.column,
                "Count": issue.count,
                "Message": issue.message,
                "Sample rows": ", ".join(map(str, issue.sample_rows)),
            }
            for issue in self.issues
        ]
        return pd.DataFrame(rows, columns=columns)


@dataclass
class ModelBundle:
    config: ModelConfig
    raw_data: pd.DataFrame
    data: pd.DataFrame
    validation: ValidationReport
    cleaning_stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class Scene:
    scene_id: str
    title: str
    kind: str
    model_names: list[str]
    filters: dict[str, Any]
    kpis: dict[str, str]
    table: pd.DataFrame
    notes: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
