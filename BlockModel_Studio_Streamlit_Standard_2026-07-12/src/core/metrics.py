from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .models import GradeSpec, ModelConfig
from .parameters import period_labels, period_year_count


TONNAGE_DIVISORS = {"t": 1.0, "Kt": 1_000.0, "kt": 1_000.0, "Mt": 1_000_000.0}
PERIODS = period_labels()


def tonnage_divisor(unit: str) -> float:
    return TONNAGE_DIVISORS.get(unit, 1_000_000.0)


def display_tonnage(tonnes: float, config: ModelConfig) -> float:
    return float(tonnes) / tonnage_divisor(config.tonnage_unit)


def tonnage_column_name(config: ModelConfig) -> str:
    return f"Tonnage ({config.tonnage_unit})"


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return float("nan")
    return float(np.average(values[valid].astype(float), weights=weights[valid].astype(float)))


def total_tonnage(data: pd.DataFrame, config: ModelConfig) -> float:
    if config.mass_column not in data.columns:
        return float("nan")
    return float(data[config.mass_column].clip(lower=0).sum())


def total_volume(data: pd.DataFrame, config: ModelConfig) -> float:
    if not config.volume_column or config.volume_column not in data.columns:
        return float("nan")
    return float(data[config.volume_column].clip(lower=0).sum())


def contained_metal(values: pd.Series, tonnes: pd.Series, unit: str) -> tuple[float, str] | None:
    valid = values.notna() & tonnes.notna() & (tonnes > 0)
    if not valid.any():
        return None
    product = float((values[valid].astype(float) * tonnes[valid].astype(float)).sum())
    if unit in {"g/t", "ppm"}:
        return product / 31.1034768, "oz"
    if unit == "%":
        return product / 100.0, "t"
    return None


def model_summary(data: pd.DataFrame, config: ModelConfig) -> dict[str, float]:
    summary: dict[str, float] = {"Rows": float(len(data)), "Tonnage": total_tonnage(data, config)}
    if config.volume_column:
        summary["Volume"] = total_volume(data, config)
    weights = data[config.mass_column]
    for spec in config.grade_specs:
        summary[f"{spec.label} ({spec.unit})"] = weighted_mean(data[spec.column], weights)
        metal = contained_metal(data[spec.column], weights, spec.unit)
        if metal:
            value, unit = metal
            summary[f"{spec.label} contained ({unit})"] = value
    return summary


def grouped_summary(
    data: pd.DataFrame,
    config: ModelConfig,
    group_columns: str | Sequence[str],
    include_volume: bool = True,
) -> pd.DataFrame:
    if isinstance(group_columns, str):
        group_columns = [group_columns]
    group_columns = [column for column in group_columns if column in data.columns]
    if not group_columns or data.empty:
        return pd.DataFrame()

    source = data.dropna(subset=group_columns).copy()
    if source.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    grouped = source.groupby(group_columns, sort=True, dropna=False, observed=True)
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: dict[str, Any] = {}
        for column, value in zip(group_columns, keys, strict=True):
            label = config.category_label(column)
            if pd.notna(value) and isinstance(value, (float, np.floating)) and float(value).is_integer():
                value = int(value)
            row[label] = value

        mass = group[config.mass_column].clip(lower=0)
        row[tonnage_column_name(config)] = mass.sum() / tonnage_divisor(config.tonnage_unit)
        if include_volume and config.volume_column:
            row[f"Volume ({config.volume_unit})"] = group[config.volume_column].clip(lower=0).sum()
        for spec in config.grade_specs:
            row[f"{spec.label} ({spec.unit})"] = weighted_mean(group[spec.column], group[config.mass_column])
        rows.append(row)

    return pd.DataFrame(rows)


def basic_stats(data: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    columns = [config.mass_column]
    if config.volume_column:
        columns.append(config.volume_column)
    columns.extend(config.grade_columns)
    columns = [column for column in columns if column in data.columns]
    if not columns:
        return pd.DataFrame()

    stats = data[columns].describe(percentiles=[0.05, 0.1, 0.5, 0.9, 0.95]).T.reset_index()
    stats = stats.rename(columns={"index": "Variable", "50%": "median"})
    return stats


def negative_counts(data: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    columns = [config.mass_column]
    if config.volume_column:
        columns.append(config.volume_column)
    columns.extend(config.grade_columns)

    rows = []
    for column in columns:
        if column in data.columns:
            rows.append(
                {
                    "Variable": column,
                    "Negative count": int((data[column] < 0).sum()),
                    "Null count": int(data[column].isna().sum()),
                    "Zero count": int((data[column] == 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def apply_categorical_filters(data: pd.DataFrame, selections: dict[str, Iterable[Any]]) -> pd.DataFrame:
    filtered = data
    for column, values in selections.items():
        selected = list(values)
        if selected and column in filtered.columns:
            filtered = filtered[filtered[column].isin(selected)]
    return filtered


def years_for_period(year_values: pd.Series, start_year: int, period: str) -> list[int]:
    available = sorted({int(value) for value in year_values.dropna() if float(value).is_integer()})
    years_count = period_year_count(period)
    if years_count is None:
        return [year for year in available if year >= start_year]
    return [year for year in available if start_year <= year < start_year + years_count]


def apply_period_filter(data: pd.DataFrame, year_column: str | None, start_year: int | None, period: str) -> tuple[pd.DataFrame, list[int]]:
    if not year_column or year_column not in data.columns or start_year is None:
        return data, []
    selected_years = years_for_period(data[year_column], start_year, period)
    return data[data[year_column].isin(selected_years)], selected_years


def period_summary(data: pd.DataFrame, config: ModelConfig, start_year: int, group_column: str | None = None) -> pd.DataFrame:
    year_column = config.column_for_role("Year")
    if not year_column or year_column not in data.columns:
        return pd.DataFrame()

    rows: list[pd.DataFrame] = []
    for period in PERIODS:
        subset, selected_years = apply_period_filter(data, year_column, start_year, period)
        if subset.empty:
            continue
        if group_column and group_column in subset.columns:
            table = grouped_summary(subset, config, group_column)
        else:
            summary = model_summary(subset, config)
            row = {
                "Rows": int(summary.get("Rows", 0)),
                tonnage_column_name(config): summary.get("Tonnage", np.nan) / tonnage_divisor(config.tonnage_unit),
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

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def format_tonnage(value_tonnes: float, unit: str, decimals: int) -> str:
    if pd.isna(value_tonnes):
        return "N/A"
    converted = value_tonnes / tonnage_divisor(unit)
    return f"{converted:,.{decimals}f} {unit}"


def format_grade(value: float, spec: GradeSpec, default_decimals: int) -> str:
    if pd.isna(value):
        return "N/A"
    decimals = spec.decimals if spec.decimals is not None else default_decimals
    return f"{value:,.{decimals}f} {spec.unit}"
