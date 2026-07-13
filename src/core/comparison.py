from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import grouped_summary, model_summary, total_volume
from .models import ModelBundle


def compare_global_volumes(bundles: dict[str, ModelBundle]) -> pd.DataFrame:
    rows = []
    volumes = []
    for name, bundle in bundles.items():
        volume = total_volume(bundle.data, bundle.config)
        volumes.append(volume)
        rows.append(
            {
                "Model": name,
                "Rows": len(bundle.data),
                "Volume column": bundle.config.volume_column or "Not configured",
                f"Volume ({bundle.config.volume_unit})": volume,
            }
        )
    table = pd.DataFrame(rows)
    valid_volumes = [value for value in volumes if not pd.isna(value)]
    if valid_volumes:
        reference = valid_volumes[0]
        volume_columns = [column for column in table.columns if column.startswith("Volume (")]
        if volume_columns:
            vol_col = volume_columns[0]
            table["Variance vs first (%)"] = np.where(
                reference != 0,
                (table[vol_col] - reference) / reference * 100,
                np.nan,
            )
    return table


def volume_gate_status(volume_table: pd.DataFrame, tolerance_pct: float) -> tuple[bool, str]:
    if volume_table.empty or "Variance vs first (%)" not in volume_table.columns:
        return False, "At least one selected model has no configured volume column."
    max_abs = volume_table["Variance vs first (%)"].abs().max(skipna=True)
    if pd.isna(max_abs):
        return False, "Unable to calculate global-volume variance."
    if max_abs <= tolerance_pct:
        return True, f"Global volumes are within ±{tolerance_pct:.4f}%. Maximum variance: {max_abs:.4f}%."
    return False, f"Global volumes exceed ±{tolerance_pct:.4f}%. Maximum variance: {max_abs:.4f}%."


def pairwise_volume_matrix(bundles: dict[str, ModelBundle]) -> pd.DataFrame:
    names = list(bundles)
    volumes = {name: total_volume(bundle.data, bundle.config) for name, bundle in bundles.items()}
    matrix = pd.DataFrame(index=names, columns=names, dtype=float)
    for left in names:
        for right in names:
            ref = volumes[left]
            value = volumes[right]
            matrix.loc[left, right] = np.nan if pd.isna(ref) or ref == 0 else (value - ref) / ref * 100
    return matrix.reset_index().rename(columns={"index": "Reference model"})


def common_category_labels(bundles: dict[str, ModelBundle]) -> list[str]:
    label_sets = []
    for bundle in bundles.values():
        label_sets.append({spec.label for spec in bundle.config.category_specs})
    if not label_sets:
        return []
    return sorted(set.intersection(*label_sets))


def common_roles(bundles: dict[str, ModelBundle]) -> list[str]:
    role_sets = []
    for bundle in bundles.values():
        role_sets.append({spec.role for spec in bundle.config.category_specs})
    if not role_sets:
        return []
    return sorted(set.intersection(*role_sets))


def common_grade_labels(bundles: dict[str, ModelBundle]) -> list[str]:
    label_sets = []
    for bundle in bundles.values():
        label_sets.append({spec.label for spec in bundle.config.grade_specs})
    if not label_sets:
        return []
    return sorted(set.intersection(*label_sets))


def incompatible_grade_units(bundles: dict[str, ModelBundle]) -> pd.DataFrame:
    rows = []
    labels = common_grade_labels(bundles)
    for label in labels:
        units = {}
        for name, bundle in bundles.items():
            spec = next((item for item in bundle.config.grade_specs if item.label == label), None)
            if spec:
                units[name] = spec.unit
        if len(set(units.values())) > 1:
            rows.append({"Grade": label, **units})
    return pd.DataFrame(rows)


def compare_model_summaries(bundles: dict[str, ModelBundle]) -> pd.DataFrame:
    rows = []
    for name, bundle in bundles.items():
        summary = model_summary(bundle.data, bundle.config)
        row = {"Model": name, "Rows": int(summary.pop("Rows", len(bundle.data)))}
        for key, value in summary.items():
            if key == "Tonnage":
                row[f"Tonnage ({bundle.config.tonnage_unit})"] = value / {"t": 1, "Kt": 1_000, "Mt": 1_000_000}.get(bundle.config.tonnage_unit, 1_000_000)
            elif key == "Volume":
                row[f"Volume ({bundle.config.volume_unit})"] = value
            else:
                row[key] = value
        rows.append(row)
    return pd.DataFrame(rows)


def category_comparison_by_role(bundles: dict[str, ModelBundle], role: str) -> pd.DataFrame:
    rows = []
    for name, bundle in bundles.items():
        column = bundle.config.column_for_role(role)
        if not column:
            continue
        table = grouped_summary(bundle.data, bundle.config, column)
        if table.empty:
            continue
        table.insert(0, "Model", name)
        rows.append(table)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()
