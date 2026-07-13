from __future__ import annotations

import numpy as np
import pandas as pd

from .models import ModelConfig


def _to_numeric(series: pd.Series, decimal_separator: str = ".", thousands_separator: str = ",") -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    text = series.astype("string").str.strip()
    if thousands_separator:
        text = text.str.replace(thousands_separator, "", regex=False)
    if decimal_separator != ".":
        text = text.str.replace(decimal_separator, ".", regex=False)
    return pd.to_numeric(text, errors="coerce")


def clean_model_data(raw_data: pd.DataFrame, config: ModelConfig) -> tuple[pd.DataFrame, dict[str, object]]:
    data = raw_data.copy()
    stats: dict[str, object] = {"input_rows": len(raw_data), "input_columns": len(raw_data.columns)}

    null_tokens = set(config.null_tokens or [])
    if null_tokens:
        data = data.replace(list(null_tokens), np.nan)

    numeric_columns = [config.mass_column, *config.grade_columns]
    if config.volume_column:
        numeric_columns.append(config.volume_column)

    for spec in config.category_specs:
        if spec.role in {"Year", "Bench"}:
            numeric_columns.append(spec.column)

    for column in dict.fromkeys(numeric_columns):
        if column in data.columns:
            before_nulls = int(data[column].isna().sum())
            data[column] = _to_numeric(data[column], config.decimal_separator, config.thousands_separator)
            after_nulls = int(data[column].isna().sum())
            stats[f"numeric_conversion_added_nulls::{column}"] = max(0, after_nulls - before_nulls)

    for spec in config.category_specs:
        if spec.column in data.columns and spec.role not in {"Year", "Bench"}:
            data[spec.column] = data[spec.column].astype("string").str.strip()
            data.loc[data[spec.column].isin(["", "nan", "None", "<NA>"]), spec.column] = pd.NA

    stats["output_rows"] = len(data)
    return data, stats
