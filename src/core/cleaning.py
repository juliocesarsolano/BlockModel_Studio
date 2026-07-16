"""
PV BlockModel Studio
====================

Core data-cleaning utilities for preparing uploaded block-model tables before
validation, evaluation, comparison and reporting.

Project
-------
PV BlockModel Studio

Module
------
src/core/cleaning.py

Purpose
-------
This module converts the user-selected model columns into the data types
expected by the rest of the application. Its responsibilities are intentionally
limited to:

- Replacing configured null tokens.
- Converting mass, volume, grade, Year and Bench columns to numeric values.
- Normalizing non-numeric categorical variables as trimmed pandas strings.
- Recording basic cleaning diagnostics without removing model rows.

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
- The public ``clean_model_data`` function name, arguments and return type are
  preserved for compatibility with ``pages.py`` and the complete application.
- The internal ``_to_numeric`` helper keeps its original signature.
- No model rows are removed in this module.
- Existing valid numeric and categorical values retain the same results.
- Values that cannot be converted to numeric continue to become ``NaN``.
- Existing cleaning-statistic keys are preserved.
- Null-token behavior remains controlled by ``ModelConfig.null_tokens``.

Cleaning conventions
--------------------
- Mass, configured grades and volume are numeric.
- Category roles ``Year`` and ``Bench`` are also numeric.
- Other configured category variables are stored as trimmed pandas strings.
- Empty strings and the standard textual missing-value representations
  ``nan``, ``None`` and ``<NA>`` are converted to ``pd.NA``.
- Decimal and thousands separators follow the model configuration.

Maintenance notes
-----------------
- Keep data-type normalization here and business exclusions in the appropriate
  filtering or validation modules.
- Do not silently remove records during cleaning.
- Preserve the diagnostic key
  ``numeric_conversion_added_nulls::<column>`` because it may be displayed or
  exported elsewhere in the application.
"""

from __future__ import annotations

# =============================================================================
# Third-party numerical and tabular libraries
# =============================================================================
# NumPy provides the canonical ``np.nan`` missing numeric value.
# pandas provides dataframe copying, replacement, string normalization and
# robust numeric conversion.
import numpy as np
import pandas as pd

# =============================================================================
# Internal project configuration model
# =============================================================================
# ``ModelConfig`` identifies the mass, volume, grade and categorical columns
# selected by the user during Model Setup.
from .models import ModelConfig


# =============================================================================
# Internal cleaning helpers
# =============================================================================
def _to_numeric(
    series: pd.Series,
    decimal_separator: str = ".",
    thousands_separator: str = ",",
) -> pd.Series:
    """Convert one series to numeric values using configured separators.

    Parameters
    ----------
    series:
        Source pandas series.
    decimal_separator:
        Character used as the decimal separator in imported text values.
    thousands_separator:
        Character used as the thousands separator in imported text values.

    Returns
    -------
    pd.Series
        Numeric series. Values that cannot be interpreted as numbers are
        returned as ``NaN``.

    Notes
    -----
    Numeric dtypes are passed directly through ``pd.to_numeric``. Text values
    are stripped before separator normalization. The thousands separator is
    removed only when it differs from the decimal separator; this defensive
    check avoids deleting decimal marks in a malformed configuration.
    """

    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    text = series.astype("string").str.strip()

    decimal = str(decimal_separator or ".")
    thousands = str(thousands_separator or "")

    if thousands and thousands != decimal:
        text = text.str.replace(
            thousands,
            "",
            regex=False,
        )

    if decimal != ".":
        text = text.str.replace(
            decimal,
            ".",
            regex=False,
        )

    return pd.to_numeric(
        text,
        errors="coerce",
    )


def _configured_numeric_columns(config: ModelConfig) -> list[str]:
    """Return configured numeric columns in stable order without duplicates."""

    columns = [
        config.mass_column,
        *config.grade_columns,
    ]

    if config.volume_column:
        columns.append(config.volume_column)

    # Year and Bench are categorical roles in the configuration layer, but
    # downstream filters and validation rules require numeric representations.
    for spec in config.category_specs:
        if spec.role in {"Year", "Bench"}:
            columns.append(spec.column)

    # ``dict.fromkeys`` removes duplicates while preserving insertion order.
    return list(dict.fromkeys(columns))


def _normalize_category_series(series: pd.Series) -> pd.Series:
    """Normalize a non-numeric categorical series.

    Values are converted to pandas' nullable string dtype and stripped of
    leading/trailing whitespace. Standard textual missing-value markers used by
    the original implementation are converted to ``pd.NA``.
    """

    normalized = series.astype("string").str.strip()
    missing_markers = ["", "nan", "None", "<NA>"]

    return normalized.mask(
        normalized.isin(missing_markers),
        pd.NA,
    )


# =============================================================================
# Public model-cleaning workflow
# =============================================================================
def clean_model_data(
    raw_data: pd.DataFrame,
    config: ModelConfig,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Clean one uploaded block-model dataframe without removing rows.

    Parameters
    ----------
    raw_data:
        Original dataframe loaded from CSV, Excel or another supported source.
    config:
        User-confirmed model configuration.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, object]]
        A cleaned dataframe and a dictionary containing cleaning diagnostics.

    Processing order
    ----------------
    1. Copy the uploaded dataframe.
    2. Record input dimensions.
    3. Replace configured null tokens.
    4. Convert configured numeric columns.
    5. Normalize remaining configured category columns.
    6. Record output row count.

    The output row count should normally equal the input row count because this
    function performs normalization only; filtering and exclusions occur later.
    """

    # Work on a copy so the original uploaded dataframe remains available for
    # audit, reconfiguration and repeat cleaning.
    data = raw_data.copy()

    stats: dict[str, object] = {
        "input_rows": len(raw_data),
        "input_columns": len(raw_data.columns),
    }

    # -------------------------------------------------------------------------
    # Configured null-token replacement
    # -------------------------------------------------------------------------
    # A list is used rather than a set so unusual but valid replacement values
    # do not need to be hashable. Duplicate tokens do not affect replacement.
    null_tokens = list(config.null_tokens or [])

    if null_tokens:
        data = data.replace(
            null_tokens,
            np.nan,
        )

    # -------------------------------------------------------------------------
    # Numeric conversion
    # -------------------------------------------------------------------------
    # Missing configured columns are ignored here. The validation module will
    # report them as Critical structural issues.
    for column in _configured_numeric_columns(config):
        if column not in data.columns:
            continue

        before_nulls = int(
            data[column].isna().sum()
        )

        data[column] = _to_numeric(
            data[column],
            config.decimal_separator,
            config.thousands_separator,
        )

        after_nulls = int(
            data[column].isna().sum()
        )

        # This diagnostic counts values that became missing specifically because
        # numeric conversion could not interpret the original content.
        stats[
            f"numeric_conversion_added_nulls::{column}"
        ] = max(
            0,
            after_nulls - before_nulls,
        )

    # -------------------------------------------------------------------------
    # Non-numeric categorical normalization
    # -------------------------------------------------------------------------
    for spec in config.category_specs:
        if (
            spec.column in data.columns
            and spec.role not in {"Year", "Bench"}
        ):
            data[spec.column] = _normalize_category_series(
                data[spec.column]
            )

    # No rows are intentionally removed by cleaning.
    stats["output_rows"] = len(data)

    return data, stats
