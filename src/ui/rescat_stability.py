"""
PV BlockModel Studio — ResCat Stability
=======================================

Dedicated Streamlit module for annual block-model resource-category stability,
conversion to Grade Control, and categorical-domain uncertainty analysis.

Design principles
-----------------
- CATEG_GC is the only resource-category variable used by this module.
- CATEG is intentionally ignored and is never used as a fallback.
- Models are aligned block-to-block from XYZ centroids.
- Analyses use the spatial intersection common to every loaded snapshot.
- Volume is the primary basis for category-conversion matrices.
- Tonnage is read directly when available or derived as Volume × Density.
- Au grades are tonnage weighted and Au content is calculated in troy ounces.
- Geological domains (Mettype, Lithology, Alteration) stratify the analysis;
  they do not define or modify the fixed spatial panel geometry.
- Fixed regular panels are optional analytical support containers and are
  independent of the mine plan.

Author / context
----------------
Julio Solano — Mineral Resource Management, Pueblo Viejo
"""

from __future__ import annotations

from dataclasses import dataclass
import html
from io import BytesIO
from pathlib import Path
import math
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


TROY_OUNCE_GRAMS = 31.1034768

RESCAT_ORDER = ["Grade Control", "Measured", "Indicated", "Inferred", "Inventory", "Unclassified"]
RESCAT_FOCUS_ORDER = ["Grade Control", "Measured", "Indicated"]

BARRICK_BLUE = "#03547C"
BARRICK_GOLD = "#A39161"
BARRICK_ORANGE = "#FDB813"
BARRICK_GRAY = "#C7C8CA"
BARRICK_DARK = "#23323B"
BARRICK_LIGHT = "#F7F9FB"
BARRICK_GRID = "#DCE4EA"

RESCAT_COLORS = {
    "Grade Control": BARRICK_ORANGE,
    "Measured": BARRICK_BLUE,
    "Indicated": BARRICK_GOLD,
    "Inferred": BARRICK_GRAY,
    "Inventory": BARRICK_GRAY,
    "Unclassified": BARRICK_GRAY,
}

DOMAIN_COLORS = {
    "Mettype": BARRICK_BLUE,
    "Lithology": BARRICK_GOLD,
    "Alteration": BARRICK_ORANGE,
}

COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "x": ("centroid_x", "x_centroid", "centroidx", "x", "xc", "easting"),
    "y": ("centroid_y", "y_centroid", "centroidy", "y", "yc", "northing"),
    "z": ("centroid_z", "z_centroid", "centroidz", "z", "zc", "elevation", "rl"),
    "volume": ("volume", "vol", "block_volume", "blk_volume", "total_volume"),
    "tonnes": ("tonnes", "tonnage", "tons", "mass", "total_mass"),
    "density": ("density", "dens", "sg", "bulk_density"),
    # IMPORTANT: intentionally excludes CATEG. CATEG_GC is the sole category source.
    "categ_gc": ("categ_gc", "category_gc", "resource_category_gc", "rescat_gc"),
    "au": ("au_ppm", "au_gpt", "au_gt", "au_g/t", "au"),
    "mettype": ("mettype_txt", "mettype", "met_type", "metallurgical_type"),
    "lithology": ("litho", "lithology", "lito", "lito_type", "litho_type"),
    "alteration": ("alt", "alteration", "alter", "alter_type"),
}

REQUIRED_FIELDS = ["x", "y", "z", "volume", "categ_gc"]
OPTIONAL_FIELDS = ["tonnes", "density", "au", "mettype", "lithology", "alteration"]

PANEL_PRESETS: list[tuple[int, int, int]] = [
    (100, 100, 30),
    (150, 150, 30),
    (200, 200, 30),
    (250, 250, 60),
    (300, 300, 60),
]


@dataclass(frozen=True)
class SnapshotMeta:
    source_name: str
    label: str
    year: int | None
    quarter: int | None
    sort_key: tuple[int, int, str]


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).casefold()).strip("_")


def _infer_column(columns: Iterable[str], field: str) -> str | None:
    normalized = {_normalize_name(column): column for column in columns}
    for alias in COLUMN_ALIASES.get(field, ()):  # exact normalized alias first
        key = _normalize_name(alias)
        if key in normalized:
            return normalized[key]
    return None


def _snapshot_meta(filename: str) -> SnapshotMeta:
    stem = Path(filename).stem
    year_match = re.search(r"(?<!\d)(20\d{2})(?!\d)", stem)
    quarter_match = re.search(r"(?i)(?:^|[^A-Za-z0-9])Q([1-4])(?:$|[^A-Za-z0-9])", stem)
    year = int(year_match.group(1)) if year_match else None
    quarter = int(quarter_match.group(1)) if quarter_match else None

    if year is not None and quarter is not None:
        label = f"{year} Q{quarter}"
    elif year is not None:
        label = str(year)
    else:
        label = stem

    return SnapshotMeta(
        source_name=filename,
        label=label,
        year=year,
        quarter=quarter,
        sort_key=(year if year is not None else 9999, quarter if quarter is not None else 9, stem.casefold()),
    )


@st.cache_data(show_spinner=False)
def _csv_header(payload: bytes) -> tuple[str, ...]:
    """Read only the CSV header using the same permissive delimiter strategy as the app."""
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "latin1"):
        try:
            # Fast path for conventional comma-delimited block-model CSVs.
            return tuple(pd.read_csv(BytesIO(payload), encoding=encoding, nrows=0).columns)
        except Exception as exc:
            last_error = exc
            try:
                # Defensive fallback for unusual delimiters.
                return tuple(
                    pd.read_csv(
                        BytesIO(payload),
                        sep=None,
                        engine="python",
                        encoding=encoding,
                        nrows=0,
                    ).columns
                )
            except Exception as fallback_exc:  # pragma: no cover
                last_error = fallback_exc
    raise ValueError(f"Unable to read CSV header: {last_error}")


@st.cache_data(show_spinner=False)
def _csv_row_count(payload: bytes) -> int:
    """Fast line-based row estimate for conventional CSV files (header excluded)."""
    # Source files used by this workflow are block-model CSVs with one physical
    # row per record. This avoids materializing the complete dataframe merely
    # to populate the Data & Alignment inventory table.
    if not payload:
        return 0
    return max(payload.count(b"\n") - 1, 0)


def _resource_category(value: Any) -> str:
    """Map CATEG_GC values to canonical resource-category labels."""
    if pd.isna(value):
        return "Unclassified"

    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(numeric):
        code = int(numeric) if float(numeric).is_integer() else numeric
        if code == 15:
            return "Grade Control"
        if code == 1:
            return "Measured"
        if code == 2:
            return "Indicated"
        if code == 3:
            return "Inferred"
        if code == 4:
            return "Inventory"
        if code == 5:
            return "Unclassified"

    text = " ".join(str(value).strip().replace("_", " ").replace("-", " ").split()).casefold()
    if "grade" in text and "control" in text:
        return "Grade Control"
    if text in {"gc", "gradecontrol"}:
        return "Grade Control"
    if "measured" in text:
        return "Measured"
    if "indicated" in text:
        return "Indicated"
    if "inferred" in text:
        return "Inferred"
    if "inventory" in text:
        return "Inventory"
    return "Unclassified"


def _clean_domain(series: pd.Series, missing_label: str = "Unclassified") -> pd.Series:
    text = series.astype("string").str.strip()
    missing = text.isna() | text.eq("") | text.str.casefold().isin({"nan", "none", "null"})
    text = text.mask(missing, str(missing_label))
    return text.astype(str)


@st.cache_data(show_spinner=False, max_entries=12)
def _load_snapshot(
    payload: bytes,
    mapping_items: tuple[tuple[str, str | None], ...],
    coordinate_decimals: int,
) -> pd.DataFrame:
    """Load only mapped columns and return the canonical ResCat dataframe."""
    mapping = dict(mapping_items)
    selected_columns = sorted({column for column in mapping.values() if column})
    if not selected_columns:
        return pd.DataFrame()

    last_error: Exception | None = None
    source: pd.DataFrame | None = None
    for encoding in ("utf-8-sig", "utf-8", "latin1"):
        try:
            # Fast C-engine path for normal comma-delimited CSVs.
            source = pd.read_csv(
                BytesIO(payload),
                encoding=encoding,
                usecols=selected_columns,
            )
            break
        except Exception as exc:
            last_error = exc
            try:
                source = pd.read_csv(
                    BytesIO(payload),
                    sep=None,
                    engine="python",
                    encoding=encoding,
                    usecols=selected_columns,
                )
                break
            except Exception as fallback_exc:  # pragma: no cover - defensive fallback
                last_error = fallback_exc
    if source is None:
        raise ValueError(f"Unable to read CSV: {last_error}")

    frame = pd.DataFrame(index=source.index)
    for field in ["x", "y", "z", "volume", "tonnes", "density", "au", "categ_gc"]:
        column = mapping.get(field)
        if column and column in source.columns:
            frame[field] = pd.to_numeric(source[column], errors="coerce")

    for field in ["mettype", "lithology", "alteration"]:
        column = mapping.get(field)
        if column and column in source.columns:
            # Cover/soil blocks legitimately carry no alteration assignment.
            # In the analytical workflow they are labelled "cv" rather than
            # "None"/"Unclassified" so the geological meaning is preserved.
            missing_label = "cv" if field == "alteration" else "Unclassified"
            frame[field] = _clean_domain(source[column], missing_label=missing_label)
        else:
            frame[field] = "cv" if field == "alteration" else "Unclassified"

    for coordinate in ["x", "y", "z"]:
        if coordinate in frame.columns:
            frame[coordinate] = frame[coordinate].round(int(coordinate_decimals))

    if "categ_gc" in frame.columns:
        frame["rescat"] = frame["categ_gc"].map(_resource_category)

    # Use a supplied mass column when mapped. Otherwise derive tonnes from
    # Volume × Density, which is the natural form of the supplied PV snapshots.
    if "tonnes" in frame.columns:
        frame["tonnes"] = frame["tonnes"].clip(lower=0)
    elif "volume" in frame.columns and "density" in frame.columns:
        frame["tonnes"] = frame["volume"].clip(lower=0) * frame["density"].clip(lower=0)
    else:
        frame["tonnes"] = np.nan

    if "volume" in frame.columns:
        frame["volume"] = frame["volume"].clip(lower=0)

    if "au" in frame.columns:
        frame["au_oz"] = frame["au"] * frame["tonnes"] / TROY_OUNCE_GRAMS
    else:
        frame["au"] = np.nan
        frame["au_oz"] = np.nan

    # Drop structurally invalid coordinates only. Null analytical variables are
    # retained so QA and downstream metrics can identify their impact.
    frame = frame.dropna(subset=[column for column in ["x", "y", "z"] if column in frame.columns]).copy()

    if all(column in frame.columns for column in ["x", "y", "z"]):
        frame = frame.set_index(["x", "y", "z"], drop=False)
        frame.index.names = ["x_idx", "y_idx", "z_idx"]
    return frame


def _mapping_signature(mapping: dict[str, str | None]) -> tuple[tuple[str, str | None], ...]:
    return tuple(sorted(mapping.items()))


def _common_index(models: dict[str, pd.DataFrame]) -> pd.MultiIndex:
    common: pd.MultiIndex | None = None
    for frame in models.values():
        idx = frame.index.unique()
        common = idx if common is None else common.intersection(idx, sort=False)
    if common is None:
        return pd.MultiIndex.from_arrays([[], [], []], names=["x_idx", "y_idx", "z_idx"])
    return common


def _aligned_model(frame: pd.DataFrame, common_index: pd.MultiIndex) -> pd.DataFrame:
    unique = frame[~frame.index.duplicated(keep="first")]
    return unique.reindex(common_index)


def _pair_frame(
    models: dict[str, pd.DataFrame],
    common_index: pd.MultiIndex,
    from_label: str,
    to_label: str,
) -> pd.DataFrame:
    left = _aligned_model(models[from_label], common_index)
    right = _aligned_model(models[to_label], common_index)

    columns = [
        "x", "y", "z",
        "volume", "tonnes", "au", "au_oz", "categ_gc", "rescat",
        "mettype", "lithology", "alteration",
    ]
    pair = pd.DataFrame(index=common_index)
    for column in columns:
        if column in left.columns:
            pair[f"{column}_from"] = left[column]
        if column in right.columns:
            pair[f"{column}_to"] = right[column]

    # Analytical eligibility is evaluated pair-by-pair so Data & Alignment
    # can still report the full spatial intersection. Resource calculations
    # use only positive CATEG_GC values and, when Au is mapped, positive Au in
    # both snapshots. This makes the generic workflow explicit rather than
    # relying on the source data to already satisfy those conditions.
    eligible = pd.Series(True, index=pair.index)
    for suffix in ["from", "to"]:
        category_col = f"categ_gc_{suffix}"
        if category_col in pair.columns:
            category = pd.to_numeric(pair[category_col], errors="coerce")
            eligible &= category.gt(0)

    au_from = pd.to_numeric(pair.get("au_from"), errors="coerce") if "au_from" in pair.columns else None
    au_to = pd.to_numeric(pair.get("au_to"), errors="coerce") if "au_to" in pair.columns else None
    if au_from is not None and au_to is not None and (au_from.notna().any() or au_to.notna().any()):
        eligible &= au_from.gt(0) & au_to.gt(0)

    pair = pair.loc[eligible].copy()

    # Coordinates are fixed by definition; prefer the earlier model values.
    for coordinate in ["x", "y", "z"]:
        if f"{coordinate}_from" in pair.columns:
            pair[coordinate] = pair[f"{coordinate}_from"]
        elif f"{coordinate}_to" in pair.columns:
            pair[coordinate] = pair[f"{coordinate}_to"]
    return pair


def _safe_pct(numerator: float, denominator: float) -> float:
    if denominator is None or not np.isfinite(denominator) or denominator == 0:
        return float("nan")
    return 100.0 * float(numerator) / float(denominator)


def _bias_pct(initial: float, reference: float) -> float:
    """Prediction bias relative to the later/reference value."""
    if reference is None or not np.isfinite(reference) or reference == 0:
        return float("nan")
    return 100.0 * (float(initial) - float(reference)) / float(reference)


def _weighted_grade(group: pd.DataFrame, grade_col: str, tonnes_col: str) -> float:
    if grade_col not in group.columns or tonnes_col not in group.columns:
        return float("nan")
    grade = pd.to_numeric(group[grade_col], errors="coerce")
    tonnes = pd.to_numeric(group[tonnes_col], errors="coerce")
    valid = grade.notna() & tonnes.gt(0)
    if not valid.any():
        return float("nan")
    return float(np.average(grade.loc[valid], weights=tonnes.loc[valid]))


def _transition_tables(
    pair: pd.DataFrame,
    categories: list[str],
    basis_col: str = "volume_from",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = pair[
        pair["rescat_from"].isin(categories)
        & pair["rescat_to"].isin(categories)
    ].copy()
    if work.empty or basis_col not in work.columns:
        empty = pd.DataFrame(index=categories, columns=categories).fillna(0.0)
        return empty, empty.copy()

    work["__basis__"] = pd.to_numeric(work[basis_col], errors="coerce").fillna(0.0).clip(lower=0)
    volume = (
        work.groupby(["rescat_from", "rescat_to"], observed=True)["__basis__"]
        .sum()
        .unstack(fill_value=0.0)
        .reindex(index=categories, columns=categories, fill_value=0.0)
    )
    denominator = volume.sum(axis=1).replace(0, np.nan)
    pct = volume.div(denominator, axis=0).mul(100.0).fillna(0.0)
    return volume, pct


def _transition_rate(volume_table: pd.DataFrame, origin: str, destination: str) -> float:
    if origin not in volume_table.index or destination not in volume_table.columns:
        return float("nan")
    denominator = float(volume_table.loc[origin].sum())
    return _safe_pct(float(volume_table.loc[origin, destination]), denominator)


def _format_pct(value: float, decimals: int = 1) -> str:
    return "N/A" if value is None or not np.isfinite(value) else f"{value:,.{decimals}f}%"


def _format_number(value: float, decimals: int = 2) -> str:
    return "N/A" if value is None or not np.isfinite(value) else f"{value:,.{decimals}f}"


def _format_oz(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "N/A"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:,.2f} Moz"
    if abs(value) >= 1_000:
        return f"{value / 1_000:,.1f} koz"
    return f"{value:,.0f} oz"


def _render_kpis(cards: list[tuple[str, str, str]]) -> None:
    """Render the same dashboard KPI-card geometry used by the existing app."""
    palette = [
        ("#DCE7EE", BARRICK_BLUE),
        ("#F0ECE2", BARRICK_GOLD),
        ("#FFF3D4", BARRICK_ORANGE),
        ("#ECEDEF", BARRICK_GRAY),
    ]
    card_html = "".join(
        '<div class="bm-kpi-card" '
        f'style="background:{palette[index % len(palette)][0]}; border-bottom-color:{palette[index % len(palette)][1]};" '
        f'title="{html.escape(str(help_text))}">'
        f'<div class="bm-kpi-value">{html.escape(str(value))}</div>'
        f'<div class="bm-kpi-label">{html.escape(str(label))}</div>'
        '</div>'
        for index, (label, value, help_text) in enumerate(cards)
    )
    st.markdown(
        f"""
        <style>
        .bm-kpi-grid {{
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 1.15rem;
            margin: 0.35rem 0 1.25rem 0;
        }}
        .bm-kpi-card {{
            min-height: 7.25rem;
            border-radius: 0;
            border: 1px solid rgba(0,0,0,0.04);
            border-bottom: 4px solid;
            padding: 1.05rem 0.95rem 0.85rem 0.95rem;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.45);
        }}
        .bm-kpi-value {{
            color: #20252B;
            font-size: clamp(1.85rem, 2.6vw, 2.55rem);
            line-height: 1.0;
            font-weight: 500;
            letter-spacing: -0.015em;
            text-align: center;
            word-break: break-word;
        }}
        .bm-kpi-label {{
            color: #5B5B5B;
            font-size: 0.92rem;
            line-height: 1.15;
            margin-top: 0.62rem;
            text-align: center;
            font-weight: 500;
        }}
        @media (max-width: 980px) {{
            .bm-kpi-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
        }}
        @media (max-width: 560px) {{
            .bm-kpi-grid {{ grid-template-columns: 1fr; }}
        }}
        </style>
        <div class="bm-kpi-grid">{card_html}</div>
        """,
        unsafe_allow_html=True,
    )


def _render_page_header(
    snapshot_count: int = 0,
    initial_label: str | None = None,
    later_label: str | None = None,
) -> None:
    """Use the exact page-header structure already established in the app."""
    chips = [
        ("Snapshots", f"{snapshot_count:,}"),
        ("Category", "CATEG_GC"),
        ("Initial", initial_label or "Oldest model"),
        ("Later", later_label or "Most recent model"),
    ]
    chip_html = "".join(
        f"""
        <div class="bm-page-header-chip">
            <span class="bm-page-header-chip-label">{html.escape(label)}</span>
            <span class="bm-page-header-chip-value">{html.escape(value)}</span>
        </div>
        """
        for label, value in chips
    )
    st.markdown(
        f"""
        <section class="bm-page-header-card">
            <div class="bm-page-header-content">
                <div>
                    <div class="bm-page-header-kicker">PV BlockModel Studio</div>
                    <div class="bm-page-header-title">ResCat Stability</div>
                    <div class="bm-page-header-subtitle">Evaluate resource-category conversion and Measured reliability against the later Grade Control benchmark.</div>
                </div>
                <div class="bm-page-header-chips">{chip_html}</div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _apply_rescat_styles() -> None:
    """Keep ResCat visually consistent with the established app design."""
    st.markdown(
        f"""
        <style>
        .rescat-note {{
            padding: .82rem 1rem;
            border-left: 4px solid {BARRICK_BLUE};
            background: #F4F8FA;
            margin: .4rem 0 1rem 0;
            color: #334A57;
            border-radius: 0 10px 10px 0;
        }}
        .rescat-method {{
            padding: .85rem 1rem;
            background: #FFFDF7;
            border: 1px solid rgba(163,145,97,.35);
            border-radius: 10px;
            color: #3F4650;
            margin-bottom: 1rem;
        }}
        .rescat-interpretation {{
            padding: .85rem 1rem;
            border: 1px solid rgba(0,84,124,.16);
            background: #F8FBFC;
            border-radius: 10px;
            margin: .35rem 0 1rem 0;
            color: #3F4D57;
            line-height: 1.45;
        }}
        .rescat-interpretation b {{ color: {BARRICK_BLUE}; }}

        /* ResCat tables: center headers while preserving cell alignment and existing app styling. */
        div[data-testid="stDataFrame"] [role="columnheader"] {{
            justify-content: center !important;
            text-align: center !important;
        }}
        div[data-testid="stDataFrame"] [role="columnheader"] > div {{
            justify-content: center !important;
            text-align: center !important;
            width: 100% !important;
        }}

        div[data-testid="stTabs"] {{ margin-top: 0.75rem; }}
        div[data-testid="stTabs"] div[role="tablist"],
        div[data-testid="stTabs"] [data-baseweb="tab-list"] {{
            display: flex !important;
            gap: 0.70rem !important;
            padding: 0.48rem !important;
            border: 1px solid rgba(0, 84, 124, 0.22) !important;
            border-radius: 13px !important;
            background: linear-gradient(180deg, #F4F8FA 0%, #EAF2F6 100%) !important;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.95), 0 7px 18px rgba(0,45,67,0.08) !important;
        }}
        div[data-testid="stTabs"] button[role="tab"],
        div[data-testid="stTabs"] [data-baseweb="tab"] {{
            flex: 0 0 auto !important;
            min-height: 2.65rem !important;
            padding: 0.62rem 1.08rem !important;
            margin: 0 !important;
            border: 1px solid rgba(0, 84, 124, 0.20) !important;
            border-radius: 9px !important;
            background: #FFFFFF !important;
            color: #004967 !important;
            box-shadow: 0 2px 7px rgba(0,45,67,0.08) !important;
            font-weight: 700 !important;
            letter-spacing: 0.01em !important;
            white-space: nowrap !important;
        }}
        div[data-testid="stTabs"] button[role="tab"] p,
        div[data-testid="stTabs"] [data-baseweb="tab"] p,
        div[data-testid="stTabs"] [data-baseweb="tab"] span {{
            margin: 0 !important;
            color: inherit !important;
            font-size: 0.93rem !important;
            font-weight: 700 !important;
        }}
        div[data-testid="stTabs"] button[role="tab"]:hover,
        div[data-testid="stTabs"] [data-baseweb="tab"]:hover {{
            transform: translateY(-1px) !important;
            border-color: rgba(163,145,97,0.78) !important;
            background: #FFFDF7 !important;
            color: #004967 !important;
            box-shadow: 0 5px 12px rgba(0,73,103,0.13) !important;
        }}
        div[data-testid="stTabs"] button[role="tab"][aria-selected="true"],
        div[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"],
        div[data-testid="stTabs"] [aria-selected="true"][role="tab"] {{
            border-color: #004967 !important;
            background: linear-gradient(135deg, #004967 0%, #006F98 100%) !important;
            color: #FFFFFF !important;
            box-shadow: 0 8px 18px rgba(0,73,103,0.24), inset 0 1px 0 rgba(255,255,255,0.22) !important;
        }}
        div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p,
        div[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] p,
        div[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] span {{ color: #FFFFFF !important; }}
        div[data-testid="stTabs"] div[data-baseweb="tab-highlight"],
        div[data-testid="stTabs"] div[data-baseweb="tab-border"] {{ display: none !important; }}
        div[data-testid="stTabs"] div[role="tabpanel"] {{ padding-top: 1.15rem !important; }}
        @media (max-width: 900px) {{
            div[data-testid="stTabs"] div[role="tablist"],
            div[data-testid="stTabs"] [data-baseweb="tab-list"] {{ overflow-x: auto !important; flex-wrap: nowrap !important; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _mapping_controls(common_columns: list[str]) -> dict[str, str | None]:
    st.markdown("#### Column mapping")
    st.caption(
        "CATEG_GC is mandatory and is the only resource-category variable used. "
        "CATEG is intentionally excluded from this workflow."
    )

    mapping: dict[str, str | None] = {}
    required_cols = st.columns(5)
    for position, field in enumerate(REQUIRED_FIELDS):
        inferred = _infer_column(common_columns, field)
        if field == "categ_gc":
            allowed_category_names = {_normalize_name(alias) for alias in COLUMN_ALIASES["categ_gc"]}
            selectable_columns = [
                column for column in common_columns
                if _normalize_name(column) in allowed_category_names
            ]
        else:
            selectable_columns = common_columns
        options: list[str | None] = [None] + selectable_columns
        index = options.index(inferred) if inferred in options else 0
        mapping[field] = required_cols[position].selectbox(
            field.upper() if field != "categ_gc" else "CATEG_GC",
            options,
            index=index,
            format_func=lambda value: "<Select>" if value is None else str(value),
            key=f"rescat_map_{field}",
        )

    optional_cols = st.columns(6)
    for position, field in enumerate(OPTIONAL_FIELDS):
        inferred = _infer_column(common_columns, field)
        options: list[str | None] = [None] + common_columns
        index = options.index(inferred) if inferred in options else 0
        label = {
            "tonnes": "Tonnes (optional)",
            "density": "Density",
            "au": "Au grade",
            "mettype": "Mettype",
            "lithology": "Lithology",
            "alteration": "Alteration",
        }[field]
        mapping[field] = optional_cols[position].selectbox(
            label,
            options,
            index=index,
            format_func=lambda value: "<None>" if value is None else str(value),
            key=f"rescat_map_{field}",
        )
    return mapping


def _validate_mapping(mapping: dict[str, str | None]) -> list[str]:
    issues: list[str] = []
    for field in REQUIRED_FIELDS:
        if not mapping.get(field):
            issues.append(f"{field} is required")
    if not mapping.get("tonnes") and not mapping.get("density"):
        issues.append("Map either Tonnes or Density so mass and Au content can be calculated")
    if not mapping.get("au"):
        issues.append("Map Au grade to activate Meas Reliability and metal-bias analyses")
    return issues


def _model_inventory(
    metas: dict[str, SnapshotMeta],
    headers: dict[str, tuple[str, ...]],
    payloads: dict[str, bytes],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, meta in sorted(metas.items(), key=lambda item: item[1].sort_key):
        rows.append(
            {
                "Snapshot": meta.label,
                "File": name,
                "Rows (approx.)": _csv_row_count(payloads[name]),
                "Columns": len(headers[name]),
                "Year": meta.year,
                "Quarter": meta.quarter,
            }
        )
    return pd.DataFrame(rows)


def _alignment_table(models: dict[str, pd.DataFrame], common_index: pd.MultiIndex) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    common_count = len(common_index)
    for label, frame in models.items():
        duplicate_count = int(frame.index.duplicated(keep=False).sum())
        unique_count = int(frame.index.nunique())
        common_pct = _safe_pct(common_count, unique_count)
        total_volume = pd.to_numeric(frame.get("volume", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
        common_volume = pd.to_numeric(_aligned_model(frame, common_index).get("volume", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
        rows.append(
            {
                "Snapshot": label,
                "Rows": len(frame),
                "Unique XYZ": unique_count,
                "Duplicate XYZ rows": duplicate_count,
                "Common XYZ": common_count,
                "Common blocks (%)": common_pct,
                "Total volume (Mm3)": total_volume / 1_000_000.0,
                "Common volume (Mm3)": common_volume / 1_000_000.0,
            }
        )
    return pd.DataFrame(rows)


def _evolution_table(models: dict[str, pd.DataFrame], common_index: pd.MultiIndex, labels: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label in labels:
        frame = _aligned_model(models[label], common_index)
        eligible = pd.Series(True, index=frame.index)
        if "categ_gc" in frame.columns:
            eligible &= pd.to_numeric(frame["categ_gc"], errors="coerce").gt(0)
        if "au" in frame.columns and pd.to_numeric(frame["au"], errors="coerce").notna().any():
            eligible &= pd.to_numeric(frame["au"], errors="coerce").gt(0)
        work = frame.loc[eligible & frame["rescat"].isin(RESCAT_ORDER)].copy()
        basis = pd.to_numeric(work["volume"], errors="coerce").fillna(0.0).clip(lower=0)
        total = float(basis.sum())
        for category in RESCAT_ORDER:
            amount = float(basis[work["rescat"].eq(category)].sum())
            rows.append(
                {
                    "Snapshot": label,
                    "Category": category,
                    "Volume (Mm3)": amount / 1_000_000.0,
                    "Volume (%)": _safe_pct(amount, total),
                }
            )
    return pd.DataFrame(rows)


def _hex_rgba(hex_color: str, alpha: float) -> str:
    text = str(hex_color).lstrip("#")
    if len(text) != 6:
        return f"rgba(191,191,191,{alpha})"
    try:
        r = int(text[0:2], 16)
        g = int(text[2:4], 16)
        b = int(text[4:6], 16)
    except ValueError:
        return f"rgba(191,191,191,{alpha})"
    return f"rgba({r},{g},{b},{alpha})"

def _apply_barrick_layout(
    fig: go.Figure,
    *,
    height: int = 480,
    title: str | None = None,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    legend_title: str | None = None,
) -> go.Figure:
    """Apply the ResCat visual standard without erasing titles set by Plotly Express."""
    layout_updates: dict[str, Any] = {
        "template": "plotly_white",
        "height": height,
        "font": dict(family="Arial, sans-serif", color=BARRICK_DARK),
        "paper_bgcolor": "white",
        "plot_bgcolor": "white",
        "margin": {"l": 35, "r": 20, "t": 70, "b": 35},
        "hoverlabel": dict(namelength=-1),
    }
    # Plotly Express figures already carry their title. Passing title=None to
    # update_layout can clear that title and, in some front ends, render it as
    # "undefined". Only overwrite title-related fields when explicitly supplied.
    if title is not None:
        layout_updates["title"] = dict(text=str(title))
    if xaxis_title is not None:
        layout_updates["xaxis_title"] = str(xaxis_title)
    if yaxis_title is not None:
        layout_updates["yaxis_title"] = str(yaxis_title)
    if legend_title is not None:
        layout_updates["legend_title_text"] = str(legend_title)

    fig.update_layout(**layout_updates)
    fig.update_xaxes(showline=True, linecolor="#AAB7C2", gridcolor=BARRICK_GRID)
    fig.update_yaxes(showline=True, linecolor="#AAB7C2", gridcolor=BARRICK_GRID, zerolinecolor="#AAB7C2")
    return fig


def _apply_histogram_style(fig: go.Figure, *, height: int = 450, title: str | None = None, xaxis_title: str | None = None, yaxis_title: str | None = None) -> go.Figure:
    fig.update_traces(marker_color=BARRICK_BLUE, marker_line_color="white", marker_line_width=1.2, hovertemplate="%{x:,.2f} Mt<br>Panels: %{y:,.0f}<extra></extra>")
    return _apply_barrick_layout(fig, height=height, title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title)


def _corporate_scale() -> list[list[float | str]]:
    return [
        [0.0, BARRICK_GRAY],
        [0.33, BARRICK_GOLD],
        [0.66, BARRICK_ORANGE],
        [1.0, BARRICK_BLUE],
    ]


def _sankey_figure(volume: pd.DataFrame, title: str) -> go.Figure:
    categories = list(volume.index)
    left_labels = [f"Initial: {category}" for category in categories]
    right_labels = [f"Later: {category}" for category in categories]
    node_labels = left_labels + right_labels
    node_colors = [RESCAT_COLORS.get(category, BARRICK_GRAY) for category in categories] * 2

    sources: list[int] = []
    targets: list[int] = []
    values: list[float] = []
    link_colors: list[str] = []
    for i, origin in enumerate(categories):
        for j, destination in enumerate(categories):
            value = float(volume.loc[origin, destination])
            if value <= 0:
                continue
            sources.append(i)
            targets.append(len(categories) + j)
            values.append(value / 1_000_000.0)
            color = RESCAT_COLORS.get(origin, BARRICK_GRAY)
            link_colors.append(_hex_rgba(color, 0.34))

    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            textfont=dict(family="Arial", size=13, color=BARRICK_DARK),
            node={
                "label": node_labels,
                "color": node_colors,
                "pad": 22,
                "thickness": 22,
                "line": {"color": "white", "width": 1.0},
            },
            link={
                "source": sources,
                "target": targets,
                "value": values,
                "color": link_colors,
                "hovertemplate": "%{source.label} → %{target.label}<br>Volume: %{value:,.2f} Mm³<extra></extra>",
            },
        )
    )
    return _apply_barrick_layout(fig, title=title, height=540)


def _heatmap_figure(table_pct: pd.DataFrame, title: str, x_title: str = "Later category", y_title: str = "Initial category") -> go.Figure:
    z = table_pct.to_numpy(dtype=float)
    text = np.vectorize(lambda value: f"{value:.1f}%")(z)
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=table_pct.columns.tolist(),
            y=table_pct.index.tolist(),
            colorscale=_corporate_scale(),
            zmin=0,
            zmax=max(100.0, float(np.nanmax(z)) if z.size else 100.0),
            text=text,
            texttemplate="%{text}",
            hovertemplate="Initial: %{y}<br>Later: %{x}<br>Share: %{z:.1f}%<extra></extra>",
            colorbar={"title": "% of initial"},
        )
    )
    return _apply_barrick_layout(fig, title=title, height=500, xaxis_title=x_title, yaxis_title=y_title)


def _pair_selectors(labels: list[str], key_prefix: str) -> tuple[str, str]:
    """Select a chronological model pair, defaulting to oldest → most recent."""
    if len(labels) < 2:
        raise ValueError("At least two snapshots are required for a pair comparison.")

    from_key = f"{key_prefix}_from"
    to_key = f"{key_prefix}_to"

    valid_initial = labels[:-1]
    existing_from = st.session_state.get(from_key)
    if existing_from not in valid_initial:
        st.session_state[from_key] = valid_initial[0]

    left, right = st.columns(2)
    initial_label = left.selectbox(
        "Initial snapshot",
        valid_initial,
        key=from_key,
        help="Defaults to the oldest loaded snapshot. You can select any snapshot that has a later model available for comparison.",
    )

    initial_position = labels.index(initial_label)
    valid_later = labels[initial_position + 1 :]
    existing_to = st.session_state.get(to_key)
    if existing_to not in valid_later:
        st.session_state[to_key] = valid_later[-1]

    later_label = right.selectbox(
        "Later snapshot",
        valid_later,
        key=to_key,
        help="Defaults to the most recent loaded snapshot and is restricted to snapshots later than the selected initial model.",
    )

    st.caption(
        "Defaults are oldest → most recent; adjust the pair when you want to test an intermediate model-to-model interval."
    )
    return initial_label, later_label


def _cohort_summary(pair: pd.DataFrame) -> pd.DataFrame:
    work = pair[
        pair["rescat_from"].isin(["Indicated", "Measured"])
        & pair["rescat_to"].eq("Grade Control")
    ].copy()
    if work.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for category in ["Measured", "Indicated"]:
        group = work[work["rescat_from"].eq(category)]
        if group.empty:
            continue

        volume = float(pd.to_numeric(group["volume_from"], errors="coerce").fillna(0).clip(lower=0).sum())
        tonnes_from = float(pd.to_numeric(group["tonnes_from"], errors="coerce").fillna(0).clip(lower=0).sum())
        tonnes_to = float(pd.to_numeric(group["tonnes_to"], errors="coerce").fillna(0).clip(lower=0).sum())
        grade_from = _weighted_grade(group, "au_from", "tonnes_from")
        grade_to = _weighted_grade(group, "au_to", "tonnes_to")
        oz_from = float(pd.to_numeric(group["au_oz_from"], errors="coerce").fillna(0).sum())
        oz_to = float(pd.to_numeric(group["au_oz_to"], errors="coerce").fillna(0).sum())

        rows.append(
            {
                "Initial category": category,
                "Blocks reaching GC": len(group),
                "Volume (Mm3)": volume / 1_000_000.0,
                "Initial tonnes (Mt)": tonnes_from / 1_000_000.0,
                "GC tonnes (Mt)": tonnes_to / 1_000_000.0,
                "Tonnes bias (%)": _bias_pct(tonnes_from, tonnes_to),
                "Initial Au (g/t)": grade_from,
                "GC Au (g/t)": grade_to,
                "Au grade bias (%)": _bias_pct(grade_from, grade_to),
                "Initial Au (oz)": oz_from,
                "GC Au (oz)": oz_to,
                "Au oz bias (%)": _bias_pct(oz_from, oz_to),
            }
        )
    return pd.DataFrame(rows)


def _panel_ids(pair: pd.DataFrame, sx: float, sy: float, sz: float, origin: tuple[float, float, float]) -> pd.Series:
    x0, y0, z0 = origin
    ix = np.floor((pd.to_numeric(pair["x"], errors="coerce") - x0) / float(sx)).astype("Int64")
    iy = np.floor((pd.to_numeric(pair["y"], errors="coerce") - y0) / float(sy)).astype("Int64")
    iz = np.floor((pd.to_numeric(pair["z"], errors="coerce") - z0) / float(sz)).astype("Int64")
    return ix.astype(str) + "_" + iy.astype(str) + "_" + iz.astype(str)


def _panel_cohort_metrics(
    pair: pd.DataFrame,
    sx: float,
    sy: float,
    sz: float,
    origin: tuple[float, float, float],
    minimum_support_mt: float,
) -> pd.DataFrame:
    work = pair[
        pair["rescat_from"].isin(["Indicated", "Measured"])
        & pair["rescat_to"].eq("Grade Control")
    ].copy()
    if work.empty:
        return pd.DataFrame()

    work["Panel"] = _panel_ids(work, sx, sy, sz, origin)
    rows: list[dict[str, Any]] = []
    for (panel, category), group in work.groupby(["Panel", "rescat_from"], observed=True):
        volume = float(pd.to_numeric(group["volume_from"], errors="coerce").fillna(0).clip(lower=0).sum())
        tonnes_from = float(pd.to_numeric(group["tonnes_from"], errors="coerce").fillna(0).clip(lower=0).sum())
        tonnes_to = float(pd.to_numeric(group["tonnes_to"], errors="coerce").fillna(0).clip(lower=0).sum())
        support_mt = tonnes_to / 1_000_000.0
        if support_mt < float(minimum_support_mt):
            continue

        grade_from = _weighted_grade(group, "au_from", "tonnes_from")
        grade_to = _weighted_grade(group, "au_to", "tonnes_to")
        oz_from = float(pd.to_numeric(group["au_oz_from"], errors="coerce").fillna(0).sum())
        oz_to = float(pd.to_numeric(group["au_oz_to"], errors="coerce").fillna(0).sum())
        rows.append(
            {
                "Panel": panel,
                "Initial category": category,
                "Blocks": len(group),
                "Volume (Mm3)": volume / 1_000_000.0,
                "Initial tonnes (t)": tonnes_from,
                "GC tonnes (t)": tonnes_to,
                "GC support (Mt)": support_mt,
                "Initial Au (g/t)": grade_from,
                "GC Au (g/t)": grade_to,
                "Initial Au (oz)": oz_from,
                "GC Au (oz)": oz_to,
                "Tonnes bias (%)": _bias_pct(tonnes_from, tonnes_to),
                "Au grade bias (%)": _bias_pct(grade_from, grade_to),
                "Au oz bias (%)": _bias_pct(oz_from, oz_to),
            }
        )
    return pd.DataFrame(rows)


def _weighted_quantile(values: pd.Series, weights: pd.Series, quantile: float) -> float:
    """Return a weighted quantile using strictly positive finite weights."""
    table = pd.DataFrame(
        {
            "value": pd.to_numeric(values, errors="coerce"),
            "weight": pd.to_numeric(weights, errors="coerce"),
        }
    ).dropna()
    table = table[np.isfinite(table["value"]) & np.isfinite(table["weight"]) & table["weight"].gt(0)].copy()
    if table.empty:
        return float("nan")
    table = table.sort_values("value", kind="stable")
    values_sorted = table["value"].to_numpy(dtype=float)
    weights_sorted = table["weight"].to_numpy(dtype=float)
    total_weight = float(weights_sorted.sum())
    if total_weight <= 0:
        return float("nan")
    # Midpoint plotting positions provide a smooth weighted percentile and
    # avoid the lower-step behavior of a raw weighted empirical CDF.
    positions = (np.cumsum(weights_sorted) - 0.5 * weights_sorted) / total_weight
    q = float(np.clip(quantile, 0.0, 1.0))
    return float(np.interp(q, positions, values_sorted, left=values_sorted[0], right=values_sorted[-1]))


def _weighted_tolerance_share(values: pd.Series, weights: pd.Series, tolerance_pct: float) -> float:
    table = pd.DataFrame(
        {
            "value": pd.to_numeric(values, errors="coerce"),
            "weight": pd.to_numeric(weights, errors="coerce"),
        }
    ).dropna()
    table = table[np.isfinite(table["value"]) & np.isfinite(table["weight"]) & table["weight"].gt(0)].copy()
    if table.empty:
        return float("nan")
    total_weight = float(table["weight"].sum())
    if total_weight <= 0:
        return float("nan")
    inside = float(table.loc[table["value"].abs().le(float(tolerance_pct)), "weight"].sum())
    return 100.0 * inside / total_weight


def _wape(initial: pd.Series, benchmark: pd.Series) -> float:
    initial_values = pd.to_numeric(initial, errors="coerce")
    benchmark_values = pd.to_numeric(benchmark, errors="coerce")
    valid = initial_values.notna() & benchmark_values.notna() & np.isfinite(initial_values) & np.isfinite(benchmark_values) & benchmark_values.gt(0)
    if not valid.any():
        return float("nan")
    denominator = float(benchmark_values.loc[valid].sum())
    if denominator <= 0:
        return float("nan")
    numerator = float((initial_values.loc[valid] - benchmark_values.loc[valid]).abs().sum())
    return 100.0 * numerator / denominator


def _uncertainty_summary(panel_metrics: pd.DataFrame, tolerance_pct: float) -> pd.DataFrame:
    """Summarize reliability with weighted metal-risk metrics plus legacy diagnostics.

    Primary metrics are designed for Measured reliability against the later
    Grade Control benchmark:
      - Aggregate Au-oz bias: signed accuracy at cohort scale.
      - Au-oz WAPE: absolute error weighted by benchmark metal.
      - Weighted P10/P50/P90: benchmark-Au-oz-weighted distribution of panel bias.
      - GC Au-oz within tolerance: share of benchmark metal represented by
        panel cells whose absolute Au-oz bias lies inside the selected band.

    The unweighted P90 absolute bias is retained as a secondary diagnostic only.
    """
    if panel_metrics.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for category in ["Measured", "Indicated"]:
        group = panel_metrics[panel_metrics["Initial category"].eq(category)].copy()
        if group.empty:
            continue

        tonnes_from = pd.to_numeric(group["Initial tonnes (t)"], errors="coerce")
        tonnes_to = pd.to_numeric(group["GC tonnes (t)"], errors="coerce")
        oz_from = pd.to_numeric(group["Initial Au (oz)"], errors="coerce")
        oz_to = pd.to_numeric(group["GC Au (oz)"], errors="coerce")
        grade_from = pd.to_numeric(group["Initial Au (g/t)"], errors="coerce")
        grade_to = pd.to_numeric(group["GC Au (g/t)"], errors="coerce")

        total_tonnes_from = float(tonnes_from.fillna(0).clip(lower=0).sum())
        total_tonnes_to = float(tonnes_to.fillna(0).clip(lower=0).sum())
        total_oz_from = float(oz_from.fillna(0).clip(lower=0).sum())
        total_oz_to = float(oz_to.fillna(0).clip(lower=0).sum())
        agg_grade_from = _weighted_grade(group, "Initial Au (g/t)", "Initial tonnes (t)")
        agg_grade_to = _weighted_grade(group, "GC Au (g/t)", "GC tonnes (t)")

        metal_bias = pd.to_numeric(group["Au oz bias (%)"], errors="coerce")
        metal_weights = oz_to.clip(lower=0)
        p10 = _weighted_quantile(metal_bias, metal_weights, 0.10)
        p50 = _weighted_quantile(metal_bias, metal_weights, 0.50)
        p90 = _weighted_quantile(metal_bias, metal_weights, 0.90)

        row: dict[str, Any] = {
            "Initial category": category,
            "Panels": len(group),
            "GC support (Mt)": total_tonnes_to / 1_000_000.0,
            "GC Au (koz)": total_oz_to / 1_000.0,
            "Aggregate tonnes bias (%)": _bias_pct(total_tonnes_from, total_tonnes_to),
            "Aggregate Au grade bias (%)": _bias_pct(agg_grade_from, agg_grade_to),
            "Aggregate Au oz bias (%)": _bias_pct(total_oz_from, total_oz_to),
            "Au oz WAPE (%)": _wape(oz_from, oz_to),
            "Weighted P10 Au oz bias (%)": p10,
            "Weighted P50 Au oz bias (%)": p50,
            "Weighted P90 Au oz bias (%)": p90,
            "Weighted P10-P90 width (%)": p90 - p10 if np.isfinite(p10) and np.isfinite(p90) else np.nan,
            f"GC Au oz within ±{tolerance_pct:g}% (%)": _weighted_tolerance_share(metal_bias, metal_weights, tolerance_pct),
        }

        # Retain the former cell-count-based diagnostics for drill-down only.
        for metric in ["Tonnes bias (%)", "Au grade bias (%)", "Au oz bias (%)"]:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"Median {metric}"] = float(values.median()) if not values.empty else np.nan
            row[f"P90 abs {metric}"] = float(values.abs().quantile(0.90)) if not values.empty else np.nan
            row[f"Cell count within ±{tolerance_pct:g}% {metric}"] = (
                100.0 * float(values.abs().le(tolerance_pct).mean()) if not values.empty else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _domain_panel_metrics(
    pair: pd.DataFrame,
    domain: str,
    sx: float,
    sy: float,
    sz: float,
    origin: tuple[float, float, float],
    minimum_support_mt: float,
    stable_domain_only: bool,
) -> pd.DataFrame:
    from_col = f"{domain}_from"
    to_col = f"{domain}_to"
    if from_col not in pair.columns:
        return pd.DataFrame()

    work = pair[
        pair["rescat_from"].isin(["Indicated", "Measured"])
        & pair["rescat_to"].eq("Grade Control")
    ].copy()
    if stable_domain_only and to_col in work.columns:
        work = work[work[from_col].astype(str).eq(work[to_col].astype(str))].copy()
    if work.empty:
        return pd.DataFrame()

    work["Panel"] = _panel_ids(work, sx, sy, sz, origin)
    work["Domain"] = _clean_domain(work[from_col])

    rows: list[dict[str, Any]] = []
    for (panel, domain_value, category), group in work.groupby(["Panel", "Domain", "rescat_from"], observed=True):
        tonnes_from = float(pd.to_numeric(group["tonnes_from"], errors="coerce").fillna(0).clip(lower=0).sum())
        tonnes_to = float(pd.to_numeric(group["tonnes_to"], errors="coerce").fillna(0).clip(lower=0).sum())
        support_mt = tonnes_to / 1_000_000.0
        if support_mt < float(minimum_support_mt):
            continue

        volume = float(pd.to_numeric(group["volume_from"], errors="coerce").fillna(0).clip(lower=0).sum())
        grade_from = _weighted_grade(group, "au_from", "tonnes_from")
        grade_to = _weighted_grade(group, "au_to", "tonnes_to")
        oz_from = float(pd.to_numeric(group["au_oz_from"], errors="coerce").fillna(0).sum())
        oz_to = float(pd.to_numeric(group["au_oz_to"], errors="coerce").fillna(0).sum())
        rows.append(
            {
                "Panel": panel,
                "Domain": str(domain_value),
                "Initial category": category,
                "Blocks": len(group),
                "Volume (Mm3)": volume / 1_000_000.0,
                "Initial tonnes (t)": tonnes_from,
                "GC tonnes (t)": tonnes_to,
                "GC support (Mt)": support_mt,
                "Initial Au (g/t)": grade_from,
                "GC Au (g/t)": grade_to,
                "Initial Au (oz)": oz_from,
                "GC Au (oz)": oz_to,
                "Tonnes bias (%)": _bias_pct(tonnes_from, tonnes_to),
                "Au grade bias (%)": _bias_pct(grade_from, grade_to),
                "Au oz bias (%)": _bias_pct(oz_from, oz_to),
            }
        )
    return pd.DataFrame(rows)


def _domain_uncertainty_summary(metrics: pd.DataFrame, tolerance_pct: float) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (domain_value, category), group in metrics.groupby(["Domain", "Initial category"], observed=True):
        metal_bias = pd.to_numeric(group["Au oz bias (%)"], errors="coerce")
        metal_weights = pd.to_numeric(group["GC Au (oz)"], errors="coerce").clip(lower=0)
        oz_from = pd.to_numeric(group["Initial Au (oz)"], errors="coerce")
        oz_to = pd.to_numeric(group["GC Au (oz)"], errors="coerce")
        tonnes_from = pd.to_numeric(group["Initial tonnes (t)"], errors="coerce")
        tonnes_to = pd.to_numeric(group["GC tonnes (t)"], errors="coerce")
        grade_from = _weighted_grade(group, "Initial Au (g/t)", "Initial tonnes (t)")
        grade_to = _weighted_grade(group, "GC Au (g/t)", "GC tonnes (t)")

        p10 = _weighted_quantile(metal_bias, metal_weights, 0.10)
        p50 = _weighted_quantile(metal_bias, metal_weights, 0.50)
        p90 = _weighted_quantile(metal_bias, metal_weights, 0.90)
        finite_metal = metal_bias.dropna()

        rows.append(
            {
                "Domain": domain_value,
                "Initial category": category,
                "Panel-domain cells": len(group),
                "GC support (Mt)": float(tonnes_to.fillna(0).clip(lower=0).sum()) / 1_000_000.0,
                "GC Au (koz)": float(oz_to.fillna(0).clip(lower=0).sum()) / 1_000.0,
                "Aggregate Au oz bias (%)": _bias_pct(float(oz_from.fillna(0).sum()), float(oz_to.fillna(0).sum())),
                "Au oz WAPE (%)": _wape(oz_from, oz_to),
                "Weighted P10 Au oz bias (%)": p10,
                "Weighted P50 Au oz bias (%)": p50,
                "Weighted P90 Au oz bias (%)": p90,
                "Weighted P10-P90 width (%)": p90 - p10 if np.isfinite(p10) and np.isfinite(p90) else np.nan,
                f"GC Au oz within ±{tolerance_pct:g}% (%)": _weighted_tolerance_share(metal_bias, metal_weights, tolerance_pct),
                "Aggregate Au grade bias (%)": _bias_pct(grade_from, grade_to),
                "Aggregate tonnes bias (%)": _bias_pct(float(tonnes_from.fillna(0).sum()), float(tonnes_to.fillna(0).sum())),
                "P90 abs Au oz bias diagnostic (%)": float(finite_metal.abs().quantile(0.90)) if not finite_metal.empty else np.nan,
            }
        )
    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values(["Au oz WAPE (%)", "GC support (Mt)"], ascending=[False, False], kind="stable")
    return table


def _domain_transition(pair: pd.DataFrame, domain: str, top_n: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    from_col = f"{domain}_from"
    to_col = f"{domain}_to"
    if from_col not in pair.columns or to_col not in pair.columns:
        return pd.DataFrame(), pd.DataFrame()

    work = pair[[from_col, to_col, "volume_from"]].copy()
    work[from_col] = _clean_domain(work[from_col])
    work[to_col] = _clean_domain(work[to_col])
    work["volume_from"] = pd.to_numeric(work["volume_from"], errors="coerce").fillna(0).clip(lower=0)

    top = (
        work.groupby(from_col, observed=True)["volume_from"].sum().nlargest(int(top_n)).index.astype(str).tolist()
    )
    work = work[work[from_col].astype(str).isin(top)].copy()
    matrix = (
        work.groupby([from_col, to_col], observed=True)["volume_from"]
        .sum()
        .unstack(fill_value=0.0)
        .reindex(index=top, fill_value=0.0)
    )
    # Keep columns that materially occur for selected initial domains and order
    # them by total volume.
    ordered_cols = matrix.sum(axis=0).sort_values(ascending=False).index.tolist()[: max(int(top_n), 1)]
    matrix = matrix.reindex(columns=ordered_cols, fill_value=0.0)
    pct = matrix.div(matrix.sum(axis=1).replace(0, np.nan), axis=0).mul(100).fillna(0.0)
    return matrix, pct


def _panel_support_distribution(
    frame: pd.DataFrame,
    common_index: pd.MultiIndex,
    sx: float,
    sy: float,
    sz: float,
    origin: tuple[float, float, float],
) -> pd.DataFrame:
    work = _aligned_model(frame, common_index).copy()
    pseudo_pair = pd.DataFrame({"x": work["x"], "y": work["y"], "z": work["z"]}, index=work.index)
    work["Panel"] = _panel_ids(pseudo_pair, sx, sy, sz, origin)
    work["tonnes"] = pd.to_numeric(work["tonnes"], errors="coerce").fillna(0).clip(lower=0)
    work["volume"] = pd.to_numeric(work["volume"], errors="coerce").fillna(0).clip(lower=0)
    return (
        work.groupby("Panel", observed=True)
        .agg(Blocks=("Panel", "size"), Tonnes=("tonnes", "sum"), Volume=("volume", "sum"))
        .reset_index()
        .assign(**{"Tonnes (Mt)": lambda table: table["Tonnes"] / 1_000_000.0, "Volume (Mm3)": lambda table: table["Volume"] / 1_000_000.0})
    )


def _support_sensitivity_table(
    models: dict[str, pd.DataFrame],
    common_index: pd.MultiIndex,
    reference_label: str,
    pair: pd.DataFrame,
    origin: tuple[float, float, float],
    minimum_support_mt: float,
    tolerance_pct: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sx, sy, sz in PANEL_PRESETS:
        support = _panel_support_distribution(models[reference_label], common_index, sx, sy, sz, origin)
        cohort = _panel_cohort_metrics(pair, sx, sy, sz, origin, minimum_support_mt)
        reliability = _uncertainty_summary(cohort, tolerance_pct)
        row: dict[str, Any] = {
            "Panel geometry": f"{sx}×{sy}×{sz} m",
            "Panels": len(support),
            "Median support (Mt)": float(support["Tonnes (Mt)"].median()) if not support.empty else np.nan,
            "P25 support (Mt)": float(support["Tonnes (Mt)"].quantile(0.25)) if not support.empty else np.nan,
            "P75 support (Mt)": float(support["Tonnes (Mt)"].quantile(0.75)) if not support.empty else np.nan,
        }
        for category in ["Measured", "Indicated"]:
            matched = reliability[reliability["Initial category"].eq(category)] if not reliability.empty else pd.DataFrame()
            row[f"{category} cells"] = int(matched["Panels"].iloc[0]) if not matched.empty else 0
            row[f"{category} Aggregate Au oz bias (%)"] = float(matched["Aggregate Au oz bias (%)"].iloc[0]) if not matched.empty else np.nan
            row[f"{category} Au oz WAPE (%)"] = float(matched["Au oz WAPE (%)"].iloc[0]) if not matched.empty else np.nan
            row[f"{category} Weighted P10 Au oz bias (%)"] = float(matched["Weighted P10 Au oz bias (%)"].iloc[0]) if not matched.empty else np.nan
            row[f"{category} Weighted P90 Au oz bias (%)"] = float(matched["Weighted P90 Au oz bias (%)"].iloc[0]) if not matched.empty else np.nan
            row[f"{category} GC Au oz within ±{tolerance_pct:g}% (%)"] = float(matched[f"GC Au oz within ±{tolerance_pct:g}% (%)"].iloc[0]) if not matched.empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows)

def _render_data_tab(
    models: dict[str, pd.DataFrame],
    common_index: pd.MultiIndex,
    labels: list[str],
    inventory: pd.DataFrame,
) -> None:
    st.subheader("Data & Alignment")
    st.markdown(
        """
        <div class="rescat-method"><b>Method.</b> Every snapshot is aligned from XYZ centroids. The analytical population is the intersection of blocks present in all loaded snapshots, so every temporal comparison uses the same spatial population. Resource categories come exclusively from <b>CATEG_GC</b>.</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Loaded snapshots")
    st.dataframe(inventory, use_container_width=True, hide_index=True)

    alignment = _alignment_table(models, common_index)
    min_common_pct = float(alignment["Common blocks (%)"].min()) if not alignment.empty else np.nan
    _render_kpis([
        ("Snapshots", f"{len(labels)}", "Number of annual/quarterly model snapshots loaded."),
        ("Common XYZ blocks", f"{len(common_index):,}", "Blocks present in every loaded model snapshot."),
        ("Minimum common coverage", _format_pct(min_common_pct, 3), "Common XYZ blocks divided by unique XYZ blocks for the least-covered snapshot."),
        ("Duplicate XYZ", f"{int(alignment['Duplicate XYZ rows'].sum()):,}" if not alignment.empty else "0", "Duplicate centroid records across the loaded files."),
    ])

    st.markdown("#### Block-grid alignment")
    st.dataframe(
        alignment.style.format({
            "Common blocks (%)": "{:,.4f}%",
            "Total volume (Mm3)": "{:,.3f}",
            "Common volume (Mm3)": "{:,.3f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    if not alignment.empty and alignment["Duplicate XYZ rows"].sum() > 0:
        st.warning("Duplicate XYZ records were detected. The analysis retains the first occurrence for alignment; the source files should be reviewed.")
    if np.isfinite(min_common_pct) and min_common_pct < 99.0:
        st.warning("Common spatial coverage is below 99%. Review model extents before interpreting stability metrics.")
    else:
        st.success("Spatial alignment is suitable for block-to-block ResCat stability analysis.")

    st.markdown("#### CATEG_GC distribution on the common population")
    evolution = _evolution_table(models, common_index, labels)
    if not evolution.empty:
        focus = evolution[evolution["Category"].isin(RESCAT_ORDER)].copy()
        fig = px.bar(
            focus,
            x="Snapshot",
            y="Volume (%)",
            color="Category",
            category_orders={"Snapshot": labels, "Category": RESCAT_ORDER},
            color_discrete_map=RESCAT_COLORS,
            text="Volume (%)",
            title="Resource-category evolution by model snapshot",
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="inside", marker_line_color="white", marker_line_width=0.8, hovertemplate="Snapshot: %{x}<br>%{fullData.name}: %{y:.1f}%<extra></extra>")
        _apply_barrick_layout(fig, height=500, yaxis_title="Volume (%)", legend_title="CATEG_GC")
        fig.update_layout(barmode="stack")
        fig.update_yaxes(range=[0, 100], ticksuffix="%")
        st.plotly_chart(fig, use_container_width=True)

        pivot = focus.pivot(index="Snapshot", columns="Category", values="Volume (%)").reindex(labels).fillna(0.0)
        pivot = pivot.reindex(columns=RESCAT_ORDER, fill_value=0.0).reset_index()
        st.dataframe(pivot.style.format({column: "{:,.1f}%" for column in RESCAT_ORDER}), use_container_width=True, hide_index=True)



def _render_conversion_tab(
    models: dict[str, pd.DataFrame],
    common_index: pd.MultiIndex,
    labels: list[str],
) -> None:
    st.subheader("ResCat Conversion")
    st.caption("Volume-weighted block-to-block transition matrices using CATEG_GC only. Conversion is not assumed to be sequential: Indicated or Inferred material may move directly to Grade Control.")
    st.markdown(
        """<div class="rescat-note"><b>Important:</b> conversion to Grade Control is descriptive, not a standalone measure of classification quality, because it also reflects where Grade Control drilling was executed. Reliability is evaluated separately in the <b>Meas Reliability</b> tab using later Grade Control as the benchmark state.</div>""",
        unsafe_allow_html=True,
    )

    from_label, to_label = _pair_selectors(labels, "rescat_conversion")
    include_inferred = st.checkbox("Include Inferred in the transition matrix", value=False, key="rescat_conversion_inferred")
    categories = RESCAT_FOCUS_ORDER.copy()
    if include_inferred:
        categories.append("Inferred")

    pair = _pair_frame(models, common_index, from_label, to_label)
    volume, pct = _transition_tables(pair, categories, "volume_from")

    _render_kpis([
        ("I → M", _format_pct(_transition_rate(volume, "Indicated", "Measured")), "Share of initial Indicated volume classified as Measured in the later snapshot."),
        ("I → GC", _format_pct(_transition_rate(volume, "Indicated", "Grade Control")), "Share of initial Indicated volume directly classified as Grade Control in the later snapshot."),
        ("M → GC", _format_pct(_transition_rate(volume, "Measured", "Grade Control")), "Share of initial Measured volume classified as Grade Control in the later snapshot."),
        ("Matched blocks", f"{len(pair):,}", "Common XYZ population used for this model pair."),
    ])

    left, right = st.columns([1.0, 1.15])
    with left:
        st.plotly_chart(
            _heatmap_figure(pct, f"CATEG_GC transition — {from_label} → {to_label}"),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(
            _sankey_figure(volume, f"Volume flow — {from_label} → {to_label}"),
            use_container_width=True,
        )

    st.markdown("#### Transition matrix — volume (Mm³)")
    display_volume = volume / 1_000_000.0
    st.dataframe(display_volume.style.format("{:,.3f}"), use_container_width=True)

    st.markdown("#### Transition matrix — % of initial category")
    st.dataframe(pct.style.format("{:,.1f}%"), use_container_width=True)

    st.markdown("#### Consecutive-model conversion rates")
    rows: list[dict[str, Any]] = []
    for initial, later in zip(labels[:-1], labels[1:], strict=True):
        annual_pair = _pair_frame(models, common_index, initial, later)
        annual_volume, _ = _transition_tables(annual_pair, RESCAT_FOCUS_ORDER, "volume_from")
        rows.append({
            "Transition": f"{initial} → {later}",
            "I → M (%)": _transition_rate(annual_volume, "Indicated", "Measured"),
            "I → GC (%)": _transition_rate(annual_volume, "Indicated", "Grade Control"),
            "M → GC (%)": _transition_rate(annual_volume, "Measured", "Grade Control"),
            "I retained (%)": _transition_rate(annual_volume, "Indicated", "Indicated"),
            "M retained (%)": _transition_rate(annual_volume, "Measured", "Measured"),
            "GC retained (%)": _transition_rate(annual_volume, "Grade Control", "Grade Control"),
        })
    trend = pd.DataFrame(rows)
    if not trend.empty:
        st.dataframe(trend.style.format({column: "{:,.1f}%" for column in trend.columns if column.endswith("(%)")}), use_container_width=True, hide_index=True)

        conversion_long = trend.melt(
            id_vars="Transition",
            value_vars=["I → M (%)", "I → GC (%)", "M → GC (%)"],
            var_name="Metric",
            value_name="Percent",
        )
        fig = px.bar(
            conversion_long,
            x="Transition",
            y="Percent",
            color="Metric",
            barmode="group",
            title="Direct conversion rates by consecutive model pair",
            color_discrete_map={
                "I → M (%)": BARRICK_BLUE,
                "I → GC (%)": BARRICK_GOLD,
                "M → GC (%)": BARRICK_ORANGE,
            },
            text="Percent",
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", marker_line_color="white", marker_line_width=1.0, hovertemplate="Transition: %{x}<br>%{fullData.name}: %{y:.1f}%<extra></extra>")
        _apply_barrick_layout(fig, height=470, yaxis_title="% of initial category", legend_title="Metric")
        st.plotly_chart(fig, use_container_width=True)

        retention_long = trend.melt(
            id_vars="Transition",
            value_vars=["I retained (%)", "M retained (%)", "GC retained (%)"],
            var_name="Metric",
            value_name="Percent",
        )
        fig = px.bar(
            retention_long,
            x="Transition",
            y="Percent",
            color="Metric",
            barmode="group",
            title="Category retention by consecutive model pair",
            color_discrete_map={
                "I retained (%)": BARRICK_GOLD,
                "M retained (%)": BARRICK_BLUE,
                "GC retained (%)": BARRICK_ORANGE,
            },
            text="Percent",
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", marker_line_color="white", marker_line_width=1.0, hovertemplate="Transition: %{x}<br>%{fullData.name}: %{y:.1f}%<extra></extra>")
        _apply_barrick_layout(fig, height=470, yaxis_title="% retained from initial category", legend_title="Metric")
        st.plotly_chart(fig, use_container_width=True)



def _render_measured_reliability_tab(
    models: dict[str, pd.DataFrame],
    common_index: pd.MultiIndex,
    labels: list[str],
    panel_settings: tuple[float, float, float, float, float],
    origin: tuple[float, float, float],
) -> None:
    st.subheader("Meas Reliability")
    st.caption(
        "Grade Control is the later benchmark state. This analysis tests how accurately and consistently "
        "the earlier Measured and Indicated classifications predicted the same material once it reached Grade Control."
    )
    st.markdown(
        """
        <div class="rescat-interpretation">
            <b>Primary reliability framework</b><br>
            <b>Aggregate Au-content bias</b> measures overall accuracy; values close to 0% are better.<br>
            <b>Au-content WAPE</b> measures absolute uncertainty without allowing positive and negative errors to cancel; lower is better.<br>
            <b>Weighted P10–P90 bias</b> shows the central 80% uncertainty interval, weighting each panel cell by later Grade Control Au ounces; a narrower interval centered near 0% is better.<br>
            <b>GC Au ounces within tolerance</b> is the percentage of later benchmark metal represented by panel cells inside the selected ± tolerance; higher is better.
        </div>
        """,
        unsafe_allow_html=True,
    )
    from_label, to_label = _pair_selectors(labels, "rescat_gc_reliability")
    sx, sy, sz, minimum_support_mt, tolerance_pct = panel_settings
    pair = _pair_frame(models, common_index, from_label, to_label)

    summary = _cohort_summary(pair)
    if summary.empty:
        st.info("No Indicated/Measured blocks in the initial snapshot reach Grade Control in the selected later snapshot.")
        return

    st.markdown("#### Aggregate cohort comparison")
    st.caption(
        "This table compares the complete initial Measured or Indicated cohort with the same blocks in the later Grade Control benchmark. "
        "It is the primary accuracy check, but it does not by itself measure local dispersion because positive and negative panel errors can offset each other."
    )
    st.dataframe(
        summary.style.format({
            "Volume (Mm3)": "{:,.3f}",
            "Initial tonnes (Mt)": "{:,.3f}",
            "GC tonnes (Mt)": "{:,.3f}",
            "Tonnes bias (%)": "{:,.2f}%",
            "Initial Au (g/t)": "{:,.3f}",
            "GC Au (g/t)": "{:,.3f}",
            "Au grade bias (%)": "{:,.2f}%",
            "Initial Au (oz)": "{:,.0f}",
            "GC Au (oz)": "{:,.0f}",
            "Au oz bias (%)": "{:,.2f}%",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### Fixed-panel reliability — primary metrics")
    st.caption(
        f"Panel geometry: {sx:g} × {sy:g} × {sz:g} m. Minimum later-GC support: {minimum_support_mt:g} Mt per panel/category cell. "
        f"Reference tolerance: ±{tolerance_pct:g}%. Metal-weighted statistics use later Grade Control Au ounces as weights."
    )
    panel_metrics = _panel_cohort_metrics(pair, sx, sy, sz, origin, minimum_support_mt)
    if panel_metrics.empty:
        st.info("No panel/category cells meet the selected support threshold.")
        return

    reliability = _uncertainty_summary(panel_metrics, tolerance_pct)
    primary_columns = [
        "Initial category",
        "Panels",
        "GC support (Mt)",
        "GC Au (koz)",
        "Aggregate Au oz bias (%)",
        "Au oz WAPE (%)",
        "Weighted P10 Au oz bias (%)",
        "Weighted P50 Au oz bias (%)",
        "Weighted P90 Au oz bias (%)",
        f"GC Au oz within ±{tolerance_pct:g}% (%)",
    ]
    primary = reliability[[column for column in primary_columns if column in reliability.columns]].copy()
    st.dataframe(
        primary.style.format({
            "GC support (Mt)": "{:,.3f}",
            "GC Au (koz)": "{:,.1f}",
            "Aggregate Au oz bias (%)": "{:,.2f}%",
            "Au oz WAPE (%)": "{:,.2f}%",
            "Weighted P10 Au oz bias (%)": "{:,.2f}%",
            "Weighted P50 Au oz bias (%)": "{:,.2f}%",
            "Weighted P90 Au oz bias (%)": "{:,.2f}%",
            f"GC Au oz within ±{tolerance_pct:g}% (%)": "{:,.1f}%",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # Accuracy and absolute uncertainty.
    chart_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    for _, row in reliability.iterrows():
        category = str(row["Initial category"])
        chart_rows.extend([
            {"Initial category": category, "Metric": "Aggregate bias", "Value (%)": float(row.get("Aggregate Au oz bias (%)", np.nan))},
            {"Initial category": category, "Metric": "WAPE", "Value (%)": float(row.get("Au oz WAPE (%)", np.nan))},
        ])
        coverage_rows.append(
            {
                "Initial category": category,
                "GC Au oz within tolerance (%)": float(row.get(f"GC Au oz within ±{tolerance_pct:g}% (%)", np.nan)),
            }
        )

    col1, col2 = st.columns(2)
    with col1:
        accuracy_df = pd.DataFrame(chart_rows)
        fig = px.bar(
            accuracy_df,
            x="Metric",
            y="Value (%)",
            color="Initial category",
            barmode="group",
            category_orders={"Initial category": ["Measured", "Indicated"]},
            color_discrete_map=RESCAT_COLORS,
            title=f"Au-content accuracy and absolute uncertainty — {from_label} → {to_label}",
            text="Value (%)",
        )
        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside",
            marker_line_color="white",
            marker_line_width=1.0,
            hovertemplate="Initial class: %{fullData.name}<br>Metric: %{x}<br>Value: %{y:.1f}%<extra></extra>",
        )
        fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="#59656D")
        _apply_barrick_layout(fig, height=500, yaxis_title="Au-content error (%)", legend_title="Initial category")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        coverage_df = pd.DataFrame(coverage_rows)
        fig = px.bar(
            coverage_df,
            x="Initial category",
            y="GC Au oz within tolerance (%)",
            color="Initial category",
            category_orders={"Initial category": ["Measured", "Indicated"]},
            color_discrete_map=RESCAT_COLORS,
            title=f"Later GC Au ounces represented within ±{tolerance_pct:g}%",
            text="GC Au oz within tolerance (%)",
        )
        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside",
            marker_line_color="white",
            marker_line_width=1.0,
            hovertemplate="Initial class: %{x}<br>GC Au oz within tolerance: %{y:.1f}%<extra></extra>",
        )
        _apply_barrick_layout(fig, height=500, yaxis_title="GC Au ounces within tolerance (%)")
        fig.update_yaxes(range=[0, 105], ticksuffix="%")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Metal-weighted P10-P90 interval.
    st.markdown("#### Metal-weighted Au-content bias interval")
    st.caption(
        "Each panel/category cell contributes according to its later Grade Control Au ounces. The horizontal segment spans weighted P10 to P90, and the marker is weighted P50."
    )
    interval = reliability.copy()
    fig = go.Figure()
    for _, row in interval.iterrows():
        category = str(row["Initial category"])
        p10 = float(row.get("Weighted P10 Au oz bias (%)", np.nan))
        p50 = float(row.get("Weighted P50 Au oz bias (%)", np.nan))
        p90 = float(row.get("Weighted P90 Au oz bias (%)", np.nan))
        if not (np.isfinite(p10) and np.isfinite(p50) and np.isfinite(p90)):
            continue
        color = RESCAT_COLORS.get(category, BARRICK_GRAY)
        fig.add_trace(
            go.Scatter(
                x=[p10, p90],
                y=[category, category],
                mode="lines",
                line=dict(color=color, width=8),
                name=category,
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[p50],
                y=[category],
                mode="markers",
                marker=dict(color="white", line=dict(color=color, width=3), size=13),
                name=category,
                showlegend=False,
                customdata=[[p10, p90]],
                hovertemplate="Initial class: %{y}<br>Weighted P10: %{customdata[0]:.1f}%<br>Weighted P50: %{x:.1f}%<br>Weighted P90: %{customdata[1]:.1f}%<extra></extra>",
            )
        )
    fig.add_vrect(x0=-tolerance_pct, x1=tolerance_pct, fillcolor=_hex_rgba(BARRICK_GOLD, 0.10), line_width=0)
    fig.add_vline(x=-tolerance_pct, line_width=1.2, line_dash="dot", line_color=BARRICK_GOLD)
    fig.add_vline(x=0, line_width=1.2, line_dash="dash", line_color="#59656D")
    fig.add_vline(x=tolerance_pct, line_width=1.2, line_dash="dot", line_color=BARRICK_GOLD)
    _apply_barrick_layout(
        fig,
        height=390,
        title=f"Weighted P10–P90 Au-content bias — {from_label} → {to_label}",
        xaxis_title="Au-content bias relative to later GC (%)",
        yaxis_title="Initial category",
    )
    fig.update_yaxes(categoryorder="array", categoryarray=["Measured", "Indicated"])
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Secondary diagnostics — unweighted panel-cell statistics", expanded=False):
        diagnostic_columns = [
            "Initial category",
            "Panels",
            "Median Tonnes bias (%)",
            "P90 abs Tonnes bias (%)",
            f"Cell count within ±{tolerance_pct:g}% Tonnes bias (%)",
            "Median Au grade bias (%)",
            "P90 abs Au grade bias (%)",
            f"Cell count within ±{tolerance_pct:g}% Au grade bias (%)",
            "Median Au oz bias (%)",
            "P90 abs Au oz bias (%)",
            f"Cell count within ±{tolerance_pct:g}% Au oz bias (%)",
        ]
        diagnostics = reliability[[column for column in diagnostic_columns if column in reliability.columns]].copy()
        formatters = {column: "{:,.2f}%" for column in diagnostics.columns if column not in {"Initial category", "Panels"}}
        st.dataframe(diagnostics.style.format(formatters), use_container_width=True, hide_index=True)
        st.caption(
            "These statistics give every occupied panel cell equal weight. They are retained for diagnostic review, but they are not the primary reliability metrics because very small cells can produce extreme percentage errors."
        )

    with st.expander("Panel-level reliability values", expanded=False):
        st.dataframe(
            panel_metrics.style.format({
                "Volume (Mm3)": "{:,.3f}",
                "Initial tonnes (t)": "{:,.0f}",
                "GC tonnes (t)": "{:,.0f}",
                "GC support (Mt)": "{:,.3f}",
                "Initial Au (g/t)": "{:,.3f}",
                "GC Au (g/t)": "{:,.3f}",
                "Initial Au (oz)": "{:,.1f}",
                "GC Au (oz)": "{:,.1f}",
                "Tonnes bias (%)": "{:,.2f}%",
                "Au grade bias (%)": "{:,.2f}%",
                "Au oz bias (%)": "{:,.2f}%",
            }),
            use_container_width=True,
            hide_index=True,
        )


def _render_domain_tab(
    models: dict[str, pd.DataFrame],
    common_index: pd.MultiIndex,
    labels: list[str],
    panel_settings: tuple[float, float, float, float, float],
    origin: tuple[float, float, float],
) -> None:
    st.subheader("Domain Uncertainty")
    st.caption(
        "Mettype, Lithology and Alteration are uncertainty strata inside the fixed spatial panels. "
        "The objective is to identify domains where material classified as Measured has historically shown low uncertainty against later Grade Control, supporting lower-risk future planning when full GC drilling is not available."
    )

    from_label, to_label = _pair_selectors(labels, "rescat_domain")
    control1, control2, control3 = st.columns([1.2, 1.0, 1.0])
    domain_options = ["Mettype", "Lithology", "Alteration"]
    if "rescat_domain_variable" not in st.session_state:
        st.session_state["rescat_domain_variable"] = "Alteration"
    domain_label = control1.selectbox(
        "Domain variable",
        domain_options,
        key="rescat_domain_variable",
    )
    domain = {"Mettype": "mettype", "Lithology": "lithology", "Alteration": "alteration"}[domain_label]
    stable_domain_only = control2.checkbox(
        "Stable domain only",
        value=False,
        key="rescat_domain_stable_only",
        help="When enabled, reliability is calculated only where the selected categorical domain has the same value in both snapshots.",
    )
    top_n = int(control3.number_input("Top classes in domain matrix", min_value=3, max_value=25, value=12, step=1, key="rescat_domain_top_n"))

    sx, sy, sz, minimum_support_mt, tolerance_pct = panel_settings
    pair = _pair_frame(models, common_index, from_label, to_label)
    metrics = _domain_panel_metrics(
        pair,
        domain,
        sx,
        sy,
        sz,
        origin,
        minimum_support_mt,
        stable_domain_only,
    )
    if metrics.empty:
        st.info("No panel-domain cells meet the selected category and support criteria.")
    else:
        summary = _domain_uncertainty_summary(metrics, tolerance_pct)
        st.markdown("#### Reliability vs later GC benchmark by initial geological domain")
        st.caption(
            f"Domain is assigned from the initial snapshot ({from_label}). Reliability uses only blocks classified as Indicated or Measured initially and Grade Control in {to_label}."
        )
        st.markdown(
            f"""
            <div class="rescat-interpretation">
                <b>Planning interpretation:</b> focus first on the <b>Measured</b> rows. Lower-risk domains should combine a small aggregate Au-content bias, low Au-content WAPE, a narrow weighted P10–P90 interval centered near zero, a high share of later GC ounces inside ±{tolerance_pct:g}%, adequate benchmark support, and a stable geological interpretation.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(
            summary.style.format({
                "GC support (Mt)": "{:,.3f}",
                "GC Au (koz)": "{:,.1f}",
                "Aggregate Au oz bias (%)": "{:,.2f}%",
                "Au oz WAPE (%)": "{:,.2f}%",
                "Weighted P10 Au oz bias (%)": "{:,.2f}%",
                "Weighted P50 Au oz bias (%)": "{:,.2f}%",
                "Weighted P90 Au oz bias (%)": "{:,.2f}%",
                "Weighted P10-P90 width (%)": "{:,.2f}%",
                f"GC Au oz within ±{tolerance_pct:g}% (%)": "{:,.1f}%",
                "Aggregate Au grade bias (%)": "{:,.2f}%",
                "Aggregate tonnes bias (%)": "{:,.2f}%",
                "P90 abs Au oz bias diagnostic (%)": "{:,.2f}%",
            }),
            use_container_width=True,
            hide_index=True,
        )

        plot_table = summary[summary["Panel-domain cells"].gt(0)].copy()
        if not plot_table.empty:
            c1, c2 = st.columns(2)
            with c1:
                fig = px.bar(
                    plot_table,
                    x="Domain",
                    y="Au oz WAPE (%)",
                    color="Initial category",
                    barmode="group",
                    category_orders={"Initial category": ["Measured", "Indicated"]},
                    color_discrete_map=RESCAT_COLORS,
                    title=f"Au-content WAPE by {domain_label}",
                    hover_data={"Panel-domain cells": True, "GC support (Mt)": ":.2f", "GC Au (koz)": ":.1f"},
                )
                fig.update_traces(
                    marker_line_color="white",
                    marker_line_width=1.0,
                    hovertemplate="Domain: %{x}<br>%{fullData.name}<br>Au-content WAPE: %{y:.1f}%<extra></extra>",
                )
                _apply_barrick_layout(fig, height=520, yaxis_title="Au-content WAPE (%)", legend_title="Initial category")
                fig.update_xaxes(tickangle=-35)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                coverage_col = f"GC Au oz within ±{tolerance_pct:g}% (%)"
                fig = px.bar(
                    plot_table,
                    x="Domain",
                    y=coverage_col,
                    color="Initial category",
                    barmode="group",
                    category_orders={"Initial category": ["Measured", "Indicated"]},
                    color_discrete_map=RESCAT_COLORS,
                    title=f"GC Au ounces within ±{tolerance_pct:g}% by {domain_label}",
                )
                fig.update_traces(
                    marker_line_color="white",
                    marker_line_width=1.0,
                    hovertemplate="Domain: %{x}<br>%{fullData.name}<br>GC Au oz within tolerance: %{y:.1f}%<extra></extra>",
                )
                _apply_barrick_layout(fig, height=520, yaxis_title="GC Au ounces within tolerance (%)", legend_title="Initial category")
                fig.update_yaxes(range=[0, 105], ticksuffix="%")
                fig.update_xaxes(tickangle=-35)
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Geological categorical stability")
    st.caption(
        "This matrix is separate from resource-category conversion. It shows whether the geological categorical interpretation itself changed between model snapshots."
    )
    matrix, pct = _domain_transition(pair, domain, top_n=top_n)
    if pct.empty:
        st.info("No geological-domain transition matrix is available for the selected field.")
    else:
        st.plotly_chart(
            _heatmap_figure(
                pct,
                f"{domain_label} stability — {from_label} → {to_label}",
                x_title=f"{domain_label} in {to_label}",
                y_title=f"{domain_label} in {from_label}",
            ),
            use_container_width=True,
        )
        with st.expander("Geological transition volumes (Mm³)", expanded=False):
            st.dataframe((matrix / 1_000_000.0).style.format("{:,.3f}"), use_container_width=True)



def _render_panel_tab(
    models: dict[str, pd.DataFrame],
    common_index: pd.MultiIndex,
    labels: list[str],
    panel_settings: tuple[float, float, float, float, float],
    origin: tuple[float, float, float],
) -> None:
    st.subheader("Panel Support")
    st.caption(
        "Panels are fixed regular spatial support containers; they are not mine-plan panels. "
        "The same XYZ-based geometry is applied to every snapshot and geological domain."
    )
    sx, sy, sz, minimum_support_mt, tolerance_pct = panel_settings
    reference_label = st.selectbox("Reference snapshot for support distribution", labels, index=0, key="rescat_panel_reference")
    support = _panel_support_distribution(models[reference_label], common_index, sx, sy, sz, origin)

    if support.empty:
        st.info("No panels could be constructed from the common block population.")
        return

    _render_kpis([
        ("Panels", f"{len(support):,}", "Number of occupied regular spatial panels."),
        ("Median support", f"{support['Tonnes (Mt)'].median():,.2f} Mt", "Median total panel tonnage in the reference snapshot."),
        ("P25–P75 support", f"{support['Tonnes (Mt)'].quantile(.25):,.2f}–{support['Tonnes (Mt)'].quantile(.75):,.2f} Mt", "Interquartile range of occupied panel tonnage."),
        ("Geometry", f"{sx:g}×{sy:g}×{sz:g} m", "Current fixed panel dimensions."),
    ])

    fig = px.histogram(
        support,
        x="Tonnes (Mt)",
        nbins=40,
        title=f"Panel-support distribution — {reference_label}",
    )
    _apply_histogram_style(fig, height=450, yaxis_title="Panel count", xaxis_title="Panel support (Mt)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Support sensitivity")
    pair_from, pair_to = _pair_selectors(labels, "rescat_support")
    pair = _pair_frame(models, common_index, pair_from, pair_to)
    sensitivity = _support_sensitivity_table(
        models,
        common_index,
        reference_label,
        pair,
        origin,
        minimum_support_mt,
        tolerance_pct,
    )
    st.dataframe(
        sensitivity.style.format({
            "Median support (Mt)": "{:,.2f}",
            "P25 support (Mt)": "{:,.2f}",
            "P75 support (Mt)": "{:,.2f}",
            "Measured Aggregate Au oz bias (%)": "{:,.2f}%",
            "Measured Au oz WAPE (%)": "{:,.2f}%",
            "Measured Weighted P10 Au oz bias (%)": "{:,.2f}%",
            "Measured Weighted P90 Au oz bias (%)": "{:,.2f}%",
            f"Measured GC Au oz within ±{tolerance_pct:g}% (%)": "{:,.1f}%",
            "Indicated Aggregate Au oz bias (%)": "{:,.2f}%",
            "Indicated Au oz WAPE (%)": "{:,.2f}%",
            "Indicated Weighted P10 Au oz bias (%)": "{:,.2f}%",
            "Indicated Weighted P90 Au oz bias (%)": "{:,.2f}%",
            f"Indicated GC Au oz within ±{tolerance_pct:g}% (%)": "{:,.1f}%",
        }),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown(
        """
        <div class="rescat-note"><b>Interpretation:</b> use the sensitivity table to verify that the relative separation between Measured and Indicated is not an artifact of one arbitrary panel size. A robust classification system should show consistently lower WAPE, a tighter metal-weighted P10–P90 interval, and higher GC-ounce tolerance coverage for Measured than for Indicated across reasonable spatial supports.</div>
        """,
        unsafe_allow_html=True,
    )




def _build_rescat_report_package(
    models: dict[str, pd.DataFrame],
    common_index: pd.MultiIndex,
    labels: list[str],
    inventory: pd.DataFrame,
    panel_settings: tuple[float, float, float, float, float],
    origin: tuple[float, float, float],
    mapping: dict[str, str | None],
) -> dict[str, Any]:
    """Build a compact report payload for the independent Report Builder.

    The payload stores report-ready summary tables and the minimum chart data
    needed to rebuild publication-quality static figures without retaining the
    full uploaded block-model dataframes in report state.
    """
    if len(labels) < 2:
        return {}

    sx, sy, sz, minimum_support_mt, tolerance_pct = panel_settings
    initial_label, later_label = labels[0], labels[-1]
    pair = _pair_frame(models, common_index, initial_label, later_label)
    volume, pct = _transition_tables(pair, RESCAT_FOCUS_ORDER, "volume_from")

    tables: list[dict[str, Any]] = []

    settings = pd.DataFrame(
        [
            {"Parameter": "Snapshots loaded", "Value": len(labels)},
            {"Parameter": "Initial snapshot", "Value": initial_label},
            {"Parameter": "Later benchmark snapshot", "Value": later_label},
            {"Parameter": "Common XYZ blocks", "Value": len(common_index)},
            {"Parameter": "Category source", "Value": "CATEG_GC"},
            {"Parameter": "Panel geometry", "Value": f"{sx:g}×{sy:g}×{sz:g} m"},
            {"Parameter": "Minimum benchmark GC support/cell", "Value": f"{minimum_support_mt:g} Mt"},
            {"Parameter": "Reference tolerance", "Value": f"±{tolerance_pct:g}%"},
        ]
    )
    tables.append(
        {
            "title": "ResCat Stability - Study Configuration",
            "table": settings,
            "notes": [
                "Snapshots are aligned block-to-block using XYZ centroids.",
                "The oldest loaded snapshot is the initial state and the most recent loaded snapshot is the later benchmark state.",
                "CATEG_GC is the sole resource-category variable used by this module.",
            ],
        }
    )

    alignment = _alignment_table(models, common_index)
    if not alignment.empty:
        tables.append(
            {
                "title": "ResCat Stability - Data Alignment",
                "table": alignment,
                "notes": ["Common XYZ defines the spatial population used in temporal comparisons."],
            }
        )

    evolution = _evolution_table(models, common_index, labels)
    if not evolution.empty:
        tables.append(
            {
                "title": "ResCat Stability - Category Evolution",
                "table": evolution,
                "notes": ["Category shares are volume-weighted on the common XYZ population."],
            }
        )

    if not volume.empty:
        tables.append(
            {
                "title": "ResCat Conversion - Transition Volume (Mm3)",
                "table": (volume / 1_000_000.0).reset_index().rename(columns={"rescat_from": "Initial category", "index": "Initial category"}),
                "notes": [f"Initial snapshot: {initial_label}.", f"Later benchmark snapshot: {later_label}."],
            }
        )
    if not pct.empty:
        tables.append(
            {
                "title": "ResCat Conversion - Transition Share of Initial Category",
                "table": pct.reset_index().rename(columns={"rescat_from": "Initial category", "index": "Initial category"}),
                "notes": ["Each row is normalized to 100% of the initial resource category."],
            }
        )

    consecutive_rows: list[dict[str, Any]] = []
    for initial, later in zip(labels[:-1], labels[1:], strict=True):
        annual_pair = _pair_frame(models, common_index, initial, later)
        annual_volume, _ = _transition_tables(annual_pair, RESCAT_FOCUS_ORDER, "volume_from")
        consecutive_rows.append(
            {
                "Transition": f"{initial} → {later}",
                "I → M (%)": _transition_rate(annual_volume, "Indicated", "Measured"),
                "I → GC (%)": _transition_rate(annual_volume, "Indicated", "Grade Control"),
                "M → GC (%)": _transition_rate(annual_volume, "Measured", "Grade Control"),
                "I retained (%)": _transition_rate(annual_volume, "Indicated", "Indicated"),
                "M retained (%)": _transition_rate(annual_volume, "Measured", "Measured"),
                "GC retained (%)": _transition_rate(annual_volume, "Grade Control", "Grade Control"),
            }
        )
    consecutive = pd.DataFrame(consecutive_rows)
    if not consecutive.empty:
        tables.append(
            {
                "title": "ResCat Conversion - Consecutive Model Rates",
                "table": consecutive,
                "notes": ["Conversion and retention are reported as percentages of the corresponding initial category."],
            }
        )

    cohort = pd.DataFrame()
    panel_metrics = pd.DataFrame()
    uncertainty = pd.DataFrame()
    support = pd.DataFrame()
    sensitivity = pd.DataFrame()
    domain_summaries: dict[str, pd.DataFrame] = {}
    domain_stability: dict[str, pd.DataFrame] = {}

    if mapping.get("au"):
        cohort = _cohort_summary(pair)
        if not cohort.empty:
            tables.append(
                {
                    "title": "Meas Reliability - Aggregate Cohort Comparison",
                    "table": cohort,
                    "notes": [
                        "Later Grade Control is used as the benchmark state.",
                        "Positive bias means the earlier estimate overpredicted the later Grade Control result; negative bias means underprediction.",
                    ],
                }
            )

        panel_metrics = _panel_cohort_metrics(pair, sx, sy, sz, origin, minimum_support_mt)
        uncertainty = _uncertainty_summary(panel_metrics, tolerance_pct)
        if not uncertainty.empty:
            tables.append(
                {
                    "title": "Meas Reliability - Fixed Panel Summary",
                    "table": uncertainty,
                    "notes": [
                        f"Fixed panel geometry: {sx:g}×{sy:g}×{sz:g} m.",
                        f"Reference tolerance: ±{tolerance_pct:g}%.",
                        "Measured reliability is assessed against the later Grade Control benchmark and compared with Indicated as a reference population.",
                    ],
                }
            )

        for domain_label, domain in (("Mettype", "mettype"), ("Lithology", "lithology"), ("Alteration", "alteration")):
            if not mapping.get(domain):
                continue
            metrics = _domain_panel_metrics(
                pair,
                domain,
                sx,
                sy,
                sz,
                origin,
                minimum_support_mt,
                False,
            )
            summary = _domain_uncertainty_summary(metrics, tolerance_pct)
            if not summary.empty:
                domain_summaries[domain_label] = summary
                tables.append(
                    {
                        "title": f"Meas Reliability by {domain_label}",
                        "table": summary,
                        "notes": [
                            f"{domain_label} is assigned from the initial snapshot.",
                            "Reliability uses blocks initially classified as Measured or Indicated that are Grade Control in the later benchmark snapshot.",
                            "Domain subvolumes retain their actual support; equal domain volumes are not imposed.",
                        ],
                    }
                )
            _, domain_pct = _domain_transition(pair, domain, top_n=12)
            if not domain_pct.empty:
                domain_stability[domain_label] = domain_pct
                tables.append(
                    {
                        "title": f"{domain_label} - Geological Categorical Stability (%)",
                        "table": domain_pct.reset_index(),
                        "notes": ["Rows are normalized to the initial geological category and show categorical interpretation stability between snapshots."],
                    }
                )

        support = _panel_support_distribution(models[initial_label], common_index, sx, sy, sz, origin)
        sensitivity = _support_sensitivity_table(
            models,
            common_index,
            initial_label,
            pair,
            origin,
            minimum_support_mt,
            tolerance_pct,
        )
        if not sensitivity.empty:
            tables.append(
                {
                    "title": "Panel Support - Sensitivity",
                    "table": sensitivity,
                    "notes": ["Support sensitivity checks whether the separation between Measured and Indicated is robust across reasonable fixed spatial panel sizes."],
                }
            )

    return {
        "module": "ResCat Stability",
        "snapshot_labels": list(labels),
        "initial_label": initial_label,
        "later_label": later_label,
        "common_blocks": int(len(common_index)),
        "panel_geometry": f"{sx:g}×{sy:g}×{sz:g} m",
        "tolerance_pct": float(tolerance_pct),
        "minimum_support_mt": float(minimum_support_mt),
        "tables": tables,
        "chart_data": {
            "evolution": evolution,
            "transition_volume_mm3": volume / 1_000_000.0 if not volume.empty else pd.DataFrame(),
            "transition_pct": pct,
            "consecutive": consecutive,
            "uncertainty": uncertainty,
            "panel_support": support,
            "domain_summaries": domain_summaries,
            "domain_stability": domain_stability,
        },
    }

def render_rescat_stability() -> None:
    """Render the dedicated ResCat Stability workflow."""
    _apply_rescat_styles()
    header_placeholder = st.empty()
    with header_placeholder.container():
        _render_page_header()

    st.markdown(
        """
        <div class="rescat-note"><b>Scope:</b> this module is independent of the mine-plan Year/Phase/Destination filters. Upload the undepleted model snapshots directly here. The analytical population is aligned by XYZ centroid and <b>CATEG_GC</b> is the sole resource-category source. The oldest loaded snapshot is treated as the initial state and the most recent loaded snapshot is treated as the benchmark later state.</div>
        """,
        unsafe_allow_html=True,
    )

    uploads = st.file_uploader(
        "Upload ResCat model snapshots (.csv)",
        type=["csv", "txt"],
        accept_multiple_files=True,
        key="rescat_snapshot_uploads",
        help="Upload two or more undepleted model snapshots ordered through their filenames or model-period labels. Each snapshot should include XYZ centroids and CATEG_GC.",
    )

    if not uploads or len(uploads) < 2:
        st.info("Upload at least two model snapshots to activate ResCat Stability. Additional intermediate snapshots are recommended when you want to evaluate temporal consistency between model updates.")
        return

    payloads = {upload.name: upload.getvalue() for upload in uploads}
    metas = {name: _snapshot_meta(name) for name in payloads}
    ordered_names = [name for name, _ in sorted(metas.items(), key=lambda item: item[1].sort_key)]

    # Labels must be unique even when filenames contain the same year/quarter.
    label_counts: dict[str, int] = {}
    labels_by_name: dict[str, str] = {}
    for name in ordered_names:
        base = metas[name].label
        label_counts[base] = label_counts.get(base, 0) + 1
        labels_by_name[name] = base if label_counts[base] == 1 else f"{base} ({label_counts[base]})"

    try:
        headers = {name: _csv_header(payloads[name]) for name in ordered_names}
    except Exception as exc:
        st.error(f"Unable to read one or more CSV headers: {exc}")
        return

    common_columns = sorted(set(headers[ordered_names[0]]).intersection(*(set(headers[name]) for name in ordered_names[1:])))
    if not common_columns:
        st.error("The uploaded snapshots do not share any common column names.")
        return

    with st.expander("Data mapping and analytical settings", expanded=True):
        mapping = _mapping_controls(common_columns)
        coordinate_decimals = int(
            st.number_input(
                "XYZ matching decimals",
                min_value=0,
                max_value=6,
                value=3,
                step=1,
                key="rescat_coordinate_decimals",
                help="Centroid coordinates are rounded to this precision before block-to-block alignment.",
            )
        )
        st.markdown("#### Fixed spatial support")
        p1, p2, p3, p4, p5 = st.columns(5)
        sx = float(p1.number_input("Panel X (m)", min_value=10.0, value=200.0, step=10.0, key="rescat_panel_x"))
        sy = float(p2.number_input("Panel Y (m)", min_value=10.0, value=200.0, step=10.0, key="rescat_panel_y"))
        sz = float(p3.number_input("Panel Z (m)", min_value=10.0, value=30.0, step=10.0, key="rescat_panel_z"))
        minimum_support_mt = float(
            p4.number_input(
                "Min benchmark GC support/cell (Mt)",
                min_value=0.0,
                value=0.0,
                step=0.05,
                key="rescat_min_support_mt",
                help="Minimum later-GC tonnage required for a panel/category or panel/domain/category cell to enter uncertainty statistics. Use 0 to retain every occupied cell.",
            )
        )
        tolerance_pct = float(
            p5.number_input(
                "Reference tolerance (±%)",
                min_value=1.0,
                max_value=100.0,
                value=15.0,
                step=1.0,
                key="rescat_tolerance_pct",
                help="Reference tolerance used for the within-band statistics. It does not change the underlying bias calculations.",
            )
        )

    mapping_issues = _validate_mapping(mapping)
    fatal_issues = [issue for issue in mapping_issues if not issue.startswith("Map Au")]
    if fatal_issues:
        st.error("Mapping required: " + "; ".join(fatal_issues) + ".")
        return
    if any(issue.startswith("Map Au") for issue in mapping_issues):
        st.warning("Au is not mapped. Category conversion remains available, but Meas Reliability and Au-content uncertainty will be limited.")

    inventory = _model_inventory(metas, headers, payloads)
    inventory["Snapshot"] = inventory["File"].map(labels_by_name)

    models: dict[str, pd.DataFrame] = {}
    try:
        with st.spinner("Loading mapped ResCat variables and aligning model snapshots..."):
            signature = _mapping_signature(mapping)
            for name in ordered_names:
                frame = _load_snapshot(payloads[name], signature, coordinate_decimals)
                models[labels_by_name[name]] = frame
    except Exception as exc:
        st.error(f"Unable to load the mapped model data: {exc}")
        return

    labels = [labels_by_name[name] for name in ordered_names]
    with header_placeholder.container():
        _render_page_header(
            snapshot_count=len(labels),
            initial_label=labels[0] if labels else None,
            later_label=labels[-1] if labels else None,
        )
    if any(frame.empty for frame in models.values()):
        st.error("At least one snapshot is empty after loading the mapped variables.")
        return

    if any(frame.index.duplicated(keep=False).any() for frame in models.values()):
        st.warning("Duplicate XYZ coordinates exist in at least one snapshot. Review Data & Alignment before interpreting results.")

    common_index = _common_index(models)
    if len(common_index) == 0:
        st.error("No common XYZ centroids were found across the uploaded snapshots.")
        return

    reference = _aligned_model(models[labels[0]], common_index)
    origin = (
        float(pd.to_numeric(reference["x"], errors="coerce").min()),
        float(pd.to_numeric(reference["y"], errors="coerce").min()),
        float(pd.to_numeric(reference["z"], errors="coerce").min()),
    )
    panel_settings = (sx, sy, sz, minimum_support_mt, tolerance_pct)

    try:
        st.session_state["rescat_report_package"] = _build_rescat_report_package(
            models,
            common_index,
            labels,
            inventory,
            panel_settings,
            origin,
            mapping,
        )
    except Exception as exc:
        # The analytical module remains usable even if report preparation fails.
        st.session_state["rescat_report_package"] = {}
        st.warning(f"ResCat analysis is available, but the Report Builder package could not be refreshed: {exc}")

    tabs = st.tabs([
        "Data & Alignment",
        "ResCat Conversion",
        "Meas Reliability",
        "Domain Uncertainty",
        "Panel Support",
    ])

    with tabs[0]:
        _render_data_tab(models, common_index, labels, inventory)
    with tabs[1]:
        _render_conversion_tab(models, common_index, labels)
    with tabs[2]:
        if mapping.get("au"):
            _render_measured_reliability_tab(models, common_index, labels, panel_settings, origin)
        else:
            st.info("Map Au grade to activate Meas Reliability.")
    with tabs[3]:
        if mapping.get("au"):
            _render_domain_tab(models, common_index, labels, panel_settings, origin)
        else:
            st.info("Map Au grade to activate Domain Uncertainty reliability metrics.")
    with tabs[4]:
        if mapping.get("au"):
            _render_panel_tab(models, common_index, labels, panel_settings, origin)
        else:
            st.info("Map Au grade to activate support-sensitivity reliability metrics.")
