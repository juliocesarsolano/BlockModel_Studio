from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.core.models import TONNAGE_UNITS
from src.core.parameters import master_filter_defaults


ROOT = Path(__file__).resolve().parents[2]

CATEGORY_COLORS = ["#03547C", "#A39161", "#48A9A6", "#F16349", "#7E57C2", "#607D8B", "#8BC34A", "#FFB300"]
MODEL_COLORS = ["#03547C", "#A39161", "#F16349", "#48A9A6", "#7E57C2"]

MASTER_BLK_MODEL_OPTIONS = {
    "1 - In situ": [1],
    "2 - Stockpiles": [2],
    "1+2 - In situ + stockpiles": [1, 2],
}


def _default_master_scope_label() -> str:
    default_value = int(master_filter_defaults().get("default_blk_model", 1))
    return "2 - Stockpiles" if default_value == 2 else "1 - In situ"


def selected_master_blk_model_values() -> list[int]:
    label = st.session_state.get("master_blk_model_scope", _default_master_scope_label())
    return MASTER_BLK_MODEL_OPTIONS.get(label, [1])


def apply_styles() -> None:
    css_path = ROOT / "assets" / "styles.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def initialize_state() -> None:
    defaults: dict[str, Any] = {
        "models": {},
        "scenes": [],
        "setup_raw": None,
        "setup_filename": "",
        "workflow_mode": "Evaluate block model",
        "nav_page": "Home",
        "master_blk_model_scope": _default_master_scope_label(),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_master_scope_sidebar() -> None:
    """Global app scope applied to calculations, tables and plots."""
    st.sidebar.markdown(
        """
        <div class="bm-side-banner">
            <div class="bm-banner-kicker">Model Source (BLK-Model)</div>
            <div class="bm-banner-title">BLK_MODEL Filter</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    options = list(MASTER_BLK_MODEL_OPTIONS)
    current = st.session_state.get("master_blk_model_scope", _default_master_scope_label())
    index = options.index(current) if current in options else 0
    st.sidebar.selectbox(
        "SOURCE",
        options=options,
        index=index,
        key="master_blk_model_scope",
        help="1 = in situ model, 2 = stockpiles. BLK_MODEL = 0 is always excluded.",
    )


def sidebar_brand() -> None:
    st.sidebar.markdown(
        """
        <div class="bm-sidebar-brand">
            <div class="bm-sidebar-title">BlockModel Studio</div>
            <div class="bm-sidebar-subtitle">Evaluation & Comparison App</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def model_status_line() -> None:
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Configured models: **{len(st.session_state.models)}**")
    st.sidebar.caption(f"Report scenes: **{len(st.session_state.scenes)}**")


def render_workspace_banners() -> None:
    left, right = st.columns([1.0, 1.45])

    with left:
        st.markdown(
            """
            <div class="bm-top-banner">
                <div class="bm-banner-kicker">Workflow</div>
                <div class="bm-banner-title">Evaluate or Compare Block Models</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        workflow = st.radio(
            "Workflow Mode",
            ["Evaluate block model", "Compare block models"],
            horizontal=True,
            key="workflow_mode",
            label_visibility="collapsed",
        )
        buttons = st.columns(2)
        if buttons[0].button("Open setup", use_container_width=True):
            move_to("Model Setup")
        if buttons[1].button("Open selected flow", use_container_width=True):
            move_to("Model Comparison" if workflow == "Compare block models" else "Model Evaluation")

    with right:
        st.markdown(
            """
            <div class="bm-top-banner">
                <div class="bm-banner-kicker">Preferences</div>
                <div class="bm-banner-title">Tonnage, decimals and reporting parameters</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if not st.session_state.models:
            st.info("Configure at least one model to activate quick preferences.")
            return

        selected_model = st.selectbox("Model preferences", list(st.session_state.models), key="banner_model_preferences")
        bundle = st.session_state.models[selected_model]
        config = bundle.config
        columns = st.columns(3)
        current_unit = config.tonnage_unit if config.tonnage_unit in TONNAGE_UNITS else "Mt"
        config.tonnage_unit = columns[0].selectbox(
            "Tonnage unit",
            TONNAGE_UNITS,
            index=TONNAGE_UNITS.index(current_unit),
            key=f"banner_tonnage_unit_{selected_model}",
        )
        config.tonnage_decimals = int(
            columns[1].number_input(
                "Tonnage decimals",
                min_value=0,
                max_value=6,
                value=int(config.tonnage_decimals),
                key=f"banner_tonnage_decimals_{selected_model}",
            )
        )
        config.grade_decimals = int(
            columns[2].number_input(
                "Grade decimals",
                min_value=0,
                max_value=6,
                value=int(config.grade_decimals),
                key=f"banner_grade_decimals_{selected_model}",
            )
        )


def move_to(page_name: str) -> None:
    st.session_state.nav_page = page_name
    st.rerun()


def _formatters(table: pd.DataFrame, config) -> dict[str, str]:
    formatters: dict[str, str] = {}
    for column in table.columns:
        lower = column.lower()
        if column.startswith("Tonnage"):
            formatters[column] = f"{{:,.{config.tonnage_decimals}f}}"
        elif column.startswith("Volume"):
            formatters[column] = "{:,.2f}"
        elif any(token in lower for token in ["grade", "au", "ag", "cu", "ppm", "g/t", "%", "s_", "c_"]):
            formatters[column] = f"{{:,.{config.grade_decimals}f}}"
        elif any(token in lower for token in ["variance", "pct", "percentage"]):
            formatters[column] = "{:,.4f}"
    return formatters


def format_table(table: pd.DataFrame, config) -> pd.io.formats.style.Styler:
    return table.style.format(_formatters(table, config), na_rep="N/A")
