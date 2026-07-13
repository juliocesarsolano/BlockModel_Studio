from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ui.common import (  # noqa: E402
    apply_styles,
    initialize_state,
    model_status_line,
    render_master_scope_sidebar,
    sidebar_brand,
)
from src.ui.pages import (  # noqa: E402
    render_comparison,
    render_evaluation,
    render_home,
    render_quality,
    render_reports,
    render_setup,
)


st.set_page_config(
    page_title="BlockModel Studio",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

initialize_state()
apply_styles()
sidebar_brand()
render_master_scope_sidebar()

PAGES = {
    "Home": render_home,
    "Model Setup": render_setup,
    "Model Description": render_quality,
    "Model Evaluation": render_evaluation,
    "Model Comparison": render_comparison,
    "Report Builder": render_reports,
}

selected_page = st.sidebar.radio(
    "Workspace",
    list(PAGES),
    key="nav_page",
    label_visibility="collapsed",
)
model_status_line()
PAGES[selected_page]()
