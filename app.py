"""
PV BlockModel Studio
====================

Application entry point for the Streamlit-based block-model evaluation,
comparison and reporting platform.

Project
-------
PV BlockModel Studio

Module
------
app.py

Purpose
-------
This module initializes the Streamlit application, loads shared interface
components, defines the main navigation structure and dispatches the selected
workspace page.

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
1.0

Year
----
2026

Documentation review
--------------------
2026-07-15

Copyright
---------
© 2026 Julio Solano. All rights reserved.

Maintenance notes
-----------------
- Keep this file limited to application bootstrap and page routing.
- Place analytical calculations and page-specific logic in the appropriate
  modules under ``src``.
- Preserve the page names used by session-state navigation helpers.
- ``st.set_page_config`` must remain the first Streamlit command executed.
- Imports marked with ``# noqa: E402`` intentionally occur after the project
  root is added to ``sys.path``.

Technical-use notice
--------------------
This application supports block-model validation, mineral-resource analysis,
model comparison and reporting. Results must be reviewed and validated by an
appropriately qualified mineral-resource professional before official use.
"""

from __future__ import annotations

# =============================================================================
# Python standard-library imports
# =============================================================================
# ``sys`` is used to register the project root in Python's module search path.
# ``Path`` provides platform-independent filesystem path handling.
import sys
from pathlib import Path

# =============================================================================
# Third-party application framework
# =============================================================================
# Streamlit supplies the web interface, session state, sidebar and page layout.
import streamlit as st


# =============================================================================
# Project path initialization
# =============================================================================
# Resolve the directory containing this file. Because ``app.py`` is the project
# entry point, this directory is also the repository/application root.
ROOT = Path(__file__).resolve().parent

# Add the project root to ``sys.path`` only when it is not already registered.
# This permits absolute imports such as ``src.ui.common`` when the app is run
# locally, from Streamlit Community Cloud or from another supported environment.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# =============================================================================
# Shared UI infrastructure
# =============================================================================
# These imports intentionally occur after the project root has been registered.
# ``# noqa: E402`` informs code-quality tools that the delayed import is
# deliberate rather than an accidental import-order violation.
from src.ui.common import (  # noqa: E402
    apply_styles,                 # Load the shared CSS and visual identity.
    initialize_state,             # Create required Streamlit session defaults.
    model_status_line,            # Show configured-model/report status.
    render_master_scope_sidebar,  # Render the global BLK_MODEL scope control.
    sidebar_brand,                # Render the application identity in sidebar.
)

# =============================================================================
# Page-rendering functions
# =============================================================================
# Each function owns the interface and workflow for one navigation destination.
# Keeping page logic in ``src.ui.pages`` leaves this entry point compact and
# focused on initialization and routing.
from src.ui.pages import (  # noqa: E402
    render_about,
    render_comparison,
    render_evaluation,
    render_home,
    render_quality,
    render_reports,
    render_setup,
)


# =============================================================================
# Streamlit page configuration
# =============================================================================
# This must be the first Streamlit command executed. It controls the browser
# title, page icon, application width and initial sidebar state.
st.set_page_config(
    page_title="BlockModel Studio",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# Global application initialization
# =============================================================================
# The initialization order is intentional:
# 1. Prepare session-state variables used by all pages.
# 2. Apply the shared visual theme.
# 3. Display the application brand.
# 4. Render the transversal BLK_MODEL master filter.
initialize_state()
apply_styles()
sidebar_brand()
render_master_scope_sidebar()


# =============================================================================
# Main navigation registry
# =============================================================================
# Dictionary order determines the order displayed in the sidebar. The values
# are function references; only the function corresponding to the selected page
# is executed at the end of this module.
#
# Important:
# - Keep ``About`` as the final informational page.
# - Preserve these labels because session-state navigation helpers may use them.
PAGES = {
    "Home": render_home,
    "Model Setup": render_setup,
    "Model Description": render_quality,
    "Model Evaluation": render_evaluation,
    "Model Comparison": render_comparison,
    "Report Builder": render_reports,
    "About": render_about,
}


# =============================================================================
# Sidebar page selector
# =============================================================================
# ``nav_page`` is shared with programmatic navigation helpers such as
# ``move_to``. Hiding the label keeps the sidebar compact while retaining the
# internal widget identity and accessibility metadata.
selected_page = st.sidebar.radio(
    "Workspace",
    list(PAGES),
    key="nav_page",
    label_visibility="collapsed",
)


# =============================================================================
# Sidebar status and selected-page dispatch
# =============================================================================
# Show the current application/model status beneath the navigation control.
model_status_line()

# Execute the renderer associated with the selected navigation label.
PAGES[selected_page]()
