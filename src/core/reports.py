"""
PV BlockModel Studio
====================

Report-export utilities for converting configured report scenes into Excel and
PDF deliverables.

Project
-------
PV BlockModel Studio

Module
------
src/core/reports.py

Purpose
-------
This module provides the export layer used by Report Builder. It converts a
collection of ``Scene`` objects into:

- A multi-sheet Excel workbook containing scene metadata and tables.
- A compact PDF summary listing the report title, scene type, associated models,
  creation date and table row count.

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
- Public function names, arguments and return types are preserved.
- The module remains compatible with ``pages.py`` and the complete Streamlit
  application.
- Excel exports retain one metadata sheet and one data sheet per scene.
- PDF exports retain the same compact scene-summary structure.
- Empty report collections continue to generate a valid explanatory export.
- Missing ReportLab support continues to return a byte message instead of
  raising an application-stopping exception.

Maintenance notes
-----------------
- Keep report-scene assembly in the UI/report builder and file serialization in
  this module.
- Preserve sheet-name uniqueness and Excel's 31-character worksheet-name limit.
- Keep PDF text defensive because scene titles and notes may contain characters
  unsupported by ReportLab's built-in Helvetica fonts.
- Use shared export helpers from ``src.core.io`` rather than duplicating Excel
  writer logic.
"""

from __future__ import annotations

# =============================================================================
# Python standard-library imports
# =============================================================================
# ``BytesIO`` stores the generated PDF entirely in memory so Streamlit can
# expose it directly through a download button without creating temporary files.
# ``re`` supports safe Excel worksheet-name cleanup.
# ``textwrap`` splits long PDF text into readable lines.
from io import BytesIO
import re
import textwrap

# =============================================================================
# Third-party tabular library
# =============================================================================
# pandas is used to construct scene metadata tables and workbook sheets.
import pandas as pd

# =============================================================================
# Internal project dependencies
# =============================================================================
# ``dataframe_to_excel_bytes`` owns the actual Excel serialization.
# ``Scene`` defines the report-scene structure consumed by both exporters.
from .io import dataframe_to_excel_bytes
from .models import Scene


# =============================================================================
# Report-export constants
# =============================================================================
# Excel limits worksheet names to 31 characters and rejects the characters
# ``[]:*?/\``. The generated names are already short, but the helper below
# keeps the rule explicit and reusable.
_EXCEL_SHEET_NAME_LIMIT = 31

# PDF layout values are centralized to keep pagination and spacing consistent.
_PDF_LEFT_MARGIN = 50
_PDF_TOP_MARGIN = 50
_PDF_BOTTOM_MARGIN = 70
_PDF_TITLE_FONT_SIZE = 14
_PDF_SCENE_TITLE_FONT_SIZE = 11
_PDF_BODY_FONT_SIZE = 8
_PDF_SCENE_GAP = 20


# =============================================================================
# Internal Excel helpers
# =============================================================================
def _safe_sheet_name(name: str, used_names: set[str]) -> str:
    """Return a valid, unique Excel worksheet name.

    Parameters
    ----------
    name:
        Proposed worksheet name.
    used_names:
        Names already assigned in the current workbook.

    Returns
    -------
    str
        A worksheet name that satisfies Excel's character and length rules.
    """

    cleaned = re.sub(r"[\[\]:*?/\\]", "_", str(name)).strip()
    cleaned = cleaned or "Sheet"
    cleaned = cleaned[:_EXCEL_SHEET_NAME_LIMIT]

    candidate = cleaned
    suffix_number = 1

    while candidate.casefold() in used_names:
        suffix = f"_{suffix_number}"
        candidate = (
            cleaned[: _EXCEL_SHEET_NAME_LIMIT - len(suffix)]
            + suffix
        )
        suffix_number += 1

    used_names.add(candidate.casefold())
    return candidate


def _scene_table(scene: Scene) -> pd.DataFrame:
    """Return a defensive dataframe copy for one report scene.

    ``Scene.table`` is expected to be a dataframe. The conversion fallback
    prevents a malformed scene payload from stopping the complete export.
    """

    if isinstance(scene.table, pd.DataFrame):
        return scene.table.copy()

    try:
        return pd.DataFrame(scene.table)
    except Exception:
        return pd.DataFrame(
            [
                {
                    "Message": (
                        "The scene table could not be converted to a dataframe."
                    )
                }
            ]
        )


def _scene_metadata(scene: Scene) -> pd.DataFrame:
    """Build the standard two-column metadata table for one scene."""

    model_names = getattr(scene, "model_names", []) or []

    return pd.DataFrame(
        [
            {
                "Field": "Title",
                "Value": str(getattr(scene, "title", "") or ""),
            },
            {
                "Field": "Kind",
                "Value": str(getattr(scene, "kind", "") or ""),
            },
            {
                "Field": "Models",
                "Value": ", ".join(map(str, model_names)),
            },
            {
                "Field": "Created",
                "Value": str(getattr(scene, "created_at", "") or ""),
            },
            {
                "Field": "Notes",
                "Value": str(getattr(scene, "notes", "") or ""),
            },
        ]
    )


# =============================================================================
# Public Excel export
# =============================================================================
def scenes_to_excel(scenes: list[Scene]) -> bytes:
    """Serialize report scenes into a multi-sheet Excel workbook.

    Each scene produces:

    - ``NN_metadata``: title, type, models, creation date and notes.
    - ``NN_table``: the analytical table stored in the scene.

    An empty scene list produces one ``Readme`` worksheet so the returned byte
    stream remains a valid workbook.
    """

    sheets: dict[str, pd.DataFrame] = {}
    used_names: set[str] = set()

    if not scenes:
        readme_name = _safe_sheet_name("Readme", used_names)
        sheets[readme_name] = pd.DataFrame(
            [
                {
                    "Message": (
                        "No report scenes have been added yet."
                    )
                }
            ]
        )

    for index, scene in enumerate(scenes, start=1):
        metadata_name = _safe_sheet_name(
            f"{index:02d}_metadata",
            used_names,
        )
        table_name = _safe_sheet_name(
            f"{index:02d}_table",
            used_names,
        )

        sheets[metadata_name] = _scene_metadata(scene)
        sheets[table_name] = _scene_table(scene)

    return dataframe_to_excel_bytes(sheets)


# =============================================================================
# Internal PDF helpers
# =============================================================================
def _pdf_safe_text(value: object) -> str:
    """Return text compatible with ReportLab's built-in Helvetica fonts.

    ReportLab's standard Type 1 fonts are not fully Unicode-capable. Encoding
    through Windows-1252 with replacement prevents unsupported characters from
    causing export failures while preserving common Western European text.
    """

    text = str(value or "")
    return (
        text
        .encode("cp1252", errors="replace")
        .decode("cp1252")
    )


def _wrapped_pdf_lines(
    value: object,
    width: int = 105,
) -> list[str]:
    """Split PDF text into lines suitable for the available page width."""

    safe_text = _pdf_safe_text(value)

    if not safe_text:
        return [""]

    return textwrap.wrap(
        safe_text,
        width=max(20, int(width)),
        break_long_words=True,
        break_on_hyphens=False,
    ) or [""]


def _start_pdf_page(
    pdf: object,
    page_height: float,
    *,
    include_report_title: bool,
) -> float:
    """Initialize a PDF page and return the first writable y-coordinate."""

    y = page_height - _PDF_TOP_MARGIN

    if include_report_title:
        pdf.setFont("Helvetica-Bold", _PDF_TITLE_FONT_SIZE)
        pdf.drawString(
            _PDF_LEFT_MARGIN,
            y,
            "PV BlockModel Studio Report",
        )
        y -= 30

    pdf.setFont("Helvetica", _PDF_BODY_FONT_SIZE)
    return y


def _ensure_pdf_space(
    pdf: object,
    y: float,
    page_height: float,
    required_height: float,
) -> float:
    """Create a new page when the current page lacks vertical space."""

    if y - required_height >= _PDF_BOTTOM_MARGIN:
        return y

    pdf.showPage()
    return _start_pdf_page(
        pdf,
        page_height,
        include_report_title=False,
    )


# =============================================================================
# Public PDF export
# =============================================================================
def scenes_to_pdf(scenes: list[Scene]) -> bytes:
    """Serialize report scenes into a compact PDF summary.

    The PDF intentionally summarizes scene metadata rather than rendering the
    complete dataframe content. Full analytical tables remain available through
    the Excel export.

    When ReportLab is unavailable, a byte message is returned so the calling UI
    can present a controlled dependency notice.
    """

    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except Exception:  # pragma: no cover
        # Preserve the existing fail-soft behavior. This path is environment
        # dependent and normally excluded from automated coverage.
        return (
            b"PDF export requires reportlab. "
            b"Install reportlab or use Excel export."
        )

    output = BytesIO()
    pdf = canvas.Canvas(output, pagesize=letter)
    _, page_height = letter

    y = _start_pdf_page(
        pdf,
        page_height,
        include_report_title=True,
    )

    if not scenes:
        pdf.drawString(
            _PDF_LEFT_MARGIN,
            y,
            "No report scenes have been added yet.",
        )

    for scene in scenes:
        title_lines = _wrapped_pdf_lines(
            getattr(scene, "title", ""),
            width=90,
        )

        model_names = getattr(scene, "model_names", []) or []
        detail_text = (
            f"Kind: {getattr(scene, 'kind', '')} | "
            f"Models: {', '.join(map(str, model_names))} | "
            f"Created: {getattr(scene, 'created_at', '')}"
        )
        detail_lines = _wrapped_pdf_lines(
            detail_text,
            width=118,
        )

        scene_table = _scene_table(scene)
        row_count_text = f"Rows in table: {len(scene_table):,}"

        required_height = (
            len(title_lines) * 14
            + len(detail_lines) * 11
            + 12
            + _PDF_SCENE_GAP
        )
        y = _ensure_pdf_space(
            pdf,
            y,
            page_height,
            required_height,
        )

        pdf.setFont(
            "Helvetica-Bold",
            _PDF_SCENE_TITLE_FONT_SIZE,
        )
        for line in title_lines:
            pdf.drawString(
                _PDF_LEFT_MARGIN,
                y,
                line,
            )
            y -= 14

        pdf.setFont(
            "Helvetica",
            _PDF_BODY_FONT_SIZE,
        )
        for line in detail_lines:
            pdf.drawString(
                _PDF_LEFT_MARGIN,
                y,
                line,
            )
            y -= 11

        pdf.drawString(
            _PDF_LEFT_MARGIN,
            y,
            row_count_text,
        )
        y -= _PDF_SCENE_GAP

    pdf.save()
    output.seek(0)
    return output.getvalue()
