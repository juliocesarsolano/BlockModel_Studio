from __future__ import annotations

from io import BytesIO

import pandas as pd

from .io import dataframe_to_excel_bytes
from .models import Scene


def scenes_to_excel(scenes: list[Scene]) -> bytes:
    sheets: dict[str, pd.DataFrame] = {}
    if not scenes:
        sheets["Readme"] = pd.DataFrame([{"Message": "No report scenes have been added yet."}])
    for index, scene in enumerate(scenes, start=1):
        meta = pd.DataFrame(
            [
                {"Field": "Title", "Value": scene.title},
                {"Field": "Kind", "Value": scene.kind},
                {"Field": "Models", "Value": ", ".join(scene.model_names)},
                {"Field": "Created", "Value": scene.created_at},
                {"Field": "Notes", "Value": scene.notes},
            ]
        )
        sheets[f"{index:02d}_metadata"] = meta
        sheets[f"{index:02d}_table"] = scene.table
    return dataframe_to_excel_bytes(sheets)


def scenes_to_pdf(scenes: list[Scene]) -> bytes:
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except Exception:  # pragma: no cover
        return b"PDF export requires reportlab. Install reportlab or use Excel export."

    output = BytesIO()
    pdf = canvas.Canvas(output, pagesize=letter)
    width, height = letter
    y = height - 50
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, y, "PV BlockModel Studio Report")
    y -= 30
    pdf.setFont("Helvetica", 9)

    if not scenes:
        pdf.drawString(50, y, "No report scenes have been added yet.")
    for scene in scenes:
        if y < 120:
            pdf.showPage()
            y = height - 50
        pdf.setFont("Helvetica-Bold", 11)
        pdf.drawString(50, y, scene.title[:90])
        y -= 16
        pdf.setFont("Helvetica", 8)
        pdf.drawString(50, y, f"Kind: {scene.kind} | Models: {', '.join(scene.model_names)} | Created: {scene.created_at}")
        y -= 14
        pdf.drawString(50, y, f"Rows in table: {len(scene.table):,}")
        y -= 20

    pdf.save()
    output.seek(0)
    return output.getvalue()
