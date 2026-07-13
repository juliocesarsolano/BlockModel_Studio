from __future__ import annotations

from io import BytesIO

import pandas as pd
import pytest

from src.core.models import CategorySpec, ModelConfig, Scene
from src.core.reports import scenes_to_excel, scenes_to_pdf
from src.ui import pages


def scope_config() -> ModelConfig:
    return ModelConfig(
        model_name="Scope fixture",
        model_type="Resource Model",
        report_title="Scope fixture",
        mass_column="TONNES",
        volume_column="VOLUME",
        category_specs=[
            CategorySpec("BLK_MODEL", "Block model", "Block Model"),
            CategorySpec("DESTINATION", "Destination", "Destination"),
        ],
    )


def test_ui_project_root_contains_embedded_vulcan_resource() -> None:
    resource = pages.ROOT / "assets" / "BlockModel_Studio_Vulcan_Model_Query.res"
    assert resource.is_file()
    assert resource.read_text(encoding="utf-8").startswith("* MAPTEK: Specifications")


def scope_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TONNES": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
            "VOLUME": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "BLK_MODEL": [0, 1, 1, 1, 2, 2, 2],
            "DESTINATION": ["H1", "H1", "L2", "MW1", "W1", "CUT", "NONE"],
        }
    )


@pytest.mark.parametrize(
    ("blk_values", "destination_mode", "expected_tonnes"),
    [
        ([1], "HG+LG", 50.0),
        ([1], "HG+LG+MW", 90.0),
        ([2], "All destinations", 50.0),
        ([1, 2], "All destinations", 140.0),
    ],
)
def test_global_scope_freezes_blk_model_and_destination_rules(
    monkeypatch: pytest.MonkeyPatch,
    blk_values: list[int],
    destination_mode: str,
    expected_tonnes: float,
) -> None:
    monkeypatch.setattr(pages, "selected_master_blk_model_values", lambda: blk_values)
    monkeypatch.setattr(pages, "_master_destination_label", lambda: destination_mode)
    monkeypatch.setattr(pages.st, "session_state", {})
    scoped = pages._apply_global_scope(scope_data(), scope_config())
    assert 0 not in scoped["BLK_MODEL"].tolist()
    assert not set(scoped["DESTINATION"]).intersection({"CUT", "NONE"})
    assert scoped["TONNES"].sum() == expected_tonnes


def report_scene() -> Scene:
    return Scene(
        scene_id="characterization-1",
        title="Synthetic resource table",
        kind="Evaluation",
        model_names=["Anonymous model"],
        filters={"Year": "LOM"},
        kpis={"Tonnage": "0.00 Mt"},
        table=pd.DataFrame({"Category": ["Measured", "Indicated"], "Tonnage (Mt)": [1.25, 2.75]}),
        notes="Synthetic characterization fixture",
        created_at="2026-01-01 00:00 UTC",
    )


def test_scene_excel_preserves_sheet_names_metadata_and_totals() -> None:
    payload = scenes_to_excel([report_scene()])
    workbook = pd.ExcelFile(BytesIO(payload))
    assert workbook.sheet_names == ["01_metadata", "01_table"]
    metadata = pd.read_excel(BytesIO(payload), sheet_name="01_metadata")
    table = pd.read_excel(BytesIO(payload), sheet_name="01_table")
    assert metadata.loc[metadata["Field"] == "Title", "Value"].iloc[0] == "Synthetic resource table"
    assert table["Tonnage (Mt)"].sum() == 4.0


def test_empty_scene_excel_has_readme_sheet() -> None:
    payload = scenes_to_excel([])
    workbook = pd.ExcelFile(BytesIO(payload))
    assert workbook.sheet_names == ["Readme"]
    readme = pd.read_excel(BytesIO(payload), sheet_name="Readme")
    assert readme.iloc[0, 0] == "No report scenes have been added yet."


def test_scene_pdf_has_valid_signature_and_content_marker() -> None:
    payload = scenes_to_pdf([report_scene()])
    assert payload.startswith(b"%PDF-")
    assert len(payload) > 1_000
    assert b"%%EOF" in payload[-32:]
