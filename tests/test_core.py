from __future__ import annotations

import math

import pandas as pd

from src.core.cleaning import clean_model_data
from src.core.comparison import compare_global_volumes, volume_gate_status
from src.core.metrics import contained_metal, grouped_summary, period_summary, weighted_mean
from src.core.models import CategorySpec, GradeSpec, ModelBundle, ModelConfig
from src.core.validation import validate_model


def sample_data() -> pd.DataFrame:
    """Small anonymous fixture; production models are always user uploads."""
    return pd.DataFrame(
        {
            "TONNES": [1000, 1500, 2000, 2500, 1200, 1800],
            "VOLUME": [400, 600, 800, 1000, 480, 720],
            "AU_PPM": [0.8, 1.2, 0.6, 1.5, 0.4, 0.9],
            "CU_PCT": [0.2, 0.25, 0.15, 0.3, 0.1, 0.18],
            "CATEG": ["Measured", "Measured", "Indicated", "Indicated", "Inferred", "Inferred"],
            "DESTINATION": ["HG", "HG", "LG", "HG", "LG", "LG"],
            "BENCH": [1000, 1010, 1020, 1030, 1040, 1050],
            "YEAR": [2025, 2026, 2027, 2028, 2030, 2035],
            "BLK_MODEL": [1, 1, 1, 1, 1, 1],
        }
    )


def sample_config() -> ModelConfig:
    return ModelConfig(
        model_name="Sample",
        model_type="Resource Model",
        report_title="Sample",
        mass_column="TONNES",
        volume_column="VOLUME",
        grade_specs=[GradeSpec("AU_PPM", "Au", "g/t", 2), GradeSpec("CU_PCT", "Cu", "%", 3)],
        category_specs=[
            CategorySpec("CATEG", "Categ", "Category"),
            CategorySpec("DESTINATION", "Destination", "Destination"),
            CategorySpec("BENCH", "Bench", "Bench"),
            CategorySpec("YEAR", "Year", "Year"),
        ],
    )


def test_weighted_mean() -> None:
    value = weighted_mean(pd.Series([1.0, 3.0]), pd.Series([1.0, 3.0]))
    assert value == 2.5


def test_weighted_mean_ignores_non_positive_weights() -> None:
    value = weighted_mean(pd.Series([100.0, 2.0, 4.0]), pd.Series([0.0, 1.0, 3.0]))
    assert value == 3.5
    assert math.isnan(weighted_mean(pd.Series([1.0]), pd.Series([0.0])))


def test_contained_metal_units_are_frozen() -> None:
    gold = contained_metal(pd.Series([1.0, 2.0]), pd.Series([100.0, 200.0]), "g/t")
    copper = contained_metal(pd.Series([1.0, 2.0]), pd.Series([100.0, 200.0]), "%")
    assert gold is not None and gold[1] == "oz"
    assert gold[0] == (1.0 * 100.0 + 2.0 * 200.0) / 31.1034768
    assert copper == (5.0, "t")


def test_load_clean_validate_sample() -> None:
    config = sample_config()
    raw = sample_data()
    data, stats = clean_model_data(raw, config)
    report = validate_model(data, config)
    assert stats["output_rows"] == len(raw)
    assert report.total_rows == len(raw)
    assert "TONNES" in data.columns


def test_grouped_summary() -> None:
    config = sample_config()
    raw = sample_data()
    data, _ = clean_model_data(raw, config)
    table = grouped_summary(data, config, "CATEG")
    assert not table.empty
    assert "Tonnage (Mt)" in table.columns


def test_period_summary() -> None:
    config = sample_config()
    raw = sample_data()
    data, _ = clean_model_data(raw, config)
    table = period_summary(data, config, start_year=2026)
    assert set(table["Period"]).issubset({"2Y", "5Y", "10Y", "LOM"})


def test_global_volume_gate_boundary() -> None:
    config_a = sample_config()
    config_b = sample_config()
    data_a, _ = clean_model_data(sample_data(), config_a)
    data_b = data_a.copy()
    data_b["VOLUME"] = data_b["VOLUME"] * 1.001
    bundles = {
        "Reference": ModelBundle(config=config_a, raw_data=data_a, data=data_a, validation=validate_model(data_a, config_a)),
        "Candidate": ModelBundle(config=config_b, raw_data=data_b, data=data_b, validation=validate_model(data_b, config_b)),
    }
    volume_table = compare_global_volumes(bundles)
    passed, _ = volume_gate_status(volume_table, tolerance_pct=0.1)
    failed, _ = volume_gate_status(volume_table, tolerance_pct=0.0999)
    assert passed is True
    assert failed is False
