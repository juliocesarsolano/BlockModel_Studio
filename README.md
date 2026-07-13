# App_BM_Evaluation V1.5

Streamlit app for mineral-resource block-model evaluation and comparison.

## What changed in V1.5

- Added a transverse **BLK_MODEL master scope** applied across the whole app before any calculation, table or plot.
  - `BLK_MODEL = 1`: in situ model, default.
  - `BLK_MODEL = 2`: stockpiles.
  - `BLK_MODEL = 0`: always excluded.
- Added a setup-page control for the master BLK_MODEL scope and a persistent sidebar master-scope banner.
- Added valid destination logic:
  - Valid destinations: `h1`, `h2`, `l1`, `l2`, `l3`, `m1`, `m2`, `m3`, `mw1`, `mw2`.
  - Destinations ending in `cut`, `none`, blank or any other invalid value are excluded from calculations, tables and plots.
- Updated the Resource Tabulation destination selector:
  - `HG+LG` = `h1`, `h2`, `l1`, `l2`, `l3`.
  - `HG only` = `h1`, `h2`.
  - `LG only` = `l1`, `l2`, `l3`.
  - `All destinations` = all valid H/L/M/MW destination codes.
- Updated sample models to include `BLK_MODEL` and the valid destination coding system.
- Added `BLK_MODEL` to the role-mapping logic and parameter aliases.

- Fixed Resource Tabulation metal calculation for real PV-style columns such as `AU_LTP` and `AG_LTP`:
  - `Au oz = sum(Au g/t × tonnes) / 31.1034768`.
  - `Ag oz = sum(Ag g/t × tonnes) / 31.1034768`.
- Added canonical display mapping so `AU_LTP`, `AG_LTP`, `CU_LTP`, `S_LTP`, `C_LTP`, `OC_LTP`, `S2_LTP`, `STOT`, and `CTOT` are shown as `Au`, `Ag`, `Cu`, `S`, `C`, `OC`, and `S2`.
- Added unit overrides for resource tables: `Au`/`Ag` as `g/t`; `Cu`, `C`, `OC`, `S`, and `S2` as `%`.
- Removed the extra top/bottom table guide lines that looked like missing columns.
- Added a **Calculation audit / reconciliation inputs** expander below Resource Tabulation to compare Streamlit totals against Power BI using the same filter scope.

## Project structure

```text
App_BM_Evaluation_v1_5/
├── app.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── README.md
├── .streamlit/
│   └── config.toml
├── assets/
├── parameters/
│   ├── app_defaults.json
│   ├── column_aliases.json
│   ├── grade_defaults.json
│   └── README.md
├── source/
│   ├── sample_model_2025.csv
│   └── sample_model_2026.csv
├── src/
│   ├── core/
│   └── ui/
└── tests/
    └── test_core.py
```

## Run locally on Windows

```bash
cd /d E:\Data2\3.Projects\App_BM_Evaluation_v1_5
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Quick test

```bash
python -m compileall app.py src tests
python -m pytest -q
```

## Demo workflow

1. Open **Model Setup**.
2. Confirm the **Master app filter**. Default is `1 - In situ`.
3. Use **Demonstration models**.
4. Click **Configure both sample models automatically**.
5. Open **Model Evaluation**.
6. Review **Variables & controls** and **Resource tabulation**.

## Notes

- If the user data contains a `BLK_MODEL` column, the app applies the selected master scope globally.
- If `BLK_MODEL` is missing, the app will not apply the BLK_MODEL filter, but valid destination filtering is still applied when a Destination role is configured.
- Invalid destinations are not deleted from the raw data; they are simply excluded from analytical views.
