# Parameters folder

This folder centralizes configuration rules that should be editable without changing the app logic.

## `app_defaults.json`

Includes display defaults, validation settings, period definitions and master filtering rules.

Important V1.4 keys:

```json
"master_filters": {
  "default_blk_model": 1,
  "valid_blk_model_values": [1, 2],
  "valid_destinations": ["h1", "h2", "l1", "l2", "l3", "m1", "m2", "m3", "mw1", "mw2"],
  "hg_destinations": ["h1", "h2"],
  "lg_destinations": ["l1", "l2", "l3"]
}
```

- `BLK_MODEL = 1`: in situ model.
- `BLK_MODEL = 2`: stockpiles.
- `BLK_MODEL = 0`: ignored.
- Values ending in `cut`, `none`, blank or outside the valid destination list are excluded from analytical calculations.

## `column_aliases.json`

Controls column detection and role inference. It now includes `BLK_MODEL`, `BLOCK_MODEL`, `BLOCK MODEL` and `BM_TYPE` as aliases for the `Block Model` role.

## `grade_defaults.json`

Controls default grade variables, labels and units.

V1.5 adds PV-style LTP aliases such as `AU_LTP`, `AG_LTP`, `CU_LTP`, `S_LTP`, `C_LTP`, `OC_LTP`, `S2_LTP`, `STOT` and `CTOT`. The resource table canonicalizes these to `Au`, `Ag`, `Cu`, `S`, `C`, `OC` and `S2`, and applies table units as follows:

- `Au`, `Ag`: `g/t`
- `Cu`, `C`, `OC`, `S`, `S2`: `%`

Contained metal is calculated only for Au and Ag using:

```text
Contained oz = sum(grade_gpt * tonnes) / 31.1034768
```
