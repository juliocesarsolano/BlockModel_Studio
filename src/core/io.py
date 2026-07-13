from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd


CSV_ENCODINGS = ("utf-8-sig", "utf-8", "latin1")


def load_dataframe(filename: str, payload: bytes) -> pd.DataFrame:
    suffix = Path(filename).suffix.casefold()
    buffer = BytesIO(payload)

    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(buffer)

    if suffix in {".csv", ".txt", ""}:
        last_error: Exception | None = None
        for encoding in CSV_ENCODINGS:
            try:
                buffer.seek(0)
                return pd.read_csv(buffer, sep=None, engine="python", encoding=encoding)
            except Exception as exc:  # pragma: no cover - defensive fallback
                last_error = exc
        raise ValueError(f"Unable to read CSV file: {last_error}")

    raise ValueError(f"Unsupported file extension: {suffix}")


def load_path(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    return load_dataframe(file_path.name, file_path.read_bytes())


def dataframe_to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for sheet_name, frame in sheets.items():
            safe_name = sheet_name[:31].replace("/", "-").replace("\\", "-")
            frame.to_excel(writer, sheet_name=safe_name, index=False)
    output.seek(0)
    return output.getvalue()
