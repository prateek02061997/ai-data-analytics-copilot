"""CSV and Excel upload validation, loading, and dataset summarisation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Callable

import pandas as pd
from pandas.api import types as pd_types


SUPPORTED_EXTENSIONS = {".csv", ".xlsx"}
PREVIEW_ROW_LIMIT = 100
DATETIME_PARSE_THRESHOLD = 0.8
LARGE_CSV_THRESHOLD_BYTES = 50 * 1024 * 1024
CSV_CHUNK_SIZE = 100_000

LoadProgressCallback = Callable[[int, int | None], None]


class UploadValidationError(ValueError):
    """Raised when an uploaded file is not supported or cannot be processed."""


@dataclass(frozen=True)
class ColumnSummary:
    """Detected metadata for one dataset column."""

    name: str
    detected_type: str
    missing_values: int


@dataclass(frozen=True)
class DatasetSummary:
    """High-level metadata for an uploaded dataset."""

    rows: int
    columns: int
    fields: list[ColumnSummary]


@dataclass(frozen=True)
class UploadedDataset:
    """Loaded dataset plus the metadata needed by the upload UI."""

    file_name: str
    dataframe: pd.DataFrame
    preview: pd.DataFrame
    summary: DatasetSummary


def validate_upload(file_name: str, file_size_bytes: int, max_upload_mb: int) -> None:
    """Validate upload name, extension, and size before parsing."""
    if not file_name:
        raise UploadValidationError("Uploaded file must have a name.")

    extension = Path(file_name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise UploadValidationError(f"Unsupported file type. Upload one of: {supported}.")

    max_size_bytes = max_upload_mb * 1024 * 1024
    if file_size_bytes <= 0:
        raise UploadValidationError("Uploaded file is empty.")
    if file_size_bytes > max_size_bytes:
        raise UploadValidationError(f"File is too large. Maximum allowed size is {max_upload_mb} MB.")


def load_uploaded_dataset(
    file: BinaryIO,
    file_name: str,
    file_size_bytes: int,
    max_upload_mb: int,
    progress_callback: LoadProgressCallback | None = None,
) -> UploadedDataset:
    """Validate and load an uploaded CSV or XLSX file into a DataFrame."""
    validate_upload(file_name=file_name, file_size_bytes=file_size_bytes, max_upload_mb=max_upload_mb)

    try:
        file.seek(0)
        dataframe = _read_file(file, file_name, file_size_bytes, progress_callback)
    except UnicodeDecodeError as exc:
        raise UploadValidationError("CSV file encoding could not be read as UTF-8.") from exc
    except ValueError as exc:
        raise UploadValidationError(f"File could not be parsed: {exc}") from exc

    if dataframe.empty:
        raise UploadValidationError("Uploaded dataset contains no rows.")

    summary = summarize_dataframe(dataframe)
    return UploadedDataset(
        file_name=file_name,
        dataframe=dataframe,
        preview=dataframe.head(PREVIEW_ROW_LIMIT),
        summary=summary,
    )


def summarize_dataframe(dataframe: pd.DataFrame) -> DatasetSummary:
    """Create row, column, and field metadata for a DataFrame."""
    fields = [
        ColumnSummary(
            name=str(column),
            detected_type=_detect_column_type(dataframe[column]),
            missing_values=int(dataframe[column].isna().sum()),
        )
        for column in dataframe.columns
    ]

    return DatasetSummary(rows=len(dataframe), columns=len(dataframe.columns), fields=fields)


def _read_file(
    file: BinaryIO,
    file_name: str,
    file_size_bytes: int,
    progress_callback: LoadProgressCallback | None,
) -> pd.DataFrame:
    extension = Path(file_name).suffix.lower()
    if extension == ".csv":
        if file_size_bytes >= LARGE_CSV_THRESHOLD_BYTES:
            return _read_csv_in_chunks(file, file_size_bytes, progress_callback)
        try:
            dataframe = pd.read_csv(file, on_bad_lines="skip")
        except (UnicodeDecodeError, Exception):
            file.seek(0)
            dataframe = pd.read_csv(file, encoding="latin1", on_bad_lines="skip")
        if progress_callback:
            progress_callback(file_size_bytes, file_size_bytes)
        return dataframe
    if extension == ".xlsx":
        dataframe = pd.read_excel(file)
        if progress_callback:
            progress_callback(file_size_bytes, file_size_bytes)
        return dataframe
    raise UploadValidationError("Unsupported file type.")


MAX_LARGE_CSV_ROWS = 100_000


def _read_csv_in_chunks(
    file: BinaryIO,
    file_size_bytes: int,
    progress_callback: LoadProgressCallback | None,
) -> pd.DataFrame:
    """Read CSV in chunks, capping at MAX_LARGE_CSV_ROWS for large files to keep UI responsive."""
    chunks: list[pd.DataFrame] = []
    rows_loaded = 0
    encodings = ["utf-8", "latin1", "utf-8-sig"]
    for enc in encodings:
        try:
            file.seek(0)
            chunks = []
            rows_loaded = 0
            for chunk in pd.read_csv(file, chunksize=CSV_CHUNK_SIZE, low_memory=True, encoding=enc, on_bad_lines="skip"):
                chunks.append(chunk)
                rows_loaded += len(chunk)
                if progress_callback:
                    progress_callback(min(rows_loaded * 200, file_size_bytes), file_size_bytes)
                if rows_loaded >= MAX_LARGE_CSV_ROWS:
                    break
            if chunks:
                break
        except Exception:
            continue

    if not chunks:
        return pd.DataFrame()
    if progress_callback:
        progress_callback(file_size_bytes, file_size_bytes)
    return pd.concat(chunks, ignore_index=True, copy=False)


def _current_file_position(file: BinaryIO) -> int:
    try:
        return int(file.tell())
    except (AttributeError, OSError):
        return 0


def _detect_column_type(series: pd.Series) -> str:
    non_null = series.dropna()
    if non_null.empty:
        return "Empty"
    if pd_types.is_bool_dtype(series):
        return "Boolean"
    if pd_types.is_numeric_dtype(series):
        return "Numeric"
    if pd_types.is_datetime64_any_dtype(series):
        return "Datetime"
    if _looks_like_datetime(non_null):
        return "Datetime"
    return "Text"


def _looks_like_datetime(series: pd.Series) -> bool:
    if not pd_types.is_object_dtype(series) and not pd_types.is_string_dtype(series):
        return False

    parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    parse_ratio = parsed.notna().mean()
    return bool(parse_ratio >= DATETIME_PARSE_THRESHOLD)