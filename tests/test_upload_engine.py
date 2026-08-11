from io import BytesIO

import pandas as pd
import pytest

import src.data_upload.upload_engine as upload_engine
from src.data_upload import UploadValidationError, load_uploaded_dataset, summarize_dataframe, validate_upload


def test_validate_upload_rejects_unsupported_extension() -> None:
    with pytest.raises(UploadValidationError, match="Unsupported file type"):
        validate_upload("data.txt", file_size_bytes=20, max_upload_mb=100)


def test_validate_upload_rejects_large_file() -> None:
    with pytest.raises(UploadValidationError, match="File is too large"):
        validate_upload("data.csv", file_size_bytes=2 * 1024 * 1024, max_upload_mb=1)


def test_validate_upload_accepts_files_up_to_2gb_when_limit_allows() -> None:
    validate_upload("large.csv", file_size_bytes=2_000 * 1024 * 1024, max_upload_mb=2048)


def test_load_uploaded_csv_returns_preview_and_summary() -> None:
    file = BytesIO(b"Revenue,Date,Customer\n100,2026-01-01,Auckland\n250,2026-01-02,Wellington\n")

    dataset = load_uploaded_dataset(file, "sales.csv", file_size_bytes=len(file.getvalue()), max_upload_mb=100)

    assert dataset.summary.rows == 2
    assert dataset.summary.columns == 3
    assert len(dataset.preview) == 2
    assert [(field.name, field.detected_type) for field in dataset.summary.fields] == [
        ("Revenue", "Numeric"),
        ("Date", "Datetime"),
        ("Customer", "Text"),
    ]


def test_load_large_csv_uses_chunk_progress(monkeypatch) -> None:
    monkeypatch.setattr(upload_engine, "LARGE_CSV_THRESHOLD_BYTES", 1)
    monkeypatch.setattr(upload_engine, "CSV_CHUNK_SIZE", 1)
    file = BytesIO(b"Revenue,Customer\n100,A\n200,B\n300,C\n")
    progress_updates: list[tuple[int, int | None]] = []

    dataset = load_uploaded_dataset(
        file,
        "large.csv",
        file_size_bytes=len(file.getvalue()),
        max_upload_mb=500,
        progress_callback=lambda bytes_read, total_bytes: progress_updates.append((bytes_read, total_bytes)),
    )

    assert dataset.summary.rows == 3
    assert dataset.dataframe["Revenue"].tolist() == [100, 200, 300]
    assert progress_updates
    assert progress_updates[-1] == (len(file.getvalue()), len(file.getvalue()))


def test_load_uploaded_xlsx_returns_summary() -> None:
    source = pd.DataFrame({"Revenue": [100, 200], "Customer": ["A", "B"]})
    file = BytesIO()
    source.to_excel(file, index=False)

    dataset = load_uploaded_dataset(file, "sales.xlsx", file_size_bytes=len(file.getvalue()), max_upload_mb=100)

    assert dataset.summary.rows == 2
    assert dataset.summary.columns == 2
    assert dataset.preview.equals(source)


def test_summarize_dataframe_counts_missing_values() -> None:
    dataframe = pd.DataFrame({"Customer": ["A", None], "Revenue": [100.0, None]})

    summary = summarize_dataframe(dataframe)

    assert summary.fields[0].missing_values == 1
    assert summary.fields[1].missing_values == 1