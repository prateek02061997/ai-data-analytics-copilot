import pandas as pd

from src.cleaning import clean_dataframe


def test_clean_dataframe_removes_duplicates_and_fills_missing_values() -> None:
    dataframe = pd.DataFrame(
        {
            "Customer": ["A", "A", None],
            "Revenue": [100.0, 100.0, 300.0],
        }
    )

    result = clean_dataframe(dataframe)

    assert len(result.cleaned_dataframe) == 2
    assert result.summary.records_processed == 3
    assert result.summary.duplicates_removed == 1
    assert result.summary.missing_values_filled == 1
    assert result.cleaned_dataframe.isna().sum().sum() == 0


def test_clean_dataframe_standardizes_location_aliases() -> None:
    dataframe = pd.DataFrame({"Region": ["Auckland", "auckland", " AKL ", "Wellington"]})

    result = clean_dataframe(dataframe)

    assert result.cleaned_dataframe["Region"].tolist() == ["Auckland", "Wellington"]
    assert result.summary.text_values_standardized == 2


def test_clean_dataframe_fixes_date_formats_and_removes_invalid_records() -> None:
    dataframe = pd.DataFrame(
        {
            "Order Date": ["2026-01-01", "31/01/2026", "not-a-date"],
            "Revenue": [100, 200, 300],
        }
    )

    result = clean_dataframe(dataframe)

    assert len(result.cleaned_dataframe) == 2
    assert pd.api.types.is_datetime64_any_dtype(result.cleaned_dataframe["Order Date"])
    assert result.summary.date_values_fixed == 2
    assert result.summary.invalid_records_removed == 1


def test_clean_dataframe_converts_numeric_text_and_removes_invalid_numeric_records() -> None:
    dataframe = pd.DataFrame({"Revenue": ["100", "200", "bad-value", "300"]})

    result = clean_dataframe(dataframe)

    assert result.cleaned_dataframe["Revenue"].tolist() == [100.0, 200.0, 300.0]
    assert result.summary.invalid_records_removed == 1


def test_clean_dataframe_detects_anomalies_without_removing_them() -> None:
    dataframe = pd.DataFrame({"Revenue": [100, 101, 99, 102, 10_000]})

    result = clean_dataframe(dataframe)

    assert result.summary.anomalies_detected == 1
    assert len(result.cleaned_dataframe) == 5


def test_cleaning_summary_separates_rows_cells_and_flags() -> None:
    dataframe = pd.DataFrame(
        {
            "Region": [" AKL ", None, "WLG", "WLG", "CHC"],
            "Revenue": ["100", "200", "300", "301", "10000"],
        }
    )

    result = clean_dataframe(dataframe)

    assert result.summary.records_processed == 5
    assert result.summary.rows_modified == 5
    assert result.summary.records_fixed == result.summary.rows_modified
    assert result.summary.cells_modified == 10
    assert result.summary.missing_values_filled == 1
    assert result.summary.duplicates_removed == 0
    assert result.summary.outliers_flagged == 1
    assert result.summary.formats_standardized == 9