import pandas as pd
import pytest

from src.analytics import generate_profile_report
from src.cleaning import clean_dataframe
from src.reports import build_executive_report


def test_build_executive_report_creates_html_and_markdown() -> None:
    dataframe = pd.DataFrame(
        {
            "Patient ID": [1, 2, 3],
            "Age": [34, 51, 51],
            "Diagnosis": ["A", "B", "B"],
        }
    )
    cleaning_result = clean_dataframe(dataframe)
    profile = generate_profile_report(cleaning_result.cleaned_dataframe)

    report = build_executive_report(
        dataframe=cleaning_result.cleaned_dataframe,
        dataset_name="Patient Data.csv",
        profile_report=profile,
        cleaning_summary=cleaning_result.summary,
    )

    assert report.html_file_name == "patient_data_csv_executive_report.html"
    assert report.markdown_file_name == "patient_data_csv_executive_report.md"
    assert "Executive Report: Patient Data.csv" in report.markdown
    assert "Rows: 3" in report.markdown
    assert "Health score" in report.markdown
    assert "Cleaning Summary" in report.markdown
    assert "Rows modified" in report.markdown
    assert "Cells modified" in report.markdown
    assert "<!doctype html>" in report.html
    assert "Patient ID" in report.html


def test_build_executive_report_rejects_empty_dataframe() -> None:
    with pytest.raises(ValueError, match="empty dataset"):
        build_executive_report(
            dataframe=pd.DataFrame(),
            dataset_name="empty.csv",
            profile_report=generate_profile_report(pd.DataFrame()),
        )