import json

import pandas as pd
import pytest

from src.powerbi import build_power_bi_export, dataframe_to_excel_bytes


def test_build_power_bi_export_creates_download_assets() -> None:
    dataframe = pd.DataFrame(
        {
            "Patient ID": [1, 2],
            "Age": [34, 51],
            "Risk Score": [0.2, 0.8],
            "Active": [True, False],
            "Diagnosis": ["A", "B"],
        }
    )

    export = build_power_bi_export(dataframe, table_name="dataset patient data")
    schema = json.loads(export.schema_json)

    assert export.csv_file_name == "dataset_patient_data.csv"
    assert export.schema_file_name == "dataset_patient_data_schema.json"
    assert export.power_query_file_name == "dataset_patient_data_power_query.m"
    assert export.csv_bytes.startswith(b"\xef\xbb\xbfPatient ID")
    assert schema["row_count"] == 2
    assert schema["column_count"] == 5
    assert {field["name"]: field["power_bi_type"] for field in schema["fields"]} == {
        "Patient ID": "Int64.Type",
        "Age": "Int64.Type",
        "Risk Score": "type number",
        "Active": "type logical",
        "Diagnosis": "type text",
    }
    assert "Table.TransformColumnTypes" in export.power_query_m
    assert "dataset_patient_data.csv" in export.power_query_m


def test_build_power_bi_export_rejects_empty_dataframe() -> None:
    with pytest.raises(ValueError, match="empty dataset"):
        build_power_bi_export(pd.DataFrame(), table_name="empty")


def test_dataframe_to_excel_bytes_generates_xlsx() -> None:
    dataframe = pd.DataFrame({"Category": ["A", "B"], "Score": [10, 20]})
    excel_bytes = dataframe_to_excel_bytes(dataframe)
    assert excel_bytes.startswith(b"PK")  # ZIP file header for xlsx