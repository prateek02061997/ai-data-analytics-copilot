import pandas as pd
import pytest

from src.database import DatabaseError, DatabaseManager


def test_save_dataset_creates_table_and_metadata(tmp_path) -> None:
    database = DatabaseManager(f"sqlite:///{tmp_path / 'analytics.db'}")
    dataframe = pd.DataFrame({"Revenue": [100, 200], "Customer": ["A", "B"]})

    metadata = database.save_dataset(
        dataframe=dataframe,
        source_file="Sales Report.csv",
        column_metadata=[
            {"column_name": "Revenue", "detected_type": "Numeric", "missing_values": 0},
            {"column_name": "Customer", "detected_type": "Text", "missing_values": 0},
        ],
        profile_health_score=95,
        cleaning_summary={"records_processed": 2, "records_fixed": 0},
    )

    saved = database.list_datasets()
    queried = database.query_table(metadata.table_name, limit=10)
    columns = database.get_column_metadata(metadata.dataset_id)

    assert len(saved) == 1
    assert saved[0].source_file == "Sales Report.csv"
    assert saved[0].row_count == 2
    assert saved[0].profile_health_score == 95
    assert queried.dataframe.equals(dataframe)
    assert columns["column_name"].tolist() == ["Revenue", "Customer"]


def test_save_dataset_preserves_previous_uploads(tmp_path) -> None:
    database = DatabaseManager(f"sqlite:///{tmp_path / 'analytics.db'}")
    dataframe = pd.DataFrame({"Value": [1]})

    database.save_dataset(dataframe, "first.csv", [{"column_name": "Value", "detected_type": "Numeric", "missing_values": 0}])
    database.save_dataset(dataframe, "second.csv", [{"column_name": "Value", "detected_type": "Numeric", "missing_values": 0}])

    saved = database.list_datasets()

    assert len(saved) == 2
    assert {item.source_file for item in saved} == {"first.csv", "second.csv"}
    assert saved[0].table_name != saved[1].table_name


def test_query_table_rejects_unknown_table(tmp_path) -> None:
    database = DatabaseManager(f"sqlite:///{tmp_path / 'analytics.db'}")

    with pytest.raises(DatabaseError, match="Unknown table"):
        database.query_table("missing_table")


def test_postgresql_url_is_rejected_until_adapter_exists() -> None:
    with pytest.raises(DatabaseError, match="PostgreSQL support is planned"):
        DatabaseManager("postgresql://user:password@localhost:5432/bi")