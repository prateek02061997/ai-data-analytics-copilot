from types import SimpleNamespace

import pandas as pd
import pytest

from src.ai import AIResponseError, NaturalLanguageSQLAgent, SQLSafetyError, validate_select_sql
from src.database import DatabaseManager


class FakeClaudeMessages:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.last_request = {}

    def create(self, **kwargs):
        self.last_request = kwargs
        return SimpleNamespace(content=[SimpleNamespace(text=self.response_text)])


def test_validate_select_sql_allows_single_select_for_allowed_table() -> None:
    sql = validate_select_sql("SELECT Customer, SUM(Revenue) AS TotalRevenue FROM sales GROUP BY Customer", "sales")

    assert sql.endswith("LIMIT 100")


def test_validate_select_sql_rejects_unsafe_statement() -> None:
    with pytest.raises(SQLSafetyError, match="Only SELECT"):
        validate_select_sql("DELETE FROM sales", "sales")


def test_validate_select_sql_rejects_wrong_table() -> None:
    with pytest.raises(SQLSafetyError, match="selected saved table"):
        validate_select_sql("SELECT * FROM other_table", "sales")


def test_natural_language_sql_agent_generates_safe_sql() -> None:
    fake_client = FakeClaudeMessages(
        '{"sql":"SELECT Customer, SUM(Revenue) AS TotalRevenue FROM sales GROUP BY Customer LIMIT 10",'
        '"explanation":"Totals revenue by customer.",'
        '"chart_recommendation":"Bar chart sorted by revenue."}'
    )
    agent = NaturalLanguageSQLAgent(api_key="", model="test-model", client=fake_client)
    columns = pd.DataFrame(
        [
            {"column_name": "Customer", "detected_type": "Text", "missing_values": 0},
            {"column_name": "Revenue", "detected_type": "Numeric", "missing_values": 0},
        ]
    )

    response = agent.generate_sql("Show top customers by revenue", "sales", columns)

    assert response.sql == "SELECT Customer, SUM(Revenue) AS TotalRevenue FROM sales GROUP BY Customer LIMIT 10"
    assert response.explanation == "Totals revenue by customer."
    assert response.chart_recommendation == "Bar chart sorted by revenue."
    assert "Show top customers by revenue" in fake_client.last_request["messages"][0]["content"]


def test_natural_language_sql_agent_rejects_invalid_json() -> None:
    agent = NaturalLanguageSQLAgent(api_key="", model="test-model", client=FakeClaudeMessages("not json"))

    with pytest.raises(AIResponseError, match="valid JSON"):
        agent.generate_sql("Show revenue", "sales", pd.DataFrame())


def test_database_manager_executes_validated_select(tmp_path) -> None:
    database = DatabaseManager(f"sqlite:///{tmp_path / 'analytics.db'}")
    metadata = database.save_dataset(
        pd.DataFrame({"Customer": ["A", "B"], "Revenue": [100, 200]}),
        "sales.csv",
        [
            {"column_name": "Customer", "detected_type": "Text", "missing_values": 0},
            {"column_name": "Revenue", "detected_type": "Numeric", "missing_values": 0},
        ],
    )
    sql = validate_select_sql(f"SELECT Customer, Revenue FROM {metadata.table_name} ORDER BY Revenue DESC", metadata.table_name)

    result = database.execute_select_query(sql)

    assert result.dataframe["Customer"].tolist() == ["B", "A"]