from src.config.health import build_health_report
from src.config.settings import Settings


def test_build_health_report_is_ready_with_sqlite_and_missing_ai_key_warning(tmp_path) -> None:
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'app.db'}",
        ai_provider="auto",
        gemini_api_key="",
        claude_api_key="",
    )

    report = build_health_report(settings)

    assert report.status == "ready"
    assert {check.name for check in report.checks} == {"Database", "AI Provider", "Upload Limit"}
    assert any(check.name == "Database" and check.status == "ok" for check in report.checks)
    assert any(check.name == "AI Provider" and check.status == "warning" for check in report.checks)


def test_build_health_report_warns_for_postgresql_until_adapter_exists() -> None:
    settings = Settings(database_url="postgresql://user:password@localhost:5432/bi")

    report = build_health_report(settings)

    assert report.status == "ready"
    assert any(check.name == "Database" and check.status == "warning" for check in report.checks)