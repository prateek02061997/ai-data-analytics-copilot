from src.config.settings import Settings, get_settings


def test_default_settings_are_valid() -> None:
    settings = Settings(ai_provider="auto")

    assert settings.app_name == "AI BI Copilot"
    assert settings.app_env == "development"
    assert settings.log_level == "INFO"
    assert settings.database_url.startswith("sqlite:///")
    assert settings.ai_provider == "auto"
    assert settings.gemini_model == "auto"
    assert settings.max_upload_mb == 2048


def test_postgresql_database_url_is_allowed() -> None:
    settings = Settings(database_url="postgresql://user:password@localhost:5432/bi")

    assert settings.database_url.startswith("postgresql://")


def test_get_settings_reflects_environment_changes(monkeypatch) -> None:
    monkeypatch.setenv("CLAUDE_API_KEY", "first-key")
    assert get_settings().claude_api_key == "first-key"

    monkeypatch.setenv("CLAUDE_API_KEY", "second-key")
    assert get_settings().claude_api_key == "second-key"