"""Application settings and environment loading."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class SettingsError(RuntimeError):
    """Raised when application settings cannot be loaded."""


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables and .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = Field(default="AI BI Copilot", min_length=1)
    app_env: Literal["development", "test", "staging", "production"] = "development"
    log_level: str = "INFO"
    database_url: str = "sqlite:///data/ai_bi_copilot.db"
    ai_provider: Literal["auto", "claude", "gemini"] = "auto"
    claude_api_key: str = ""
    claude_model: str = "auto"
    gemini_api_key: str = ""
    gemini_model: str = "auto"
    max_upload_mb: int = Field(default=2048, gt=0, le=2048)

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        normalized = value.upper().strip()
        allowed_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if normalized not in allowed_levels:
            raise ValueError(f"LOG_LEVEL must be one of {', '.join(sorted(allowed_levels))}")
        return normalized

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("DATABASE_URL cannot be empty")
        if not (normalized.startswith("sqlite:///") or normalized.startswith("postgresql://")):
            raise ValueError("DATABASE_URL must use sqlite:/// or postgresql://")
        return normalized


def get_settings() -> Settings:
    """Return application settings from the current environment."""
    try:
        return Settings()
    except ValidationError as exc:
        raise SettingsError(str(exc)) from exc