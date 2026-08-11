"""Operational health checks for deployment readiness."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from src.config.settings import Settings


@dataclass(frozen=True)
class HealthCheck:
    """One deployment readiness check."""

    name: str
    status: str
    message: str


@dataclass(frozen=True)
class HealthReport:
    """Overall application health state."""

    status: str
    checks: list[HealthCheck]


def build_health_report(settings: Settings) -> HealthReport:
    """Build a lightweight readiness report for the current runtime."""
    checks = [
        _check_database(settings.database_url),
        _check_ai_provider(settings),
        _check_upload_limit(settings.max_upload_mb),
    ]
    status = "ready" if all(check.status in {"ok", "warning"} for check in checks) else "error"
    return HealthReport(status=status, checks=checks)


def _check_database(database_url: str) -> HealthCheck:
    if database_url.startswith("postgresql://"):
        return HealthCheck("Database", "warning", "PostgreSQL URL configured; adapter is planned but not active yet.")
    if not database_url.startswith("sqlite:///"):
        return HealthCheck("Database", "error", "DATABASE_URL must use sqlite:/// for the current runtime.")

    database_path = Path(database_url.removeprefix("sqlite:///"))
    try:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(database_path) as connection:
            connection.execute("SELECT 1")
    except sqlite3.Error as exc:
        return HealthCheck("Database", "error", f"SQLite connection failed: {exc}")
    except OSError as exc:
        return HealthCheck("Database", "error", f"SQLite path is not writable: {exc}")
    return HealthCheck("Database", "ok", f"SQLite database is reachable at {database_path}.")


def _check_ai_provider(settings: Settings) -> HealthCheck:
    if settings.ai_provider == "gemini" and not settings.gemini_api_key:
        return HealthCheck("AI Provider", "warning", "Gemini selected but GEMINI_API_KEY is not configured.")
    if settings.ai_provider == "claude" and not settings.claude_api_key:
        return HealthCheck("AI Provider", "warning", "Claude selected but CLAUDE_API_KEY is not configured.")
    if settings.ai_provider == "auto" and not (settings.gemini_api_key or settings.claude_api_key):
        return HealthCheck("AI Provider", "warning", "No AI key configured; non-AI dashboard features remain available.")
    return HealthCheck("AI Provider", "ok", f"AI provider mode is {settings.ai_provider}.")


def _check_upload_limit(max_upload_mb: int) -> HealthCheck:
    if max_upload_mb <= 0:
        return HealthCheck("Upload Limit", "error", "MAX_UPLOAD_MB must be greater than zero.")
    return HealthCheck("Upload Limit", "ok", f"Uploads are limited to {max_upload_mb} MB.")