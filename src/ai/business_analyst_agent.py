"""AI-powered business analyst agent."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

import pandas as pd

from src.analytics import ProfileReport
from src.cleaning import CleaningSummary


DEFAULT_CLAUDE_MODEL = "auto"
CLAUDE_MODEL_FALLBACKS = [
    "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-20240620",
    "claude-3-haiku-20240307",
    "claude-3-sonnet-20240229",
]
GEMINI_MODEL_FALLBACKS = [
    "gemini-flash-latest",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash",
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash",
]
MAX_SAMPLE_ROWS = 10


class AIConfigurationError(RuntimeError):
    """Raised when the AI analyst cannot be configured."""


class AIResponseError(RuntimeError):
    """Raised when the AI provider returns an unusable response."""


class ClaudeMessageClient(Protocol):
    """Minimal protocol for an AI messages client."""

    def create(self, **kwargs: Any) -> Any:
        """Create an AI message."""


@dataclass(frozen=True)
class BusinessAnalysisResponse:
    """Structured business analysis returned by the AI analyst."""

    summary: str
    key_findings: list[str]
    business_recommendations: list[str]
    raw_response: str


class BusinessAnalystAgent:
    """Generate business analysis from dataset context using the configured AI provider."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_CLAUDE_MODEL,
        client: ClaudeMessageClient | None = None,
        provider: str = "claude",
    ) -> None:
        if not api_key and client is None:
            raise AIConfigurationError("An AI API key is required to use the AI Business Analyst Agent.")

        self.provider = provider
        self.models = _model_candidates(model, provider=provider)
        self._client = client or _build_provider_client(provider=provider, api_key=api_key, model=model)

    def answer_question(
        self,
        question: str,
        dataframe: pd.DataFrame,
        profile_report: ProfileReport,
        cleaning_summary: CleaningSummary,
        business_context: str = "",
        history: list[dict[str, str]] | None = None,
    ) -> BusinessAnalysisResponse:
        """Answer a business question using dataset structure, quality context, and chat history."""
        clean_question = question.strip()
        if not clean_question:
            raise ValueError("Business question cannot be empty.")

        prompt = _build_prompt(
            question=clean_question,
            dataframe=dataframe,
            profile_report=profile_report,
            cleaning_summary=cleaning_summary,
            business_context=business_context,
            history=history,
        )
        response = create_message_with_model_fallback(
            client=self._client,
            models=self.models,
            provider=self.provider,
            max_tokens=1200,
            temperature=0.2,
            system=(
                "You are a senior business intelligence analyst. "
                "Use only the provided dataset context. Do not invent facts. "
                "If the answer is not present in the dataset context, reply exactly: This information is not available in the uploaded dataset. "
                "Return concise, executive-ready analysis as valid JSON."
            ),
            messages=[{"role": "user", "content": prompt}],
        )
        raw_text = _extract_text(response)
        return _parse_response(raw_text)


def _build_anthropic_client(api_key: str) -> ClaudeMessageClient:
    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise AIConfigurationError("Install the anthropic package before using the Claude API.") from exc

    return Anthropic(api_key=api_key).messages


def _build_provider_client(provider: str, api_key: str, model: str) -> ClaudeMessageClient:
    if provider == "gemini":
        return _build_gemini_client(api_key=api_key, model=model)
    return _build_anthropic_client(api_key)


class GeminiMessageClient:
    """Adapter that exposes Gemini responses through the existing message client shape."""

    def __init__(self, api_key: str, model: str) -> None:
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise AIConfigurationError("Install google-generativeai before using the Gemini API.") from exc

        genai.configure(api_key=api_key)
        self._genai = genai
        self._model_name = model or "auto"

    def create(self, **kwargs: Any) -> Any:
        model_name = kwargs.get("model") or self._model_name
        model = self._genai.GenerativeModel(
            model_name=model_name,
            system_instruction=kwargs.get("system", ""),
        )
        prompt = "\n\n".join(str(message.get("content", "")) for message in kwargs.get("messages", []))
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": kwargs.get("temperature", 0.2),
                "max_output_tokens": kwargs.get("max_tokens", 1200),
                "response_mime_type": "application/json",
            },
        )
        return _text_response(getattr(response, "text", ""))


def _build_gemini_client(api_key: str, model: str) -> GeminiMessageClient:
    return GeminiMessageClient(api_key=api_key, model=model)


def _text_response(text: str) -> Any:
    from types import SimpleNamespace

    return SimpleNamespace(content=[SimpleNamespace(text=text)])


def create_message_with_model_fallback(client: ClaudeMessageClient, models: list[str], provider: str = "claude", **kwargs: Any) -> Any:
    """Create an AI message, trying fallback models only when a model is not found."""
    not_found_models: list[str] = []
    for model in models:
        try:
            return client.create(model=model, **kwargs)
        except Exception as exc:
            if not _is_model_not_found_error(exc):
                raise AIResponseError(_provider_error_message(exc, model, provider=provider)) from exc
            not_found_models.append(model)

    tried = ", ".join(not_found_models)
    provider_name = _provider_display_name(provider)
    raise AIResponseError(f"No configured {provider_name} model was available for this API key. Tried: {tried}.")


def _model_candidates(model: str, provider: str = "claude") -> list[str]:
    normalized = model.strip()
    if not normalized or normalized.casefold() == "auto":
        if provider == "gemini":
            return GEMINI_MODEL_FALLBACKS
        return CLAUDE_MODEL_FALLBACKS
    return [normalized]


def _provider_error_message(exc: Exception, model: str, provider: str = "claude") -> str:
    status_code = getattr(exc, "status_code", None)
    provider_name = _provider_display_name(provider)
    key_name = "GEMINI_API_KEY" if provider == "gemini" else "CLAUDE_API_KEY"
    model_name = "GEMINI_MODEL" if provider == "gemini" else "CLAUDE_MODEL"
    if status_code == 401:
        return f"{provider_name} authentication failed. Check {key_name} in `.env`."
    if _is_model_not_found_error(exc):
        return f"{provider_name} model `{model}` was not found. Set {model_name}=auto in `.env` to try fallback models."
    if status_code == 429:
        return f"{provider_name} rate limit reached. Wait and try again."
    if status_code is not None:
        return f"{provider_name} API request failed with status {status_code}."
    return f"{provider_name} API request failed: {exc}"


def _is_model_not_found_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code == 404:
        return True
    message = str(exc).casefold()
    return "404" in message and "model" in message and ("not found" in message or "no longer available" in message)


def _provider_display_name(provider: str) -> str:
    return "Gemini" if provider == "gemini" else "Claude"


def _build_prompt(
    question: str,
    dataframe: pd.DataFrame,
    profile_report: ProfileReport,
    cleaning_summary: CleaningSummary,
    business_context: str,
    history: list[dict[str, str]] | None = None,
) -> str:
    context_payload = {
        "business_context": business_context.strip() or "Not provided",
        "chat_history": history[-6:] if history else [],
        "question": question,
        "dataset": {
            "rows": len(dataframe),
            "columns": len(dataframe.columns),
            "column_names": [str(column) for column in dataframe.columns],
            "data_types": {str(column): str(dtype) for column, dtype in dataframe.dtypes.items()},
            "sample_rows": dataframe.head(MAX_SAMPLE_ROWS).to_dict(orient="records"),
        },
        "profiling": {
            "health_score": profile_report.health_score,
            "missing_values": profile_report.missing_values,
            "duplicate_records": profile_report.duplicate_records,
            "issues": [
                {
                    "category": issue.category,
                    "field": issue.field,
                    "severity": issue.severity,
                    "affected_rows": issue.affected_rows,
                    "message": issue.message,
                }
                for issue in profile_report.issues
            ],
        },
        "cleaning": {
            "records_processed": cleaning_summary.records_processed,
            "rows_modified": cleaning_summary.rows_modified,
            "cells_modified": cleaning_summary.cells_modified,
            "records_removed": cleaning_summary.records_removed,
            "missing_values_filled": cleaning_summary.missing_values_filled,
            "duplicates_removed": cleaning_summary.duplicates_removed,
            "outliers_flagged": cleaning_summary.outliers_flagged,
            "formats_standardized": cleaning_summary.formats_standardized,
            "anomalies_detected": cleaning_summary.anomalies_detected,
        },
    }
    return (
        "Analyze the business question using the provided dataset context and previous conversation history. "
        "Do not use generic knowledge, assumptions, or facts outside this uploaded dataset. "
        "If a requested answer cannot be derived from the columns, sample rows, profiling, or cleaning context, "
        "return JSON where summary is exactly 'This information is not available in the uploaded dataset.' and both arrays contain that same sentence. "
        "Every answer must explicitly include dataset fields used, calculations performed, assumptions, and limitations. "
        "Return JSON with exactly these keys: summary, key_findings, business_recommendations. "
        "key_findings and business_recommendations must be arrays of strings.\n\n"
        f"Context:\n{json.dumps(context_payload, default=str, indent=2)}"
    )


def _extract_text(response: Any) -> str:
    content = getattr(response, "content", None)
    if not content:
        raise AIResponseError("AI provider returned an empty response.")

    parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if text:
            parts.append(str(text))

    raw_text = "\n".join(parts).strip()
    if not raw_text:
        raise AIResponseError("AI provider response did not contain text.")
    return raw_text


def _parse_response(raw_text: str) -> BusinessAnalysisResponse:
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise AIResponseError("AI provider response was not valid JSON.") from exc

    summary = str(payload.get("summary", "")).strip()
    key_findings = _string_list(payload.get("key_findings"))
    business_recommendations = _string_list(payload.get("business_recommendations"))

    if not summary or not key_findings or not business_recommendations:
        raise AIResponseError("AI provider response did not include the required analysis sections.")

    return BusinessAnalysisResponse(
        summary=summary,
        key_findings=key_findings,
        business_recommendations=business_recommendations,
        raw_response=raw_text,
    )


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]