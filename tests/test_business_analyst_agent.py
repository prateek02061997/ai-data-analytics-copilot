from types import SimpleNamespace

import pandas as pd
import pytest

from src.ai import AIConfigurationError, AIResponseError, BusinessAnalystAgent
from src.analytics import generate_profile_report
from src.cleaning import clean_dataframe


class FakeClaudeMessages:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.last_request = {}

    def create(self, **kwargs):
        self.last_request = kwargs
        return SimpleNamespace(content=[SimpleNamespace(text=self.response_text)])


class FailingClaudeMessages:
    def create(self, **kwargs):
        error = RuntimeError("not found")
        error.status_code = 404
        raise error


class FallbackClaudeMessages:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.models = []

    def create(self, **kwargs):
        self.models.append(kwargs["model"])
        if len(self.models) == 1:
            error = RuntimeError("not found")
            error.status_code = 404
            raise error
        return SimpleNamespace(content=[SimpleNamespace(text=self.response_text)])


class FallbackGeminiMessages:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.models = []

    def create(self, **kwargs):
        self.models.append(kwargs["model"])
        if len(self.models) == 1:
            raise RuntimeError("404 This model models/gemini-2.5-flash is no longer available to new users")
        return SimpleNamespace(content=[SimpleNamespace(text=self.response_text)])


def test_business_analyst_agent_returns_structured_response() -> None:
    dataframe = pd.DataFrame({"Revenue": [100, 120], "Region": ["Auckland", "Wellington"]})
    cleaning_result = clean_dataframe(dataframe)
    profile = generate_profile_report(dataframe)
    fake_client = FakeClaudeMessages(
        '{"summary":"Revenue increased.",'
        '"key_findings":["Wellington revenue is higher."],'
        '"business_recommendations":["Review regional growth drivers."]}'
    )
    agent = BusinessAnalystAgent(api_key="", client=fake_client)

    response = agent.answer_question(
        question="Which region performs best?",
        dataframe=cleaning_result.cleaned_dataframe,
        profile_report=profile,
        cleaning_summary=cleaning_result.summary,
        business_context="Retail branch sales",
    )

    assert response.summary == "Revenue increased."
    assert response.key_findings == ["Wellington revenue is higher."]
    assert response.business_recommendations == ["Review regional growth drivers."]
    assert "Which region performs best?" in fake_client.last_request["messages"][0]["content"]
    assert "Retail branch sales" in fake_client.last_request["messages"][0]["content"]


def test_business_analyst_agent_accepts_gemini_provider_with_client() -> None:
    dataframe = pd.DataFrame({"Patient Age": [30, 40], "Diagnosis": ["A", "B"]})
    cleaning_result = clean_dataframe(dataframe)
    profile = generate_profile_report(dataframe)
    fake_client = FakeClaudeMessages(
        '{"summary":"Patient data is available.",'
        '"key_findings":["Two patient records are present."],'
        '"business_recommendations":["Review diagnosis distribution."]}'
    )
    agent = BusinessAnalystAgent(api_key="", model="gemini-1.5-flash", provider="gemini", client=fake_client)

    response = agent.answer_question(
        question="Summarise patient records.",
        dataframe=cleaning_result.cleaned_dataframe,
        profile_report=profile,
        cleaning_summary=cleaning_result.summary,
    )

    assert response.summary == "Patient data is available."
    assert fake_client.last_request["model"] == "gemini-1.5-flash"


def test_business_analyst_agent_requires_api_key_without_client() -> None:
    with pytest.raises(AIConfigurationError, match="AI API key"):
        BusinessAnalystAgent(api_key="")


def test_business_analyst_agent_rejects_invalid_json_response() -> None:
    dataframe = pd.DataFrame({"Revenue": [100]})
    cleaning_result = clean_dataframe(dataframe)
    profile = generate_profile_report(dataframe)
    agent = BusinessAnalystAgent(api_key="", client=FakeClaudeMessages("not json"))

    with pytest.raises(AIResponseError, match="valid JSON"):
        agent.answer_question(
            question="Why did revenue decrease?",
            dataframe=cleaning_result.cleaned_dataframe,
            profile_report=profile,
            cleaning_summary=cleaning_result.summary,
        )


def test_business_analyst_agent_wraps_provider_model_errors() -> None:
    dataframe = pd.DataFrame({"Revenue": [100]})
    cleaning_result = clean_dataframe(dataframe)
    profile = generate_profile_report(dataframe)
    agent = BusinessAnalystAgent(api_key="", model="bad-model", client=FailingClaudeMessages())

    with pytest.raises(AIResponseError, match="bad-model"):
        agent.answer_question(
            question="Why did revenue decrease?",
            dataframe=cleaning_result.cleaned_dataframe,
            profile_report=profile,
            cleaning_summary=cleaning_result.summary,
        )


def test_business_analyst_agent_auto_model_fallback_tries_next_model() -> None:
    dataframe = pd.DataFrame({"Revenue": [100]})
    cleaning_result = clean_dataframe(dataframe)
    profile = generate_profile_report(dataframe)
    fake_client = FallbackClaudeMessages(
        '{"summary":"Revenue is stable.",'
        '"key_findings":["Revenue is available."],'
        '"business_recommendations":["Continue monitoring revenue."]}'
    )
    agent = BusinessAnalystAgent(api_key="", model="auto", client=fake_client)

    response = agent.answer_question(
        question="How is revenue performing?",
        dataframe=cleaning_result.cleaned_dataframe,
        profile_report=profile,
        cleaning_summary=cleaning_result.summary,
    )

    assert response.summary == "Revenue is stable."
    assert len(fake_client.models) == 2


def test_business_analyst_agent_gemini_auto_model_fallback_tries_next_model() -> None:
    dataframe = pd.DataFrame({"Patient Age": [30, 40], "Diagnosis": ["A", "B"]})
    cleaning_result = clean_dataframe(dataframe)
    profile = generate_profile_report(dataframe)
    fake_client = FallbackGeminiMessages(
        '{"summary":"Patient data is ready.",'
        '"key_findings":["Diagnosis values are available."],'
        '"business_recommendations":["Review diagnosis mix."]}'
    )
    agent = BusinessAnalystAgent(api_key="", model="auto", provider="gemini", client=fake_client)

    response = agent.answer_question(
        question="Summarise patient records.",
        dataframe=cleaning_result.cleaned_dataframe,
        profile_report=profile,
        cleaning_summary=cleaning_result.summary,
    )

    assert response.summary == "Patient data is ready."
    assert fake_client.models[:2] == ["gemini-flash-latest", "gemini-3.5-flash"]