from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from dmf_bench.models import LLMResponse, TokenUsage
from dmf_bench.providers.judge_prompts import JUDGE_SYSTEM_PROMPT, build_judge_user_prompt
from dmf_bench.providers.openai_compatible import (
    OpenAIClient,
    ProviderRequestError,
    ProviderResponseError,
    resolve_provider_runtime_config,
)
from dmf_bench.adapters.base import AnswererRequest, JudgeRequest
from dmf_bench.adapters.providers import (
    OpenAICompatibleAnswererAdapter,
    OpenAICompatibleJudgeAdapter,
    ProviderModelConfig,
    answerer_factories,
    judge_factories,
    judge_fingerprint,
)
from dmf_bench.runtime import RuntimeFactories, assemble_runtime, benchmark_factories


class FakeTransport:
    def __init__(self, response: LLMResponse) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def generate_with_usage(self, **kwargs: Any) -> LLMResponse:
        self.calls.append(dict(kwargs))
        return self.response


def settings(role: str, *, provider: str = "openai") -> ProviderModelConfig:
    return ProviderModelConfig(
        role=role,
        provider=provider,
        requested_model=f"requested-{role}",
        temperature=0.0,
        max_tokens=256,
        reasoning_effort="low" if role == "judge" else None,
        timeout_seconds=30.0,
        rpm=100,
        max_retries=2,
    )


def response(text: str, *, model: str = "returned-model") -> LLMResponse:
    return LLMResponse(
        response=text,
        token_usage=TokenUsage(
            prompt_tokens_total=11,
            completion_tokens=7,
            total_tokens=18,
        ),
        model=model,
        finish_reason="stop",
    )


def experiment_config(
    *,
    benchmark: str = "locomo",
    answerer_provider: str = "openai",
    judge_provider: str = "openai",
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "benchmark": benchmark,
        "framework": "dmf",
        "models": {
            "answerer": {
                "provider": answerer_provider,
                "requested_model": "requested-answerer",
                "parameters": {"temperature": 0, "max_tokens": 256},
                "runtime": {"timeout_seconds": 30, "rpm": 100, "max_retries": 2},
            },
            "judge": {
                "provider": judge_provider,
                "requested_model": "requested-judge",
                "parameters": {
                    "temperature": 0,
                    "max_tokens": 256,
                    "reasoning_effort": "low",
                },
                "runtime": {"timeout_seconds": 30, "rpm": 100, "max_retries": 2},
            },
        },
    }


def test_answerer_preserves_prompt_and_provider_response_metadata() -> None:
    transport = FakeTransport(response("The answer."))
    adapter = OpenAICompatibleAnswererAdapter(
        settings=settings("answerer"),
        transport=transport,
    )

    result = adapter.generate(
        AnswererRequest(
            system_prompt="system prompt",
            user_prompt="user prompt",
            metadata={"question_id": "q-1"},
        )
    )

    assert transport.calls == [
        {
            "system_prompt": "system prompt",
            "user_prompt": "user prompt",
            "temperature": 0.0,
            "max_tokens": 256,
            "reasoning_effort": None,
        }
    ]
    assert result == {
        "generated_answer": "The answer.",
        "answerer_provider": "openai",
        "answerer_requested_model": "requested-answerer",
        "answerer_model": "returned-model",
        "answerer_finish_reason": "stop",
        "answerer_usage": {
            "prompt_tokens_total": 11,
            "completion_tokens": 7,
            "total_tokens": 18,
        },
    }


@pytest.mark.parametrize("benchmark", ["locomo", "longmemeval"])
def test_judge_reuses_legacy_prompt_surface_and_parser(benchmark: str) -> None:
    transport = FakeTransport(response('{"reasoning":"same fact","label":"CORRECT"}'))
    adapter = OpenAICompatibleJudgeAdapter(
        benchmark=benchmark,
        settings=settings("judge"),
        transport=transport,
    )
    prediction = {
        "question": "When did it happen?",
        "ground_truth_answer": "Yesterday",
        "generated_answer": "Yesterday",
        "question_type": "temporal",
        "question_date": "2026-07-18",
        "category": 2,
        "evidence": ["D1:3"],
    }

    result = adapter.judge(
        JudgeRequest(prediction=prediction)
    )

    expected_prompt = build_judge_user_prompt(
        question="When did it happen?",
        ground_truth_answer="Yesterday",
        generated_answer="Yesterday",
        question_type="temporal" if benchmark == "longmemeval" else "",
        question_date="2026-07-18" if benchmark == "longmemeval" else "",
    )
    assert transport.calls[0]["system_prompt"] == JUDGE_SYSTEM_PROMPT
    assert transport.calls[0]["user_prompt"] == expected_prompt
    assert result == {
        "judgment": "CORRECT",
        "score": 1.0,
        "reason": "same fact",
        "judge_provider": "openai",
        "judge_requested_model": "requested-judge",
        "judge_model": "returned-model",
        "judge_finish_reason": "stop",
        "judge_usage": {
            "prompt_tokens_total": 11,
            "completion_tokens": 7,
            "total_tokens": 18,
        },
        "judge_fingerprint": judge_fingerprint(benchmark),
    }


def test_malformed_judge_response_fails_closed() -> None:
    adapter = OpenAICompatibleJudgeAdapter(
        benchmark="locomo",
        settings=settings("judge"),
        transport=FakeTransport(response("I cannot decide.")),
    )

    with pytest.raises(ProviderResponseError, match="no recognized verdict"):
        adapter.judge(JudgeRequest(prediction={}))


def test_transport_retries_only_retryable_failures() -> None:
    sleeps: list[float] = []
    client = OpenAIClient(
        model="test-model",
        api_key="test-key",
        max_retries=2,
        rpm=0,
        sleeper=sleeps.append,
    )
    attempts = 0

    def create(**_kwargs: Any) -> Any:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TimeoutError("transient")
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="recovered"),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=2, completion_tokens=1, total_tokens=3),
            model="returned-model",
            model_dump=lambda: {},
        )

    client.client.chat.completions.create = create

    result = client.generate_with_usage("system", "user")

    assert result.response == "recovered"
    assert attempts == 2
    assert len(sleeps) == 1


def test_transport_does_not_retry_non_retryable_failures() -> None:
    sleeps: list[float] = []
    client = OpenAIClient(
        model="test-model",
        api_key="test-key",
        max_retries=2,
        rpm=0,
        sleeper=sleeps.append,
    )
    attempts = 0

    def create(**_kwargs: Any) -> Any:
        nonlocal attempts
        attempts += 1
        raise ValueError("invalid request")

    client.client.chat.completions.create = create

    with pytest.raises(ProviderRequestError) as failure:
        client.generate_with_usage("system", "user")

    assert failure.value.retryable is False
    assert failure.value.attempts == 1
    assert attempts == 1
    assert sleeps == []


@pytest.mark.parametrize("provider", ["openai", "openrouter", "ollama"])
def test_provider_factories_cover_supported_provider_without_network(provider: str) -> None:
    answer_response = response("answer")
    judge_response = response('{"reasoning":"ok","label":"CORRECT"}')

    def transport_builder(
        model_settings: ProviderModelConfig,
        _metrics: Any,
    ) -> FakeTransport:
        return FakeTransport(
            answer_response if model_settings.role == "answerer" else judge_response
        )

    config = experiment_config(
        answerer_provider=provider,
        judge_provider=provider,
    )
    factories = RuntimeFactories(
        benchmarks=benchmark_factories(),
        frameworks={"dmf": lambda _config: SimpleNamespace(name="dmf")},
        answerers=answerer_factories(transport_builder=transport_builder),
        judges=judge_factories(transport_builder=transport_builder),
    )

    components = assemble_runtime(config, factories=factories)

    assert components.answerer.name == provider
    assert components.judge.settings.provider == provider


@pytest.mark.parametrize(
    ("provider", "environment", "message"),
    [
        ("openai", ("OPENAI_API_KEY", "OPENAI_BASE_URL"), "OPENAI_API_KEY"),
        (
            "openrouter",
            ("OPENROUTER_API_KEY", "OPENROUTER_BASE_URL"),
            "OPENROUTER_API_KEY",
        ),
        ("ollama", ("OLLAMA_BASE_URL",), "OLLAMA_BASE_URL"),
    ],
)
def test_missing_provider_environment_fails_during_runtime_assembly(
    provider: str,
    environment: tuple[str, ...],
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in environment:
        monkeypatch.delenv(name, raising=False)
    config = experiment_config(
        answerer_provider=provider,
        judge_provider=provider,
    )
    framework_builds: list[str] = []

    def build_framework(_config: dict[str, Any]) -> SimpleNamespace:
        framework_builds.append("dmf")
        return SimpleNamespace(name="dmf")

    factories = RuntimeFactories(
        benchmarks=benchmark_factories(),
        frameworks={"dmf": build_framework},
        answerers=answerer_factories(),
        judges=judge_factories(),
    )

    with pytest.raises(ValueError, match=message):
        assemble_runtime(config, factories=factories)
    assert framework_builds == []


def test_openrouter_missing_endpoint_fails_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)

    with pytest.raises(ValueError, match="OPENROUTER_BASE_URL"):
        resolve_provider_runtime_config("openrouter")


def test_openai_empty_optional_endpoint_uses_sdk_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "")

    assert resolve_provider_runtime_config("openai") == ("test-key", None)


def test_provider_model_config_rejects_unhandled_scientific_parameter() -> None:
    config = experiment_config()
    config["models"]["answerer"]["parameters"]["top_p"] = 0.9

    with pytest.raises(ValueError, match="unsupported fields"):
        ProviderModelConfig.from_experiment(config, role="answerer")
