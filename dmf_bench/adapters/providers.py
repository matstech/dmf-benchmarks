"""Provider-backed answerer and benchmark-specific judge adapters."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from dmf_bench.providers.judge_prompts import (
    JUDGE_RETRY_INSTRUCTION,
    JUDGE_SYSTEM_PROMPT,
    build_judge_user_prompt,
)
from dmf_bench.models import LLMResponse
from dmf_bench.providers.openai_compatible import (
    OpenAIClient,
    ProviderResponseError,
    normalize_provider_name,
    resolve_provider_runtime_config,
)
from dmf_bench.fingerprints import judge_fingerprint
from dmf_bench.metrics import BenchmarkMetrics

from .base import AnswererRequest, JudgeRequest


SUPPORTED_BENCHMARKS = ("locomo", "longmemeval")


class LLMTransport(Protocol):
    """Narrow transport surface used by both provider adapters."""

    def generate_with_usage(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        include_raw: bool = False,
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        """Return provider text and normalized response metadata."""


@dataclass(frozen=True)
class ProviderModelConfig:
    """Scientific and operational settings for one provider role."""

    role: str
    provider: str
    requested_model: str
    temperature: float
    max_tokens: int
    reasoning_effort: str | None
    timeout_seconds: float
    rpm: int
    max_retries: int
    response_max_retries: int = 0

    @classmethod
    def from_experiment(
        cls,
        config: dict[str, Any],
        *,
        role: str,
    ) -> "ProviderModelConfig":
        if role not in {"answerer", "judge"}:
            raise ValueError(f"Unsupported model role: {role!r}.")
        models = _required_mapping(config, "models")
        model = _required_mapping(models, role)
        parameters = _required_mapping(model, "parameters")
        runtime = _required_mapping(model, "runtime")

        provider = normalize_provider_name(_required_string(model, "provider"))
        requested_model = _required_string(model, "requested_model")
        temperature = _number(parameters, "temperature", default=0.0)
        max_tokens = _positive_integer(parameters, "max_tokens", default=4096)
        raw_reasoning_effort = parameters.get("reasoning_effort")
        if raw_reasoning_effort is not None and (
            not isinstance(raw_reasoning_effort, str)
            or not raw_reasoning_effort.strip()
        ):
            raise ValueError(
                f"models.{role}.parameters.reasoning_effort must be a non-empty string or null."
            )
        unsupported = sorted(
            set(parameters) - {"temperature", "max_tokens", "reasoning_effort"}
        )
        if unsupported:
            raise ValueError(
                f"models.{role}.parameters contains unsupported fields: {unsupported}."
            )

        return cls(
            role=role,
            provider=provider,
            requested_model=requested_model,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=(
                raw_reasoning_effort.strip()
                if isinstance(raw_reasoning_effort, str)
                else None
            ),
            timeout_seconds=_positive_number(runtime, "timeout_seconds"),
            rpm=_positive_integer(runtime, "rpm"),
            max_retries=_non_negative_integer(runtime, "max_retries"),
            response_max_retries=_non_negative_integer(
                runtime,
                "response_max_retries",
                default=1 if role == "judge" else 0,
            ),
        )


class OpenAICompatibleAnswererAdapter:
    """Generate benchmark answers through an injected OpenAI-compatible transport."""

    def __init__(
        self,
        *,
        settings: ProviderModelConfig,
        transport: LLMTransport,
        metrics: BenchmarkMetrics | None = None,
    ) -> None:
        if settings.role != "answerer":
            raise ValueError("Answerer adapter requires answerer model settings.")
        self.name = settings.provider
        self.settings = settings
        self.transport = transport
        self.metrics = metrics

    def generate(self, request: AnswererRequest) -> dict[str, Any]:
        response = self._call(
            system_prompt=request.system_prompt,
            user_prompt=request.user_prompt,
        )
        return {
            "generated_answer": response.response,
            "answerer_provider": self.settings.provider,
            "answerer_requested_model": self.settings.requested_model,
            "answerer_model": response.model,
            "answerer_finish_reason": response.finish_reason,
            "answerer_usage": _usage_payload(response),
        }

    def _call(self, *, system_prompt: str, user_prompt: str) -> LLMResponse:
        try:
            response = self.transport.generate_with_usage(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_tokens,
                reasoning_effort=self.settings.reasoning_effort,
            )
            if not response.response.strip():
                raise ProviderResponseError("Answerer provider returned an empty response.")
        except Exception:
            _record_llm(self.metrics, role="answerer", provider=self.settings.provider, outcome="failed")
            raise
        _record_llm(
            self.metrics,
            role="answerer",
            provider=self.settings.provider,
            outcome="completed",
            response=response,
        )
        return response


class OpenAICompatibleJudgeAdapter:
    """Apply the current benchmark judge rubric through a provider transport."""

    def __init__(
        self,
        *,
        benchmark: str,
        settings: ProviderModelConfig,
        transport: LLMTransport,
        metrics: BenchmarkMetrics | None = None,
    ) -> None:
        if benchmark not in SUPPORTED_BENCHMARKS:
            raise ValueError(f"Unsupported judge benchmark: {benchmark!r}.")
        if settings.role != "judge":
            raise ValueError("Judge adapter requires judge model settings.")
        self.name = f"{benchmark}-{settings.provider}-judge"
        self.benchmark = benchmark
        self.settings = settings
        self.transport = transport
        self.metrics = metrics
        self.judge_fingerprint = judge_fingerprint(benchmark)

    def judge(self, request: JudgeRequest) -> dict[str, Any]:
        prediction = request.prediction
        prompt = build_judge_user_prompt(
            question=str(prediction.get("question", "")),
            ground_truth_answer=str(prediction.get("ground_truth_answer", "")),
            generated_answer=str(
                prediction.get("generated_answer", prediction.get("prediction", ""))
            ),
            question_type=(
                str(prediction.get("question_type", ""))
                if self.benchmark == "longmemeval"
                else ""
            ),
            question_date=(
                str(prediction.get("question_date", ""))
                if self.benchmark == "longmemeval"
                else ""
            ),
        )
        responses: list[LLMResponse] = []
        total_attempts = self.settings.response_max_retries + 1
        for attempt in range(total_attempts):
            try:
                response = self.transport.generate_with_usage(
                    system_prompt=JUDGE_SYSTEM_PROMPT,
                    user_prompt=(
                        prompt if attempt == 0 else prompt + JUDGE_RETRY_INSTRUCTION
                    ),
                    temperature=self.settings.temperature,
                    max_tokens=self.settings.max_tokens,
                    reasoning_effort=self.settings.reasoning_effort,
                )
                responses.append(response)
                if response.finish_reason not in {None, "stop"}:
                    raise ProviderResponseError(
                        "Judge provider returned an incomplete response "
                        f"(finish_reason={response.finish_reason!r})."
                    )
                judgment, score, reason = _parse_judge_response_strict(
                    self.benchmark,
                    response.response,
                )
            except ProviderResponseError:
                _record_llm(
                    self.metrics,
                    role="judge",
                    provider=self.settings.provider,
                    outcome="failed",
                    response=responses[-1] if responses else None,
                )
                if attempt + 1 >= total_attempts:
                    raise
                if self.metrics is not None:
                    self.metrics.record_llm_retry(
                        role="judge",
                        provider=self.settings.provider,
                    )
                continue
            except Exception:
                _record_llm(
                    self.metrics,
                    role="judge",
                    provider=self.settings.provider,
                    outcome="failed",
                )
                raise
            _record_llm(
                self.metrics,
                role="judge",
                provider=self.settings.provider,
                outcome="completed",
                response=response,
            )
            break
        else:  # pragma: no cover - loop exits by return or exception
            raise AssertionError("Judge response retry loop exited unexpectedly.")

        return {
            "judgment": judgment,
            "score": score,
            "reason": reason,
            "judge_provider": self.settings.provider,
            "judge_requested_model": self.settings.requested_model,
            "judge_model": response.model,
            "judge_finish_reason": response.finish_reason,
            "judge_usage": _cumulative_usage_payload(responses),
            "judge_attempts": len(responses),
            "judge_fingerprint": self.judge_fingerprint,
        }


TransportBuilder = Callable[[ProviderModelConfig, BenchmarkMetrics | None], LLMTransport]


def answerer_factories(
    *,
    metrics: BenchmarkMetrics | None = None,
    transport_builder: TransportBuilder | None = None,
) -> dict[str, Callable[[dict[str, Any]], OpenAICompatibleAnswererAdapter]]:
    """Return explicit factories for every supported answerer provider."""
    builder = transport_builder or _default_transport

    def build(config: dict[str, Any]) -> OpenAICompatibleAnswererAdapter:
        settings = ProviderModelConfig.from_experiment(config, role="answerer")
        return OpenAICompatibleAnswererAdapter(
            settings=settings,
            transport=builder(settings, metrics),
            metrics=metrics,
        )

    return {provider: build for provider in ("openai", "openrouter", "ollama")}


def judge_factories(
    *,
    metrics: BenchmarkMetrics | None = None,
    transport_builder: TransportBuilder | None = None,
) -> dict[tuple[str, str], Callable[[dict[str, Any]], OpenAICompatibleJudgeAdapter]]:
    """Return benchmark/provider judge factories without dynamic discovery."""
    builder = transport_builder or _default_transport

    def for_benchmark(
        benchmark: str,
    ) -> Callable[[dict[str, Any]], OpenAICompatibleJudgeAdapter]:
        def build(config: dict[str, Any]) -> OpenAICompatibleJudgeAdapter:
            settings = ProviderModelConfig.from_experiment(config, role="judge")
            return OpenAICompatibleJudgeAdapter(
                benchmark=benchmark,
                settings=settings,
                transport=builder(settings, metrics),
                metrics=metrics,
            )

        return build

    return {
        (benchmark, provider): for_benchmark(benchmark)
        for benchmark in SUPPORTED_BENCHMARKS
        for provider in ("openai", "openrouter", "ollama")
    }


def _default_transport(
    settings: ProviderModelConfig,
    metrics: BenchmarkMetrics | None,
) -> OpenAIClient:
    api_key, base_url = resolve_provider_runtime_config(settings.provider)

    def record_retry(_exc: BaseException, _attempts: int) -> None:
        if metrics is not None:
            metrics.record_llm_retry(role=settings.role, provider=settings.provider)

    return OpenAIClient(
        model=settings.requested_model,
        api_key=api_key,
        base_url=base_url,
        max_retries=settings.max_retries,
        timeout=settings.timeout_seconds,
        rpm=settings.rpm,
        retry_callback=record_retry,
    )


def _parse_judge_response_strict(
    benchmark: str,
    text: str,
) -> tuple[str, float, str]:
    normalized = text.strip()
    if not normalized:
        raise ProviderResponseError("Judge provider returned an empty response.")

    try:
        payload = json.loads(normalized)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict):
        raw_label = str(payload.get("label", payload.get("judgment", ""))).strip().upper()
        allowed = {"CORRECT", "WRONG"}
        if benchmark == "longmemeval":
            allowed.update({"PASS", "FAIL"})
        if raw_label not in allowed:
            raise ProviderResponseError("Judge response JSON has no recognized label.")
    elif not re.search(
        r"(?:VERDICT|JUDGMENT)\s*:\s*(CORRECT|WRONG|PASS|FAIL)",
        normalized,
        flags=re.IGNORECASE,
    ):
        raise ProviderResponseError("Judge response has no recognized verdict.")

    if benchmark == "locomo":
        from dmf_bench.benchmarks.locomo.judge import parse_judge_response
    else:
        from dmf_bench.benchmarks.longmemeval.judge import parse_judge_response
    return parse_judge_response(normalized)


def _usage_payload(response: LLMResponse) -> dict[str, int]:
    usage = response.token_usage
    return {
        "prompt_tokens_total": usage.prompt_tokens_total,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }


def _cumulative_usage_payload(responses: list[LLMResponse]) -> dict[str, int]:
    return {
        "prompt_tokens_total": sum(
            response.token_usage.prompt_tokens_total for response in responses
        ),
        "completion_tokens": sum(
            response.token_usage.completion_tokens for response in responses
        ),
        "total_tokens": sum(response.token_usage.total_tokens for response in responses),
    }


def _record_llm(
    metrics: BenchmarkMetrics | None,
    *,
    role: str,
    provider: str,
    outcome: str,
    response: LLMResponse | None = None,
) -> None:
    if metrics is None:
        return
    usage = response.token_usage if response is not None else None
    metrics.record_llm_request(
        role=role,
        provider=provider,
        outcome=outcome,
        prompt_tokens=usage.prompt_tokens_total if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
    )


def _required_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object.")
    return value


def _required_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string.")
    return value.strip()


def _number(data: dict[str, Any], key: str, *, default: float) -> float:
    value = data.get(key, default)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{key} must be a number.")
    return float(value)


def _positive_number(data: dict[str, Any], key: str) -> float:
    value = _number(data, key, default=0.0)
    if value <= 0:
        raise ValueError(f"{key} must be a positive number.")
    return value


def _positive_integer(
    data: dict[str, Any],
    key: str,
    *,
    default: int | None = None,
) -> int:
    value = data.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{key} must be a positive integer.")
    return value


def _non_negative_integer(
    data: dict[str, Any],
    key: str,
    *,
    default: int | None = None,
) -> int:
    value = data.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{key} must be a non-negative integer.")
    return value
