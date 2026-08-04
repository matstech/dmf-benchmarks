# Copyright (c) 2026-present matstech
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT

"""
OpenAI client for benchmark answer generation and LLM-as-judge calls.

This module keeps the API intentionally small:
- one client class
- one text-only helper
- one text+usage helper

The normalized usage shape matches the benchmark conventions already used
in the Mem0 benchmark:
    - prompt_tokens_total
    - completion_tokens
    - total_tokens
"""

from __future__ import annotations

import logging
import os
import random
import time
from typing import Any, Callable

import openai
from openai import OpenAI

from .models import LLMResponse, TokenUsage

logger = logging.getLogger(__name__)

SUPPORTED_PROVIDERS = frozenset({"openai", "openrouter", "ollama"})
REASONING_EFFORT_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")


class ProviderRequestError(RuntimeError):
    """Fail-closed provider failure with bounded-retry diagnostics."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool,
        attempts: int,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.attempts = attempts


class ProviderResponseError(RuntimeError):
    """Raised when a provider returns a structurally unusable response."""


def is_retryable_provider_error(exc: BaseException) -> bool:
    """Classify transient transport failures without retrying caller errors."""
    if isinstance(
        exc,
        (
            TimeoutError,
            ConnectionError,
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.RateLimitError,
            openai.InternalServerError,
        ),
    ):
        return True

    if isinstance(exc, openai.APIStatusError):
        status_code = int(getattr(exc, "status_code", 0) or 0)
        return status_code in {408, 409, 425, 429} or status_code >= 500

    return False


def normalize_provider_name(provider: str) -> str:
    normalized = provider.strip().lower()
    if normalized not in SUPPORTED_PROVIDERS:
        supported = ", ".join(sorted(SUPPORTED_PROVIDERS))
        raise ValueError(
            f"Unsupported provider '{provider}'. Supported providers: {supported}."
        )
    return normalized


def resolve_provider_runtime_config(provider: str) -> tuple[str | None, str | None]:
    normalized = normalize_provider_name(provider)

    if normalized == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL") or None
        if not api_key:
            raise ValueError(
                "Provider 'openai' requires OPENAI_API_KEY in the environment or .env file."
            )
        return api_key, base_url

    if normalized == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        base_url = os.getenv("OPENROUTER_BASE_URL")
        if not api_key:
            raise ValueError(
                "Provider 'openrouter' requires OPENROUTER_API_KEY in the environment or .env file."
            )
        if not base_url:
            raise ValueError(
                "Provider 'openrouter' requires OPENROUTER_BASE_URL in the environment or .env file."
            )
        return api_key, base_url

    base_url = os.getenv("OLLAMA_BASE_URL")
    if not base_url:
        raise ValueError(
            "Provider 'ollama' requires OLLAMA_BASE_URL in the environment or .env file."
        )
    return "ollama", base_url


def model_supports_reasoning_effort(model: str) -> bool:
    normalized = model.strip().lower()
    if not normalized:
        return False
    normalized = normalized.split("/", 1)[-1]
    return normalized.startswith(REASONING_EFFORT_MODEL_PREFIXES)


class OpenAIClient:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = 5,
        timeout: float = 120.0,
        rpm: int = 200,
        sleeper: Callable[[float], None] = time.sleep,
        retry_callback: Callable[[BaseException, int], None] | None = None,
    ):
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative.")
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self.rpm = rpm
        self._sleeper = sleeper
        self._retry_callback = retry_callback
        self._min_request_interval_seconds = 60.0 / rpm if rpm > 0 else 0.0
        self._last_request_ts = 0.0
        self._reasoning_effort_unsupported_logged = False
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url,
            timeout=openai.Timeout(timeout, connect=10.0),
            max_retries=0,
        )

    def _token_limit_kwargs(self, max_tokens: int) -> dict[str, Any]:
        model_name = self.model.lower()
        if model_name.startswith(("gpt-5", "o1", "o3", "o4")):
            return {"max_completion_tokens": max_tokens}
        return {"max_tokens": max_tokens}

    def _temperature_kwargs(self, temperature: float) -> dict[str, Any]:
        model_name = self.model.lower()
        if model_name.startswith(("gpt-5", "o1", "o3", "o4")):
            return {}
        return {"temperature": temperature}

    @staticmethod
    def _retry_sleep_seconds(attempt: int) -> float:
        base_seconds = min(2**attempt, 16)
        jitter_seconds = random.uniform(0.0, 0.5)
        return base_seconds + jitter_seconds

    def _wait_for_rate_limit_slot(self) -> None:
        if self._min_request_interval_seconds <= 0:
            return

        now = time.monotonic()
        elapsed = now - self._last_request_ts
        sleep_seconds = self._min_request_interval_seconds - elapsed
        if sleep_seconds > 0:
            self._sleeper(sleep_seconds)
        self._last_request_ts = time.monotonic()

    def _reasoning_effort_kwargs(self, reasoning_effort: str | None) -> dict[str, Any]:
        if not reasoning_effort:
            return {}
        if not model_supports_reasoning_effort(self.model):
            if not self._reasoning_effort_unsupported_logged:
                logger.info(
                    "Skipping reasoning_effort for unsupported chat.completions model=%s",
                    self.model,
                )
                self._reasoning_effort_unsupported_logged = True
            return {}
        return {"reasoning_effort": reasoning_effort}

    @staticmethod
    def empty_usage() -> TokenUsage:
        return TokenUsage(
            prompt_tokens_total=0,
            completion_tokens=0,
            total_tokens=0,
        )

    @classmethod
    def extract_usage(cls, response: Any) -> TokenUsage:
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0) if usage else 0

        if total_tokens == 0:
            total_tokens = prompt_tokens + completion_tokens

        return TokenUsage(
            prompt_tokens_total=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        response = self.generate_with_usage(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.response

    def generate_with_usage(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        include_raw: bool = False,
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        total_attempts = self.max_retries + 1
        for attempt in range(total_attempts):
            try:
                self._wait_for_rate_limit_slot()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **self._temperature_kwargs(temperature),
                    **self._token_limit_kwargs(max_tokens),
                    **self._reasoning_effort_kwargs(reasoning_effort),
                )

                content = response.choices[0].message.content
                finish_reason = response.choices[0].finish_reason

                if content is None:
                    raise ProviderResponseError(
                        "Provider returned no text content "
                        f"(finish_reason={finish_reason!r})."
                    )

                return LLMResponse(
                    response=content.strip(),
                    token_usage=self.extract_usage(response),
                    model=response.model,
                    finish_reason=finish_reason,
                    raw=response.model_dump() if include_raw else None,
                )
            except Exception as exc:
                retryable = is_retryable_provider_error(exc)
                attempts = attempt + 1
                logger.warning(
                    "Provider generation attempt %d/%d failed "
                    "(retryable=%s, error_type=%s)",
                    attempts,
                    total_attempts,
                    retryable,
                    type(exc).__name__,
                )
                if not retryable or attempts >= total_attempts:
                    raise ProviderRequestError(
                        "Provider generation failed "
                        f"after {attempts} attempt(s): {type(exc).__name__}.",
                        retryable=retryable,
                        attempts=attempts,
                    ) from exc
                if self._retry_callback is not None:
                    self._retry_callback(exc, attempts)
                self._sleeper(self._retry_sleep_seconds(attempt))

        raise AssertionError("Provider retry loop exited unexpectedly.")

    def call_judge(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
    ) -> str:
        return self.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
            max_tokens=max_tokens,
        )

    def call_judge_with_usage(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
        include_raw: bool = False,
    ) -> LLMResponse:
        return self.generate_with_usage(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
            max_tokens=max_tokens,
            include_raw=include_raw,
        )
