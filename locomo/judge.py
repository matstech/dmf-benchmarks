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
LLM-as-judge phase for already generated QA results.

This module updates per-question results with:
- judgment
- score
- reason
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable
from typing import Any

from common.judge_prompts import JUDGE_SYSTEM_PROMPT, build_judge_user_prompt
from common.models import JudgeSettings
from common.native_reporting import apply_native_primary_judge_score
from common.openai_client import (
    OpenAIClient,
    model_supports_reasoning_effort,
    normalize_provider_name,
    resolve_provider_runtime_config,
)

logger = logging.getLogger(__name__)


def resolve_judge_settings(
    provider_override: str | None = None,
    model_override: str | None = None,
    reasoning_effort: str | None = None,
) -> JudgeSettings:
    provider = normalize_provider_name(provider_override or "openai")
    model = model_override or os.getenv("DMF_BENCHMARK_JUDGE_MODEL", "gpt-4o-mini")
    if not model.strip():
        raise ValueError("Judge model cannot be empty.")
    max_tokens = int(os.getenv("DMF_BENCHMARK_JUDGE_MAX_TOKENS", "4096"))
    rpm = int(os.getenv("DMF_BENCHMARK_JUDGE_RPM", "200"))
    timeout = float(os.getenv("DMF_BENCHMARK_JUDGE_TIMEOUT", "120.0"))
    _, base_url = resolve_provider_runtime_config(provider)
    resolved_reasoning_effort = reasoning_effort
    if resolved_reasoning_effort is None and model_supports_reasoning_effort(model):
        resolved_reasoning_effort = "low"
    return JudgeSettings(
        provider=provider,
        model=model,
        base_url=base_url,
        max_tokens=max_tokens,
        rpm=rpm,
        timeout=timeout,
        reasoning_effort=resolved_reasoning_effort,
    )


def log_judge_settings(settings: JudgeSettings) -> None:
    logger.info(
        "Judge config | provider=%s model=%s base_url=%s max_tokens=%s rpm=%s timeout=%s reasoning_effort=%s",
        settings.provider,
        settings.model,
        settings.base_url or "<default>",
        settings.max_tokens,
        settings.rpm,
        settings.timeout,
        settings.reasoning_effort,
    )


def build_judge(settings: JudgeSettings) -> OpenAIClient:
    api_key, base_url = resolve_provider_runtime_config(settings.provider)
    return OpenAIClient(
        model=settings.model,
        api_key=api_key,
        base_url=base_url,
        timeout=settings.timeout,
        rpm=settings.rpm,
    )


def parse_judge_response(text: str) -> tuple[str, float, str]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict):
        raw_label = str(payload.get("label", "")).strip().upper()
        verdict = raw_label if raw_label in {"CORRECT", "WRONG"} else "WRONG"
        reasoning = str(payload.get("reasoning", "")).strip()
        reason = reasoning if reasoning else text.strip()
        score = 1.0 if verdict == "CORRECT" else 0.0
        return verdict, score, reason

    verdict_match = re.search(r"VERDICT:\s*(CORRECT|WRONG)", text, flags=re.IGNORECASE)
    reason_match = re.search(r"REASON:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)

    verdict = verdict_match.group(1).upper() if verdict_match else "WRONG"
    reason = reason_match.group(1).strip() if reason_match else text.strip()
    score = 1.0 if verdict == "CORRECT" else 0.0
    return verdict, score, reason


def judge_one_result(
    result: dict[str, Any],
    judge_client: OpenAIClient,
    settings: JudgeSettings,
) -> dict[str, Any]:
    question = str(result.get("question", ""))
    ground_truth_answer = str(result.get("ground_truth_answer", ""))

    if "generated_answer" in result or "prediction" in result:
        generated_answer = str(result.get("generated_answer", result.get("prediction", "")))
        judge_prompt = build_judge_user_prompt(
            question=question,
            ground_truth_answer=ground_truth_answer,
            generated_answer=generated_answer,
        )
        judge_response = judge_client.generate_with_usage(
            system_prompt=JUDGE_SYSTEM_PROMPT,
            user_prompt=judge_prompt,
            temperature=0.0,
            max_tokens=settings.max_tokens,
            reasoning_effort=settings.reasoning_effort,
        )
        judgment, score, reason = parse_judge_response(judge_response.response)
        return apply_native_primary_judge_score(
            result,
            judgment=judgment,
            score=score,
            reason=reason,
            judge_provider=settings.provider,
            judge_model=judge_response.model,
        )

    cutoff_results = result.get("cutoff_results")
    if not isinstance(cutoff_results, dict):
        return result

    for cutoff in cutoff_results.values():
        if not isinstance(cutoff, dict):
            continue

        nested_generated_answer = str(cutoff.get("generated_answer", ""))
        judge_prompt = build_judge_user_prompt(
            question=question,
            ground_truth_answer=ground_truth_answer,
            generated_answer=nested_generated_answer,
        )
        judge_response = judge_client.generate_with_usage(
            system_prompt=JUDGE_SYSTEM_PROMPT,
            user_prompt=judge_prompt,
            temperature=0.0,
            max_tokens=settings.max_tokens,
            reasoning_effort=settings.reasoning_effort,
        )
        judgment, score, reason = parse_judge_response(judge_response.response)
        cutoff["judgment"] = judgment
        cutoff["score"] = score
        cutoff["reason"] = reason
        cutoff["judge_provider"] = settings.provider
        cutoff["judge_model"] = judge_response.model

    return result


def run_judge_for_conversation(
    results: list[dict[str, Any]],
    settings: JudgeSettings,
    on_question_completed: Callable[[], None] | None = None,
) -> list[dict[str, Any]]:
    judge_client = build_judge(settings)

    judged_results: list[dict[str, Any]] = []
    for result in results:
        judged_results.append(
            judge_one_result(
                result=result,
                judge_client=judge_client,
                settings=settings,
            )
        )
        if on_question_completed is not None:
            on_question_completed()

    return judged_results
