"""Shared LoCoMo question selection and answer normalization helpers."""

from __future__ import annotations

import logging
import os
from typing import Any

from common.models import QASettings
from common.openai_client import (
    normalize_provider_name,
    resolve_provider_runtime_config,
)
from locomo.prompts import normalize_category_5_prediction


logger = logging.getLogger(__name__)


def filter_qa_items(
    qa_items: list[dict[str, Any]],
    allowed_categories: list[int] | None = None,
) -> list[dict[str, Any]]:
    if not allowed_categories:
        return list(qa_items)
    allowed = set(allowed_categories)
    return [
        qa_item
        for qa_item in qa_items
        if int(qa_item.get("category", 0) or 0) in allowed
    ]


def enumerate_filtered_qa_items(
    qa_items: list[dict[str, Any]],
    allowed_categories: list[int] | None = None,
) -> list[tuple[int, dict[str, Any]]]:
    if not allowed_categories:
        return list(enumerate(qa_items))
    allowed = set(allowed_categories)
    return [
        (question_idx, qa_item)
        for question_idx, qa_item in enumerate(qa_items)
        if int(qa_item.get("category", 0) or 0) in allowed
    ]


def resolve_qa_settings(
    provider_override: str | None = None,
    model_override: str | None = None,
) -> QASettings:
    provider = normalize_provider_name(provider_override or "openai")
    model = model_override or os.getenv("DMF_BENCHMARK_ANSWERER_MODEL", "gpt-4o-mini")
    if not model.strip():
        raise ValueError("Answerer model cannot be empty.")
    _, base_url = resolve_provider_runtime_config(provider)
    return QASettings(
        provider=provider,
        model=model,
        base_url=base_url,
        temperature=float(os.getenv("DMF_BENCHMARK_ANSWERER_TEMPERATURE", "0.0")),
        max_tokens=int(os.getenv("DMF_BENCHMARK_ANSWERER_MAX_TOKENS", "4096")),
        rpm=int(os.getenv("DMF_BENCHMARK_ANSWERER_RPM", "200")),
        timeout=float(os.getenv("DMF_BENCHMARK_ANSWERER_TIMEOUT", "120.0")),
    )


def log_qa_settings(settings: QASettings) -> None:
    logger.info(
        "QA config | provider=%s model=%s base_url=%s temperature=%s max_tokens=%s rpm=%s timeout=%s",
        settings.provider,
        settings.model,
        settings.base_url or "<default>",
        settings.temperature,
        settings.max_tokens,
        settings.rpm,
        settings.timeout,
    )


def category_name(category: int) -> str:
    return {
        1: "multi-hop",
        2: "temporal",
        3: "open-domain",
        4: "single-hop",
        5: "adversarial",
    }.get(category, f"category-{category}")


def normalize_generated_answer_for_category(
    *,
    category: int,
    generated_answer: str,
    ground_truth_answer: str,
) -> str:
    if category == 5:
        return normalize_category_5_prediction(
            generated_answer,
            ground_truth_answer,
        )
    return generated_answer
