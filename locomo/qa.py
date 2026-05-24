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

"""QA phase for one LOCOMO conversation already ingested into DMF or Mem0."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from dmf.analysis.embedding_engine import EmbeddingEngine
from dmf.memory.api import Memory
from dmf.utils.config import VectorConfig
from dmf.utils.config_loader import DMFConfig

from common.mem0_local import LocalMem0ConversationBackend
from common.models import IngestedConversationBundle, QASettings, TokenUsage
from common.openai_client import (
    OpenAIClient,
    normalize_provider_name,
    resolve_provider_runtime_config,
)
from . import utils
from .prompts import (
    ANSWERER_SYSTEM_PROMPT,
    build_answerer_user_prompt,
    format_strict_dialogs_as_context,
    normalize_category_5_prediction,
    official_ground_truth_answer,
)

logger = logging.getLogger(__name__)

_METADATA_KEYS = (
    "benchmark",
    "conversation_idx",
    "source_unit_type",
    "source_unit_id",
    "source_unit_ids",
)


def filter_qa_items(
    qa_items: list[dict[str, Any]],
    allowed_categories: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Return only QA items whose category is in the allowed set."""
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
    """Return filtered QA items together with their original dataset index."""
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
    temperature = float(os.getenv("DMF_BENCHMARK_ANSWERER_TEMPERATURE", "0.0"))
    max_tokens = int(os.getenv("DMF_BENCHMARK_ANSWERER_MAX_TOKENS", "4096"))
    rpm = int(os.getenv("DMF_BENCHMARK_ANSWERER_RPM", "200"))
    timeout = float(os.getenv("DMF_BENCHMARK_ANSWERER_TIMEOUT", "120.0"))
    _, base_url = resolve_provider_runtime_config(provider)

    return QASettings(
        provider=provider,
        model=model,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        rpm=rpm,
        timeout=timeout,
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


def build_memory_api(
    bundle: IngestedConversationBundle,
    config: DMFConfig,
) -> Memory:
    embedding_engine = build_embedding_engine(config)
    return Memory.from_dmf_config(
        config,
        bundle.memory_engine,
        embedding_engine,
    )


def build_embedding_engine(config: DMFConfig) -> EmbeddingEngine:
    return EmbeddingEngine(
        VectorConfig(
            model_name=config.nlp.model_name,
            vector_dim=config.nlp.vector_dim,
            window_size=config.capacity.window_size,
        )
    )


def build_answerer(settings: QASettings) -> OpenAIClient:
    api_key, base_url = resolve_provider_runtime_config(settings.provider)
    return OpenAIClient(
        model=settings.model,
        api_key=api_key,
        base_url=base_url,
        timeout=settings.timeout,
        rpm=settings.rpm,
    )


def normalize_answerer_usage(token_usage: TokenUsage) -> dict[str, int]:
    return {
        "prompt_tokens_total": int(token_usage.prompt_tokens_total),
        "completion_tokens": int(token_usage.completion_tokens),
        "total_tokens": int(token_usage.total_tokens),
    }


def category_name(category: int) -> str:
    mapping = {
        1: "multi-hop",
        2: "temporal",
        3: "open-domain",
        4: "single-hop",
        5: "adversarial",
    }
    return mapping.get(category, f"category-{category}")


def normalize_generated_answer_for_category(
    *,
    category: int,
    generated_answer: str,
    ground_truth_answer: str,
) -> str:
    """Apply LoCoMo category-specific answer normalization."""
    if category == 5:
        return normalize_category_5_prediction(
            generated_answer,
            ground_truth_answer,
        )
    return generated_answer



def _normalize_created_at(value: Any) -> str | None:
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        return raw

    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()

    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()

    return None


def _minimal_search_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    minimal: dict[str, Any] = {}
    for key in _METADATA_KEYS:
        value = metadata.get(key)
        if value is None:
            continue
        if key == "source_unit_ids" and isinstance(value, list):
            filtered = [item for item in value if item is not None]
            if filtered:
                minimal[key] = filtered
            continue
        minimal[key] = value
    return minimal


def _candidate_score(candidate: dict[str, Any]) -> float:
    score = candidate.get("recall_score")
    if score is None:
        score = candidate.get("similarity_score")
    if score is None:
        return 0.0
    return float(score)


def build_search_results(
    candidates: list[dict[str, Any]],
    bundle: IngestedConversationBundle,
) -> list[dict[str, Any]]:
    search_results: list[dict[str, Any]] = []

    ordered_candidates = sorted(
        candidates,
        key=_candidate_score,
        reverse=True,
    )
    for candidate in ordered_candidates:
        record = candidate.get("record", {})
        record_id = str(record.get("record_id", ""))
        metadata = bundle.record_index.get(record_id, {})

        result: dict[str, Any] = {
            "memory": str(metadata.get("text", "") or record.get("text", "")),
            "score": _candidate_score(candidate),
            "id": record_id,
        }
        created_at = _normalize_created_at(record.get("created_at"))
        if created_at:
            result["created_at"] = created_at
        minimal_metadata = _minimal_search_metadata(metadata)
        if minimal_metadata:
            result["metadata"] = minimal_metadata
        search_results.append(result)

    return search_results


def _extract_source_dia_ids(search_result: dict[str, Any]) -> list[str]:
    metadata = search_result.get("metadata")
    if not isinstance(metadata, dict):
        return []

    dia_ids: list[str] = []
    single_id = metadata.get("source_unit_id")
    if single_id is not None:
        normalized = str(single_id).strip()
        if normalized:
            dia_ids.append(normalized)

    multiple_ids = metadata.get("source_unit_ids")
    if isinstance(multiple_ids, list):
        for raw_id in multiple_ids:
            normalized = str(raw_id).strip()
            if normalized:
                dia_ids.append(normalized)

    seen: set[str] = set()
    unique_dia_ids: list[str] = []
    for dia_id in dia_ids:
        if dia_id in seen:
            continue
        seen.add(dia_id)
        unique_dia_ids.append(dia_id)
    return unique_dia_ids


def select_strict_dia_ids(search_results: list[dict[str, Any]]) -> list[str]:
    """Select unique final `dia_id` values from framework hits in retrieval order."""
    seen: set[str] = set()
    selected_dia_ids: list[str] = []
    for search_result in search_results:
        for dia_id in _extract_source_dia_ids(search_result):
            if dia_id in seen:
                continue
            seen.add(dia_id)
            selected_dia_ids.append(dia_id)
    return selected_dia_ids


def build_strict_dialog_context(
    conversation: dict[str, Any],
    search_results: list[dict[str, Any]],
) -> str:
    """Rebuild the strict reader context from original LoCoMo dialogs."""
    conversation_data = conversation.get("conversation")
    if not isinstance(conversation_data, dict):
        return "(No relevant context found)"

    dialog_substrate = utils.build_locomo_dialog_substrate(conversation_data)
    dialog_by_dia_id = utils.map_locomo_dia_ids_to_strict_dialogs(conversation_data)
    selected_dia_ids = select_strict_dia_ids(search_results)

    selected_dialog_ids: set[str] = set()
    unmappable_dia_ids: list[str] = []
    for dia_id in selected_dia_ids:
        dialog = dialog_by_dia_id.get(dia_id)
        if dialog is None:
            unmappable_dia_ids.append(dia_id)
            continue
        selected_dialog_ids.add(dialog["dialog_id"])

    if unmappable_dia_ids:
        logger.warning(
            "LoCoMo strict context skipped unmappable dia_ids: %s",
            ", ".join(unmappable_dia_ids),
        )

    serialized_dialogs = [
        dialog["serialized_dialog"]
        for dialog_id, dialog in dialog_substrate.items()
        if dialog_id in selected_dialog_ids
    ]
    return format_strict_dialogs_as_context(serialized_dialogs)


def build_dmf_strict_retrieval_artifacts(
    *,
    conversation: dict[str, Any],
    bundle: IngestedConversationBundle,
    final_candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    """Return DMF retrieval provenance and the dataset-side strict reader context.

    DMF remains responsible for retrieval and ranking only. The answerer never
    receives DMF's native textual surface directly; its final context is always
    rebuilt from original LoCoMo dialogs selected via `dia_id`.
    """
    search_results = build_search_results(
        candidates=final_candidates,
        bundle=bundle,
    )
    strict_context = build_strict_dialog_context(
        conversation=conversation,
        search_results=search_results,
    )
    return search_results, strict_context


def build_mem0_strict_retrieval_artifacts(
    *,
    conversation: dict[str, Any],
    mem0_backend: LocalMem0ConversationBackend,
    mem0_user_id: str,
    question: str,
    retrieval_depth: int,
) -> tuple[list[dict[str, Any]], str]:
    """Return Mem0 retrieval provenance and the dataset-side strict reader context.

    Mem0 remains responsible for retrieval and usage accounting only. The
    answerer never receives Mem0's native memory surface directly; its final
    context is always rebuilt from original LoCoMo dialogs selected via
    `dia_id` provenance in the Mem0 hits.
    """
    search_results = mem0_backend.search(
        question,
        user_id=mem0_user_id,
        top_k=retrieval_depth,
    )
    strict_context = build_strict_dialog_context(
        conversation=conversation,
        search_results=search_results,
    )
    return search_results, strict_context


def build_recall_diagnostics(
    recall_diagnostics: dict[str, Any],
    bundle: IngestedConversationBundle,
) -> dict[str, Any]:
    ranked_candidates = list(recall_diagnostics.get("ranked_candidates", []))
    final_candidates = list(recall_diagnostics.get("final_candidates", []))

    return {
        "raw_candidates": list(recall_diagnostics.get("raw_candidates", [])),
        "ranked_candidates": ranked_candidates,
        "final_candidates": final_candidates,
        "suppressed": list(recall_diagnostics.get("suppressed", [])),
        "ranked_candidates_canonical": build_search_results(
            ranked_candidates,
            bundle=bundle,
        ),
        "final_candidates_canonical": build_search_results(
            final_candidates,
            bundle=bundle,
        ),
    }


def build_question_result(
    *,
    bundle: IngestedConversationBundle,
    question_idx: int,
    qa_item: dict[str, Any],
    generated_answer: str,
    answerer_provider: str,
    answerer_model: str,
    answerer_usage: dict[str, int],
    question: str,
    context: str,
    search_latency_ms: float,
    retrieval_pipeline_latency_ms: float,
    backend_search_latency_ms: float,
    answer_generation_ms: float,
    search_results: list[dict[str, Any]],
    recall_diagnostics: dict[str, Any],
    context_metrics: dict[str, int],
    cutoff_label_value: str,
) -> dict[str, Any]:
    category = int(qa_item.get("category", 0) or 0)
    question_id = f"conv{bundle.conversation_idx}_q{question_idx}"

    return {
        "question_id": question_id,
        "conversation_idx": bundle.conversation_idx,
        "framework": bundle.framework,
        "question": question,
        "ground_truth_answer": official_ground_truth_answer(
            category,
            str(qa_item.get("answer", "")),
        ),
        "category": category,
        "category_name": category_name(category),
        "evidence": list(qa_item.get("evidence", [])),
        "cutoff_label": cutoff_label_value,
        "answerer_provider": answerer_provider,
        "answerer_model": answerer_model,
        "generated_answer": generated_answer,
        "answerer_usage": answerer_usage,
        "retrieval": {
            "search_query": question,
            "context": context,
            "search_latency_ms": round(search_latency_ms, 1),
            "retrieval_pipeline_latency_ms": round(retrieval_pipeline_latency_ms, 1),
            "backend_search_latency_ms": round(backend_search_latency_ms, 1),
            "search_results": search_results,
            "total_results": len(search_results),
            "memories_evaluated": len(search_results),
            "recall_diagnostics": recall_diagnostics,
            "context_metrics": context_metrics,
        },
        "pipeline_timing": {
            "retrieval_pipeline_ms": retrieval_pipeline_latency_ms,
            "retrieval_pipeline_scope": "question",
            "backend_search_ms": backend_search_latency_ms,
            "backend_search_scope": "question",
            "answer_generation_ms": answer_generation_ms,
            "answer_generation_scope": "question",
        },
    }


def build_mem0_backend(bundle: IngestedConversationBundle) -> LocalMem0ConversationBackend:
    if bundle.framework != "mem0":
        raise RuntimeError("Mem0 backend requested for a non-Mem0 conversation bundle.")

    backend = bundle.backend_state.get("mem0_backend")
    if not isinstance(backend, LocalMem0ConversationBackend):
        raise TypeError(
            "Invalid Mem0 backend_state: missing LocalMem0ConversationBackend instance."
        )
    return backend


def build_mem0_user_id(bundle: IngestedConversationBundle) -> str:
    if bundle.framework != "mem0":
        raise RuntimeError("Mem0 user_id requested for a non-Mem0 conversation bundle.")

    user_id = bundle.backend_state.get("user_id")
    if not isinstance(user_id, str) or not user_id:
        raise TypeError("Invalid Mem0 backend_state: missing user_id.")
    return user_id


def run_qa_for_conversation(
    conversation: dict[str, Any],
    bundle: IngestedConversationBundle,
    config: DMFConfig | None,
    settings: QASettings,
    retrieval_depth: int,
    allowed_categories: list[int] | None = None,
    on_question_completed: Callable[[], None] | None = None,
) -> list[dict[str, Any]]:
    answerer = build_answerer(settings)
    embedding_engine = build_embedding_engine(config) if bundle.framework == "dmf" and config is not None else None
    mem0_backend = build_mem0_backend(bundle) if bundle.framework == "mem0" else None
    mem0_user_id = build_mem0_user_id(bundle) if bundle.framework == "mem0" else None

    results: list[dict[str, Any]] = []
    qa_items = enumerate_filtered_qa_items(
        conversation.get("qa", []),
        allowed_categories=allowed_categories,
    )
    cutoff_label_value = f"top_{retrieval_depth}"

    for question_idx, qa_item in qa_items:
        question = str(qa_item.get("question", ""))
        category = int(qa_item.get("category", 0) or 0)
        ground_truth_answer = str(qa_item.get("answer", ""))
        recall_diagnostics: dict[str, Any]
        context_metrics: dict[str, int]
        search_results: list[dict[str, Any]]

        if bundle.framework == "dmf":
            if embedding_engine is None:
                raise RuntimeError("DMF embedding engine was not initialized.")
            pipeline_start = time.monotonic()
            query_vector = embedding_engine.get_embedding(question)
            backend_search_start = time.monotonic()
            raw_hits = bundle.memory_engine.get_raw_recall_hits(
                query_vector,
                k=retrieval_depth,
            )
            backend_search_latency_ms = (time.monotonic() - backend_search_start) * 1000
            bundle.memory_engine.rerank_contextualized_recall_candidates(
                bundle.memory_engine.contextualize_raw_recall_hits(raw_hits)
            )
            raw_recall_diagnostics = bundle.memory_engine.get_recall_diagnostics()
            final_candidates = list(raw_recall_diagnostics.get("final_candidates", []))
            retrieval_pipeline_latency_ms = (time.monotonic() - pipeline_start) * 1000
            search_results, context = build_dmf_strict_retrieval_artifacts(
                conversation=conversation,
                bundle=bundle,
                final_candidates=final_candidates,
            )
            context_metrics = bundle.memory_engine.get_context_metrics()
            recall_diagnostics = build_recall_diagnostics(
                recall_diagnostics=raw_recall_diagnostics,
                bundle=bundle,
            )
        elif bundle.framework == "mem0":
            if mem0_backend is None or mem0_user_id is None:
                raise RuntimeError("Mem0 QA path is missing the local Mem0 backend or user_id.")
            pipeline_start = time.monotonic()
            backend_search_start = time.monotonic()
            search_results, context = build_mem0_strict_retrieval_artifacts(
                conversation=conversation,
                mem0_backend=mem0_backend,
                mem0_user_id=mem0_user_id,
                question=question,
                retrieval_depth=retrieval_depth,
            )
            backend_search_latency_ms = (time.monotonic() - backend_search_start) * 1000
            retrieval_pipeline_latency_ms = (time.monotonic() - pipeline_start) * 1000
            recall_diagnostics = {}
            context_metrics = {}
        else:
            raise ValueError(f"Unsupported framework: {bundle.framework}")

        user_prompt = build_answerer_user_prompt(
            context=context,
            question=question,
            category=category,
            ground_truth_answer=ground_truth_answer,
        )
        answer_generation_start = time.monotonic()
        response = answerer.generate_with_usage(
            system_prompt=ANSWERER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )
        answer_generation_ms = (time.monotonic() - answer_generation_start) * 1000
        generated_answer = normalize_generated_answer_for_category(
            category=category,
            generated_answer=response.response,
            ground_truth_answer=ground_truth_answer,
        )
        result = build_question_result(
            bundle=bundle,
            question_idx=question_idx,
            qa_item=qa_item,
            generated_answer=generated_answer,
            answerer_provider=settings.provider,
            answerer_model=response.model,
            answerer_usage=normalize_answerer_usage(response.token_usage),
            question=question,
            context=context,
            search_latency_ms=retrieval_pipeline_latency_ms,
            retrieval_pipeline_latency_ms=retrieval_pipeline_latency_ms,
            backend_search_latency_ms=backend_search_latency_ms,
            answer_generation_ms=answer_generation_ms,
            search_results=search_results,
            recall_diagnostics=recall_diagnostics,
            context_metrics=context_metrics,
            cutoff_label_value=cutoff_label_value,
        )
        results.append(result)
        if on_question_completed is not None:
            on_question_completed()

    return results
