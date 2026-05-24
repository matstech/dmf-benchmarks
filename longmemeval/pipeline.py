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

"""LongMemEval benchmark pipeline.

Orchestrates per-question memory isolation, haystack ingestion,
retrieval, answer generation, and result persistence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")

from dotenv import load_dotenv

from dmf.analysis.embedding_engine import EmbeddingEngine
from dmf.analysis.nlp_engine import NLPEngine
from dmf.analysis.scoring_engine import ScoringEngine
from dmf.memory.api import Memory
from dmf.memory.chroma_ltm import ChromaLTMHook
from dmf.memory.temporal_memory import TemporalMemory
from dmf.runtime.pipeline import InteractionPipeline, InteractionProvenance
from dmf.utils.config import NLPConfig, VectorConfig
from dmf.utils.config_loader import DMFConfig, load_dmf_config

from common.mem0_config import Mem0Config, load_mem0_config
from common.mem0_local import (
    LocalMem0BenchmarkItemBackend,
    add_memory_internal_usage,
    empty_memory_internal_usage,
)
from common.models import (
    IngestedQuestionBundle,
    JudgeSettings,
    MemoryFramework,
    QASettings,
    TokenUsage,
)
from common.openai_client import OpenAIClient, resolve_provider_runtime_config
from common import results_io
from longmemeval import judge
from longmemeval.prompts import (
    build_answerer_system_prompt,
    build_answerer_user_prompt,
    format_strict_sessions_as_history_chats,
)
from longmemeval.utils import (
    build_longmemeval_strict_session_substrate,
    dedupe_longmemeval_strict_sessions_by_session_id,
    ensure_longmemeval_dataset,
    filter_questions_by_ids,
    get_default_dataset_path,
    LongMemEvalStrictSession,
    normalize_longmemeval_haystack,
    parse_longmemeval_date,
    parse_longmemeval_date_human,
    pair_turns,
    render_longmemeval_pair_for_context,
    sample_questions_stratified,
    serialize_longmemeval_pair_for_mem0,
    sort_longmemeval_strict_sessions_chronologically,
    sort_sessions_chronologically,
)

log = logging.getLogger("longmemeval")
logging.getLogger("chromadb.telemetry.product.posthog").disabled = True
logging.getLogger("chromadb.segment.impl.vector.local_persistent_hnsw").setLevel(logging.ERROR)
logging.getLogger("chromadb.segment.impl.vector.local_hnsw").setLevel(logging.ERROR)
logging.getLogger("posthog").disabled = True
logging.getLogger("mem0").setLevel(logging.WARNING)


def _safe_question_key(question_id: str) -> str:
    """Return a filesystem- and collection-safe identifier for one question."""
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", question_id).strip("_")
    return normalized or "question"


def _stable_question_namespace(question_id: str) -> str:
    """Return a namespace that remains unique even when truncated."""
    safe_key = _safe_question_key(question_id).lower()
    digest = hashlib.sha1(question_id.encode("utf-8")).hexdigest()[:10]
    return f"{safe_key[:24]}_{digest}"


def _namespaced_cards_collection_name(base_name: str, question_id: str) -> str:
    """Return the per-question cards collection name for LongMemEval."""
    question_namespace = _stable_question_namespace(question_id)
    collection_name = f"{base_name}_longmemeval"
    suffix = f"_{question_namespace}"
    if len(collection_name) + len(suffix) <= 63:
        return f"{collection_name}{suffix}"
    return f"{collection_name[: 63 - len(suffix)]}{suffix}"


def _namespaced_cards_path(
    base_path: str | Path | None,
    *,
    question_id: str,
    persist_directory: str | Path,
) -> Path:
    """Return the per-question cards audit path for LongMemEval."""
    path = Path(base_path) if base_path is not None else Path(persist_directory) / "ltm_cards.jsonl"
    suffix = path.suffix or ".jsonl"
    question_namespace = _stable_question_namespace(question_id)
    return path.with_name(f"{path.stem}_longmemeval_{question_namespace}{suffix}")

def _banner_print(msg: str) -> None:
    """Print a rich-formatted banner message (lazy Console import)."""
    try:
        from rich.console import Console
        Console().print(msg)
    except ImportError:
        print(msg)


def _validate_startup_secrets(args: argparse.Namespace) -> None:
    """Fail fast on missing provider runtime config required by the selected run mode."""
    if _prediction_enabled(args):
        resolve_provider_runtime_config(getattr(args, "answerer_provider", None) or "openai")

    if _judging_enabled(args):
        resolve_provider_runtime_config(getattr(args, "judge_provider", None) or "openai")


def _print_startup_configuration(
    *,
    args: argparse.Namespace,
    config: "ActiveBenchmarkConfig",
    dataset_path: str | None,
    prediction_artifacts_path: Path,
    selected_type_counts: dict[str, int],
    answerer_settings: QASettings | None,
    judge_settings: JudgeSettings | None,
) -> None:
    """Print the main runtime configuration resolved for this benchmark run."""
    mode = _run_mode(args)
    _banner_print("[bold]LongMemEval startup configuration[/bold]")
    _banner_print(f"  Run mode: [bold]{mode}[/bold]")
    _banner_print(f"  Project: [bold]{args.project_name}[/bold]")
    _banner_print(f"  Framework: [bold]{args.framework}[/bold]")
    _banner_print(f"  Framework config: {Path(args.config).resolve()}")
    if dataset_path is not None:
        _banner_print(f"  Dataset file: {Path(dataset_path).resolve()}")
    _banner_print(f"  Prediction artifacts: {prediction_artifacts_path.resolve()}")
    _banner_print(f"  Retrieval depth: [bold]{config.retrieval_depth}[/bold]")
    if answerer_settings is not None:
        _banner_print(
            "  Answerer: "
            f"[bold]{answerer_settings.model}[/bold] ({answerer_settings.provider})"
        )
    else:
        _banner_print("  Answerer: skipped in this run mode")
    if judge_settings is not None:
        _banner_print(
            "  Judge: "
            f"[bold]{judge_settings.model}[/bold] ({judge_settings.provider})"
        )
    else:
        _banner_print("  Judge: skipped in this run mode")
    _banner_print(
        "  Offline evaluator: "
        + ("enabled" if _evaluation_enabled(args) else "skipped")
    )
    _banner_print(
        "  Selected question types: "
        + ", ".join(f"{qtype}={count}" for qtype, count in sorted(selected_type_counts.items()))
    )


@dataclass(frozen=True)
class ActiveBenchmarkConfig:
    """Framework-aware config wrapper for LongMemEval Phase 1."""

    framework: MemoryFramework
    dmf_config: DMFConfig | None = None
    mem0_config: Mem0Config | None = None

    @property
    def retrieval_depth(self) -> int:
        """Return the retrieval operating point from the active framework config."""
        if self.framework == "dmf":
            if self.dmf_config is None:
                raise RuntimeError("DMF config not loaded.")
            return int(self.dmf_config.ltm.recall_limit)
        if self.framework == "mem0":
            if self.mem0_config is None:
                raise RuntimeError("Mem0 config not loaded.")
            return self.mem0_config.top_k
        raise ValueError(f"Unsupported framework: {self.framework}")


def _validate_framework_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    """Validate CLI arguments whose meaning depends on the selected framework."""
    config_path = Path(args.config)
    suffix = config_path.suffix.lower()

    if args.framework == "dmf" and suffix != ".toml":
        parser.error("--config must point to a DMF TOML file when --framework=dmf.")

    if args.framework == "mem0" and suffix not in {".yaml", ".yml"}:
        parser.error("--config must point to a Mem0 YAML file when --framework=mem0.")


def _validate_mode_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    """Reject invalid runner mode combinations."""
    selected_modes = sum(
        bool(flag)
        for flag in (
            getattr(args, "predict_only", False),
            getattr(args, "evaluate_only", False),
            getattr(args, "judge_only", False),
        )
    )
    if selected_modes > 1:
        parser.error(
            "--predict-only, --evaluate-only, and --judge-only are mutually exclusive."
        )


def _load_active_config(args: argparse.Namespace) -> ActiveBenchmarkConfig:
    """Load the framework-specific runtime config selected by the CLI."""
    if args.framework == "dmf":
        return ActiveBenchmarkConfig(
            framework="dmf",
            dmf_config=load_dmf_config(path=args.config),
        )

    if args.framework == "mem0":
        return ActiveBenchmarkConfig(
            framework="mem0",
            mem0_config=load_mem0_config(path=args.config),
        )

    raise ValueError(f"Unsupported framework: {args.framework}")


def build_memory_engine_for_question(
    question_id: str,
    config: DMFConfig,
) -> TemporalMemory:
    """Create a fresh, isolated TemporalMemory for one LongMemEval question."""
    nlp_engine = NLPEngine(
        NLPConfig(
            spacy_model=config.nlp.spacy_model,
            analyze_system_prompt=False,
        )
    )

    if not config.ltm.enabled or config.ltm.storage_type != "chroma":
        return TemporalMemory.from_dmf_config(
            config=config,
            nlp_engine=nlp_engine,
        )

    vector_config = VectorConfig(
        model_name=config.nlp.model_name,
        vector_dim=config.nlp.vector_dim,
        window_size=config.capacity.window_size,
    )
    collection_name = _namespaced_cards_collection_name(
        config.ltm.collection_name,
        question_id,
    )
    ltm_hook = ChromaLTMHook(
        collection_name=collection_name,
        persist_directory=config.ltm.chroma_path,
        distance_threshold=config.ltm.distance_threshold,
        vector_config=vector_config,
        cards_enabled=config.ltm.cards_enabled,
        cards_path=_namespaced_cards_path(
            config.ltm.cards_path,
            question_id=question_id,
            persist_directory=config.ltm.chroma_path,
        ),
        cards_collection_name=_namespaced_cards_collection_name(
            config.ltm.cards_collection_name,
            question_id,
        ),
    )
    return TemporalMemory.from_dmf_config(
        config=config,
        ltm_hook=ltm_hook,
        nlp_engine=nlp_engine,
    )


def ingest_question_dmf(
    question: dict,
    config: DMFConfig,
    on_turn_completed: Any | None = None,
) -> IngestedQuestionBundle:
    """Ingest all haystack sessions for one question into a fresh DMF memory."""
    question_id = question["question_id"]

    pipeline = InteractionPipeline.from_dmf_config(config=config)
    scoring = ScoringEngine.from_dmf_config(config=config)
    memory_engine = build_memory_engine_for_question(question_id, config)
    record_index: dict[str, dict[str, Any]] = {}

    sorted_sessions = sort_sessions_chronologically(question)

    for session_id, date_str, session in sorted_sessions:
        session_ts = parse_longmemeval_date(date_str)
        pairs = pair_turns(session)

        for pair in pairs:
            for msg in pair:
                text = msg["content"]
                role = msg["role"]

                if not text.strip():
                    continue

                report, vector = pipeline.analyze_interaction_with_vector(
                    text=text,
                    is_system=False,
                    provenance=InteractionProvenance(role=role),
                )

                report.raw_metadata["benchmark"] = "longmemeval"
                report.raw_metadata["question_id"] = question_id
                report.raw_metadata["source_unit_type"] = "session"
                report.raw_metadata["source_unit_id"] = session_id
                report.raw_metadata["session_date_raw"] = date_str

                scoring.calculate_score(report, text=text)

                entry = memory_engine.add_interaction(text, report, vector)
                if session_ts is not None:
                    entry.timestamp = session_ts

                record_index[entry.record_id] = {
                    "benchmark": "longmemeval",
                    "question_id": question_id,
                    "source_unit_type": "session",
                    "source_unit_id": session_id,
                    "session_date_raw": date_str,
                    "role": role,
                    "text": text,
                }
                if on_turn_completed is not None:
                    on_turn_completed()

    return IngestedQuestionBundle(
        question_id=question_id,
        framework="dmf",
        backend_state={"memory_engine": memory_engine},
        record_index=record_index,
    )


def ingest_question_mem0(
    question: dict,
    *,
    config: Mem0Config,
    project_name: str,
    run_id: str,
    on_turn_completed: Any | None = None,
) -> IngestedQuestionBundle:
    """Ingest all haystack sessions for one question into a fresh Mem0 backend."""
    question_id = str(question["question_id"])
    mem0_backend = LocalMem0BenchmarkItemBackend(
        benchmark_name="longmemeval",
        project_name=project_name,
        item_kind="question",
        item_id=question_id,
        config=config,
    )
    user_id = f"longmemeval_{question_id}_{run_id}"
    record_index: dict[str, dict[str, Any]] = {}

    for session in normalize_longmemeval_haystack(question):
        session_id = session["session_id"]
        session_ts = session["session_timestamp"]
        session_date_raw = session["session_date_raw"]
        metadata = {
            "benchmark": "longmemeval",
            "question_id": question_id,
            "source_unit_type": "session",
            "source_unit_id": session_id,
        }

        for pair in session["pairs"]:
            messages = serialize_longmemeval_pair_for_mem0(pair)
            if not messages:
                if on_turn_completed is not None:
                    on_turn_completed()
                continue

            mem0_backend.add(
                messages,
                user_id=user_id,
                timestamp=session_ts,
                metadata=metadata,
            )
            record_id = f"{session_id}:pair:{pair['pair_index']}"
            record_index[record_id] = {
                "benchmark": "longmemeval",
                "question_id": question_id,
                "source_unit_type": "session",
                "source_unit_id": session_id,
                "source_unit_ids": [session_id],
                "session_id": session_id,
                "session_date_raw": session_date_raw,
                "session_timestamp": session_ts,
                "pair_index": pair["pair_index"],
                "text": render_longmemeval_pair_for_context(pair),
            }
            if on_turn_completed is not None:
                on_turn_completed()

    return IngestedQuestionBundle(
        question_id=question_id,
        framework="mem0",
        backend_state={
            "mem0_backend": mem0_backend,
            "user_id": user_id,
        },
        record_index=record_index,
    )


def ingest_question(
    question: dict,
    config: ActiveBenchmarkConfig,
    *,
    project_name: str | None = None,
    mem0_run_id: str | None = None,
    on_turn_completed: Any | None = None,
) -> IngestedQuestionBundle:
    """Ingest one question into the selected backend and return its bundle."""
    if config.framework == "dmf":
        if config.dmf_config is None:
            raise RuntimeError("DMF config not loaded.")
        return ingest_question_dmf(
            question,
            config.dmf_config,
            on_turn_completed=on_turn_completed,
        )

    if config.framework == "mem0":
        if config.mem0_config is None:
            raise RuntimeError("Mem0 config not loaded.")
        if not project_name:
            raise ValueError("project_name is required for LongMemEval Mem0 ingestion.")
        if not mem0_run_id:
            raise ValueError("mem0_run_id is required for LongMemEval Mem0 ingestion.")
        return ingest_question_mem0(
            question,
            config=config.mem0_config,
            project_name=project_name,
            run_id=mem0_run_id,
            on_turn_completed=on_turn_completed,
        )

    raise ValueError(f"Unsupported framework: {config.framework}")


def build_search_results(
    final_candidates: list[dict[str, Any]],
    bundle: IngestedQuestionBundle,
) -> list[dict[str, Any]]:
    """Shape DMF recall candidates into the evaluator-expected raw provenance format."""
    results: list[dict[str, Any]] = []

    for candidate in final_candidates:
        record = candidate.get("record", {})
        record_id = str(record.get("record_id", ""))
        metadata = dict(bundle.record_index.get(record_id, {}))

        score = candidate.get("recall_score")
        if score is None:
            score = candidate.get("similarity_score")
        if score is None:
            score = 0.0

        session_id = metadata.get("source_unit_id") or metadata.get("session_id")
        if session_id is not None:
            metadata.setdefault("source_unit_ids", [session_id])
            metadata.setdefault("session_id", session_id)

        results.append({
            "memory": str(record.get("text", "")),
            "score": float(score),
            "id": record_id,
            "created_at": _normalize_created_at(record.get("created_at")),
            "metadata": metadata,
        })

    return results


def run_dmf_retrieval_for_strict_reader(
    question_text: str,
    bundle: IngestedQuestionBundle,
    config: ActiveBenchmarkConfig,
) -> tuple[list[dict[str, Any]], float, float]:
    """Run DMF retrieval while keeping strict reader context dataset-side.

    DMF remains responsible for retrieval/ranking only. The returned
    `search_results` preserve the framework-native hit payload for auditing and
    evaluator compatibility, but the strict answerer context must still be
    reconstructed later from `haystack_sessions` via `session_id`.
    """
    if config.dmf_config is None:
        raise RuntimeError("DMF config not loaded for LongMemEval answerer path.")

    embedding_engine = _build_embedding_engine(config.dmf_config)
    pipeline_start = time.monotonic()
    query_vector = embedding_engine.get_embedding(question_text)
    backend_search_start = time.monotonic()
    raw_hits = bundle.memory_engine.get_raw_recall_hits(
        query_vector,
        k=config.retrieval_depth,
    )
    backend_search_latency_ms = (time.monotonic() - backend_search_start) * 1000
    bundle.memory_engine.rerank_contextualized_recall_candidates(
        bundle.memory_engine.contextualize_raw_recall_hits(raw_hits)
    )
    diagnostics = bundle.memory_engine.get_recall_diagnostics()
    final_candidates = list(diagnostics.get("final_candidates", []))
    retrieval_pipeline_latency_ms = (time.monotonic() - pipeline_start) * 1000
    search_results = build_search_results(final_candidates, bundle)
    return search_results, retrieval_pipeline_latency_ms, backend_search_latency_ms


def _dedupe_session_ids_preserving_order(session_ids: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for session_id in session_ids:
        normalized = str(session_id or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _selected_session_ids_from_search_results(
    search_results: list[dict[str, Any]],
) -> list[str]:
    session_ids: list[str] = []
    for item in search_results:
        metadata = item.get("metadata") or {}
        raw_ids = metadata.get("source_unit_ids")
        if isinstance(raw_ids, list):
            session_ids.extend(str(session_id) for session_id in raw_ids if str(session_id).strip())
            continue

        session_id = metadata.get("source_unit_id") or metadata.get("session_id")
        if session_id is not None and str(session_id).strip():
            session_ids.append(str(session_id))

    return _dedupe_session_ids_preserving_order(session_ids)


def build_strict_reader_context(
    question: dict[str, Any],
    search_results: list[dict[str, Any]],
) -> tuple[list[str], str]:
    """Map framework-side hits to dataset-side sessions and render strict history chats.

    The reader never consumes framework-native text directly. It only receives
    original dataset sessions selected through hit provenance (`session_id`).
    """
    strict_substrate = build_longmemeval_strict_session_substrate(question)
    selected_session_ids = _selected_session_ids_from_search_results(search_results)

    selected_sessions: list[LongMemEvalStrictSession] = []
    missing_session_ids: list[str] = []
    for session_id in selected_session_ids:
        strict_session = strict_substrate.get(session_id)
        if strict_session is None:
            missing_session_ids.append(session_id)
            continue
        selected_sessions.append(strict_session)

    if missing_session_ids:
        log.warning(
            "LongMemEval strict context skipped %d unmappable session ids for question %s: %s",
            len(missing_session_ids),
            question.get("question_id"),
            missing_session_ids,
        )

    deduped_selected_sessions = dedupe_longmemeval_strict_sessions_by_session_id(
        selected_sessions
    )
    ordered_sessions = sort_longmemeval_strict_sessions_chronologically(
        deduped_selected_sessions
    )
    mapped_session_ids = [
        str(session["session_id"])
        for session in deduped_selected_sessions
    ]
    strict_context = format_strict_sessions_as_history_chats(ordered_sessions)
    return mapped_session_ids, strict_context


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


def _build_session_metadata_index(
    bundle: IngestedQuestionBundle,
) -> dict[str, dict[str, Any]]:
    session_index: dict[str, dict[str, Any]] = {}
    for metadata in bundle.record_index.values():
        if not isinstance(metadata, dict):
            continue
        session_id = metadata.get("source_unit_id") or metadata.get("session_id")
        if session_id is None:
            continue
        session_index[str(session_id)] = metadata
    return session_index


def normalize_mem0_search_results(
    raw_results: list[dict[str, Any]],
    bundle: IngestedQuestionBundle,
) -> list[dict[str, Any]]:
    """Shape Mem0 search results into canonical raw provenance for strict reconstruction.

    The `memory` field is preserved for audit/debug and evaluator compatibility
    only. Strict reader context must be rebuilt separately from dataset-side
    `haystack_sessions` via `source_unit_id` / `session_id`.
    """
    session_index = _build_session_metadata_index(bundle)
    normalized: list[dict[str, Any]] = []

    for item in raw_results:
        if not isinstance(item, dict):
            continue

        metadata = dict(item.get("metadata") or {})
        session_id = metadata.get("source_unit_id")
        if session_id is not None:
            metadata.setdefault("source_unit_ids", [session_id])

        reference = session_index.get(str(session_id), {}) if session_id is not None else {}
        for key in ("benchmark", "question_id", "source_unit_type", "session_id", "session_date_raw"):
            if key not in metadata and key in reference:
                metadata[key] = reference[key]

        created_at = _normalize_created_at(item.get("created_at"))
        if created_at is None and "session_timestamp" in reference:
            created_at = _normalize_created_at(reference.get("session_timestamp"))

        normalized_item = {
            "memory": str(item.get("memory", "") or ""),
            "score": float(item.get("score", 0.0) or 0.0),
            "id": str(item.get("id", "") or ""),
            "metadata": metadata,
        }
        if created_at:
            normalized_item["created_at"] = created_at
        normalized.append(normalized_item)

    return normalized


def run_mem0_retrieval_for_strict_reader(
    question_text: str,
    bundle: IngestedQuestionBundle,
    config: ActiveBenchmarkConfig,
) -> tuple[list[dict[str, Any]], float, float]:
    """Run Mem0 retrieval while keeping strict reader context dataset-side.

    Mem0 remains responsible for ranking and raw hit provenance only. The final
    answerer context must still be reconstructed later from
    `haystack_sessions` via selected `session_id` values.
    """
    mem0_backend = _build_mem0_backend(bundle)
    mem0_user_id = _build_mem0_user_id(bundle)
    pipeline_start = time.monotonic()
    backend_search_start = time.monotonic()
    raw_results = mem0_backend.search(
        question_text,
        user_id=mem0_user_id,
        top_k=config.retrieval_depth,
    )
    backend_search_latency_ms = (time.monotonic() - backend_search_start) * 1000
    search_results = normalize_mem0_search_results(raw_results, bundle)
    retrieval_pipeline_latency_ms = (time.monotonic() - pipeline_start) * 1000
    return search_results, retrieval_pipeline_latency_ms, backend_search_latency_ms


def _build_memory_api(
    bundle: IngestedQuestionBundle,
    config: DMFConfig,
) -> Memory:
    embedding_engine = _build_embedding_engine(config)
    return Memory.from_dmf_config(
        config,
        bundle.memory_engine,
        embedding_engine,
    )


def _build_embedding_engine(config: DMFConfig) -> EmbeddingEngine:
    return EmbeddingEngine(
        VectorConfig(
            model_name=config.nlp.model_name,
            vector_dim=config.nlp.vector_dim,
            window_size=config.capacity.window_size,
        )
    )


def _build_answerer(settings: QASettings) -> OpenAIClient:
    api_key, base_url = resolve_provider_runtime_config(settings.provider)
    return OpenAIClient(
        model=settings.model,
        api_key=api_key,
        base_url=base_url,
        timeout=settings.timeout,
        rpm=settings.rpm,
    )


def _normalize_usage(usage: TokenUsage) -> dict[str, int]:
    return {
        "prompt_tokens_total": int(usage.prompt_tokens_total),
        "completion_tokens": int(usage.completion_tokens),
        "total_tokens": int(usage.total_tokens),
    }


def _build_mem0_backend(bundle: IngestedQuestionBundle) -> LocalMem0BenchmarkItemBackend:
    if bundle.framework != "mem0":
        raise RuntimeError("Mem0 backend requested for a non-Mem0 question bundle.")

    backend = bundle.backend_state.get("mem0_backend")
    if not isinstance(backend, LocalMem0BenchmarkItemBackend):
        raise TypeError(
            "Invalid Mem0 backend_state: missing LocalMem0BenchmarkItemBackend instance."
        )
    return backend


def _build_mem0_user_id(bundle: IngestedQuestionBundle) -> str:
    if bundle.framework != "mem0":
        raise RuntimeError("Mem0 user_id requested for a non-Mem0 question bundle.")

    user_id = bundle.backend_state.get("user_id")
    if not isinstance(user_id, str) or not user_id:
        raise TypeError("Invalid Mem0 backend_state: missing user_id.")
    return user_id


def run_answerer_for_question(
    question: dict,
    bundle: IngestedQuestionBundle,
    config: ActiveBenchmarkConfig,
    settings: QASettings,
) -> dict[str, Any]:
    """Retrieve context and generate one answer for a LongMemEval question."""
    qa_start = time.monotonic()
    question_text = question["question"]
    question_date = question.get("question_date", "")
    question_date_human = (
        parse_longmemeval_date_human(question_date) if question_date else ""
    )

    answerer = _build_answerer(settings)

    if config.framework == "dmf":
        search_results, retrieval_pipeline_latency_ms, backend_search_latency_ms = (
            run_dmf_retrieval_for_strict_reader(
                question_text,
                bundle,
                config,
            )
        )
    elif config.framework == "mem0":
        search_results, retrieval_pipeline_latency_ms, backend_search_latency_ms = (
            run_mem0_retrieval_for_strict_reader(
                question_text,
                bundle,
                config,
            )
        )
    else:
        raise ValueError(f"Unsupported framework: {config.framework}")

    strict_session_ids, strict_context = build_strict_reader_context(
        question,
        search_results,
    )

    # Answer generation
    system_prompt = build_answerer_system_prompt(question_date_human)
    user_prompt = build_answerer_user_prompt(strict_context, question_text, question_date_human)

    answer_generation_start = time.monotonic()
    response = answerer.generate_with_usage(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )
    answer_generation_ms = (time.monotonic() - answer_generation_start) * 1000
    qa_ms = (time.monotonic() - qa_start) * 1000

    cutoff_label = f"top_{config.retrieval_depth}"

    return {
        "generated_answer": response.response,
        "answerer_usage": _normalize_usage(response.token_usage),
        "answerer_provider": settings.provider,
        "answerer_model": response.model,
        "search_results": search_results,
        "search_latency_ms": round(retrieval_pipeline_latency_ms, 1),
        "retrieval_pipeline_latency_ms": round(retrieval_pipeline_latency_ms, 1),
        "backend_search_latency_ms": round(backend_search_latency_ms, 1),
        "memories_evaluated": len(search_results),
        "cutoff_label": cutoff_label,
        "strict_session_ids": strict_session_ids,
        "strict_context": strict_context,
        "context": strict_context,
        "pipeline_timing": {
            "qa_ms": qa_ms,
            "qa_scope": "question",
            "retrieval_pipeline_ms": retrieval_pipeline_latency_ms,
            "retrieval_pipeline_scope": "question",
            "backend_search_ms": backend_search_latency_ms,
            "backend_search_scope": "question",
            "answer_generation_ms": answer_generation_ms,
            "answer_generation_scope": "question",
        },
    }


def build_question_result(
    question: dict,
    answerer_output: dict[str, Any],
    *,
    framework: MemoryFramework = "dmf",
) -> dict[str, Any]:
    """Assemble one evaluation item in flat mono-cutoff schema."""
    question_id = question["question_id"]
    strict_context = str(
        answerer_output.get("strict_context")
        or answerer_output.get("context", "")
        or results_io.LONGMEMEVAL_NO_HISTORY_CHATS
    )

    result = {
        "question_id": question_id,
        "question_type": question["question_type"],
        "question": question["question"],
        "ground_truth_answer": str(question["answer"]),
        "question_date": question.get("question_date", ""),
        "is_abstention": question_id.endswith("_abs"),
        "answer_session_ids": question.get("answer_session_ids", []),
        "framework": framework,
        "cutoff_label": answerer_output["cutoff_label"],
        "answerer_provider": answerer_output["answerer_provider"],
        "answerer_model": answerer_output["answerer_model"],
        "generated_answer": answerer_output["generated_answer"],
        "answerer_usage": answerer_output["answerer_usage"],
        "retrieval": {
            "search_query": question["question"],
            "search_results": answerer_output["search_results"],
            "search_latency_ms": answerer_output["search_latency_ms"],
            "retrieval_pipeline_latency_ms": answerer_output["retrieval_pipeline_latency_ms"],
            "backend_search_latency_ms": answerer_output["backend_search_latency_ms"],
            "total_results": len(answerer_output["search_results"]),
            "memories_evaluated": answerer_output["memories_evaluated"],
            "strict_session_ids": list(answerer_output.get("strict_session_ids", [])),
            "strict_context": strict_context,
            "context": strict_context,
        },
    }
    pipeline_timing = results_io.normalize_pipeline_timing(
        answerer_output.get("pipeline_timing")
    )
    if pipeline_timing:
        result["pipeline_timing"] = pipeline_timing
    result.update(results_io.longmemeval_strict_protocol_metadata())

    for key in ("judgment", "score", "reason", "judge_provider", "judge_model"):
        if key in answerer_output:
            result[key] = answerer_output[key]

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LongMemEval benchmark with DMF or Mem0.",
    )
    parser.add_argument(
        "--framework", default="dmf",
        choices=("dmf", "mem0"),
        help="Memory framework to run.",
    )
    parser.add_argument(
        "--config", required=True,
        help="Framework config path: DMF TOML for dmf, Mem0 YAML for mem0.",
    )
    parser.add_argument(
        "--project-name", required=True,
        help="Name for this benchmark run.",
    )
    parser.add_argument(
        "--dataset-path", default=None,
        help="Path to longmemeval dataset JSON (default: datasets/longmemeval/...).",
    )
    parser.add_argument(
        "--all-questions", action="store_true",
        help="Process all 500 questions.",
    )
    parser.add_argument(
        "--per-type", type=int, default=5,
        help="Questions per question_type for stratified sampling (default: 5).",
    )
    parser.add_argument(
        "--question-types", default=None,
        help="Comma-separated question types to include.",
    )
    parser.add_argument(
        "--question-ids", default=None,
        help="Comma-separated question_ids to process.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for stratified sampling.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip questions whose per-question JSON already exists.",
    )
    parser.add_argument(
        "--predict-only", action="store_true",
        help="Run prediction only and skip offline evaluation.",
    )
    parser.add_argument(
        "--evaluate-only", action="store_true",
        help="Skip prediction and evaluate only saved per-question outputs.",
    )
    parser.add_argument(
        "--judge-only", action="store_true",
        help="Skip prediction and run only the judge on saved per-question outputs.",
    )
    parser.add_argument(
        "--answerer-provider", default="openai",
        choices=("openai", "openrouter", "ollama"),
        help="Provider for answer generation.",
    )
    parser.add_argument(
        "--answerer-model", default="gpt-4.1-mini",
        help="Model for answer generation.",
    )
    parser.add_argument(
        "--judge-provider", default=None,
        choices=("openai", "openrouter", "ollama"),
        help="Provider for judge.",
    )
    parser.add_argument(
        "--judge-model", default=None,
        help="Model for judge.",
    )
    parser.add_argument(
        "--judge-reasoning-effort",
        choices=("low", "medium", "high"),
        help="Reasoning effort for the judge (only supported by o-series models).",
    )
    args = parser.parse_args()
    _validate_framework_args(args, parser)
    _validate_mode_args(args, parser)
    return args


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _count_question_turns(question: dict) -> int:
    return sum(len(s) for s in question.get("haystack_sessions", []))


def _count_question_ingest_units(
    question: dict,
    framework: MemoryFramework,
) -> int:
    if framework == "mem0":
        return sum(
            len(pair_turns(session))
            for session in question.get("haystack_sessions", [])
        )
    return _count_question_turns(question)


def _validate_selected_questions(
    *,
    all_questions: list[dict[str, Any]] | None,
    selected_questions: list[dict[str, Any]],
    question_ids_arg: str | None,
    question_types_arg: str | None,
    all_questions_flag: bool,
    per_type: int,
    evaluate_only: bool,
    judge_only: bool,
    project_name: str,
) -> None:
    """Raise a clear error when CLI filters select zero questions."""
    if selected_questions:
        return

    if evaluate_only or judge_only:
        raise ValueError(
            "No saved prediction artifacts matched the requested scope for "
            f"project {project_name!r}."
        )

    if question_ids_arg:
        requested_ids = [
            question_id.strip()
            for question_id in question_ids_arg.split(",")
            if question_id.strip()
        ]
        available_ids = {
            str(question.get("question_id", ""))
            for question in (all_questions or [])
        }
        missing_ids = [
            question_id
            for question_id in requested_ids
            if question_id not in available_ids
        ]
        sample_ids = sorted(available_ids)[:10]
        raise ValueError(
            "No dataset questions matched --question-ids. "
            f"Requested: {requested_ids}. Missing: {missing_ids}. "
            f"LongMemEval question_ids are dataset-defined hashes, for example: {sample_ids}"
        )

    if question_types_arg:
        raise ValueError(
            "No dataset questions matched --question-types="
            f"{question_types_arg!r}."
        )

    if all_questions_flag:
        raise ValueError("No questions available for the requested LongMemEval scope.")

    raise ValueError(
        "Question selection returned zero items. "
        f"Check --per-type={per_type} and any active filters."
    )


def _load_question_result_json(path: Path) -> dict[str, Any]:
    """Load one saved per-question result JSON."""
    return json.loads(path.read_text(encoding="utf-8"))


def _expected_cutoff_label(retrieval_depth: int) -> str:
    """Return the mono-cutoff label expected for the active config."""
    return f"top_{retrieval_depth}"


_SNAPSHOT_SUFFIX_RE = re.compile(r"^(?P<base>.+)-\d{4}-\d{2}-\d{2}$")


def _normalize_model_identity(model_name: str) -> str:
    """Normalize provider-returned model ids for compatibility checks."""
    normalized = model_name.strip()
    if "/" in normalized:
        normalized = normalized.rsplit("/", 1)[-1]
    match = _SNAPSHOT_SUFFIX_RE.match(normalized)
    if match:
        return match.group("base")
    return normalized


def _models_are_compatible(saved_model: str, expected_model: str) -> bool:
    """Return whether CLI and provider-resolved model names refer to the same model."""
    if saved_model == expected_model:
        return True
    return _normalize_model_identity(saved_model) == _normalize_model_identity(
        expected_model
    )


def _saved_result_compatibility_errors(
    saved_result: dict[str, Any],
    *,
    expected_question: dict[str, Any] | None,
    framework: MemoryFramework,
    retrieval_depth: int,
    answerer_provider: str | None,
    answerer_model: str | None,
) -> list[str]:
    """Return semantic compatibility errors for a saved per-question result."""
    errors: list[str] = []

    try:
        results_io.ensure_longmemeval_strict_evaluation(
            saved_result,
            source_description="saved LongMemEval result",
        )
    except ValueError as exc:
        errors.append(str(exc))
        return errors

    saved_framework = saved_result.get("framework")
    if isinstance(saved_framework, str) and saved_framework:
        if saved_framework != framework:
            errors.append(f"framework={saved_framework!r} (expected {framework!r})")
    elif framework == "mem0":
        # Missing framework metadata is ambiguous once Mem0 outputs exist.
        errors.append("missing framework metadata for a mem0 run")

    expected_cutoff = _expected_cutoff_label(retrieval_depth)
    saved_cutoff = saved_result.get("cutoff_label")
    cutoff_results = saved_result.get("cutoff_results")
    if isinstance(saved_cutoff, str) and saved_cutoff:
        if saved_cutoff != expected_cutoff:
            errors.append(
                f"cutoff_label={saved_cutoff!r} (expected {expected_cutoff!r})"
            )
    elif isinstance(cutoff_results, dict) and cutoff_results:
        if expected_cutoff not in cutoff_results:
            errors.append(
                "saved legacy cutoff_results do not include "
                f"{expected_cutoff!r}"
            )
    else:
        errors.append("missing cutoff metadata")

    saved_provider = saved_result.get("answerer_provider")
    if (
        answerer_provider is not None
        and (
        isinstance(saved_provider, str)
        and saved_provider
        and saved_provider != answerer_provider
        )
    ):
        errors.append(
            f"answerer_provider={saved_provider!r} "
            f"(expected {answerer_provider!r})"
        )

    saved_model = saved_result.get("answerer_model")
    if (
        answerer_model is not None
        and (
        isinstance(saved_model, str)
        and saved_model
        and not _models_are_compatible(saved_model, answerer_model)
        )
    ):
        errors.append(
            f"answerer_model={saved_model!r} (expected {answerer_model!r})"
        )

    if expected_question is None:
        return errors

    expected_answer_session_ids = list(expected_question.get("answer_session_ids", []))
    expected_answer = (
        str(expected_question["answer"])
        if "answer" in expected_question
        else str(expected_question.get("ground_truth_answer", ""))
    )
    comparisons = (
        ("question_id", saved_result.get("question_id"), expected_question["question_id"]),
        ("question_type", saved_result.get("question_type"), expected_question["question_type"]),
        ("question", saved_result.get("question"), expected_question["question"]),
        (
            "ground_truth_answer",
            saved_result.get("ground_truth_answer"),
            expected_answer,
        ),
        (
            "question_date",
            saved_result.get("question_date", ""),
            expected_question.get("question_date", ""),
        ),
        (
            "answer_session_ids",
            list(saved_result.get("answer_session_ids", [])),
            expected_answer_session_ids,
        ),
    )
    for label, actual, expected in comparisons:
        if actual != expected:
            errors.append(f"{label} mismatch")

    return errors


def _looks_like_legacy_or_non_strict_longmemeval_result(
    saved_result: dict[str, Any],
) -> bool:
    """Return whether a saved question result is a legacy/non-strict artifact."""
    if isinstance(saved_result.get("cutoff_results"), dict):
        return True
    if "generated_answer" in saved_result:
        return True
    retrieval = saved_result.get("retrieval")
    return isinstance(retrieval, dict) and "search_results" in retrieval


def _is_complete_question_result(path: Path) -> bool:
    """Return whether a saved per-question result is a complete strict artifact."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    try:
        results_io.ensure_longmemeval_strict_evaluation(
            data,
            source_description=f"saved LongMemEval result {path.name}",
        )
    except ValueError:
        return False
    return True


def _is_judged_question_result(result: dict[str, Any]) -> bool:
    """Return whether one flat per-question result already contains judge outputs."""
    return all(key in result for key in ("judgment", "score", "reason"))


def _run_mode(args: argparse.Namespace) -> str:
    if getattr(args, "predict_only", False):
        return "predict_only"
    if getattr(args, "judge_only", False):
        return "judge_only"
    if getattr(args, "evaluate_only", False):
        return "evaluate_only"
    return "full"


def _prediction_enabled(args: argparse.Namespace) -> bool:
    return not getattr(args, "evaluate_only", False) and not getattr(args, "judge_only", False)


def _judging_enabled(args: argparse.Namespace) -> bool:
    return not getattr(args, "predict_only", False) and not getattr(args, "evaluate_only", False)


def _evaluation_enabled(args: argparse.Namespace) -> bool:
    return not getattr(args, "predict_only", False) and not getattr(args, "judge_only", False)


def _preflight_prediction_run(
    *,
    project_name: str,
    question_ids: list[str],
) -> None:
    """Refuse fresh prediction runs on top of existing checkpoints."""
    existing = [
        question_id
        for question_id in question_ids
        if results_io.is_question_predicted(
            project_name=project_name,
            question_id=question_id,
            benchmark_name="longmemeval",
        )
    ]
    if not existing:
        return

    raise ValueError(
        "Fresh prediction runs require clean checkpoints for the selected "
        f"questions. Found existing prediction artifacts for: {existing}. "
        "Use --resume to continue, or delete the project results directory and "
        "clean the framework storage before rerunning predictions."
    )


def _print_storage_reset_banner(
    *,
    framework: MemoryFramework,
    resume: bool,
    project_name: str,
    dmf_config: DMFConfig | None = None,
) -> None:
    """Print a startup reminder about cleaning local storage on fresh runs."""
    if resume:
        return

    predicted_path = (
        Path("results") / "longmemeval" / f"predicted_{project_name}"
    ).resolve()

    if framework == "dmf":
        chroma_path = "<unknown>"
        if dmf_config is not None:
            chroma_path = str(Path(dmf_config.ltm.chroma_path).resolve())
        _banner_print(
            "[bold yellow]Benchmark reminder[/bold yellow]: "
            "you are starting a DMF benchmark without `--resume`. "
            f"If you want a fresh run, make sure you have cleaned the local Chroma directory `{chroma_path}` "
            f"and the project results/checkpoints directory `{predicted_path}`."
        )
        return

    if framework == "mem0":
        mem0_storage = (
            Path("results") / "longmemeval" / ".mem0_local" / project_name
        ).resolve()
        _banner_print(
            "[bold yellow]Benchmark reminder[/bold yellow]: "
            "you are starting a Mem0 benchmark without `--resume`. "
            f"If you want a fresh run, make sure you have cleaned the local storage `{mem0_storage}` "
            f"and the project results/checkpoints directory `{predicted_path}`."
        )


def _restore_mem0_usage_for_question(
    *,
    project_name: str,
    question_id: str,
    aggregate_usage: dict[str, Any],
    missing_ok: bool = False,
) -> dict[str, Any]:
    """Restore one saved Mem0 usage sidecar into the aggregate counter."""
    try:
        usage = results_io.load_mem0_question_usage(
            project_name=project_name,
            question_id=question_id,
            benchmark_name="longmemeval",
        )
    except FileNotFoundError:
        if missing_ok:
            log.warning(
                "Missing Mem0 usage sidecar for question %s; token accounting "
                "will be partial for this run.",
                question_id,
            )
            return aggregate_usage
        raise

    return add_memory_internal_usage(
        aggregate_usage,
        usage,
    )


def _validate_saved_question_results(
    *,
    project_name: str,
    question_ids: list[str],
    framework: MemoryFramework,
    retrieval_depth: int,
    answerer_provider: str | None,
    answerer_model: str | None,
    expected_questions_by_id: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Load and validate complete saved results for the selected questions."""
    results = results_io.load_question_results_for_ids(
        project_name=project_name,
        question_ids=question_ids,
        benchmark_name="longmemeval",
    )
    saved_by_id = {
        str(result.get("question_id", "")): result
        for result in results
    }
    missing: list[str] = []
    incompatible: list[str] = []
    for question_id in question_ids:
        path = results_io.question_result_path(
            project_name=project_name,
            question_id=question_id,
            benchmark_name="longmemeval",
        )
        if not path.exists():
            missing.append(question_id)
            continue
        if not _is_complete_question_result(path):
            saved_result = _load_question_result_json(path)
            if _looks_like_legacy_or_non_strict_longmemeval_result(saved_result):
                errors = _saved_result_compatibility_errors(
                    saved_result,
                    expected_question=(
                        expected_questions_by_id.get(question_id)
                        if expected_questions_by_id is not None
                        else None
                    ),
                    framework=framework,
                    retrieval_depth=retrieval_depth,
                    answerer_provider=answerer_provider,
                    answerer_model=answerer_model,
                )
                incompatible.append(f"{question_id}: {', '.join(errors)}")
                continue
            missing.append(question_id)
            continue
        if question_id not in saved_by_id:
            missing.append(question_id)
            continue

        errors = _saved_result_compatibility_errors(
            saved_by_id[question_id],
            expected_question=(
                expected_questions_by_id.get(question_id)
                if expected_questions_by_id is not None
                else None
            ),
            framework=framework,
            retrieval_depth=retrieval_depth,
            answerer_provider=answerer_provider,
            answerer_model=answerer_model,
        )
        if errors:
            incompatible.append(f"{question_id}: {', '.join(errors)}")

    if missing:
        raise ValueError(
            "Evaluation requires saved prediction outputs for every selected "
            f"question. Missing or incomplete: {missing}"
        )
    if incompatible:
        raise ValueError(
            "Evaluation found saved prediction outputs incompatible with the "
            f"current CLI/config: {incompatible}"
        )

    return [saved_by_id[question_id] for question_id in question_ids]


def _select_saved_questions_for_evaluation(
    *,
    project_name: str,
    all_questions: bool,
    question_ids_arg: str | None,
    per_type: int,
    seed: int,
    selected_types: list[str] | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Select evaluation scope directly from saved per-question artifacts."""
    saved_results = results_io.load_question_results(
        project_name=project_name,
        benchmark_name="longmemeval",
    )
    if not saved_results:
        raise ValueError(
            "Evaluate-only mode requires saved prediction outputs, but none "
            f"were found for project {project_name!r}."
        )

    if question_ids_arg:
        requested_ids = [
            question_id.strip()
            for question_id in question_ids_arg.split(",")
            if question_id.strip()
        ]
        saved_by_id = {
            str(result.get("question_id", "")): result
            for result in saved_results
        }
        selected = [
            saved_by_id[question_id]
            for question_id in requested_ids
            if question_id in saved_by_id
        ]
        return selected, requested_ids

    if all_questions:
        selected = saved_results
        if selected_types:
            selected_type_set = set(selected_types)
            selected = [
                result
                for result in selected
                if result.get("question_type") in selected_type_set
            ]
        selected.sort(key=lambda result: str(result.get("question_id", "")))
        return selected, [str(result["question_id"]) for result in selected]

    selected = sample_questions_stratified(
        saved_results,
        per_type=per_type,
        seed=seed,
        selected_types=selected_types,
    )
    return selected, [str(result["question_id"]) for result in selected]


def _run_offline_evaluator(input_path: Path) -> subprocess.CompletedProcess[str]:
    """Launch the deterministic LongMemEval evaluator for one saved bundle."""
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "longmemeval.evaluate_rigorous",
            "--input",
            str(input_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )


def main() -> None:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.progress import (
        Progress, SpinnerColumn, BarColumn,
        TaskProgressColumn, TimeElapsedColumn, TextColumn,
    )

    load_dotenv()
    args = parse_args()
    _validate_startup_secrets(args)

    console = Console()
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )
    logging.getLogger("chromadb.telemetry.product.posthog").disabled = True
    logging.getLogger("posthog").disabled = True

    config = _load_active_config(args)

    selected_types = None
    if args.question_types:
        selected_types = [t.strip() for t in args.question_types.split(",") if t.strip()]

    dataset_path: str | None = None
    if args.evaluate_only or getattr(args, "judge_only", False):
        questions, selected_question_ids = _select_saved_questions_for_evaluation(
            project_name=args.project_name,
            all_questions=bool(args.all_questions),
            question_ids_arg=args.question_ids,
            per_type=args.per_type,
            seed=args.seed,
            selected_types=selected_types,
        )
        console.print(
            "Prediction artifacts: "
            f"[bold]{len(questions)}[/bold] selected from project {args.project_name}"
        )
    else:
        dataset_path, all_questions = ensure_longmemeval_dataset(args.dataset_path)
        console.print(
            f"Dataset: [bold]{len(all_questions)}[/bold] questions from {dataset_path}"
        )
        if args.question_ids:
            ids = [i.strip() for i in args.question_ids.split(",") if i.strip()]
            questions = filter_questions_by_ids(all_questions, ids)
        elif args.all_questions:
            questions = all_questions
            if selected_types:
                type_set = set(selected_types)
                questions = [q for q in questions if q["question_type"] in type_set]
        else:
            questions = sample_questions_stratified(
                all_questions,
                per_type=args.per_type,
                seed=args.seed,
                selected_types=selected_types,
            )
        selected_question_ids = [str(question["question_id"]) for question in questions]

    _validate_selected_questions(
        all_questions=locals().get("all_questions"),
        selected_questions=questions,
        question_ids_arg=args.question_ids,
        question_types_arg=args.question_types,
        all_questions_flag=bool(args.all_questions),
        per_type=args.per_type,
        evaluate_only=bool(args.evaluate_only),
        judge_only=bool(getattr(args, "judge_only", False)),
        project_name=args.project_name,
    )

    qa_settings: QASettings | None = None
    if _prediction_enabled(args):
        qa_settings = QASettings(
            provider=args.answerer_provider,
            model=args.answerer_model,
            base_url=None,
            temperature=0.0,
            max_tokens=4096,
            rpm=200,
            timeout=120.0,
        )

    judge_settings: JudgeSettings | None = None
    if _judging_enabled(args):
        judge_settings = judge.resolve_judge_settings(
            provider_override=getattr(args, "judge_provider", None),
            model_override=getattr(args, "judge_model", None),
            reasoning_effort=getattr(args, "judge_reasoning_effort", None),
        )
        judge.log_judge_settings(judge_settings)
        args.judge_provider = judge_settings.provider
        args.judge_model = judge_settings.model

    # Print summary
    type_counts: dict[str, int] = defaultdict(int)
    for q in questions:
        type_counts[q["question_type"]] += 1
    console.print(f"Selected [bold]{len(questions)}[/bold] questions:")
    for qtype in sorted(type_counts):
        console.print(f"  {qtype}: {type_counts[qtype]}")
    _print_startup_configuration(
        args=args,
        config=config,
        dataset_path=dataset_path,
        prediction_artifacts_path=results_io.predicted_results_dir(
            project_name=args.project_name,
            benchmark_name="longmemeval",
        ),
        selected_type_counts=type_counts,
        answerer_settings=qa_settings,
        judge_settings=judge_settings,
    )
    console.print()

    # Ensure output directory
    results_io.ensure_predicted_results_dir(
        project_name=args.project_name,
        benchmark_name="longmemeval",
    )

    completed = 0
    skipped = 0
    mem0_run_id = uuid.uuid4().hex[:8] if config.framework == "mem0" else None
    memory_internal_usage = empty_memory_internal_usage(framework=args.framework)
    judge_client = (
        judge.build_judge(judge_settings)
        if judge_settings is not None
        else None
    )

    if _prediction_enabled(args) and not args.resume:
        _print_storage_reset_banner(
            framework=config.framework,
            resume=args.resume,
            project_name=args.project_name,
            dmf_config=config.dmf_config,
        )
        _preflight_prediction_run(
            project_name=args.project_name,
            question_ids=selected_question_ids,
        )

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )

    with progress:
        global_task = progress.add_task(
            "LongMemEval benchmark",
            total=len(questions),
        )
        phase_task = progress.add_task(
            "Waiting...",
            total=1,
            visible=False,
        )

        if _prediction_enabled(args):
            for question in questions:
                qid = question["question_id"]
                qtype = question["question_type"]

                # Resume: skip only structurally complete per-question outputs
                if args.resume:
                    existing = results_io.question_result_path(
                        project_name=args.project_name,
                        question_id=qid,
                        benchmark_name="longmemeval",
                    )
                    if existing.exists():
                        saved_result = _load_question_result_json(existing)
                        if not _is_complete_question_result(existing):
                            if _looks_like_legacy_or_non_strict_longmemeval_result(saved_result):
                                compatibility_errors = _saved_result_compatibility_errors(
                                    saved_result,
                                    expected_question=question,
                                    framework=config.framework,
                                    retrieval_depth=config.retrieval_depth,
                                    answerer_provider=args.answerer_provider,
                                    answerer_model=args.answerer_model,
                                )
                                raise ValueError(
                                    "Resume found incompatible saved prediction output "
                                    f"for question {qid}: {compatibility_errors}"
                                )
                        else:
                            compatibility_errors = _saved_result_compatibility_errors(
                                saved_result,
                                expected_question=question,
                                framework=config.framework,
                                retrieval_depth=config.retrieval_depth,
                                answerer_provider=args.answerer_provider,
                                answerer_model=args.answerer_model,
                            )
                            if compatibility_errors:
                                raise ValueError(
                                    "Resume found incompatible saved prediction output "
                                    f"for question {qid}: {compatibility_errors}"
                                )
                            if config.framework == "mem0":
                                memory_internal_usage = _restore_mem0_usage_for_question(
                                    project_name=args.project_name,
                                    question_id=qid,
                                    aggregate_usage=memory_internal_usage,
                                    missing_ok=True,
                                )
                            if _judging_enabled(args) and not _is_judged_question_result(saved_result):
                                progress.update(
                                    phase_task,
                                    description=f"Judge {qid} ({qtype})",
                                    total=1,
                                    completed=0,
                                    visible=True,
                                )
                                if judge_client is None or judge_settings is None:
                                    raise RuntimeError("Judge client/settings are not initialized.")
                                judged_result = judge.judge_one_result(
                                    result=saved_result,
                                    judge_client=judge_client,
                                    settings=judge_settings,
                                )
                                results_io.save_question_result(
                                    judged_result,
                                    project_name=args.project_name,
                                    benchmark_name="longmemeval",
                                )
                                progress.update(phase_task, completed=1)
                                completed += 1
                            else:
                                skipped += 1
                            progress.advance(global_task)
                            continue

                # --- Ingestion phase ---
                turn_count = _count_question_ingest_units(question, config.framework)
                progress.update(
                    phase_task,
                    description=f"Ingestion {qid} ({qtype})",
                    total=max(1, turn_count),
                    completed=0,
                    visible=True,
                )

                t0 = time.monotonic()
                bundle = ingest_question(
                    question,
                    config,
                    project_name=args.project_name,
                    mem0_run_id=mem0_run_id,
                    on_turn_completed=lambda: progress.advance(phase_task),
                )
                ingest_s = time.monotonic() - t0

                log.info(
                    "%s ingested %d records (%d sessions) in %.1fs",
                    qid, len(bundle.record_index),
                    len(question["haystack_sessions"]), ingest_s,
                )

                # --- Answerer phase ---
                progress.update(
                    phase_task,
                    description=f"Answerer {qid} ({qtype})",
                    total=1,
                    completed=0,
                )

                answerer_output = run_answerer_for_question(
                    question, bundle, config, qa_settings,
                )
                result = build_question_result(
                    question, answerer_output, framework=config.framework,
                )
                pipeline_timing = results_io.normalize_pipeline_timing(
                    result.get("pipeline_timing")
                )
                pipeline_timing["ingestion_ms"] = ingest_s * 1000
                pipeline_timing["ingestion_scope"] = "question"
                result["pipeline_timing"] = pipeline_timing

                results_io.save_question_result(
                    result,
                    project_name=args.project_name,
                    benchmark_name="longmemeval",
                )
                results_io.mark_question_predicted(
                    project_name=args.project_name,
                    question_id=str(qid),
                    benchmark_name="longmemeval",
                )

                if config.framework == "mem0":
                    backend = _build_mem0_backend(bundle)
                    question_usage = backend.get_usage()
                    memory_internal_usage = add_memory_internal_usage(
                        memory_internal_usage,
                        question_usage,
                    )
                    results_io.save_mem0_question_usage(
                        project_name=args.project_name,
                        question_id=str(qid),
                        usage=question_usage,
                        benchmark_name="longmemeval",
                    )

                progress.update(phase_task, completed=1)

                if _judging_enabled(args):
                    progress.update(
                        phase_task,
                        description=f"Judge {qid} ({qtype})",
                        total=1,
                        completed=0,
                        visible=True,
                    )
                    if judge_client is None or judge_settings is None:
                        raise RuntimeError("Judge client/settings are not initialized.")
                    result = judge.judge_one_result(
                        result=result,
                        judge_client=judge_client,
                        settings=judge_settings,
                    )
                    results_io.save_question_result(
                        result,
                        project_name=args.project_name,
                        benchmark_name="longmemeval",
                    )
                    progress.update(phase_task, completed=1)

                answer_preview = answerer_output["generated_answer"][:80]
                gold_preview = str(question["answer"])[:80]
                log.info("%s answer: %s", qid, answer_preview)
                log.info("%s gold:   %s", qid, gold_preview)

                completed += 1
                progress.advance(global_task)
        elif _judging_enabled(args):
            for result in questions:
                qid = str(result["question_id"])
                qtype = str(result["question_type"])
                if config.framework == "mem0":
                    memory_internal_usage = _restore_mem0_usage_for_question(
                        project_name=args.project_name,
                        question_id=qid,
                        aggregate_usage=memory_internal_usage,
                        missing_ok=True,
                    )

                if args.resume and _is_judged_question_result(result):
                    skipped += 1
                    progress.advance(global_task)
                    continue

                progress.update(
                    phase_task,
                    description=f"Judge {qid} ({qtype})",
                    total=1,
                    completed=0,
                    visible=True,
                )
                if judge_client is None or judge_settings is None:
                    raise RuntimeError("Judge client/settings are not initialized.")
                judged_result = judge.judge_one_result(
                    result=result,
                    judge_client=judge_client,
                    settings=judge_settings,
                )
                results_io.save_question_result(
                    judged_result,
                    project_name=args.project_name,
                    benchmark_name="longmemeval",
                )
                progress.update(phase_task, completed=1)
                completed += 1
                progress.advance(global_task)
        else:
            for question_id in selected_question_ids:
                if config.framework == "mem0":
                    memory_internal_usage = _restore_mem0_usage_for_question(
                        project_name=args.project_name,
                        question_id=question_id,
                        aggregate_usage=memory_internal_usage,
                        missing_ok=True,
                    )
                skipped += 1
                progress.advance(global_task)

        progress.update(phase_task, visible=False)

    # Final bundle + offline evaluation
    console.print()
    console.print(f"Completed: [green]{completed}[/green], Skipped: {skipped}")
    if selected_question_ids:
        expected_questions_by_id = {
            str(question["question_id"]): question
            for question in questions
        }
        _validate_saved_question_results(
            project_name=args.project_name,
            question_ids=selected_question_ids,
            framework=config.framework,
            retrieval_depth=config.retrieval_depth,
            answerer_provider=(
                args.answerer_provider if not getattr(args, "judge_only", False) else None
            ),
            answerer_model=(
                args.answerer_model if not getattr(args, "judge_only", False) else None
            ),
            expected_questions_by_id=expected_questions_by_id,
        )
        final_path = results_io.save_final_longmemeval_results(
            project_name=args.project_name,
            mode=_run_mode(args),
            framework=args.framework,
            run_metadata={
                "dataset_path": dataset_path,
                "all_questions": bool(args.all_questions),
                "per_type": None if args.all_questions else args.per_type,
                "seed": args.seed,
                "selected_question_ids": selected_question_ids,
                "question_types_filter": selected_types or [],
                "framework": args.framework,
                "retrieval_depth": config.retrieval_depth,
                "answerer_provider": (
                    None if getattr(args, "judge_only", False) else args.answerer_provider
                ),
                "answerer_model": (
                    None if getattr(args, "judge_only", False) else args.answerer_model
                ),
                "judge_provider": getattr(args, "judge_provider", None),
                "judge_model": getattr(args, "judge_model", None),
                "run_mode": _run_mode(args),
            },
            question_ids=selected_question_ids,
            token_accounting={"memory_internal": memory_internal_usage},
        )
        console.print(f"Results saved to: [bold]{final_path}[/bold]")
        if _evaluation_enabled(args):
            completed_process = _run_offline_evaluator(final_path)
            if completed_process.stdout:
                console.print(completed_process.stdout.rstrip())
            if completed_process.returncode != 0:
                if completed_process.stderr:
                    console.print(completed_process.stderr.rstrip(), style="bold red")
                raise SystemExit(completed_process.returncode)


if __name__ == "__main__":
    main()
