"""Framework-aware ingestion retained by the legacy native LongMemEval runner."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")

from dmf.analysis.nlp_engine import NLPEngine
from dmf.analysis.scoring_engine import ScoringEngine
from dmf.memory.chroma_ltm import ChromaLTMHook
from dmf.memory.temporal_memory import TemporalMemory
from dmf.runtime.pipeline import InteractionPipeline, InteractionProvenance
from dmf.utils.config import NLPConfig, VectorConfig
from dmf.utils.config_loader import DMFConfig, load_dmf_config

from dmf_bench.frameworks.mem0_config import Mem0Config, load_mem0_config
from dmf_bench.frameworks.mem0_runtime import LocalMem0BenchmarkItemBackend
from dmf_bench.models import IngestedQuestionBundle, MemoryFramework
from longmemeval.utils import (
    normalize_longmemeval_haystack,
    pair_turns,
    parse_longmemeval_date,
    render_longmemeval_pair_for_context,
    serialize_longmemeval_pair_for_mem0,
    sort_sessions_chronologically,
)


def _safe_question_key(question_id: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", question_id).strip("_")
    return normalized or "question"


def _stable_question_namespace(question_id: str) -> str:
    safe_key = _safe_question_key(question_id).lower()
    digest = hashlib.sha1(question_id.encode("utf-8")).hexdigest()[:10]
    return f"{safe_key[:24]}_{digest}"


def _namespaced_cards_collection_name(base_name: str, question_id: str) -> str:
    namespace = _stable_question_namespace(question_id)
    collection_name = f"{base_name}_longmemeval"
    suffix = f"_{namespace}"
    if len(collection_name) + len(suffix) <= 63:
        return f"{collection_name}{suffix}"
    return f"{collection_name[: 63 - len(suffix)]}{suffix}"


def _namespaced_cards_path(
    base_path: str | Path | None,
    *,
    question_id: str,
    persist_directory: str | Path,
) -> Path:
    path = (
        Path(base_path)
        if base_path is not None
        else Path(persist_directory) / "ltm_cards.jsonl"
    )
    suffix = path.suffix or ".jsonl"
    namespace = _stable_question_namespace(question_id)
    return path.with_name(f"{path.stem}_longmemeval_{namespace}{suffix}")


@dataclass(frozen=True)
class ActiveBenchmarkConfig:
    framework: MemoryFramework
    dmf_config: DMFConfig | None = None
    mem0_config: Mem0Config | None = None

    @property
    def retrieval_depth(self) -> int:
        if self.framework == "dmf":
            if self.dmf_config is None:
                raise RuntimeError("DMF config not loaded.")
            return int(self.dmf_config.ltm.recall_limit)
        if self.mem0_config is None:
            raise RuntimeError("Mem0 config not loaded.")
        return self.mem0_config.top_k


def load_active_config(args: argparse.Namespace) -> ActiveBenchmarkConfig:
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
    ltm_hook = ChromaLTMHook(
        collection_name=_namespaced_cards_collection_name(
            config.ltm.collection_name,
            question_id,
        ),
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
    question: dict[str, Any],
    config: DMFConfig,
    on_turn_completed: Any | None = None,
) -> IngestedQuestionBundle:
    question_id = str(question["question_id"])
    pipeline = InteractionPipeline.from_dmf_config(config=config)
    scoring = ScoringEngine.from_dmf_config(config=config)
    memory_engine = build_memory_engine_for_question(question_id, config)
    record_index: dict[str, dict[str, Any]] = {}

    for session_id, date_str, session in sort_sessions_chronologically(question):
        session_ts = parse_longmemeval_date(date_str)
        for pair in pair_turns(session):
            for message in pair:
                text = str(message.get("content", ""))
                if not text.strip():
                    continue
                role = str(message.get("role", ""))
                report, vector = pipeline.analyze_interaction_with_vector(
                    text=text,
                    is_system=False,
                    provenance=InteractionProvenance(role=role),
                )
                report.raw_metadata.update(
                    {
                        "benchmark": "longmemeval",
                        "question_id": question_id,
                        "source_unit_type": "session",
                        "source_unit_id": session_id,
                        "session_date_raw": date_str,
                    }
                )
                scoring.calculate_score(report, text=text)
                entry = memory_engine.add_interaction(text, report, vector)
                if session_ts is not None:
                    entry.timestamp = session_ts
                record_index[entry.record_id] = {
                    "benchmark": "longmemeval",
                    "question_id": question_id,
                    "source_unit_type": "session",
                    "source_unit_id": session_id,
                    "source_unit_ids": [session_id],
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
    question: dict[str, Any],
    *,
    config: Mem0Config,
    project_name: str,
    run_id: str,
    on_turn_completed: Any | None = None,
) -> IngestedQuestionBundle:
    question_id = str(question["question_id"])
    backend = LocalMem0BenchmarkItemBackend(
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
            backend.add(
                messages,
                user_id=user_id,
                timestamp=session["session_timestamp"],
                metadata=metadata,
            )
            record_id = f"{session_id}:pair:{pair['pair_index']}"
            record_index[record_id] = {
                **metadata,
                "source_unit_ids": [session_id],
                "session_id": session_id,
                "session_date_raw": session["session_date_raw"],
                "session_timestamp": session["session_timestamp"],
                "pair_index": pair["pair_index"],
                "text": render_longmemeval_pair_for_context(pair),
            }
            if on_turn_completed is not None:
                on_turn_completed()

    return IngestedQuestionBundle(
        question_id=question_id,
        framework="mem0",
        backend_state={"mem0_backend": backend, "user_id": user_id},
        record_index=record_index,
    )


def ingest_question(
    question: dict[str, Any],
    config: ActiveBenchmarkConfig,
    *,
    project_name: str | None = None,
    mem0_run_id: str | None = None,
    on_turn_completed: Any | None = None,
) -> IngestedQuestionBundle:
    if config.framework == "dmf":
        if config.dmf_config is None:
            raise RuntimeError("DMF config not loaded.")
        return ingest_question_dmf(
            question,
            config.dmf_config,
            on_turn_completed=on_turn_completed,
        )
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


def count_question_ingest_units(
    question: dict[str, Any],
    framework: MemoryFramework,
) -> int:
    if framework == "mem0":
        return sum(
            len(pair_turns(session))
            for session in question.get("haystack_sessions", [])
        )
    return sum(len(session) for session in question.get("haystack_sessions", []))
