"""Controlled retrieval-QA baselines with no framework-specific runner branches."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable
from urllib.request import Request, urlopen

from dmf_bench.context import _encoding, _token_count, render_memory
from dmf_bench.atomic_io import write_json_atomic

from .base import (
    BenchmarkUnit, CanonicalRetrievalResult, FrameworkRunContext, MemoryEvent,
    MemoryQuery, OwnedResource, PreparedMemoryUnit, RetrievedMemory,
)
from .embedded_lifecycle import cleanup_owned_embedded_resource


def _profile(config: dict[str, Any], expected: str) -> dict[str, Any]:
    path = Path(config["framework_config"]["path"])
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("baseline") != expected or data.get("version") != "v1":
        raise ValueError(f"Expected {expected} baseline profile.")
    return data


def _check_events(events: tuple[MemoryEvent, ...]) -> None:
    if not isinstance(events, tuple) or any(not isinstance(event, MemoryEvent) for event in events):
        raise ValueError("Baseline ingestion requires normalized MemoryEvent values.")


def _check_query(unit: BenchmarkUnit, query: MemoryQuery) -> None:
    if not isinstance(query, MemoryQuery) or query.query_id not in unit.item_ids:
        raise ValueError("Query does not belong to the prepared unit.")


class ModelOnlyAdapter:
    name = "model-only"

    def __init__(self, config: dict[str, Any]) -> None:
        if set(_profile(config, self.name)) != {"baseline", "version"}:
            raise ValueError("Model-only profile has unsupported fields.")

    def resources_for_unit_v3(self, unit: BenchmarkUnit, config: dict[str, Any],
                              run_context: FrameworkRunContext) -> tuple[OwnedResource, ...]:
        return ()

    def cleanup_unit_v3(self, unit: BenchmarkUnit, resources: tuple[OwnedResource, ...],
                        config: dict[str, Any], run_context: FrameworkRunContext) -> dict[str, Any]:
        if resources:
            raise ValueError("Model-only owns no resources.")
        return {"verified": True, "resources": []}

    def prepare_unit_v3(self, unit: BenchmarkUnit, events: tuple[MemoryEvent, ...],
                        config: dict[str, Any], run_context: FrameworkRunContext) -> PreparedMemoryUnit:
        _check_events(events)
        return PreparedMemoryUnit(None, (), {}, {}, {"discarded_event_count": len(events)})

    def verify_prepared_v3(self, unit: BenchmarkUnit, prepared: PreparedMemoryUnit,
                           config: dict[str, Any], run_context: FrameworkRunContext) -> dict[str, Any]:
        if prepared.handle is not None or prepared.resources:
            raise ValueError("Model-only retained memory state.")
        return {"verified": True, "retained_event_count": 0}

    def retrieve_v3(self, unit: BenchmarkUnit, query: MemoryQuery, prepared: PreparedMemoryUnit,
                    config: dict[str, Any], run_context: FrameworkRunContext) -> CanonicalRetrievalResult:
        _check_query(unit, query)
        self.verify_prepared_v3(unit, prepared, config, run_context)
        return CanonicalRetrievalResult((), diagnostics={"baseline_family": "model-only"})


class FullContextAdapter:
    name = "full-context"

    def __init__(self, config: dict[str, Any]) -> None:
        profile = _profile(config, self.name)
        if set(profile) != {"baseline", "version", "model_context_window", "reserved_prompt_tokens", "truncation_policy"}:
            raise ValueError("Full-context profile has unsupported fields.")
        if profile["truncation_policy"] != "keep-latest-whole-turns-v1":
            raise ValueError("Unsupported full-context truncation policy.")
        for key in ("model_context_window", "reserved_prompt_tokens"):
            if type(profile[key]) is not int or profile[key] < 0:
                raise ValueError(f"{key} must be a non-negative integer.")
        self.profile = profile

    def resources_for_unit_v3(self, unit: BenchmarkUnit, config: dict[str, Any],
                              run_context: FrameworkRunContext) -> tuple[OwnedResource, ...]:
        return ()

    def cleanup_unit_v3(self, unit: BenchmarkUnit, resources: tuple[OwnedResource, ...],
                        config: dict[str, Any], run_context: FrameworkRunContext) -> dict[str, Any]:
        if resources:
            raise ValueError("Full-context owns no persistent resources.")
        return {"verified": True, "resources": []}

    def prepare_unit_v3(self, unit: BenchmarkUnit, events: tuple[MemoryEvent, ...],
                        config: dict[str, Any], run_context: FrameworkRunContext) -> PreparedMemoryUnit:
        _check_events(events)
        return PreparedMemoryUnit(events, (), {}, {}, {"event_count": len(events)})

    def verify_prepared_v3(self, unit: BenchmarkUnit, prepared: PreparedMemoryUnit,
                           config: dict[str, Any], run_context: FrameworkRunContext) -> dict[str, Any]:
        _check_events(prepared.handle)
        if prepared.resources:
            raise ValueError("Full-context has unexpected resources.")
        return {"verified": True, "event_count": len(prepared.handle)}

    def retrieve_v3(self, unit: BenchmarkUnit, query: MemoryQuery, prepared: PreparedMemoryUnit,
                    config: dict[str, Any], run_context: FrameworkRunContext) -> CanonicalRetrievalResult:
        _check_query(unit, query)
        self.verify_prepared_v3(unit, prepared, config, run_context)
        started = time.perf_counter()
        encoding = _encoding(config["context_budget"]["tokenizer"])
        answer_tokens = config["models"]["answerer"]["parameters"]["max_tokens"]
        available = self.profile["model_context_window"] - self.profile["reserved_prompt_tokens"]
        available -= _token_count(encoding, query.text) + answer_tokens
        budget = min(config["context_budget"]["max_tokens"], available)
        if budget < 0:
            raise ValueError("Full-context prompt and answer reserves exceed the model window.")
        events = prepared.handle
        max_results = config["retrieval"]["max_results"]
        selected: list[MemoryEvent] = []
        for event in reversed(events):
            candidate = [event, *selected]
            if len(candidate) > max_results:
                break
            items = [_event_memory(item, rank) for rank, item in enumerate(candidate, 1)]
            rendered = "\n\n".join(render_memory(item) for item in items)
            if _token_count(encoding, rendered) > budget:
                break
            selected = candidate
        retained = {event.event_id for event in selected}
        dropped = [event.event_id for event in events if event.event_id not in retained]
        items = tuple(_event_memory(event, rank) for rank, event in enumerate(selected, 1))
        return CanonicalRetrievalResult(
            items,
            diagnostics={"baseline_family": "bounded-full-context" if dropped else "full-context",
                         "truncation_policy": self.profile["truncation_policy"],
                         "dropped_event_ids": dropped, "available_history_tokens": budget,
                         "total_event_count": len(events)},
            timing={"retrieval_pipeline_ms": (time.perf_counter() - started) * 1000},
        )


def _event_memory(event: MemoryEvent, rank: int) -> RetrievedMemory:
    return RetrievedMemory(
        memory_id=f"turn:{event.event_id}", rank=rank,
        content=f"Session: {event.session_id}; turn: {event.sequence}; role: {event.role}\n{event.content}",
        occurred_at=event.occurred_at, source_event_ids=(event.event_id,),
        metadata={"source_refs": list(event.source_refs)},
    )


def _ollama_embed(model: str, inputs: list[str], base_url: str) -> list[list[float]]:
    request = Request(
        base_url.rstrip("/").removesuffix("/v1") + "/api/embed",
        data=json.dumps({"model": model, "input": inputs, "truncate": False}).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urlopen(request, timeout=120) as response:
        payload = json.load(response)
    vectors = payload.get("embeddings") if isinstance(payload, dict) else None
    if not isinstance(vectors, list) or len(vectors) != len(inputs):
        raise ValueError("Ollama returned an invalid embedding batch.")
    for vector in vectors:
        if not isinstance(vector, list) or not vector or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in vector
        ):
            raise ValueError("Ollama returned an invalid embedding vector.")
    if len({len(vector) for vector in vectors}) != 1:
        raise ValueError("Ollama embedding dimensions differ.")
    return vectors


def _verify_ollama_model(model: str, digest: str, base_url: str) -> None:
    with urlopen(base_url.rstrip("/").removesuffix("/v1") + "/api/tags", timeout=10) as response:
        payload = json.load(response)
    models = payload.get("models", []) if isinstance(payload, dict) else []
    if not any(item.get("name") == model and item.get("digest") == digest for item in models):
        raise ValueError(f"Pinned Ollama embedding model is unavailable: {model}.")


def _validate_vectors(vectors: Any, count: int) -> None:
    if not isinstance(vectors, list) or len(vectors) != count:
        raise ValueError("Embedder returned an incomplete batch.")
    if count == 0:
        return
    if any(not isinstance(vector, list) or not vector for vector in vectors):
        raise ValueError("Embedder returned an empty vector.")
    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) != 1 or any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
        for vector in vectors for value in vector
    ):
        raise ValueError("Embedder returned invalid vector values or dimensions.")


class VectorRagAdapter:
    name = "vector-rag"

    def __init__(self, config: dict[str, Any], *, embed: Callable[[str, list[str], str], list[list[float]]] = _ollama_embed) -> None:
        profile = _profile(config, self.name)
        if set(profile) != {"baseline", "version", "chunk_events", "overlap_events", "embedding_model", "embedding_model_digest", "top_k"}:
            raise ValueError("Vector-RAG profile has unsupported fields.")
        if any(type(profile[key]) is not int or profile[key] <= 0 for key in ("chunk_events", "top_k")):
            raise ValueError("Vector-RAG chunk size and top-k must be positive integers.")
        if type(profile["overlap_events"]) is not int or not 0 <= profile["overlap_events"] < profile["chunk_events"]:
            raise ValueError("Vector-RAG overlap must be smaller than chunk size.")
        if not isinstance(profile["embedding_model"], str) or not profile["embedding_model"].strip():
            raise ValueError("Vector-RAG embedding model is required.")
        digest = profile["embedding_model_digest"]
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("Vector-RAG embedding model digest must be SHA-256.")
        self.profile = profile
        self.embed = embed

    def resources_for_unit_v3(self, unit: BenchmarkUnit, config: dict[str, Any],
                              run_context: FrameworkRunContext) -> tuple[OwnedResource, ...]:
        digest = hashlib.sha256(unit.unit_id.encode("utf-8")).hexdigest()[:24]
        locator = f"baseline/vector-rag/{digest}"
        return (OwnedResource(f"{run_context.run_id}:{unit.unit_id}:vector-index", "embedded-path", "primary", locator),)

    def cleanup_unit_v3(self, unit: BenchmarkUnit, resources: tuple[OwnedResource, ...],
                        config: dict[str, Any], run_context: FrameworkRunContext) -> dict[str, Any]:
        if resources != self.resources_for_unit_v3(unit, config, run_context):
            raise ValueError("Vector-RAG resource manifest mismatch.")
        run_context.run_dir.mkdir(parents=True, exist_ok=True)
        result = cleanup_owned_embedded_resource(resources[0], root=run_context.run_dir,
                                                 owned_resources=resources)
        return {"verified": True, "cleanup": result}

    def prepare_unit_v3(self, unit: BenchmarkUnit, events: tuple[MemoryEvent, ...],
                        config: dict[str, Any], run_context: FrameworkRunContext) -> PreparedMemoryUnit:
        _check_events(events)
        started = time.perf_counter()
        resources = self.resources_for_unit_v3(unit, config, run_context)
        directory = run_context.run_dir / resources[0].locator
        if directory.exists():
            raise ValueError("Vector-RAG index already exists for this unit.")
        chunks: list[dict[str, Any]] = []
        size = self.profile["chunk_events"]
        step = size - self.profile["overlap_events"]
        sessions: list[list[MemoryEvent]] = []
        for event in events:
            if not sessions or sessions[-1][-1].session_id != event.session_id:
                sessions.append([])
            sessions[-1].append(event)
        for session in sessions:
            for offset in range(0, len(session), step):
                group = session[offset:offset + size]
                if not group:
                    break
                content = "\n".join(
                    f"Session: {event.session_id}; turn: {event.sequence}; role: {event.role}\n{event.content}"
                    for event in group
                )
                chunks.append({"id": f"chunk:{len(chunks):08d}", "content": content,
                               "source_event_ids": [event.event_id for event in group],
                               "source_refs": sorted({ref for event in group for ref in event.source_refs}),
                               "occurred_at": group[-1].occurred_at})
                if offset + size >= len(session):
                    break
        base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        if self.embed is _ollama_embed:
            _verify_ollama_model(self.profile["embedding_model"], self.profile["embedding_model_digest"], base_url)
        vectors = self.embed(self.profile["embedding_model"], [chunk["content"] for chunk in chunks], base_url) if chunks else []
        _validate_vectors(vectors, len(chunks))
        directory.mkdir(parents=True)
        index_path = directory / "index.json"
        payload = {"chunks": chunks, "vectors": vectors, "model": self.profile["embedding_model"]}
        write_json_atomic(index_path, payload)
        digest = hashlib.sha256(index_path.read_bytes()).hexdigest()
        encoding = _encoding(config["context_budget"]["tokenizer"])
        embedding_tokens = sum(_token_count(encoding, chunk["content"]) for chunk in chunks)
        return PreparedMemoryUnit(
            handle={"unit_id": unit.unit_id, "chunks": chunks, "vectors": vectors,
                    "sha256": digest, "path": str(index_path)},
            resources=resources,
            ingestion_usage={"embedding_input_count": len(chunks),
                             "embedding_input_tokens_harness": embedding_tokens},
            ingestion_timing={"ingestion_ms": (time.perf_counter() - started) * 1000},
            diagnostics={"chunk_count": len(chunks), "index_bytes": index_path.stat().st_size,
                         "index_sha256": digest,
                         "embedding_model_digest": self.profile["embedding_model_digest"]},
        )

    def verify_prepared_v3(self, unit: BenchmarkUnit, prepared: PreparedMemoryUnit,
                           config: dict[str, Any], run_context: FrameworkRunContext) -> dict[str, Any]:
        if prepared.resources != self.resources_for_unit_v3(unit, config, run_context):
            raise ValueError("Vector-RAG resource manifest mismatch.")
        handle = prepared.handle
        if handle["unit_id"] != unit.unit_id or not Path(handle["path"]).is_file():
            raise ValueError("Vector-RAG index is missing or belongs to another unit.")
        if hashlib.sha256(Path(handle["path"]).read_bytes()).hexdigest() != handle["sha256"]:
            raise ValueError("Vector-RAG index digest mismatch.")
        return {"verified": True, "chunk_count": len(handle["chunks"]),
                "index_sha256": handle["sha256"]}

    def retrieve_v3(self, unit: BenchmarkUnit, query: MemoryQuery, prepared: PreparedMemoryUnit,
                    config: dict[str, Any], run_context: FrameworkRunContext) -> CanonicalRetrievalResult:
        _check_query(unit, query)
        self.verify_prepared_v3(unit, prepared, config, run_context)
        started = time.perf_counter()
        handle = prepared.handle
        if not handle["chunks"]:
            return CanonicalRetrievalResult((), diagnostics={"baseline_family": self.name})
        base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        if self.embed is _ollama_embed:
            _verify_ollama_model(self.profile["embedding_model"], self.profile["embedding_model_digest"], base_url)
        query_vectors = self.embed(self.profile["embedding_model"], [query.text], base_url)
        _validate_vectors(query_vectors, 1)
        query_vector = query_vectors[0]
        if len(query_vector) != len(handle["vectors"][0]):
            raise ValueError("Query and index embedding dimensions differ.")
        scored = sorted(
            ((-_cosine(query_vector, vector), chunk["id"], chunk) for chunk, vector in zip(handle["chunks"], handle["vectors"], strict=True)),
            key=lambda row: (row[0], row[1]),
        )
        limit = min(self.profile["top_k"], config["retrieval"]["max_results"])
        items = tuple(
            RetrievedMemory(
                memory_id=chunk["id"], content=chunk["content"], rank=rank,
                native_score=-negative_score, native_score_kind="cosine-similarity",
                occurred_at=chunk["occurred_at"],
                source_event_ids=tuple(chunk["source_event_ids"]),
                metadata={"source_refs": chunk["source_refs"]},
            )
            for rank, (negative_score, _, chunk) in enumerate(scored[:limit], 1)
        )
        return CanonicalRetrievalResult(
            items, diagnostics={"baseline_family": self.name, "indexed_chunks": len(scored)},
            usage={"embedding_input_count": 1,
                   "embedding_input_tokens_harness": _token_count(
                       _encoding(config["context_budget"]["tokenizer"]), query.text,
                   )},
            timing={"retrieval_pipeline_ms": (time.perf_counter() - started) * 1000},
        )


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norms = math.sqrt(sum(a * a for a in left) * sum(b * b for b in right))
    return dot / norms if norms else 0.0
