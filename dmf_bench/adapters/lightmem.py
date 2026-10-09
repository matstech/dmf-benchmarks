"""Pinned LightMem retrieval adapter for an isolated Python 3.11 image."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit
from urllib.request import urlopen

from dmf_bench.adapters.base import (
    BenchmarkUnit, CanonicalRetrievalResult, FrameworkRunContext, MemoryEvent,
    MemoryQuery, OwnedResource, PreparedMemoryUnit, RetrievedMemory,
)
from dmf_bench.adapters.embedded_lifecycle import cleanup_owned_embedded_resource
from dmf_bench.atomic_io import write_json_atomic
from dmf_bench.contracts import hash_canonical_json


LIGHTMEM_COMMIT = "8449d574df6bae1bdf3314a1564da65e2f37e046"
LIGHTMEM_VERSION = "0.1.0"
_ALLOWED_OLLAMA_HOSTS = frozenset({"127.0.0.1", "localhost", "host.docker.internal"})


class LightMemRuntimeError(RuntimeError):
    """Raised when LightMem cannot satisfy the retrieval lifecycle contract."""


def _load_profile(config: dict[str, Any]) -> dict[str, Any]:
    path = Path(config["framework_config"]["path"])
    profile = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(profile, dict) or profile.get("framework") != "lightmem":
        raise LightMemRuntimeError("Expected a LightMem framework profile.")
    if profile.get("version") != "local-v1" or profile.get("source_commit") != LIGHTMEM_COMMIT:
        raise LightMemRuntimeError("LightMem profile source or schema does not match the pinned adapter.")
    if profile.get("manager_backend") != "openai-compatible-local-ollama":
        raise LightMemRuntimeError("LightMem requires the pinned local Ollama manager route.")
    if profile.get("pre_compress") is not False or profile.get("kv_cache") is not False:
        raise LightMemRuntimeError("The local LightMem profile disables optional compression and KV cache.")
    if profile.get("topic_segment") is not True:
        raise LightMemRuntimeError("The primary LightMem profile requires sensory topic segmentation.")
    for model in ("manager", "embedding", "segmenter"):
        if not isinstance(profile.get(model), dict):
            raise LightMemRuntimeError(f"LightMem {model} profile is missing.")
    update = profile.get("offline_update")
    if not isinstance(update, dict) or set(update) != {"top_k", "keep_top_n", "score_threshold", "max_workers"}:
        raise LightMemRuntimeError("LightMem offline update profile is incomplete.")
    for key in ("top_k", "keep_top_n", "max_workers"):
        if type(update[key]) is not int or update[key] < 1:
            raise LightMemRuntimeError(f"LightMem {key} must be a positive integer.")
    if update["max_workers"] != 1 or update["keep_top_n"] > update["top_k"]:
        raise LightMemRuntimeError("LightMem update must use one worker and a bounded queue.")
    if type(update["score_threshold"]) not in (int, float) or not 0 <= update["score_threshold"] <= 1:
        raise LightMemRuntimeError("LightMem update score threshold must be in [0, 1].")
    if type(profile.get("embedding_dimensions")) is not int or profile["embedding_dimensions"] < 1:
        raise LightMemRuntimeError("LightMem embedding dimensions must be positive.")
    return profile


def _ollama_url() -> str:
    base = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    parsed = urlsplit(base)
    if parsed.scheme != "http" or parsed.hostname not in _ALLOWED_OLLAMA_HOSTS or parsed.path not in ("", "/", "/v1"):
        raise LightMemRuntimeError("LightMem Ollama endpoint must be a local HTTP host.")
    return base.removesuffix("/v1")


def _verify_model_file(section: dict[str, Any], env_name: str) -> Path:
    raw = os.environ.get(env_name)
    if not raw:
        raise LightMemRuntimeError(f"{env_name} must point to a provisioned local model directory.")
    root = Path(raw).resolve()
    if not root.is_dir():
        raise LightMemRuntimeError(f"{env_name} is not a local model directory.")
    files = section.get("files")
    if not isinstance(files, dict) or not files or section.get("weights_file") not in files:
        raise LightMemRuntimeError("LightMem model file manifest is invalid.")
    for name, digest in files.items():
        if not isinstance(name, str):
            raise LightMemRuntimeError("LightMem model file pin is invalid.")
        relative = Path(name)
        if (relative.is_absolute() or ".." in relative.parts
                or not isinstance(digest, str) or len(digest) != 64):
            raise LightMemRuntimeError("LightMem model file pin is invalid.")
        path = root / relative
        if not path.is_file():
            raise LightMemRuntimeError(f"{env_name} is missing pinned model file {name}.")
        actual = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                actual.update(chunk)
        if actual.hexdigest() != digest:
            raise LightMemRuntimeError(f"{env_name} model file {name} does not match its pin.")
    return root


def _preflight(profile: dict[str, Any]) -> tuple[Path, Path, str]:
    if sys.version_info[:2] != (3, 11):
        raise LightMemRuntimeError("LightMem requires the dedicated Python 3.11 image.")
    try:
        version = metadata.version("lightmem")
    except metadata.PackageNotFoundError as exc:
        raise LightMemRuntimeError("Pinned LightMem package is missing from the dedicated image.") from exc
    if version != LIGHTMEM_VERSION or os.getenv("LIGHTMEM_SOURCE_COMMIT") != LIGHTMEM_COMMIT:
        raise LightMemRuntimeError("LightMem package version or source commit does not match the profile.")
    if os.getenv("OPENROUTER_API_KEY"):
        raise LightMemRuntimeError("OpenRouter environment variables are forbidden in the local LightMem image.")
    embedding = _verify_model_file(profile["embedding"], "LIGHTMEM_EMBEDDING_MODEL_PATH")
    segmenter = _verify_model_file(profile["segmenter"], "LIGHTMEM_SEGMENTER_MODEL_PATH")
    base = _ollama_url()
    with urlopen(base + "/api/tags", timeout=10) as response:
        models = json.load(response).get("models", [])
    expected = profile["manager"]
    if not any(item.get("name") == expected.get("model") and item.get("digest") == expected.get("digest") for item in models):
        raise LightMemRuntimeError("Pinned local Ollama model is unavailable.")
    return embedding, segmenter, base


def _build_engine(profile: dict[str, Any], root: Path, environment: tuple[Path, Path, str]) -> Any:
    from lightmem.memory.lightmem import LightMemory

    class LocalLightMemory(LightMemory):
        def __init__(self, config: Any) -> None:
            # Upstream passes self.compressor to the segmenter even when compression is disabled.
            self.compressor = None
            super().__init__(config)

    embedding, segmenter, ollama_base = environment
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    manager = profile["manager"]
    return LocalLightMemory.from_config({
        "pre_compress": False,
        "topic_segment": True,
        "precomp_topic_shared": False,
        "topic_segmenter": {"model_name": "llmlingua-2", "configs": {
            "model_name": str(segmenter), "device_map": "cpu", "buffer_len": 512,
            "model_config": {"attn_implementation": "eager"},
        }},
        "messages_use": "user_only", "metadata_generate": True, "text_summary": True,
        "extraction_mode": "flat", "index_strategy": "embedding", "retrieve_strategy": "embedding",
        "memory_manager": {"model_name": "openai", "configs": {
            "model": manager["model"], "api_key": "ollama", "openai_base_url": ollama_base + "/v1",
            "seed": manager["seed"], "temperature": 0, "top_p": 1,
            "max_tokens": manager["max_tokens"],
        }},
        "text_embedder": {"model_name": "huggingface", "configs": {
            "model": str(embedding), "embedding_dims": profile["embedding_dimensions"],
            "model_kwargs": {"device": "cpu", "local_files_only": True},
        }},
        "embedding_retriever": {"model_name": "qdrant", "configs": {
            "collection_name": "lightmem", "embedding_model_dims": profile["embedding_dimensions"],
            "path": str(root / "index"), "on_disk": True,
        }},
        "history_db_path": str(root / "history.db"), "update": "offline",
        "kv_cache": False, "kv_cache_path": str(root / "kv-cache"), "graph_mem": False,
        "summary_retriever": None,
        "logging": {"level": "WARNING", "console_enabled": False, "file_enabled": False},
    })


def _close_engine(engine: Any) -> None:
    if engine is None:
        return
    client = getattr(getattr(engine, "embedding_retriever", None), "client", None)
    close = getattr(client, "close", None)
    if callable(close):
        close()


def _entries(engine: Any) -> dict[str, dict[str, Any]]:
    rows = engine.embedding_retriever.get_all()
    if not isinstance(rows, list):
        raise LightMemRuntimeError("LightMem did not return a complete index listing.")
    entries = {str(row["id"]): row for row in rows}
    if len(entries) != len(rows) or any(not isinstance(row.get("payload"), dict) for row in rows):
        raise LightMemRuntimeError("LightMem index contains duplicate IDs or invalid payloads.")
    return entries


def _state_digest(entries: dict[str, dict[str, Any]]) -> str:
    return hash_canonical_json({key: entries[key] for key in sorted(entries)})


def _event_message(event: MemoryEvent) -> list[dict[str, str]]:
    timestamp = event.occurred_at or "2000-01-01T00:00:00Z"
    return [
        {"role": "user", "content": f"[{event.role}] {event.content}", "time_stamp": timestamp},
        {"role": "assistant", "content": "Acknowledged.", "time_stamp": timestamp},
    ]


def _state_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


@dataclass
class _Handle:
    unit_id: str
    engine: Any
    digest: str
    entry_count: int
    lineage: dict[str, str]


class LightMemAdapter:
    name = "lightmem"
    adapter_version = "lightmem-v1"

    def __init__(
        self, config: dict[str, Any], *,
        preflight: Callable[[dict[str, Any]], tuple[Path, Path, str]] = _preflight,
        engine_factory: Callable[[dict[str, Any], Path, tuple[Path, Path, str]], Any] = _build_engine,
    ) -> None:
        self.profile = _load_profile(config)
        self.environment = preflight(self.profile)
        self.engine_factory = engine_factory
        self._engines: dict[str, Any] = {}

    def resources_for_unit_v3(self, unit: BenchmarkUnit, config: dict[str, Any],
                              run_context: FrameworkRunContext) -> tuple[OwnedResource, ...]:
        fragment = hashlib.sha256(unit.unit_id.encode("utf-8")).hexdigest()[:24]
        locator = f"lightmem/{fragment}"
        return (OwnedResource(f"{run_context.run_id}:{unit.unit_id}:lightmem-state", "embedded-path", "primary", locator),)

    def cleanup_unit_v3(self, unit: BenchmarkUnit, resources: tuple[OwnedResource, ...],
                        config: dict[str, Any], run_context: FrameworkRunContext) -> dict[str, Any]:
        if resources != self.resources_for_unit_v3(unit, config, run_context):
            raise LightMemRuntimeError("LightMem cleanup manifest does not match the owned unit.")
        _close_engine(self._engines.pop(resources[0].resource_id, None))
        run_context.run_dir.mkdir(parents=True, exist_ok=True)
        evidence = cleanup_owned_embedded_resource(resources[0], root=run_context.run_dir,
                                                   owned_resources=resources)
        return {"verified": True, "cleanup": evidence}

    def prepare_unit_v3(self, unit: BenchmarkUnit, events: tuple[MemoryEvent, ...],
                        config: dict[str, Any], run_context: FrameworkRunContext) -> PreparedMemoryUnit:
        if not isinstance(events, tuple) or any(not isinstance(event, MemoryEvent) for event in events):
            raise LightMemRuntimeError("LightMem requires normalized memory events.")
        resources = self.resources_for_unit_v3(unit, config, run_context)
        root = run_context.run_dir / resources[0].locator
        if root.exists():
            raise LightMemRuntimeError("LightMem unit state already exists before ingestion.")
        root.mkdir(parents=True)
        engine: Any = None
        started = time.perf_counter()
        try:
            engine = self.engine_factory(self.profile, root, self.environment)
            lineage: dict[str, str] = {}
            before: set[str] = set()
            for event in events:
                engine.add_memory(_event_message(event), force_segment=True, force_extract=True)
                current = _entries(engine)
                new_ids = set(current) - before
                lineage.update({memory_id: event.event_id for memory_id in new_ids})
                before = set(current)
            indexed = _entries(engine)
            queue_started = time.perf_counter()
            update = self.profile["offline_update"]
            engine.construct_update_queue_all_entries(
                top_k=update["top_k"], keep_top_n=update["keep_top_n"], max_workers=1,
            )
            queued = _entries(engine)
            if set(queued) != set(indexed) or any("update_queue" not in row["payload"] for row in queued.values()):
                raise LightMemRuntimeError("LightMem update queue did not complete for every indexed entry.")
            candidates = {
                memory_id for memory_id in queued
                if any(
                    candidate.get("id") == memory_id and candidate.get("score", -1) >= update["score_threshold"]
                    for other in queued.values() for candidate in other["payload"]["update_queue"]
                )
            }
            update_calls_before = int(engine.get_token_statistics()["llm"]["update"]["calls"])
            update_method = engine.manager._call_update_llm
            actions: dict[str, tuple[str | None, str | None]] = {}

            def tracked_update(prompt: str, target: dict[str, Any], sources: list[dict[str, Any]]) -> Any:
                memory_id = str(target["id"])
                response = update_method(prompt, target, sources)
                if not isinstance(response, dict) or memory_id in actions:
                    raise LightMemRuntimeError("LightMem update returned an invalid or duplicate action.")
                actions[memory_id] = (response.get("action"), response.get("new_memory"))
                return response

            engine.manager._call_update_llm = tracked_update
            try:
                engine.offline_update_all_entries(score_threshold=update["score_threshold"], max_workers=1)
            finally:
                engine.manager._call_update_llm = update_method
            usage = engine.get_token_statistics()
            update_calls = int(usage["llm"]["update"]["calls"]) - update_calls_before
            if update_calls != len(candidates) or set(actions) != candidates:
                raise LightMemRuntimeError("LightMem offline updates did not finish for all candidates.")
            final = _entries(engine)
            for memory_id, (action, new_memory) in actions.items():
                if action == "delete":
                    committed = memory_id not in final
                elif action == "update":
                    committed = (isinstance(new_memory, str) and bool(new_memory.strip())
                                 and memory_id in final and final[memory_id]["payload"].get("memory") == new_memory)
                elif action == "ignore":
                    committed = (memory_id in final and final[memory_id]["payload"].get("memory")
                                 == queued[memory_id]["payload"].get("memory"))
                else:
                    committed = False
                if not committed:
                    raise LightMemRuntimeError("LightMem offline update action was not committed.")
            for memory_id in tuple(lineage):
                if memory_id not in final or (
                    memory_id in candidates and final[memory_id]["payload"].get("memory") != queued[memory_id]["payload"].get("memory")
                ):
                    lineage.pop(memory_id)
            update_ms = (time.perf_counter() - queue_started) * 1000
            digest = _state_digest(final)
            write_json_atomic(root / "lineage.json", {"source_event_ids": lineage, "state_sha256": digest})
            state_bytes = _state_bytes(root)
        except Exception:
            _close_engine(engine)
            self.cleanup_unit_v3(unit, resources, config, run_context)
            raise
        handle = _Handle(unit.unit_id, engine, digest, len(final), lineage)
        self._engines[resources[0].resource_id] = engine
        return PreparedMemoryUnit(
            handle, resources,
            ingestion_usage={"lightmem": usage},
            ingestion_timing={"total_ms": (time.perf_counter() - started) * 1000,
                              "offline_update_ms": update_ms},
            diagnostics={"event_count": len(events), "indexed_before_update": len(indexed),
                         "indexed_after_update": len(final), "offline_update_candidates": len(candidates),
                         "offline_update_calls": update_calls, "state_bytes": state_bytes,
                         "state_sha256": digest, "exact_lineage_count": len(lineage),
                         "source_commit": LIGHTMEM_COMMIT},
        )

    def verify_prepared_v3(self, unit: BenchmarkUnit, prepared: PreparedMemoryUnit,
                           config: dict[str, Any], run_context: FrameworkRunContext) -> dict[str, Any]:
        if prepared.resources != self.resources_for_unit_v3(unit, config, run_context):
            raise LightMemRuntimeError("LightMem prepared resource manifest mismatch.")
        handle = prepared.handle
        if not isinstance(handle, _Handle) or handle.unit_id != unit.unit_id:
            raise LightMemRuntimeError("LightMem unit handle mismatch.")
        current = _entries(handle.engine)
        if len(current) != handle.entry_count or _state_digest(current) != handle.digest:
            raise LightMemRuntimeError("LightMem committed index differs from the prepared snapshot.")
        root = run_context.run_dir / prepared.resources[0].locator
        manifest = json.loads((root / "lineage.json").read_text(encoding="utf-8"))
        if manifest != {"source_event_ids": handle.lineage, "state_sha256": handle.digest}:
            raise LightMemRuntimeError("LightMem lineage evidence is missing or changed.")
        return {"verified": True, "entry_count": len(current), "state_sha256": handle.digest,
                "offline_update_calls": prepared.diagnostics["offline_update_calls"]}

    def retrieve_v3(self, unit: BenchmarkUnit, query: MemoryQuery, prepared: PreparedMemoryUnit,
                    config: dict[str, Any], run_context: FrameworkRunContext) -> CanonicalRetrievalResult:
        if not isinstance(query, MemoryQuery) or query.query_id not in unit.item_ids:
            raise LightMemRuntimeError("LightMem query does not belong to the prepared unit.")
        self.verify_prepared_v3(unit, prepared, config, run_context)
        started = time.perf_counter()
        handle: _Handle = prepared.handle
        vector = handle.engine.text_embedder.embed(query.text)
        rows = handle.engine.embedding_retriever.search(
            query_vector=vector, limit=config["retrieval"]["max_results"], return_full=True,
        )
        if not isinstance(rows, list):
            raise LightMemRuntimeError("LightMem retrieval did not return a ranked list.")
        items: list[RetrievedMemory] = []
        for rank, row in enumerate(rows, 1):
            payload = row.get("payload")
            if not isinstance(payload, dict):
                raise LightMemRuntimeError("LightMem retrieval payload is missing.")
            memory_id = str(row["id"])
            content = payload.get("memory")
            if not isinstance(content, str) or not content.strip():
                raise LightMemRuntimeError("LightMem returned an empty memory.")
            timestamp = payload.get("time_stamp")
            occurred_at = None
            if timestamp:
                occurred_at = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00")).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
            source = handle.lineage.get(memory_id)
            items.append(RetrievedMemory(
                memory_id=memory_id, content=content, rank=rank,
                native_score=float(row["score"]), native_score_kind="qdrant-cosine-similarity",
                occurred_at=occurred_at, source_event_ids=(source,) if source else (),
                metadata={"topic_summary": payload.get("topic_summary") or "",
                          "lineage_status": "exact" if source else "unverified-after-update"},
            ))
        usage = handle.engine.get_token_statistics()
        return CanonicalRetrievalResult(
            tuple(items), raw_payload=rows,
            diagnostics={"lightmem": {"indexed_entries": handle.entry_count,
                                      "source_commit": LIGHTMEM_COMMIT}},
            usage={"lightmem": usage},
            timing={"retrieval_pipeline_ms": (time.perf_counter() - started) * 1000},
        )
