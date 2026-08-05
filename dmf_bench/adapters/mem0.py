"""Executable Mem0 runtime backed exclusively by Qdrant Server."""

from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Protocol

from dmf_bench.frameworks.mem0_config import (
    Mem0Config,
    build_mem0_qdrant_server_runtime_config,
    load_mem0_config,
)
from dmf_bench.frameworks.mem0_runtime import (
    add_mem0_with_observation_timestamp,
    add_memory_internal_usage,
    empty_memory_internal_usage,
    normalize_mem0_search_response,
    subtract_memory_internal_usage,
)
from dmf_bench.frameworks.mem0_context import build_mem0_native_context_surface
from dmf_bench.metrics import BenchmarkMetrics
from dmf_bench.benchmarks.locomo import dataset as locomo_utils
from dmf_bench.benchmarks.locomo.adapter import LoCoMoQuestion
from longmemeval.utils import (
    normalize_longmemeval_haystack,
    render_longmemeval_pair_for_context,
    serialize_longmemeval_pair_for_mem0,
)

from .base import (
    BenchmarkUnit,
    FrameworkCapability,
    FrameworkRunContext,
    ResumeCapability,
    RetrievalResult,
)
from .qdrant_lifecycle import (
    CleanupManifest,
    CollectionRole,
    QdrantClientProtocol,
    QdrantLifecycleError,
    QdrantLifecycleManager,
    build_cleanup_manifest,
    stable_hash,
)


REQUIRED_MEM0_QDRANT_COMMIT = "8db3430d20f8b76cb7f80fb30df048321863392f"
REQUIRED_MEM0_VERSION = "2.0.2"
_DISABLED_BOOLEAN_VALUES = frozenset({"0", "false", "no", "off"})
_MEM0_USAGE_FIELDS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "calls",
)
_FORBIDDEN_VECTOR_RUNTIME_FIELDS = frozenset(
    {
        "api_key",
        "client",
        "collection_name",
        "host",
        "path",
        "port",
        "url",
    }
)
_SECRET_FIELD_NAMES = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "credentials",
        "password",
        "secret",
        "token",
    }
)


class Mem0RuntimeError(RuntimeError):
    """Raised when the executable Mem0 runtime violates an invariant."""


@dataclass(frozen=True)
class Mem0EngineBundle:
    """One unit-isolated Mem0 memory instance and its benchmark wrapper."""

    memory: Any
    backend: "Mem0RuntimeBackend"


class Mem0EngineBuilder(Protocol):
    def build(
        self,
        *,
        mem0_config: Mem0Config,
        cleanup_manifest: CleanupManifest,
        qdrant_client: QdrantClientProtocol,
        history_path: Path,
    ) -> Mem0EngineBundle:
        """Build one unit-isolated Mem0 engine without selecting a fallback."""


MemoryFactory = Callable[[dict[str, Any]], Any]


@dataclass
class DefaultMem0EngineBuilder:
    """Build the pinned Mem0 fork around one injected Qdrant Server client."""

    memory_factory: MemoryFactory | None = None

    def build(
        self,
        *,
        mem0_config: Mem0Config,
        cleanup_manifest: CleanupManifest,
        qdrant_client: QdrantClientProtocol,
        history_path: Path,
    ) -> Mem0EngineBundle:
        _require_mem0_telemetry_disabled()
        from mem0 import Memory
        from mem0.vector_stores.qdrant import Qdrant

        collections = {
            collection.role: collection
            for collection in cleanup_manifest.collections
        }
        primary = collections[CollectionRole.PRIMARY]
        entities = collections[CollectionRole.ENTITIES]
        runtime_config = build_mem0_qdrant_server_runtime_config(
            mem0_config,
            collection_name=primary.name,
            history_db_path=str(history_path),
            qdrant_client=qdrant_client,
        )
        memory = (
            self.memory_factory(runtime_config)
            if self.memory_factory is not None
            else Memory.from_config(runtime_config)
        )
        _validate_memory_surface(memory, primary.name)

        vector_config = _mapping(
            _mapping(mem0_config.memory_config.get("vector_store"), "vector_store").get(
                "config"
            ),
            "vector_store.config",
        )
        entity_store = Qdrant(
            collection_name=entities.name,
            embedding_model_dims=entities.vector_size,
            client=qdrant_client,
            on_disk=bool(vector_config.get("on_disk", False)),
        )
        # The fork otherwise derives `<primary>_entities` lazily. Injecting the
        # store here makes the independently owned cleanup manifest authoritative.
        memory._entity_store = entity_store
        memory.reset_llm_usage()
        return Mem0EngineBundle(
            memory=memory,
            backend=Mem0RuntimeBackend(memory),
        )


@dataclass(frozen=True)
class Mem0RuntimeBackend:
    """Narrow Mem0 surface used by ingestion and retrieval."""

    memory: Any

    def add(
        self,
        messages: list[dict[str, str]],
        *,
        user_id: str,
        timestamp: int | None,
        metadata: dict[str, Any],
    ) -> int:
        effective_metadata = dict(metadata)
        created_at = _timestamp_to_created_at(timestamp)
        if created_at is not None:
            effective_metadata.setdefault("created_at", created_at)
        usage_before = _read_mem0_usage(self.memory)
        response = add_mem0_with_observation_timestamp(
            self.memory,
            messages,
            timestamp=timestamp,
            user_id=user_id,
            metadata=effective_metadata,
        )
        usage_after = _read_mem0_usage(self.memory)
        usage_delta = subtract_memory_internal_usage(usage_after, usage_before)
        if usage_delta["calls"] < 1:
            raise Mem0RuntimeError(
                "Mem0 additive ingestion completed without recording its required "
                "LLM extraction call."
            )
        if not isinstance(response, dict) or not isinstance(response.get("results"), list):
            raise Mem0RuntimeError("Mem0 add returned an incompatible response.")
        results = response["results"]
        for result in results:
            if not isinstance(result, dict) or str(result.get("event", "")).upper() != "ADD":
                raise Mem0RuntimeError(
                    "Pinned Mem0 additive ingestion returned a non-ADD result."
                )
        return len(results)

    def search(
        self,
        query: str,
        *,
        user_id: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        return normalize_mem0_search_response(
            self.search_raw(query, user_id=user_id, top_k=top_k)
        )

    def search_raw(
        self,
        query: str,
        *,
        user_id: str,
        top_k: int,
    ) -> Any:
        return self.memory.search(
            query,
            top_k=top_k,
            filters={"user_id": user_id},
        )

    def get_usage(self) -> dict[str, Any]:
        return _read_mem0_usage(self.memory)


@dataclass(frozen=True)
class Mem0PreparedUnit:
    unit_id: str
    resource_namespace: str
    cleanup_manifest: CleanupManifest
    engine: Mem0EngineBundle
    user_id: str
    record_index: dict[str, dict[str, Any]]
    ingested_batches: int
    persisted_memory_count: int
    collection_counts: dict[CollectionRole, int]
    ingestion_usage: dict[str, Any]
    ingestion_usage_shares: dict[str, dict[str, Any]]


QdrantClientFactory = Callable[[str, str | None, float], QdrantClientProtocol]


@dataclass
class Mem0QdrantFrameworkAdapter:
    """Mem0 ingestion/retrieval runtime shared by LoCoMo and LongMemEval."""

    vector_size: int
    sqlite_root: Path = Path(".")
    qdrant_client: QdrantClientProtocol | None = None
    mem0_config: Mem0Config | None = None
    engine_builder: Mem0EngineBuilder = field(default_factory=DefaultMem0EngineBuilder)
    native_surface_builder: Callable[..., Any] = build_mem0_native_context_surface
    metrics: BenchmarkMetrics | None = None
    name: str = "mem0"
    resume_capability: ResumeCapability = ResumeCapability.RESTART_UNIT
    capabilities: frozenset[FrameworkCapability] = frozenset(
        {
            FrameworkCapability.NATIVE_SURFACE,
            FrameworkCapability.USAGE,
            FrameworkCapability.QDRANT_SERVER,
            FrameworkCapability.CLEANUP_MANIFEST,
        }
    )

    @classmethod
    def from_experiment(
        cls,
        config: dict[str, Any],
        *,
        metrics: BenchmarkMetrics | None = None,
        engine_builder: Mem0EngineBuilder | None = None,
        client_factory: QdrantClientFactory | None = None,
    ) -> "Mem0QdrantFrameworkAdapter":
        _require_mem0_telemetry_disabled()
        framework_config = _mapping(config.get("framework_config"), "framework_config")
        config_path = Path(_required_string(framework_config, "path"))
        mem0_config = load_mem0_config(config_path)
        vector_size = _validate_mem0_qdrant_config(mem0_config)

        qdrant = _mapping(config.get("qdrant"), "qdrant")
        endpoint_env = _required_string(qdrant, "endpoint_env")
        endpoint = os.getenv(endpoint_env)
        if not endpoint or not endpoint.strip():
            raise ValueError(
                f"Mem0 Qdrant runtime requires {endpoint_env} in the environment."
            )
        timeout = float(qdrant.get("request_timeout_seconds", 10.0))
        api_key = os.getenv("QDRANT_API_KEY") or None
        factory = client_factory or _default_qdrant_client
        client = factory(endpoint.strip(), api_key, timeout)
        cache_root = Path(
            _required_string(_mapping(config.get("runtime"), "runtime"), "cache_dir")
        ).resolve()
        adapter = cls(
            vector_size=vector_size,
            sqlite_root=cache_root / "mem0-history",
            qdrant_client=client,
            mem0_config=mem0_config,
            engine_builder=engine_builder or DefaultMem0EngineBuilder(),
            metrics=metrics,
        )
        adapter.validate_runtime()
        adapter._observe_qdrant("health", adapter._lifecycle().check_ready)
        return adapter

    def resources_for_unit(self, run_hash: str, unit_id: str) -> CleanupManifest:
        unit_fragment = unit_id.replace("/", "_")
        history_path = self.sqlite_root / f"{unit_fragment}.sqlite"
        return build_cleanup_manifest(
            run_hash=run_hash,
            framework=self.name,
            unit_id=unit_id,
            roles=(CollectionRole.PRIMARY, CollectionRole.ENTITIES),
            vector_size=self.vector_size,
            local_paths=_sqlite_owned_paths(history_path),
        )

    def validate_runtime(self) -> None:
        _require_mem0_telemetry_disabled()
        try:
            import mem0
            from mem0 import Memory
            from mem0.vector_stores.qdrant import Qdrant
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Mem0 Qdrant Server mode is unavailable. Pin the approved fork "
                f"to commit {REQUIRED_MEM0_QDRANT_COMMIT}."
            ) from exc
        try:
            installed_version = metadata.version("mem0ai")
        except metadata.PackageNotFoundError as exc:
            raise RuntimeError("The mem0ai distribution is unavailable.") from exc
        if installed_version != REQUIRED_MEM0_VERSION:
            raise RuntimeError(
                "Unsupported Mem0 distribution version: "
                f"expected {REQUIRED_MEM0_VERSION}, got {installed_version}."
            )
        if (
            not hasattr(mem0, "Memory")
            or not callable(getattr(Memory, "get_llm_usage", None))
            or not callable(getattr(Memory, "reset_llm_usage", None))
            or Qdrant is None
        ):
            raise RuntimeError("The pinned Mem0 runtime exposes an incompatible API.")
        if self.mem0_config is not None:
            configured_size = _validate_mem0_qdrant_config(self.mem0_config)
            if configured_size != self.vector_size:
                raise Mem0RuntimeError(
                    "Mem0 vector size does not match the framework adapter."
                )

    def cleanup_unit(
        self,
        unit: BenchmarkUnit,
        _item: dict[str, Any],
        config: dict[str, Any],
        *,
        run_context: FrameworkRunContext,
    ) -> None:
        manifest = self._manifest_for_context(unit, config, run_context)
        self._observe_qdrant(
            "delete_collection",
            lambda: self._lifecycle().delete_and_wait(manifest),
        )
        self._delete_local_paths(manifest, config)

    def prepare_unit(
        self,
        unit: BenchmarkUnit,
        item: dict[str, Any],
        config: dict[str, Any],
        *,
        run_context: FrameworkRunContext,
    ) -> dict[str, Any]:
        benchmark = _required_string(config, "benchmark")
        if benchmark not in {"locomo", "longmemeval"}:
            raise Mem0RuntimeError(f"Unsupported Mem0 benchmark: {benchmark!r}.")
        manifest = self._manifest_for_context(unit, config, run_context)
        lifecycle = self._lifecycle()
        lifecycle.assert_absent(manifest)
        self._assert_local_paths_absent(manifest)
        history_path = Path(manifest.local_paths[0])
        history_path.parent.mkdir(parents=True, exist_ok=True)
        user_id = self._user_id(benchmark, unit, run_context)

        try:
            engine = self._engine_builder().build(
                mem0_config=self._config(),
                cleanup_manifest=manifest,
                qdrant_client=self._client(),
                history_path=history_path,
            )
            record_index, ingested_batches, persisted_memory_count = self._ingest(
                benchmark=benchmark,
                unit=unit,
                item=item,
                backend=engine.backend,
                user_id=user_id,
            )
            counts = self._observe_qdrant(
                "count",
                lambda: lifecycle.collection_counts(manifest),
            )
            primary_count = counts.get(CollectionRole.PRIMARY, 0)
            if primary_count != persisted_memory_count:
                raise QdrantLifecycleError(
                    "Mem0 ingestion barrier mismatch: "
                    f"reported_adds={persisted_memory_count}, primary={primary_count}."
                )
            _verify_sqlite_history(history_path)
            ingestion_usage = engine.backend.get_usage()
            usage_shares = _apportion_usage_by_item(ingestion_usage, unit.item_ids)
        except Exception:
            lifecycle.delete_and_wait(manifest)
            self._delete_local_paths(manifest, config)
            raise

        prepared = Mem0PreparedUnit(
            unit_id=unit.unit_id,
            resource_namespace=self._resource_namespace(run_context),
            cleanup_manifest=manifest,
            engine=engine,
            user_id=user_id,
            record_index=record_index,
            ingested_batches=ingested_batches,
            persisted_memory_count=persisted_memory_count,
            collection_counts=counts,
            ingestion_usage=ingestion_usage,
            ingestion_usage_shares=usage_shares,
        )
        return {
            "mem0_prepared_unit": prepared,
            "cleanup_manifest": manifest.to_dict(),
            "qdrant_commit_barrier": {
                "verified": True,
                "ingested_batches": ingested_batches,
                "persisted_memory_count": persisted_memory_count,
                "collection_counts": {
                    role.value: count for role, count in counts.items()
                },
                "sqlite_history_integrity": "ok",
            },
        }

    def retrieve(
        self,
        unit: BenchmarkUnit,
        question: dict[str, Any],
        config: dict[str, Any],
        prepared: dict[str, Any],
        *,
        run_context: FrameworkRunContext,
    ) -> RetrievalResult:
        self._assert_prepared(unit, prepared, run_context)
        return self._retrieve_question_text(
            question_id=str(question.get("question_id", unit.unit_id)),
            question_text=str(question.get("question", "")),
            config=config,
            prepared=self._prepared(prepared),
        )

    def retrieve_question(
        self,
        unit: BenchmarkUnit,
        _conversation: dict[str, Any],
        question: LoCoMoQuestion,
        config: dict[str, Any],
        prepared: dict[str, Any],
        *,
        run_context: FrameworkRunContext,
    ) -> RetrievalResult:
        self._assert_prepared(unit, prepared, run_context)
        return self._retrieve_question_text(
            question_id=question.question_id,
            question_text=str(question.qa_item.get("question", "")),
            config=config,
            prepared=self._prepared(prepared),
        )

    def _retrieve_question_text(
        self,
        *,
        question_id: str,
        question_text: str,
        config: dict[str, Any],
        prepared: Mem0PreparedUnit,
    ) -> RetrievalResult:
        return self._retrieve_native(question_id, question_text, prepared)

    def _retrieve_native(
        self,
        question_id: str,
        question_text: str,
        prepared: Mem0PreparedUnit,
    ) -> RetrievalResult:
        baseline = prepared.engine.backend.get_usage()
        started = time.perf_counter()
        surface = self.native_surface_builder(
            mem0_backend=prepared.engine.backend,
            query_text=question_text,
            user_id=prepared.user_id,
            top_k=self._config().top_k,
        )
        elapsed_seconds = time.perf_counter() - started
        search_results = normalize_mem0_search_response(surface.raw_search_output)
        usage = self._question_usage(question_id, prepared, baseline)
        self._record_retrieval_metric("retrieve", elapsed_seconds)
        return RetrievalResult(
            cutoff_label="native",
            search_results=tuple(search_results),
            recall_diagnostics={
                "retrieval_stack": "mem0_search",
                "normalized_result_count": len(search_results),
            },
            native_context=surface.native_context,
            native_surface_diagnostics={
                "surface_marker": surface.surface_marker,
                "result_count": surface.result_count,
                "search_kwargs": dict(surface.search_kwargs),
                "raw_search_output": surface.raw_search_output,
            },
            memory_internal_usage=usage,
            memories_evaluated=surface.result_count,
            timing={
                "retrieval_pipeline_ms": elapsed_seconds * 1000,
                "retrieval_pipeline_scope": "question",
                "backend_search_ms": elapsed_seconds * 1000,
                "backend_search_scope": "question",
            },
        )

    def _question_usage(
        self,
        question_id: str,
        prepared: Mem0PreparedUnit,
        baseline: dict[str, Any],
    ) -> dict[str, Any]:
        current = prepared.engine.backend.get_usage()
        retrieval_delta = subtract_memory_internal_usage(current, baseline)
        ingestion_share = prepared.ingestion_usage_shares.get(
            question_id,
            empty_memory_internal_usage(available=True, framework="mem0"),
        )
        return add_memory_internal_usage(ingestion_share, retrieval_delta)

    def _ingest(
        self,
        *,
        benchmark: str,
        unit: BenchmarkUnit,
        item: dict[str, Any],
        backend: Mem0RuntimeBackend,
        user_id: str,
    ) -> tuple[dict[str, dict[str, Any]], int, int]:
        if benchmark == "locomo":
            return _ingest_locomo(
                item,
                backend,
                user_id=user_id,
                conversation_idx=int(unit.metadata.get("conversation_idx", 0)),
            )
        return _ingest_longmemeval(item, backend, user_id=user_id)

    def _manifest_for_context(
        self,
        unit: BenchmarkUnit,
        config: dict[str, Any],
        run_context: FrameworkRunContext,
    ) -> CleanupManifest:
        cache_root = Path(
            _required_string(_mapping(config.get("runtime"), "runtime"), "cache_dir")
        ).resolve()
        namespace = self._resource_namespace(run_context)
        history_path = (
            cache_root
            / "mem0-history"
            / namespace[:16]
            / f"{stable_hash(unit.unit_id)}.sqlite"
        )
        return build_cleanup_manifest(
            run_hash=namespace,
            framework=self.name,
            unit_id=unit.unit_id,
            roles=(CollectionRole.PRIMARY, CollectionRole.ENTITIES),
            vector_size=self.vector_size,
            local_paths=_sqlite_owned_paths(history_path),
        )

    @staticmethod
    def _resource_namespace(run_context: FrameworkRunContext) -> str:
        return stable_hash(
            f"{run_context.run_id}:{run_context.scientific_fingerprint}",
            length=64,
        )

    @staticmethod
    def _user_id(
        benchmark: str,
        unit: BenchmarkUnit,
        run_context: FrameworkRunContext,
    ) -> str:
        namespace = Mem0QdrantFrameworkAdapter._resource_namespace(run_context)
        return f"{benchmark}_{namespace[:16]}_{stable_hash(unit.unit_id)}"

    def _assert_prepared(
        self,
        unit: BenchmarkUnit,
        prepared: dict[str, Any],
        run_context: FrameworkRunContext,
    ) -> None:
        state = self._prepared(prepared)
        if state.unit_id != unit.unit_id:
            raise Mem0RuntimeError("Prepared Mem0 unit does not match retrieval unit.")
        if state.resource_namespace != self._resource_namespace(run_context):
            raise Mem0RuntimeError("Prepared Mem0 unit belongs to another run.")

    @staticmethod
    def _prepared(prepared: dict[str, Any]) -> Mem0PreparedUnit:
        state = prepared.get("mem0_prepared_unit")
        if not isinstance(state, Mem0PreparedUnit):
            raise Mem0RuntimeError("Missing Mem0 prepared unit state.")
        return state

    @staticmethod
    def _assert_local_paths_absent(manifest: CleanupManifest) -> None:
        existing = [path for path in manifest.local_paths if Path(path).exists()]
        if existing:
            raise Mem0RuntimeError(
                "Mem0 history resources already exist for unit: "
                + ", ".join(existing)
            )

    def _delete_local_paths(
        self,
        manifest: CleanupManifest,
        config: dict[str, Any],
    ) -> None:
        cache_root = Path(
            _required_string(_mapping(config.get("runtime"), "runtime"), "cache_dir")
        ).resolve()
        for raw_path in manifest.local_paths:
            path = Path(raw_path).resolve()
            if path != cache_root and cache_root not in path.parents:
                raise Mem0RuntimeError(
                    f"Refusing to clean Mem0 path outside runtime.cache_dir: {path}"
                )
            if path.exists():
                path.unlink()

    def _observe_qdrant(self, operation: str, callback: Callable[[], Any]) -> Any:
        started = time.perf_counter()
        try:
            result = callback()
        except Exception:
            if self.metrics is not None:
                self.metrics.record_qdrant_operation(
                    operation=operation,
                    outcome="failed",
                    seconds=time.perf_counter() - started,
                )
            raise
        if self.metrics is not None:
            self.metrics.record_qdrant_operation(
                operation=operation,
                outcome="completed",
                seconds=time.perf_counter() - started,
            )
        return result

    def _record_retrieval_metric(self, operation: str, seconds: float) -> None:
        if self.metrics is not None:
            self.metrics.record_qdrant_operation(
                operation=operation,
                outcome="completed",
                seconds=seconds,
            )

    def _client(self) -> QdrantClientProtocol:
        if self.qdrant_client is None:
            raise Mem0RuntimeError("Mem0 runtime has no Qdrant Server client.")
        return self.qdrant_client

    def _lifecycle(self) -> QdrantLifecycleManager:
        return QdrantLifecycleManager(self._client())

    def _config(self) -> Mem0Config:
        if self.mem0_config is None:
            raise Mem0RuntimeError("Mem0 runtime has no loaded framework config.")
        return self.mem0_config

    def _engine_builder(self) -> Mem0EngineBuilder:
        return self.engine_builder


def mem0_framework_factories(
    *,
    metrics: BenchmarkMetrics | None = None,
    engine_builder: Mem0EngineBuilder | None = None,
    client_factory: QdrantClientFactory | None = None,
) -> dict[str, Callable[[dict[str, Any]], Mem0QdrantFrameworkAdapter]]:
    """Return the explicit Mem0 runtime factory for runtime assembly."""

    def build(config: dict[str, Any]) -> Mem0QdrantFrameworkAdapter:
        return Mem0QdrantFrameworkAdapter.from_experiment(
            config,
            metrics=metrics,
            engine_builder=engine_builder,
            client_factory=client_factory,
        )

    return {"mem0": build}


def _ingest_locomo(
    conversation: dict[str, Any],
    backend: Mem0RuntimeBackend,
    *,
    user_id: str,
    conversation_idx: int,
) -> tuple[dict[str, dict[str, Any]], int, int]:
    conversation_data = _mapping(conversation.get("conversation"), "conversation")
    speaker_a = str(conversation_data.get("speaker_a", "") or "")
    record_index: dict[str, dict[str, Any]] = {}
    session_rows: list[tuple[float, str, str]] = []
    for session_key, session_value in conversation_data.items():
        if not session_key.startswith("session_") or session_key.endswith("_date_time"):
            continue
        if not isinstance(session_value, list):
            raise Mem0RuntimeError(f"LoCoMo {session_key} must be a list.")
        time_str = _required_string(conversation_data, f"{session_key}_date_time")
        session_rows.append(
            (locomo_utils.parse_locomo_date(date_str=time_str), session_key, time_str)
        )
    session_rows.sort(key=lambda row: row[0])

    ingested_batches = 0
    persisted_memory_count = 0
    for current_ts, session_key, time_str in session_rows:
        for turn in conversation_data[session_key]:
            if not isinstance(turn, dict):
                raise Mem0RuntimeError("LoCoMo turn must be an object.")
            ingest_text = locomo_utils.serialize_locomo_turn_for_mem0(turn)
            if not ingest_text:
                continue
            dia_id = _required_string(turn, "dia_id")
            role = "user" if str(turn.get("speaker", "")) == speaker_a else "assistant"
            persisted_memory_count += backend.add(
                [{"role": role, "content": ingest_text}],
                user_id=user_id,
                timestamp=int(current_ts),
                metadata={
                    "benchmark": "locomo",
                    "conversation_idx": conversation_idx,
                    "source_unit_type": "dia",
                    "source_unit_id": dia_id,
                    "framework": "mem0",
                },
            )
            ingested_batches += 1
            record_index[dia_id] = {
                "benchmark": "locomo",
                "conversation_idx": conversation_idx,
                "source_unit_type": "dia",
                "source_unit_id": dia_id,
                "source_unit_ids": [dia_id],
                "session_key": session_key,
                "session_datetime_raw": time_str,
                "speaker": str(turn.get("speaker", "")),
                "text": locomo_utils.render_locomo_turn_for_context(turn),
                "ingest_text": ingest_text,
                "raw_text": str(turn.get("text", "") or ""),
                "query": str(turn.get("query", "") or ""),
                "blip_caption": str(turn.get("blip_caption", "") or ""),
            }
    return record_index, ingested_batches, persisted_memory_count


def _ingest_longmemeval(
    question: dict[str, Any],
    backend: Mem0RuntimeBackend,
    *,
    user_id: str,
) -> tuple[dict[str, dict[str, Any]], int, int]:
    question_id = _required_string(question, "question_id")
    record_index: dict[str, dict[str, Any]] = {}
    ingested_batches = 0
    persisted_memory_count = 0
    for session in normalize_longmemeval_haystack(question):
        session_id = session["session_id"]
        session_ts = session["session_timestamp"]
        session_date_raw = session["session_date_raw"]
        metadata_payload = {
            "benchmark": "longmemeval",
            "question_id": question_id,
            "source_unit_type": "session",
            "source_unit_id": session_id,
        }
        for pair in session["pairs"]:
            messages = serialize_longmemeval_pair_for_mem0(pair)
            if not messages:
                continue
            persisted_memory_count += backend.add(
                messages,
                user_id=user_id,
                timestamp=session_ts,
                metadata=metadata_payload,
            )
            ingested_batches += 1
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
    return record_index, ingested_batches, persisted_memory_count


def _validate_mem0_qdrant_config(config: Mem0Config) -> int:
    memory_config = config.memory_config
    secret_paths = _find_secret_paths(memory_config)
    if secret_paths:
        raise Mem0RuntimeError(
            "Mem0 framework config contains inline secret fields: "
            + ", ".join(secret_paths)
        )
    vector_store = _mapping(memory_config.get("vector_store"), "vector_store")
    if vector_store.get("provider") != "qdrant":
        raise Mem0RuntimeError(
            "Mem0 runtime requires vector_store.provider='qdrant'; "
            "Chroma and embedded stores are forbidden."
        )
    vector_config = _mapping(vector_store.get("config"), "vector_store.config")
    forbidden = sorted(_FORBIDDEN_VECTOR_RUNTIME_FIELDS.intersection(vector_config))
    if forbidden:
        raise Mem0RuntimeError(
            "Mem0 static vector config contains runtime-only fields: "
            + ", ".join(forbidden)
        )
    unsupported = sorted(set(vector_config) - {"embedding_model_dims", "on_disk"})
    if unsupported:
        raise Mem0RuntimeError(
            "Mem0 static vector config contains unsupported fields: "
            + ", ".join(unsupported)
        )
    vector_size = _positive_integer(vector_config, "embedding_model_dims")
    if not isinstance(vector_config.get("on_disk", False), bool):
        raise Mem0RuntimeError("vector_store.config.on_disk must be a boolean.")
    embedder = _mapping(memory_config.get("embedder"), "embedder")
    embedder_config = _mapping(embedder.get("config"), "embedder.config")
    embedder_size = _positive_integer(embedder_config, "embedding_dims")
    if vector_size != embedder_size:
        raise Mem0RuntimeError(
            "Mem0 vector_store.config.embedding_model_dims must match "
            "embedder.config.embedding_dims."
        )
    if "history_db_path" in memory_config:
        raise Mem0RuntimeError(
            "Mem0 history_db_path is runtime-only and cannot be declared statically."
        )
    if str(memory_config.get("version", "")) != "v1.1":
        raise Mem0RuntimeError("Mem0 runtime requires config version='v1.1'.")
    return vector_size


def _find_secret_paths(value: Any, *, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            if key_text.lower() in _SECRET_FIELD_NAMES:
                paths.append(path)
            paths.extend(_find_secret_paths(nested, prefix=path))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            paths.extend(_find_secret_paths(nested, prefix=f"{prefix}[{index}]"))
    return paths


def _read_mem0_usage(memory: Any) -> dict[str, Any]:
    getter = getattr(memory, "get_llm_usage", None)
    if not callable(getter):
        raise Mem0RuntimeError("Pinned Mem0 usage sidecar is unavailable.")
    raw = getter()
    if not isinstance(raw, dict) or not isinstance(raw.get("scopes"), dict):
        raise Mem0RuntimeError("Pinned Mem0 usage sidecar returned an incompatible snapshot.")
    usage = empty_memory_internal_usage(available=True, framework="mem0")
    for field_name in _MEM0_USAGE_FIELDS:
        value = raw.get(field_name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise Mem0RuntimeError(
                f"Pinned Mem0 usage field {field_name!r} must be a non-negative integer."
            )
        usage[field_name] = value
    return usage


def _apportion_usage_by_item(
    usage: dict[str, Any],
    item_ids: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    if not item_ids:
        return {}
    shares = {
        item_id: empty_memory_internal_usage(available=True, framework="mem0")
        for item_id in item_ids
    }
    for field_name in _MEM0_USAGE_FIELDS:
        base, remainder = divmod(int(usage[field_name]), len(item_ids))
        for index, item_id in enumerate(item_ids):
            shares[item_id][field_name] = base + (1 if index < remainder else 0)
    return shares


def _require_mem0_telemetry_disabled() -> None:
    raw = os.getenv("MEM0_TELEMETRY")
    if raw is None or raw.strip().lower() not in _DISABLED_BOOLEAN_VALUES:
        raise Mem0RuntimeError(
            "Mem0 Qdrant runtime requires MEM0_TELEMETRY=false before process startup "
            "to prevent creation of an unowned local telemetry store."
        )
    try:
        from mem0.memory import main as memory_main
    except ModuleNotFoundError:
        return
    if bool(memory_main.MEM0_TELEMETRY):
        raise Mem0RuntimeError(
            "Mem0 was imported before MEM0_TELEMETRY=false took effect; restart the process."
        )


def _validate_memory_surface(memory: Any, expected_collection: str) -> None:
    for method_name in ("add", "search", "get_llm_usage", "reset_llm_usage"):
        if not callable(getattr(memory, method_name, None)):
            raise Mem0RuntimeError(
                f"Pinned Mem0 memory is missing required method {method_name!r}."
            )
    vector_store = getattr(memory, "vector_store", None)
    if vector_store is None or getattr(vector_store, "collection_name", None) != expected_collection:
        raise Mem0RuntimeError("Mem0 primary collection does not match the cleanup manifest.")
    if bool(getattr(vector_store, "is_local", True)):
        raise Mem0RuntimeError("Mem0 selected a local vector store instead of Qdrant Server.")


def _verify_sqlite_history(path: Path) -> None:
    if not path.is_file():
        raise Mem0RuntimeError(f"Mem0 history database was not created: {path}")
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
            row = connection.execute("PRAGMA integrity_check").fetchone()
    except sqlite3.Error as exc:
        raise Mem0RuntimeError(f"Mem0 history database cannot be verified: {exc}") from exc
    if not row or str(row[0]).lower() != "ok":
        raise Mem0RuntimeError(f"Mem0 history database integrity check failed: {row!r}")


def _sqlite_owned_paths(history_path: Path) -> tuple[str, ...]:
    raw = str(history_path)
    return (raw, f"{raw}-wal", f"{raw}-shm")


def _timestamp_to_created_at(timestamp: int | None) -> str | None:
    if timestamp is None:
        return None
    from datetime import datetime, timezone

    return datetime.fromtimestamp(int(timestamp), tz=timezone.utc).isoformat()


def _default_qdrant_client(
    endpoint: str,
    api_key: str | None,
    timeout: float,
) -> QdrantClientProtocol:
    from qdrant_client import QdrantClient

    return QdrantClient(url=endpoint, api_key=api_key, timeout=timeout)


def _mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object.")
    return value


def _required_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string.")
    return value.strip()


def _positive_integer(data: dict[str, Any], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise Mem0RuntimeError(f"{key} must be a positive integer.")
    return value
