"""Executable DMF framework runtime backed exclusively by Qdrant Server."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Protocol

from dmf_bench.frameworks.dmf_context import build_dmf_native_context_surface
from dmf_bench.frameworks.mem0_runtime import empty_memory_internal_usage
from dmf_bench.metrics import BenchmarkMetrics

from .base import (
    BenchmarkUnit,
    CanonicalRetrievalResult,
    FrameworkCapability,
    FrameworkRunContext,
    MemoryEvent,
    MemoryQuery,
    OwnedResource,
    PreparedMemoryUnit,
    ProgressReporter,
    ProgressUpdate,
    RetrievedMemory,
    ResumeCapability,
    RetrievalResult,
)
from .qdrant_lifecycle import (
    CleanupManifest,
    CollectionRole,
    QdrantClientProtocol,
    QdrantLifecycleError,
    QdrantLifecycleManager,
    QDRANT_COLLECTION_NAMESPACE,
    QDRANT_COLLECTION_NAMESPACE_V3,
    build_cleanup_manifest,
    stable_hash,
)


class DmfRuntimeError(RuntimeError):
    """Raised when the executable DMF runtime violates an invariant."""


@dataclass(frozen=True)
class DmfEngineBundle:
    """One isolated DMF memory engine and its shared scientific components."""

    pipeline: Any
    scoring: Any
    memory_engine: Any
    embedding_engine: Any
    memory_api: Any


class DmfEngineBuilder(Protocol):
    def build(
        self,
        *,
        dmf_config: Any,
        cleanup_manifest: CleanupManifest,
        qdrant_client: QdrantClientProtocol,
        cards_path: Path,
    ) -> DmfEngineBundle:
        """Build one unit-isolated DMF engine without selecting a fallback backend."""


@dataclass
class DefaultDmfEngineBuilder:
    """Build the pinned DMF stack while allowing a deterministic test embedder."""

    embedding_cache_dir: Path
    embedding_factory: Callable[[Any], Any] | None = None
    ablation_profile: str | None = None

    def build(
        self,
        *,
        dmf_config: Any,
        cleanup_manifest: CleanupManifest,
        qdrant_client: QdrantClientProtocol,
        cards_path: Path,
    ) -> DmfEngineBundle:
        from dmf.analysis.embedding_engine import EmbeddingEngine
        from dmf.analysis.scoring_engine import ScoringEngine
        from dmf.memory.api import Memory
        from dmf.memory.ltm_hooks import QdrantLTMHook
        from dmf.memory.temporal_memory import TemporalMemory
        from dmf.runtime.pipeline import InteractionPipeline
        from dmf.utils.config import VectorConfig

        collections = {
            collection.role: collection
            for collection in cleanup_manifest.collections
        }
        primary = collections[CollectionRole.PRIMARY]
        cards = collections[CollectionRole.CARDS]
        vector_config = VectorConfig(
            model_name=dmf_config.nlp.model_name,
            vector_dim=dmf_config.nlp.vector_dim,
            cache_dir=str(self.embedding_cache_dir),
            window_size=dmf_config.capacity.window_size,
        )
        embedding_engine = (
            self.embedding_factory(vector_config)
            if self.embedding_factory is not None
            else EmbeddingEngine(vector_config)
        )
        pipeline = InteractionPipeline.from_dmf_config(
            dmf_config,
            analyze_system_prompt=False,
        )
        # One injected embedder owns ingestion, archival and queries. This is
        # required for deterministic tests and avoids loading the model twice.
        pipeline._embedding_engine = embedding_engine
        nlp_engine = pipeline._nlp_engine
        ltm_hook = QdrantLTMHook(
            collection_name=primary.name,
            distance_threshold=dmf_config.ltm.distance_threshold,
            vector_config=vector_config,
            embed_text=embedding_engine.get_embedding,
            cards_enabled=dmf_config.ltm.cards_enabled,
            cards_path=cards_path,
            cards_collection_name=cards.name,
            client=qdrant_client,
        )
        memory_engine = TemporalMemory.from_dmf_config(
            config=dmf_config,
            ltm_hook=ltm_hook,
            nlp_engine=nlp_engine,
        )
        scoring = ScoringEngine.from_dmf_config(config=dmf_config)
        if getattr(self, "ablation_profile", None) == "dmf-no-salience-scoring":
            from dmf_bench.ablations import NeutralScoringEngine

            scoring = NeutralScoringEngine()
        return DmfEngineBundle(
            pipeline=pipeline,
            scoring=scoring,
            memory_engine=memory_engine,
            embedding_engine=embedding_engine,
            memory_api=Memory.from_dmf_config(
                dmf_config,
                memory_engine,
                embedding_engine,
            ),
        )


@dataclass(frozen=True)
class DmfPreparedUnit:
    unit_id: str
    resource_namespace: str
    cleanup_manifest: CleanupManifest
    engine: DmfEngineBundle
    record_index: dict[str, dict[str, Any]]
    ingested_count: int
    collection_counts: dict[CollectionRole, int]


QdrantClientFactory = Callable[[str, str | None, float], QdrantClientProtocol]


@dataclass
class DmfQdrantFrameworkAdapter:
    """DMF ingestion/retrieval runtime shared by LoCoMo and LongMemEval."""

    vector_size: int
    qdrant_client: QdrantClientProtocol | None = None
    dmf_config: Any | None = None
    engine_builder: DmfEngineBuilder | None = None
    native_surface_builder: Callable[..., Any] = build_dmf_native_context_surface
    metrics: BenchmarkMetrics | None = None
    ablation_profile: str | None = None
    name: str = "dmf"
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
        engine_builder: DmfEngineBuilder | None = None,
        client_factory: QdrantClientFactory | None = None,
    ) -> "DmfQdrantFrameworkAdapter":
        from dmf.utils.config_loader import load_dmf_config

        framework_config = _mapping(config.get("framework_config"), "framework_config")
        config_path = Path(_required_string(framework_config, "path"))
        dmf_config = load_dmf_config(path=config_path)
        if "ablation" in config:
            from dmf_bench.ablations import ablation_identity

            ablation_identity(config)
        _validate_qdrant_server_config(dmf_config)

        storage = config.get("storage")
        if storage is not None:
            qdrant = _mapping(storage, "storage")
            if qdrant.get("kind") != "qdrant-server":
                raise DmfRuntimeError("DMF v3 requires storage.kind='qdrant-server'.")
        else:
            qdrant = _mapping(config.get("qdrant"), "qdrant")
        endpoint_env = _required_string(qdrant, "endpoint_env")
        endpoint = os.getenv(endpoint_env)
        if not endpoint or not endpoint.strip():
            raise ValueError(
                f"DMF Qdrant runtime requires {endpoint_env} in the environment."
            )
        timeout = float(qdrant.get("request_timeout_seconds", 10.0))
        api_key = os.getenv("QDRANT_API_KEY") or None
        factory = client_factory or _default_qdrant_client
        client = factory(endpoint.strip(), api_key, timeout)
        runtime = _mapping(config.get("runtime"), "runtime")
        cache_root = Path(_required_string(runtime, "cache_dir")).resolve()
        adapter = cls(
            vector_size=int(dmf_config.nlp.vector_dim),
            qdrant_client=client,
            dmf_config=dmf_config,
            engine_builder=engine_builder
            or DefaultDmfEngineBuilder(
                embedding_cache_dir=cache_root / "models" / "embeddings",
                ablation_profile=(framework_config["profile"] if "ablation" in config else None),
            ),
            metrics=metrics,
            ablation_profile=(framework_config["profile"] if "ablation" in config else None),
        )
        adapter.validate_runtime()
        adapter._observe_qdrant("health", adapter._lifecycle().check_ready)
        return adapter

    def resources_for_unit(self, run_hash: str, unit_id: str) -> CleanupManifest:
        return build_cleanup_manifest(
            run_hash=run_hash,
            framework=self.name,
            unit_id=unit_id,
            roles=(CollectionRole.PRIMARY, CollectionRole.CARDS),
            vector_size=self.vector_size,
        )

    def resources_for_unit_v3(
        self,
        unit: BenchmarkUnit,
        config: dict[str, Any],
        run_context: FrameworkRunContext,
    ) -> tuple[OwnedResource, ...]:
        manifest = self._manifest_for_context(unit, config, run_context)
        return _owned_resources(manifest)

    def cleanup_unit_v3(
        self,
        unit: BenchmarkUnit,
        resources: tuple[OwnedResource, ...],
        config: dict[str, Any],
        run_context: FrameworkRunContext,
    ) -> dict[str, Any]:
        expected = self.resources_for_unit_v3(unit, config, run_context)
        if resources != expected:
            raise DmfRuntimeError("Refusing cleanup of resources outside this run/unit manifest.")
        manifest = self._manifest_for_context(unit, config, run_context)
        self._observe_qdrant("delete_collection", lambda: self._lifecycle().delete_and_wait(manifest))
        self._delete_local_paths(manifest, config)
        if any(self._client().collection_exists(item.name) for item in manifest.collections):
            raise DmfRuntimeError("DMF collection remains after cleanup.")
        if any(Path(path).exists() for path in manifest.local_paths):
            raise DmfRuntimeError("DMF local resource remains after cleanup.")
        return {"verified": True, "resources": [resource.to_dict() for resource in resources]}

    def prepare_unit_v3(
        self,
        unit: BenchmarkUnit,
        events: tuple[MemoryEvent, ...],
        config: dict[str, Any],
        run_context: FrameworkRunContext,
    ) -> PreparedMemoryUnit:
        if not isinstance(events, tuple) or any(not isinstance(event, MemoryEvent) for event in events):
            raise DmfRuntimeError("DMF v3 requires normalized MemoryEvent values.")
        manifest = self._manifest_for_context(unit, config, run_context)
        resources = self.resources_for_unit_v3(unit, config, run_context)
        lifecycle = self._lifecycle()
        lifecycle.assert_absent(manifest)
        cards_path = Path(manifest.local_paths[0])
        started = time.perf_counter()
        try:
            self._observe_qdrant("create_collection", lambda: lifecycle.create_collections(manifest))
            engine = self._engine_builder().build(
                dmf_config=self._config(), cleanup_manifest=manifest,
                qdrant_client=self._client(), cards_path=cards_path,
            )
            record_index, ingested_count = _ingest_events_v3(
                events, engine, progress=run_context.report_progress,
            )
            counts = self._observe_qdrant("count", lambda: lifecycle.collection_counts(manifest))
            active_count = int(engine.memory_engine.size)
            primary_count = counts.get(CollectionRole.PRIMARY, 0)
            if active_count + primary_count != ingested_count:
                raise DmfRuntimeError("DMF v3 ingestion barrier count mismatch.")
        except Exception:
            lifecycle.delete_and_wait(manifest)
            self._delete_local_paths(manifest, config)
            raise
        handle = DmfPreparedUnit(
            unit_id=unit.unit_id,
            resource_namespace=self._resource_namespace(run_context),
            cleanup_manifest=manifest,
            engine=engine,
            record_index=record_index,
            ingested_count=ingested_count,
            collection_counts=counts,
        )
        return PreparedMemoryUnit(
            handle=handle,
            resources=resources,
            ingestion_usage={},
            ingestion_timing={"ingestion_ms": (time.perf_counter() - started) * 1000},
            diagnostics={"ingested_count": ingested_count, "active_count": active_count,
                         "collection_counts": {role.value: count for role, count in counts.items()}},
        )

    def verify_prepared_v3(
        self,
        unit: BenchmarkUnit,
        prepared: PreparedMemoryUnit,
        config: dict[str, Any],
        run_context: FrameworkRunContext,
    ) -> dict[str, Any]:
        handle = self._assert_prepared_v3(unit, prepared, config, run_context)
        counts = self._observe_qdrant(
            "count", lambda: self._lifecycle().collection_counts(handle.cleanup_manifest),
        )
        active_count = int(handle.engine.memory_engine.size)
        if active_count + counts.get(CollectionRole.PRIMARY, 0) != handle.ingested_count:
            raise DmfRuntimeError("DMF v3 commit barrier count mismatch.")
        return {"verified": True, "ingested_count": handle.ingested_count,
                "active_count": active_count,
                "collection_counts": {role.value: count for role, count in counts.items()}}

    def retrieve_v3(
        self,
        unit: BenchmarkUnit,
        query: MemoryQuery,
        prepared: PreparedMemoryUnit,
        config: dict[str, Any],
        run_context: FrameworkRunContext,
    ) -> CanonicalRetrievalResult:
        del config
        handle = self._assert_prepared_v3(unit, prepared, None, run_context)
        if query.query_id not in unit.item_ids:
            raise DmfRuntimeError("DMF query does not belong to prepared unit.")
        started = time.perf_counter()
        surface = self.native_surface_builder(
            memory=handle.engine.memory_api,
            query_text=query.text,
            record_index=handle.record_index,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        self._record_retrieval_metric("retrieve", elapsed_ms / 1000)
        raw_outputs = dict(surface.raw_retrieval_outputs)
        native_rows = raw_outputs.get("search_results", [])
        if not isinstance(native_rows, list):
            raise DmfRuntimeError("DMF search_results must be a list.")
        items = tuple(
            _canonical_dmf_memory(row, rank, handle.record_index)
            for rank, row in enumerate(native_rows, start=1)
        )
        return CanonicalRetrievalResult(
            items=items,
            raw_payload=raw_outputs,
            diagnostics={"dmf": {"surface_marker": surface.surface_marker,
                                  "result_count": surface.result_count}},
            usage={},
            timing={"retrieval_pipeline_ms": elapsed_ms},
        )

    def _assert_prepared_v3(
        self,
        unit: BenchmarkUnit,
        prepared: PreparedMemoryUnit,
        config: dict[str, Any] | None,
        run_context: FrameworkRunContext,
    ) -> DmfPreparedUnit:
        if not isinstance(prepared, PreparedMemoryUnit) or not isinstance(prepared.handle, DmfPreparedUnit):
            raise DmfRuntimeError("Missing DMF v3 prepared state.")
        handle = prepared.handle
        if handle.unit_id != unit.unit_id or handle.resource_namespace != self._resource_namespace(run_context):
            raise DmfRuntimeError("DMF prepared state belongs to another run/unit.")
        if config is not None and prepared.resources != self.resources_for_unit_v3(unit, config, run_context):
            raise DmfRuntimeError("DMF prepared resource manifest mismatch.")
        if _owned_resources(handle.cleanup_manifest) != prepared.resources:
            raise DmfRuntimeError("DMF prepared resource manifest mismatch.")
        return handle

    def validate_runtime(self) -> None:
        try:
            from dmf.memory.ltm_hooks import QdrantLTMHook
            from dmf.memory.ltm_hooks.qdrant_client import QdrantConnectionMode
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "DMF Qdrant Server mode is unavailable. Install dmf-memory with "
                "qdrant-client available in the runtime environment."
            ) from exc
        if QdrantLTMHook is None or QdrantConnectionMode.SERVER.value != "server":
            raise RuntimeError("DMF Qdrant Server mode has an incompatible API.")
        if self.dmf_config is not None:
            _validate_qdrant_server_config(self.dmf_config)
            if int(self.dmf_config.nlp.vector_dim) != self.vector_size:
                raise DmfRuntimeError(
                    "DMF vector size does not match the framework adapter."
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
            raise DmfRuntimeError(f"Unsupported DMF benchmark: {benchmark!r}.")
        manifest = self._manifest_for_context(unit, config, run_context)
        lifecycle = self._lifecycle()
        lifecycle.assert_absent(manifest)
        cards_path = Path(manifest.local_paths[0])

        try:
            run_context.report_progress(
                ProgressUpdate(
                    stage="memory_initialization",
                    label="Initializing memory system",
                    completed=0,
                    total=1,
                    item_label="step",
                )
            )
            self._observe_qdrant(
                "create_collection",
                lambda: lifecycle.create_collections(manifest),
            )
            engine = self._engine_builder().build(
                dmf_config=self._config(),
                cleanup_manifest=manifest,
                qdrant_client=self._client(),
                cards_path=cards_path,
            )
            run_context.report_progress(
                ProgressUpdate(
                    stage="memory_initialization",
                    label="Initializing memory system",
                    completed=1,
                    total=1,
                    item_label="step",
                )
            )
            record_index, ingested_count = self._ingest(
                benchmark=benchmark,
                unit=unit,
                item=item,
                engine=engine,
                progress=run_context.report_progress,
            )
            counts = self._observe_qdrant(
                "count",
                lambda: lifecycle.collection_counts(manifest),
            )
            primary_count = counts.get(CollectionRole.PRIMARY, 0)
            active_count = int(engine.memory_engine.size)
            if active_count + primary_count != ingested_count:
                raise QdrantLifecycleError(
                    "DMF ingestion barrier mismatch: "
                    f"ingested={ingested_count}, active={active_count}, "
                    f"archived={primary_count}."
                )
        except Exception:
            lifecycle.delete_and_wait(manifest)
            self._delete_local_paths(manifest, config)
            raise

        prepared = DmfPreparedUnit(
            unit_id=unit.unit_id,
            resource_namespace=self._resource_namespace(run_context),
            cleanup_manifest=manifest,
            engine=engine,
            record_index=record_index,
            ingested_count=ingested_count,
            collection_counts=counts,
        )
        return {
            "dmf_prepared_unit": prepared,
            "cleanup_manifest": manifest.to_dict(),
            "qdrant_commit_barrier": {
                "verified": True,
                "ingested_count": ingested_count,
                "active_count": active_count,
                "collection_counts": {
                    role.value: count for role, count in counts.items()
                },
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
            question_text=str(question.get("question", "")),
            config=config,
            prepared=self._prepared(prepared),
        )

    def retrieve_question(
        self,
        unit: BenchmarkUnit,
        _conversation: dict[str, Any],
        question: Any,
        config: dict[str, Any],
        prepared: dict[str, Any],
        *,
        run_context: FrameworkRunContext,
    ) -> RetrievalResult:
        self._assert_prepared(unit, prepared, run_context)
        return self._retrieve_question_text(
            question_text=str(question.qa_item.get("question", "")),
            config=config,
            prepared=self._prepared(prepared),
        )

    def _retrieve_question_text(
        self,
        *,
        question_text: str,
        config: dict[str, Any],
        prepared: DmfPreparedUnit,
    ) -> RetrievalResult:
        return self._retrieve_native(question_text, prepared)

    def _retrieve_native(
        self,
        question_text: str,
        prepared: DmfPreparedUnit,
    ) -> RetrievalResult:
        started = time.perf_counter()
        surface = self.native_surface_builder(
            memory=prepared.engine.memory_api,
            query_text=question_text,
            record_index=prepared.record_index,
        )
        elapsed_seconds = time.perf_counter() - started
        raw_outputs = dict(surface.raw_retrieval_outputs)
        search_results = list(raw_outputs.get("search_results", []))
        # ``Memory.retrieve`` owns a distinct structured retrieval stack and
        # does not populate TemporalMemory recall diagnostics for this path.
        # Reading them here would therefore expose empty or stale data.
        # The public native surface returns final, answerability-ranked
        # evidence. Project that auditable evidence into the canonical
        # ranked/final stages and state explicitly that the pre-rerank raw
        # stage is unavailable.
        canonical_diagnostics = {
            "diagnostics_available": True,
            "diagnostic_source": "dmf_structured_native_final_projection",
            "raw_stage_available": False,
            "raw_candidates": [],
            "ranked_candidates": list(
                raw_outputs.get("retrieved_evidence", [])
            ),
            "final_candidates": list(
                raw_outputs.get("retrieved_evidence", [])
            ),
            "suppressed": [],
            "ranked_candidates_canonical": search_results,
            "final_candidates_canonical": search_results,
            "context_metrics": dict(surface.context_metrics),
        }
        self._record_retrieval_metric("retrieve", elapsed_seconds)
        return RetrievalResult(
            cutoff_label="native",
            search_results=tuple(search_results),
            recall_diagnostics=canonical_diagnostics,
            native_context=surface.native_context,
            native_surface_diagnostics={
                "surface_marker": surface.surface_marker,
                "recalled_section_present": surface.recalled_section_present,
                "active_section_present": surface.active_section_present,
                "result_count": surface.result_count,
                "context_metrics": dict(surface.context_metrics),
                "raw_retrieval_outputs": raw_outputs,
            },
            memory_internal_usage=empty_memory_internal_usage(framework="dmf"),
            memories_evaluated=surface.result_count,
            timing={
                "retrieval_pipeline_ms": elapsed_seconds * 1000,
                "retrieval_pipeline_scope": "question",
                "backend_search_ms": elapsed_seconds * 1000,
                "backend_search_scope": "question",
            },
        )

    def _ingest(
        self,
        *,
        benchmark: str,
        unit: BenchmarkUnit,
        item: dict[str, Any],
        engine: DmfEngineBundle,
        progress: ProgressReporter,
    ) -> tuple[dict[str, dict[str, Any]], int]:
        from .legacy_v2_dmf import _ingest_locomo, _ingest_longmemeval

        if benchmark == "locomo":
            return _ingest_locomo(
                item,
                engine,
                conversation_idx=int(unit.metadata.get("conversation_idx", 0)),
                progress=progress,
            )
        return _ingest_longmemeval(item, engine, progress=progress)

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
        cards_path = (
            cache_root
            / "dmf-cards"
            / namespace[:16]
            / f"{stable_hash(unit.unit_id)}.jsonl"
        )
        return build_cleanup_manifest(
            run_hash=namespace,
            framework=self.name,
            unit_id=unit.unit_id,
            roles=(CollectionRole.PRIMARY, CollectionRole.CARDS),
            vector_size=self.vector_size,
            local_paths=(str(cards_path),),
            namespace=(
                QDRANT_COLLECTION_NAMESPACE_V3
                if config.get("schema_version") == 3
                else QDRANT_COLLECTION_NAMESPACE
            ),
        )

    @staticmethod
    def _resource_namespace(run_context: FrameworkRunContext) -> str:
        return stable_hash(
            f"{run_context.run_id}:{run_context.scientific_fingerprint}",
            length=64,
        )

    def _assert_prepared(
        self,
        unit: BenchmarkUnit,
        prepared: dict[str, Any],
        run_context: FrameworkRunContext,
    ) -> None:
        state = self._prepared(prepared)
        if state.unit_id != unit.unit_id:
            raise DmfRuntimeError("Prepared DMF unit does not match retrieval unit.")
        if state.resource_namespace != self._resource_namespace(run_context):
            raise DmfRuntimeError("Prepared DMF unit belongs to another run.")

    @staticmethod
    def _prepared(prepared: dict[str, Any]) -> DmfPreparedUnit:
        state = prepared.get("dmf_prepared_unit")
        if not isinstance(state, DmfPreparedUnit):
            raise DmfRuntimeError("Missing DMF prepared unit state.")
        return state

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
                raise DmfRuntimeError(
                    f"Refusing to clean DMF path outside runtime.cache_dir: {path}"
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
            raise DmfRuntimeError("DMF runtime has no Qdrant Server client.")
        return self.qdrant_client

    def _lifecycle(self) -> QdrantLifecycleManager:
        return QdrantLifecycleManager(self._client())

    def _config(self) -> Any:
        if self.dmf_config is None:
            raise DmfRuntimeError("DMF runtime has no loaded framework config.")
        return self.dmf_config

    def _engine_builder(self) -> DmfEngineBuilder:
        if self.engine_builder is None:
            raise DmfRuntimeError(
                "DMF runtime has no engine builder with an explicit cache path."
            )
        return self.engine_builder


def dmf_framework_factories(
    *,
    metrics: BenchmarkMetrics | None = None,
    engine_builder: DmfEngineBuilder | None = None,
    client_factory: QdrantClientFactory | None = None,
) -> dict[str, Callable[[dict[str, Any]], DmfQdrantFrameworkAdapter]]:
    """Return the explicit DMF runtime factory for runtime assembly."""

    def build(config: dict[str, Any]) -> DmfQdrantFrameworkAdapter:
        return DmfQdrantFrameworkAdapter.from_experiment(
            config,
            metrics=metrics,
            engine_builder=engine_builder,
            client_factory=client_factory,
        )

    return {"dmf": build}


def _owned_resources(manifest: CleanupManifest) -> tuple[OwnedResource, ...]:
    return tuple(
        OwnedResource(
            resource_id=collection.name,
            kind="qdrant-collection",
            role=collection.role.value,
            locator=collection.name,
        ) for collection in manifest.collections
    ) + tuple(
        OwnedResource(
            resource_id=stable_hash(path, length=32),
            kind="local-file",
            role="cards",
            locator=path,
        ) for path in manifest.local_paths
    )


def _ingest_events_v3(
    events: tuple[MemoryEvent, ...],
    engine: DmfEngineBundle,
    *,
    progress: ProgressReporter,
) -> tuple[dict[str, dict[str, Any]], int]:
    from dmf.runtime.pipeline import InteractionProvenance

    record_index: dict[str, dict[str, Any]] = {}
    progress(ProgressUpdate(
        stage="memory_ingestion", label="Ingesting source memory",
        completed=0, total=len(events), item_label="events",
    ))
    for offset, event in enumerate(events):
        report, vector = engine.pipeline.analyze_interaction_with_vector(
            text=event.content,
            is_system=event.role == "system",
            provenance=InteractionProvenance(role=event.role),
        )
        report.raw_metadata.update({
            "source_event_ids": [event.event_id],
            "source_refs": list(event.source_refs),
            "session_id": event.session_id,
            "event_sequence": event.sequence,
        })
        engine.scoring.calculate_score(report, text=event.content)
        entry = engine.memory_engine.add_interaction(event.content, report, vector)
        if event.occurred_at is not None:
            entry.timestamp = datetime.fromisoformat(
                event.occurred_at.replace("Z", "+00:00")
            ).timestamp()
        record_index[entry.record_id] = {
            "content": event.content,
            "source_event_ids": [event.event_id],
            "source_refs": list(event.source_refs),
            "session_id": event.session_id,
            "occurred_at": event.occurred_at,
        }
        progress(ProgressUpdate(
            stage="memory_ingestion", label="Ingesting source memory",
            completed=offset + 1, total=len(events), item_label="events",
        ))
    return record_index, len(events)


def _canonical_dmf_memory(
    row: Any,
    rank: int,
    record_index: dict[str, dict[str, Any]],
) -> RetrievedMemory:
    if not isinstance(row, dict):
        raise DmfRuntimeError("DMF retrieval result must be an object.")
    memory_id = str(row.get("id") or row.get("memory_id") or "")
    indexed = record_index.get(memory_id, {})
    metadata = row.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise DmfRuntimeError("DMF retrieval metadata must be an object.")
    content = str(row.get("memory") or row.get("text") or indexed.get("content") or "")
    raw_score = row.get("score")
    score = float(raw_score) if raw_score is not None else None
    record_ids = metadata.get("source_record_ids") or []
    if not isinstance(record_ids, (tuple, list)):
        raise DmfRuntimeError("DMF source_record_ids must be a sequence.")
    source_ids: list[str] = []
    for record_id in (memory_id, *record_ids):
        for event_id in record_index.get(str(record_id), {}).get("source_event_ids", []):
            if event_id not in source_ids:
                source_ids.append(event_id)
    if not source_ids:
        fallback = metadata.get("source_event_ids") or []
        if not isinstance(fallback, (tuple, list)):
            raise DmfRuntimeError("DMF source_event_ids must be a sequence.")
        source_ids = [str(value) for value in fallback]
    source_refs: list[str] = []
    for record_id in (memory_id, *record_ids):
        for source_ref in record_index.get(str(record_id), {}).get("source_refs", []):
            if source_ref not in source_refs:
                source_refs.append(str(source_ref))
    return RetrievedMemory(
        memory_id=memory_id,
        content=content,
        rank=rank,
        native_score=score,
        native_score_kind="dmf-native" if score is not None else None,
        occurred_at=indexed.get("occurred_at"),
        source_event_ids=tuple(str(value) for value in source_ids),
        metadata={"source_refs": source_refs},
    )


def _validate_qdrant_server_config(dmf_config: Any) -> None:
    if not bool(dmf_config.ltm.enabled):
        raise DmfRuntimeError("DMF runtime requires ltm.enabled=true.")
    if str(dmf_config.ltm.storage_type) != "qdrant":
        raise DmfRuntimeError(
            "DMF runtime requires ltm.storage_type='qdrant'; "
            "Chroma, file and null backends are forbidden."
        )
    if str(dmf_config.ltm.qdrant_mode) != "server":
        raise DmfRuntimeError(
            "DMF runtime requires ltm.qdrant_mode='server'; memory mode is forbidden."
        )
    if int(dmf_config.nlp.vector_dim) <= 0:
        raise DmfRuntimeError("DMF runtime requires a positive nlp.vector_dim.")


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
