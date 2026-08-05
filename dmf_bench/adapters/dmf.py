"""Executable DMF framework runtime backed exclusively by Qdrant Server."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from dmf_bench.frameworks.dmf_context import build_dmf_native_context_surface
from dmf_bench.frameworks.mem0_runtime import empty_memory_internal_usage
from dmf_bench.metrics import BenchmarkMetrics
from dmf_bench.benchmarks.locomo import dataset as locomo_utils
from dmf_bench.benchmarks.locomo.adapter import LoCoMoQuestion
from longmemeval.utils import (
    pair_turns,
    parse_longmemeval_date,
    sort_sessions_chronologically,
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


REQUIRED_DMF_QDRANT_COMMIT = "5c4318e36120c60c1a4f4322b999f964cefa42d4"


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
        return DmfEngineBundle(
            pipeline=pipeline,
            scoring=ScoringEngine.from_dmf_config(config=dmf_config),
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
        _validate_qdrant_server_config(dmf_config)

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
                embedding_cache_dir=cache_root / "models" / "embeddings"
            ),
            metrics=metrics,
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

    def validate_runtime(self) -> None:
        try:
            from dmf.memory.ltm_hooks import QdrantLTMHook
            from dmf.memory.ltm_hooks.qdrant_client import QdrantConnectionMode
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "DMF Qdrant Server mode is unavailable. Pin dmf-memory[qdrant] "
                f"to commit {REQUIRED_DMF_QDRANT_COMMIT} or a later approved release."
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
            record_index, ingested_count = self._ingest(
                benchmark=benchmark,
                unit=unit,
                item=item,
                engine=engine,
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
        question: LoCoMoQuestion,
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
        # does not populate TemporalMemory's legacy recall diagnostics. Reading
        # them here would therefore expose empty or stale data.
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
    ) -> tuple[dict[str, dict[str, Any]], int]:
        if benchmark == "locomo":
            return _ingest_locomo(
                item,
                engine,
                conversation_idx=int(unit.metadata.get("conversation_idx", 0)),
            )
        return _ingest_longmemeval(item, engine)

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


def _ingest_locomo(
    conversation: dict[str, Any],
    engine: DmfEngineBundle,
    *,
    conversation_idx: int,
) -> tuple[dict[str, dict[str, Any]], int]:
    from dmf.runtime.pipeline import InteractionProvenance

    conversation_data = _mapping(conversation.get("conversation"), "conversation")
    record_index: dict[str, dict[str, Any]] = {}
    session_rows: list[tuple[float, str, str]] = []
    for session_key, session_value in conversation_data.items():
        if not session_key.startswith("session_") or session_key.endswith("_date_time"):
            continue
        if not isinstance(session_value, list):
            raise DmfRuntimeError(f"LoCoMo {session_key} must be a list.")
        time_str = _required_string(conversation_data, f"{session_key}_date_time")
        session_rows.append(
            (locomo_utils.parse_locomo_date(date_str=time_str), session_key, time_str)
        )
    session_rows.sort(key=lambda row: row[0])

    ingested_count = 0
    for current_ts, session_key, time_str in session_rows:
        for turn in conversation_data[session_key]:
            if not isinstance(turn, dict):
                raise DmfRuntimeError("LoCoMo turn must be an object.")
            text = locomo_utils.serialize_locomo_turn_for_dmf(turn)
            if not text:
                continue
            context_text = locomo_utils.render_locomo_turn_for_context(turn)
            report, vector = engine.pipeline.analyze_interaction_with_vector(
                text=text,
                is_system=False,
                provenance=InteractionProvenance(
                    role=str(turn.get("speaker", "")).lower()
                ),
            )
            dia_id = _required_string(turn, "dia_id")
            report.raw_metadata.update(
                {
                    "benchmark": "locomo",
                    "conversation_idx": conversation_idx,
                    "source_unit_type": "dia",
                    "source_unit_id": dia_id,
                    "session_key": session_key,
                    "session_datetime_raw": time_str,
                    "framework": "dmf",
                }
            )
            engine.scoring.calculate_score(report, text=text)
            entry = engine.memory_engine.add_interaction(text, report, vector)
            entry.timestamp = current_ts
            record_index[entry.record_id] = {
                "benchmark": "locomo",
                "conversation_idx": conversation_idx,
                "source_unit_type": "dia",
                "source_unit_id": dia_id,
                "source_unit_ids": [dia_id],
                "session_key": session_key,
                "session_datetime_raw": time_str,
                "speaker": str(turn.get("speaker", "")),
                "text": context_text,
                "analysis_text": text,
                "raw_text": str(turn.get("text", "") or ""),
                "query": str(turn.get("query", "") or ""),
                "blip_caption": str(turn.get("blip_caption", "") or ""),
            }
            ingested_count += 1
    return record_index, ingested_count


def _ingest_longmemeval(
    question: dict[str, Any],
    engine: DmfEngineBundle,
) -> tuple[dict[str, dict[str, Any]], int]:
    from dmf.runtime.pipeline import InteractionProvenance

    question_id = _required_string(question, "question_id")
    record_index: dict[str, dict[str, Any]] = {}
    ingested_count = 0
    for session_id, date_str, session in sort_sessions_chronologically(question):
        session_ts = parse_longmemeval_date(date_str)
        for pair in pair_turns(session):
            for message in pair:
                text = str(message.get("content", ""))
                role = str(message.get("role", ""))
                if not text.strip():
                    continue
                report, vector = engine.pipeline.analyze_interaction_with_vector(
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
                engine.scoring.calculate_score(report, text=text)
                entry = engine.memory_engine.add_interaction(text, report, vector)
                if session_ts is not None:
                    entry.timestamp = session_ts
                record_index[entry.record_id] = {
                    "benchmark": "longmemeval",
                    "question_id": question_id,
                    "source_unit_type": "session",
                    "source_unit_id": session_id,
                    "source_unit_ids": [session_id],
                    "session_id": session_id,
                    "session_date_raw": date_str,
                    "session_timestamp": session_ts,
                    "role": role,
                    "text": text,
                }
                ingested_count += 1
    return record_index, ingested_count


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
