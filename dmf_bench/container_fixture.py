"""Explicit deterministic container harness for lifecycle certification.

This module is selected only by the dedicated internal application environment
used by the Compose fixture. It certifies Docker stop/resume without a remote
provider or implicit model download while still entering through ``dmf-bench``
and using Qdrant Server.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any, Callable

from qdrant_client import QdrantClient, models

from dmf_bench.adapters.base import (
    AnswererRequest,
    BenchmarkUnit,
    CanonicalRetrievalResult,
    FrameworkRunContext,
    JudgeRequest,
    MemoryEvent,
    MemoryQuery,
    OwnedResource,
    PreparedMemoryUnit,
    RetrievedMemory,
    RetrievalResult,
)
from dmf_bench.adapters.qdrant_lifecycle import (
    CollectionRole,
    CleanupManifest,
    QdrantLifecycleManager,
    QDRANT_COLLECTION_NAMESPACE_V3,
    build_cleanup_manifest,
)
from dmf_bench.cli import main as cli_main
from dmf_bench.fingerprints import judge_fingerprint
from dmf_bench.logging_config import JsonEventLogger
from dmf_bench.metrics import BenchmarkMetrics
from dmf_bench.runtime import (
    RuntimeApplication,
    RuntimeFactories,
    assemble_application,
    benchmark_factories,
)


FIXTURE_PROFILE = "docker-qdrant-fixture-v1"
VECTOR_SIZE = 4


class DeterministicQdrantFramework:
    """Minimal framework boundary that performs real Qdrant writes and reads."""

    def __init__(
        self,
        *,
        name: str,
        config: dict[str, Any],
        metrics: BenchmarkMetrics | None,
    ) -> None:
        _require_fixture_profile(config)
        storage = config.get("storage") if config.get("schema_version") == 3 else config.get("qdrant")
        endpoint_env = str((storage or {}).get("endpoint_env", "QDRANT_URL"))
        endpoint = os.getenv(endpoint_env)
        if not endpoint:
            raise ValueError(f"Container fixture requires {endpoint_env}.")
        timeout = float((storage or {}).get("request_timeout_seconds", 10))
        self.name = name
        self.metrics = metrics
        self.client = QdrantClient(
            url=endpoint,
            api_key=os.getenv("QDRANT_API_KEY") or None,
            timeout=timeout,
        )
        self.lifecycle = QdrantLifecycleManager(self.client)
        self._observe("health", self.lifecycle.check_ready)

    def resources_for_unit_v3(
        self, unit: BenchmarkUnit, _config: dict[str, Any],
        run_context: FrameworkRunContext,
    ) -> tuple[OwnedResource, ...]:
        manifest = self._manifest_v3(unit, run_context)
        return tuple(
            OwnedResource(item.name, "qdrant-collection", item.role.value, item.name)
            for item in manifest.collections
        )

    def cleanup_unit_v3(
        self, unit: BenchmarkUnit, resources: tuple[OwnedResource, ...],
        config: dict[str, Any], run_context: FrameworkRunContext,
    ) -> dict[str, Any]:
        if resources != self.resources_for_unit_v3(unit, config, run_context):
            raise ValueError("Container fixture refuses cleanup outside the owned manifest.")
        manifest = self._manifest_v3(unit, run_context)
        self._observe("delete_collection", lambda: self.lifecycle.delete_and_wait(manifest))
        if self.client.collection_exists(manifest.collections[0].name):
            raise RuntimeError("Container fixture collection remains after cleanup.")
        return {"verified": True, "resources": [item.to_dict() for item in resources]}

    def prepare_unit_v3(
        self, unit: BenchmarkUnit, events: tuple[MemoryEvent, ...],
        config: dict[str, Any], run_context: FrameworkRunContext,
    ) -> PreparedMemoryUnit:
        if not events or any(not isinstance(event, MemoryEvent) for event in events):
            raise ValueError("Container fixture requires normalized memory events.")
        manifest = self._manifest_v3(unit, run_context)
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{run_context.run_id}:{self.name}:{unit.unit_id}:v3"))
        first = events[0]
        self.lifecycle.assert_absent(manifest)
        try:
            self._observe("create_collection", lambda: self.lifecycle.create_collections(manifest))
            self._observe("upsert", lambda: self.client.upsert(
                collection_name=manifest.collections[0].name,
                wait=True,
                points=[models.PointStruct(
                    id=point_id, vector=[1.0, 0.0, 0.0, 0.0],
                    payload={
                        "run_id": run_context.run_id,
                        "unit_id": unit.unit_id,
                        "content": first.content,
                        "source_event_ids": [first.event_id],
                        "source_refs": list(first.source_refs),
                    },
                )],
            ))
        except Exception:
            self.lifecycle.delete_and_wait(manifest)
            raise
        return PreparedMemoryUnit(
            handle={"manifest": manifest, "point_id": point_id},
            resources=self.resources_for_unit_v3(unit, config, run_context),
            ingestion_usage={}, ingestion_timing={},
            diagnostics={"ingested_event_count": len(events)},
        )

    def verify_prepared_v3(
        self, unit: BenchmarkUnit, prepared: PreparedMemoryUnit,
        config: dict[str, Any], run_context: FrameworkRunContext,
    ) -> dict[str, Any]:
        if prepared.resources != self.resources_for_unit_v3(unit, config, run_context):
            raise ValueError("Container fixture prepared manifest mismatch.")
        handle = prepared.handle
        manifest = self._manifest_v3(unit, run_context)
        self._observe("count", lambda: self.lifecycle.verify_counts(
            manifest, minimum_count_by_role={CollectionRole.PRIMARY: 1},
        ))
        self._assert_qdrant_point(
            {"collection_name": manifest.collections[0].name,
             "point_id": handle["point_id"]}, run_context,
        )
        return {"verified": True, "point_id": handle["point_id"]}

    def retrieve_v3(
        self, unit: BenchmarkUnit, query: MemoryQuery,
        prepared: PreparedMemoryUnit, config: dict[str, Any],
        run_context: FrameworkRunContext,
    ) -> CanonicalRetrievalResult:
        if query.query_id not in unit.item_ids:
            raise ValueError("Container fixture query is outside its unit.")
        handle = prepared.handle
        manifest = self._manifest_v3(unit, run_context)
        records = self._observe("retrieve", lambda: self.client.retrieve(
            collection_name=manifest.collections[0].name,
            ids=[handle["point_id"]], with_payload=True, with_vectors=False,
        ))
        if len(records) != 1 or (records[0].payload or {}).get("run_id") != run_context.run_id:
            raise RuntimeError("Container fixture could not read its Qdrant point.")
        payload = records[0].payload or {}
        item = RetrievedMemory(
            memory_id=str(records[0].id), content=str(payload["content"]), rank=1,
            source_event_ids=tuple(payload["source_event_ids"]),
            metadata={"source_refs": list(payload["source_refs"])},
        )
        return CanonicalRetrievalResult(
            items=(item,), diagnostics={"fixture_qdrant_roundtrip": True},
        )

    def _manifest_v3(
        self, unit: BenchmarkUnit, run_context: FrameworkRunContext,
    ) -> CleanupManifest:
        return build_cleanup_manifest(
            run_hash=run_context.scientific_fingerprint,
            framework=self.name, unit_id=unit.unit_id,
            roles=(CollectionRole.PRIMARY,), vector_size=VECTOR_SIZE,
            namespace=QDRANT_COLLECTION_NAMESPACE_V3,
        )

    def cleanup_unit(
        self,
        unit: BenchmarkUnit,
        _item: dict[str, Any],
        _config: dict[str, Any],
        *,
        run_context: FrameworkRunContext,
    ) -> None:
        manifest = self._manifest(unit, run_context)
        self._observe("delete_collection", lambda: self.lifecycle.delete_and_wait(manifest))

    def prepare_unit(
        self,
        unit: BenchmarkUnit,
        _item: dict[str, Any],
        _config: dict[str, Any],
        *,
        run_context: FrameworkRunContext,
    ) -> dict[str, Any]:
        manifest = self._manifest(unit, run_context)
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{run_context.run_id}:{self.name}:{unit.unit_id}"))
        self.lifecycle.assert_absent(manifest)
        try:
            self._observe("create_collection", lambda: self.lifecycle.create_collections(manifest))
            self._observe(
                "upsert",
                lambda: self.client.upsert(
                    collection_name=manifest.collections[0].name,
                    wait=True,
                    points=[
                        models.PointStruct(
                            id=point_id,
                            vector=[1.0, 0.0, 0.0, 0.0],
                            payload={
                                "run_id": run_context.run_id,
                                "framework": self.name,
                                "unit_id": unit.unit_id,
                            },
                        )
                    ],
                ),
            )
            self._observe(
                "count",
                lambda: self.lifecycle.verify_counts(
                    manifest,
                    minimum_count_by_role={CollectionRole.PRIMARY: 1},
                ),
            )
        except Exception:
            self.lifecycle.delete_and_wait(manifest)
            raise
        return {
            "qdrant_commit_barrier": True,
            "collection_name": manifest.collections[0].name,
            "point_id": point_id,
            "cleanup_manifest": manifest.to_dict(),
        }

    def retrieve(
        self,
        _unit: BenchmarkUnit,
        question: dict[str, Any],
        _config: dict[str, Any],
        prepared: dict[str, Any],
        *,
        run_context: FrameworkRunContext,
    ) -> RetrievalResult:
        self._assert_qdrant_point(prepared, run_context)
        source_ids = [str(item) for item in question.get("answer_session_ids", [])]
        return self._retrieval_result(source_ids, prepared)

    def retrieve_question(
        self,
        _unit: BenchmarkUnit,
        _conversation: dict[str, Any],
        question: Any,
        _config: dict[str, Any],
        prepared: dict[str, Any],
        *,
        run_context: FrameworkRunContext,
    ) -> RetrievalResult:
        self._assert_qdrant_point(prepared, run_context)
        source_ids = [str(item) for item in question.qa_item.get("evidence", [])]
        return self._retrieval_result(source_ids, prepared)

    def _manifest(
        self,
        unit: BenchmarkUnit,
        run_context: FrameworkRunContext,
    ) -> CleanupManifest:
        return build_cleanup_manifest(
            run_hash=run_context.scientific_fingerprint,
            framework=self.name,
            unit_id=unit.unit_id,
            roles=(CollectionRole.PRIMARY,),
            vector_size=VECTOR_SIZE,
        )

    def _assert_qdrant_point(
        self,
        prepared: dict[str, Any],
        run_context: FrameworkRunContext,
    ) -> None:
        records = self._observe(
            "retrieve",
            lambda: self.client.retrieve(
                collection_name=str(prepared["collection_name"]),
                ids=[str(prepared["point_id"])],
                with_payload=True,
                with_vectors=False,
            ),
        )
        if len(records) != 1 or (records[0].payload or {}).get("run_id") != run_context.run_id:
            raise RuntimeError("Container fixture could not read its Qdrant point.")

    def _retrieval_result(
        self,
        source_ids: list[str],
        prepared: dict[str, Any],
    ) -> RetrievalResult:
        search_results = (
            {
                "memory": "deterministic Qdrant fixture hit",
                "metadata": {"source_unit_ids": source_ids},
            },
        )
        recall_diagnostics: dict[str, Any] = {"qdrant_roundtrip": True}
        if self.name == "dmf":
            recall_diagnostics.update(
                {
                    "diagnostics_available": True,
                    "diagnostic_source": "deterministic_qdrant_fixture",
                    "raw_stage_available": False,
                    "raw_candidates": [],
                    "ranked_candidates_canonical": list(search_results),
                    "final_candidates_canonical": list(search_results),
                }
            )
        return RetrievalResult(
            cutoff_label="fixture-qdrant",
            search_results=search_results,
            native_context={
                "surface": "deterministic-qdrant-fixture",
                "collection_name": prepared["collection_name"],
                "source_unit_ids": source_ids,
            },
            native_surface_diagnostics={"result_count": 1},
            recall_diagnostics=recall_diagnostics,
            memories_evaluated=1,
        )

    def _observe(self, operation: str, callback: Callable[[], Any]) -> Any:
        started_at = time.perf_counter()
        try:
            result = callback()
        except Exception:
            if self.metrics is not None:
                self.metrics.record_qdrant_operation(
                    operation=operation,
                    outcome="failed",
                    seconds=time.perf_counter() - started_at,
                )
            raise
        if self.metrics is not None:
            self.metrics.record_qdrant_operation(
                operation=operation,
                outcome="completed",
                seconds=time.perf_counter() - started_at,
            )
        return result


class DeterministicAnswerer:
    name = "docker-fixture-answerer"

    def __init__(self, config: dict[str, Any]) -> None:
        _require_fixture_profile(config)
        self.requested_model = str(
            (((config.get("models") or {}).get("answerer") or {}).get("requested_model"))
        )
        self.delay_seconds = _non_negative_float(
            os.getenv("DMF_BENCH_FIXTURE_ANSWER_DELAY_SECONDS", "0")
        )

    def generate(self, _request: AnswererRequest) -> dict[str, Any]:
        deadline = time.monotonic() + self.delay_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(0.25, remaining))
        return {
            "generated_answer": "deterministic fixture answer",
            "answerer_provider": "fixture",
            "answerer_requested_model": self.requested_model,
            "answerer_model": self.requested_model,
            "answerer_finish_reason": "stop",
            "answerer_usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }


class DeterministicJudge:
    name = "docker-fixture-judge"

    def __init__(self, config: dict[str, Any]) -> None:
        _require_fixture_profile(config)
        self.benchmark = str(config.get("benchmark", ""))
        self.requested_model = str(
            (((config.get("models") or {}).get("judge") or {}).get("requested_model"))
        )

    def judge(self, _request: JudgeRequest) -> dict[str, Any]:
        return {
            "judgment": "CORRECT",
            "score": 1.0,
            "reason": "deterministic offline container fixture",
            "judge_provider": "fixture",
            "judge_requested_model": self.requested_model,
            "judge_model": self.requested_model,
            "judge_finish_reason": "stop",
            "judge_usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "judge_fingerprint": judge_fingerprint(self.benchmark),
        }


def fixture_application_builder(
    config: dict[str, Any],
    *,
    metrics: BenchmarkMetrics,
    events: JsonEventLogger,
) -> RuntimeApplication:
    _require_fixture_profile(config)
    factories = RuntimeFactories(
        benchmarks=benchmark_factories(),
        frameworks={
            "dmf": lambda selected: DeterministicQdrantFramework(
                name="dmf", config=selected, metrics=metrics
            ),
            "mem0": lambda selected: DeterministicQdrantFramework(
                name="mem0", config=selected, metrics=metrics
            ),
        },
        answerers={"fixture": lambda selected: DeterministicAnswerer(selected)},
        judges={
            ("locomo", "fixture"): lambda selected: DeterministicJudge(selected),
            ("longmemeval", "fixture"): lambda selected: DeterministicJudge(selected),
        },
    )
    return assemble_application(
        config,
        metrics=metrics,
        factories=factories,
        events=events,
    )


def main(argv: list[str] | None = None) -> int:
    return cli_main(argv, application_builder=fixture_application_builder)


def _require_fixture_profile(config: dict[str, Any]) -> None:
    if config.get("schema_version") == 3:
        if config.get("scientific_profile") != FIXTURE_PROFILE:
            raise ValueError(f"Container fixture requires scientific_profile={FIXTURE_PROFILE!r}.")
        models = config.get("models") or {}
        providers = {str((models.get("answerer") or {}).get("provider", ""))}
        providers.update(str(judge.get("provider", "")) for judge in models.get("judges", []))
    else:
        runtime = config.get("runtime") or {}
        if runtime.get("execution_profile") != FIXTURE_PROFILE:
            raise ValueError(
                f"Container fixture requires runtime.execution_profile={FIXTURE_PROFILE!r}."
            )
        providers = {
            str(((config.get("models") or {}).get(role) or {}).get("provider", ""))
            for role in ("answerer", "judge")
        }
    if providers != {"fixture"}:
        raise ValueError("Container fixture requires explicit fixture answerer and judge providers.")


def _non_negative_float(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError("DMF_BENCH_FIXTURE_ANSWER_DELAY_SECONDS must be numeric.") from exc
    if result < 0:
        raise ValueError("DMF_BENCH_FIXTURE_ANSWER_DELAY_SECONDS cannot be negative.")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
