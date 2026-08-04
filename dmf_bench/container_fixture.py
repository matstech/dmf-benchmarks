"""Explicit deterministic container harness for lifecycle certification.

This module is never selected by the production CLI.  The dedicated Compose
override invokes it directly so Docker stop/resume can be certified without a
remote provider or implicit model download while still using Qdrant Server.
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
    FrameworkRunContext,
    JudgeRequest,
    RetrievalResult,
)
from dmf_bench.adapters.qdrant_lifecycle import (
    CollectionRole,
    CleanupManifest,
    QdrantLifecycleManager,
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
        endpoint_env = str((config.get("qdrant") or {}).get("endpoint_env", "QDRANT_URL"))
        endpoint = os.getenv(endpoint_env)
        if not endpoint:
            raise ValueError(f"Container fixture requires {endpoint_env}.")
        timeout = float((config.get("qdrant") or {}).get("request_timeout_seconds", 10))
        self.name = name
        self.metrics = metrics
        self.client = QdrantClient(
            url=endpoint,
            api_key=os.getenv("QDRANT_API_KEY") or None,
            timeout=timeout,
        )
        self.lifecycle = QdrantLifecycleManager(self.client)
        self._observe("health", self.lifecycle.check_ready)

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
