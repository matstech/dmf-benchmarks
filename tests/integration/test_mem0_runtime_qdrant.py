from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from dmf_bench.adapters.base import (
    AnswererRequest,
    FrameworkRunContext,
    JudgeRequest,
)
from dmf_bench.benchmarks.locomo.adapter import LoCoMoAdapter
from dmf_bench.benchmarks.longmemeval.adapter import LongMemEvalAdapter
from dmf_bench.adapters.mem0 import (
    DefaultMem0EngineBuilder,
    Mem0EngineBundle,
    Mem0QdrantFrameworkAdapter,
)
from dmf_bench.artifacts import LocalArtifactStore
from dmf_bench.atomic_io import read_json
from dmf_bench.contracts import sha256_file
from dmf_bench.evaluation.finalizer import OfflineLifecycleFinalizer
from dmf_bench.fingerprints import judge_fingerprint
from dmf_bench.runner import LoCoMoPredictOnlyRunner, LongMemEvalPredictOnlyRunner
from dmf_bench.state import load_manifest


FIXTURE_DIR = Path(__file__).parents[1] / "fixtures"
CONFIG_DIR = Path(__file__).parents[2] / "config"


class DeterministicMem0Embedder:
    def __init__(self, vector_size: int = 768) -> None:
        self.vector_size = vector_size

    def embed(self, text: str, memory_action: str | None = None) -> list[float]:
        del text, memory_action
        vector = [0.0] * self.vector_size
        vector[0] = 1.0
        return vector

    def embed_batch(
        self,
        texts: list[str],
        memory_action: str | None = None,
    ) -> list[list[float]]:
        return [self.embed(text, memory_action) for text in texts]


class DeterministicMem0Llm:
    def __init__(self) -> None:
        self.calls = 0
        self.collector: Any | None = None

    def generate_response(self, **kwargs: Any) -> str:
        del kwargs
        self.calls += 1
        if self.collector is not None:
            self.collector.record_usage(
                prompt_tokens=5,
                completion_tokens=2,
                total_tokens=7,
                calls=1,
            )
        return json.dumps(
            {"memory": [{"text": f"deterministic memory {self.calls}"}]}
        )


class TrackingMem0EngineBuilder:
    def __init__(self) -> None:
        self.build_count = 0
        self.llms: list[DeterministicMem0Llm] = []
        self.delegate = DefaultMem0EngineBuilder(memory_factory=self._memory_factory)

    def _memory_factory(self, runtime_config: dict[str, Any]) -> Any:
        from mem0 import Memory
        from mem0.utils.factory import EmbedderFactory, LlmFactory

        llm = DeterministicMem0Llm()
        with (
            patch.object(
                EmbedderFactory,
                "create",
                return_value=DeterministicMem0Embedder(),
            ),
            patch.object(LlmFactory, "create", return_value=llm),
        ):
            memory = Memory.from_config(runtime_config)
        llm.collector = memory._llm_usage
        self.llms.append(llm)
        return memory

    def build(self, **kwargs: Any) -> Mem0EngineBundle:
        self.build_count += 1
        bundle = self.delegate.build(**kwargs)
        # The production fork uses FastEmbed BM25. Disable only that optional
        # sparse encoder in this offline gate to forbid an implicit model download.
        bundle.memory.vector_store._bm25_encoder = False
        bundle.memory._entity_store._bm25_encoder = False
        return bundle


class FakeAnswerer:
    name = "offline-fixture-answerer"

    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.calls: list[AnswererRequest] = []

    def generate(self, request: AnswererRequest) -> dict[str, Any]:
        self.calls.append(request)
        return {
            "generated_answer": self.answers.pop(0),
            "answerer_provider": "offline-fixture",
            "answerer_requested_model": "deterministic-answerer",
            "answerer_model": "deterministic-answerer",
            "answerer_finish_reason": "stop",
            "answerer_usage": {
                "prompt_tokens_total": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }


class FakeJudge:
    name = "offline-fixture-judge"

    def judge(self, request: JudgeRequest) -> dict[str, Any]:
        return {
            "judgment": "CORRECT",
            "score": 1.0,
            "reason": "deterministic Qdrant integration fixture",
            "judge_provider": "offline-fixture",
            "judge_requested_model": "unused-judge",
            "judge_model": "unused-judge",
            "judge_finish_reason": "stop",
            "judge_usage": {"total_tokens": 0},
            "judge_fingerprint": judge_fingerprint(
                str(request.metadata["benchmark"])
            ),
        }


def experiment_config(
    tmp_path: Path,
    *,
    benchmark: str,
) -> dict[str, Any]:
    dataset_path = FIXTURE_DIR / f"{benchmark}-mini.json"
    framework_path = CONFIG_DIR / f"{benchmark}_mem0_qdrant_settings.yaml"
    experiment_id = f"integration-{benchmark}-mem0-{uuid.uuid4().hex[:10]}"
    selection = (
        {
            "ordered_item_ids": ["conversation-0001"],
            "filters": {"categories": [1, 2]},
            "seed": 7,
        }
        if benchmark == "locomo"
        else {
            "ordered_item_ids": ["lme-001"],
            "filters": {},
            "seed": 7,
        }
    )
    return {
        "schema_version": 2,
        "experiment_id": experiment_id,
        "benchmark": benchmark,
        "framework": "mem0",
        "runtime": {
            "root": str(tmp_path),
            "runs_dir": str(tmp_path / "runs"),
            "cache_dir": str(tmp_path / "cache"),
            "metrics_port": 9464,
            "log_level": "INFO",
        },
        "framework_config": {
            "path": str(framework_path),
            "sha256": sha256_file(framework_path),
            "format": "yaml",
            "profile": "qdrant-integration",
        },
        "qdrant": {
            "endpoint_env": "DMF_BENCH_TEST_QDRANT_URL",
            "retention": "keep",
            "request_timeout_seconds": 10,
        },
        "dataset": {
            "name": benchmark,
            "source": "fixture",
            "revision": "mini",
            "sha256": sha256_file(dataset_path),
            "path": str(dataset_path),
        },
        "selection": selection,
        "models": {
            "answerer": {
                "provider": "offline-fixture",
                "endpoint_identity": "offline://fixture",
                "requested_model": "deterministic-answerer",
                "parameters": {"temperature": 0, "max_tokens": 128},
                "runtime": {
                    "timeout_seconds": 1,
                    "rpm": 1000,
                    "max_retries": 0,
                },
            },
            "judge": {
                "provider": "offline-fixture",
                "endpoint_identity": "offline://fixture",
                "requested_model": "unused-judge",
                "parameters": {"temperature": 0, "max_tokens": 128},
                "runtime": {
                    "timeout_seconds": 1,
                    "rpm": 1000,
                    "max_retries": 0,
                },
            },
        },
        "evaluation": {"required": [], "optional": []},
        "artifact_store": {
            "type": "local",
            "uri": str(tmp_path / "runs"),
        },
    }


def require_qdrant() -> str:
    endpoint = os.getenv("DMF_BENCH_TEST_QDRANT_URL")
    if not endpoint:
        pytest.skip("Set DMF_BENCH_TEST_QDRANT_URL to run Qdrant integration.")
    return endpoint


def cleanup_committed_unit(
    *,
    adapter: Mem0QdrantFrameworkAdapter,
    unit: Any,
    item: dict[str, Any],
    config: dict[str, Any],
    run_dir: Path,
) -> None:
    if not (run_dir / "run-manifest.json").is_file():
        return
    manifest = load_manifest(run_dir)
    adapter.cleanup_unit(
        unit,
        item,
        config,
        run_context=FrameworkRunContext(
            run_id=manifest.run_id,
            scientific_fingerprint=manifest.scientific_fingerprint,
            run_dir=run_dir,
        ),
    )


@pytest.mark.integration
@pytest.mark.parametrize("benchmark", ["locomo", "longmemeval"])
def test_mem0_predict_only_on_qdrant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    benchmark: str,
) -> None:
    require_qdrant()
    monkeypatch.setenv("MEM0_TELEMETRY", "false")
    from mem0.utils import spacy_models

    monkeypatch.setattr(spacy_models, "get_nlp_full", lambda: None)
    monkeypatch.setattr(spacy_models, "get_nlp_lemma", lambda: None)

    config = experiment_config(tmp_path, benchmark=benchmark)
    builder = TrackingMem0EngineBuilder()
    adapter = Mem0QdrantFrameworkAdapter.from_experiment(
        config,
        engine_builder=builder,
    )
    store = LocalArtifactStore(Path(config["runtime"]["runs_dir"]))
    if benchmark == "locomo":
        benchmark_adapter: Any = LoCoMoAdapter()
        unit = benchmark_adapter.enumerate_units(config)[0]
        _index, item, _questions = benchmark_adapter.selected_conversations_by_id(config)[
            unit.unit_id
        ]
        answerer = FakeAnswerer(["Pixel", "near the kitchen window"])
        runner: Any = LoCoMoPredictOnlyRunner(
            artifact_store=store,
            framework=adapter,
            answerer=answerer,
        )
    else:
        benchmark_adapter = LongMemEvalAdapter()
        unit = benchmark_adapter.enumerate_units(config)[0]
        item = benchmark_adapter.selected_questions_by_id(config)[unit.unit_id]
        answerer = FakeAnswerer(["almonds"])
        runner = LongMemEvalPredictOnlyRunner(
            artifact_store=store,
            framework=adapter,
            answerer=answerer,
        )
    run_dir = Path(config["runtime"]["runs_dir"]) / config["experiment_id"]

    try:
        result = runner.run(config)

        assert result.committed_unit_ids == (unit.unit_id,)
        assert builder.build_count == 1
        cleanup_manifest = read_json(
            run_dir / "items" / unit.unit_id / "cleanup-manifest.json"
        )
        counts = {
            collection["role"]: int(
                adapter.qdrant_client.count(collection["name"], exact=True).count
            )
            for collection in cleanup_manifest["collections"]
        }
        assert counts["primary"] == (3 if benchmark == "locomo" else 2)
        assert counts["entities"] == 0
        history_path = Path(cleanup_manifest["local_paths"][0])
        assert history_path.is_file()

        if benchmark == "locomo":
            predictions = read_json(
                run_dir / "items" / unit.unit_id / "predictions.json"
            )["predictions"]
        else:
            predictions = [
                read_json(run_dir / "items" / unit.unit_id / "prediction.json")
            ]
        assert all(prediction["schema_version"] == 2 for prediction in predictions)
        assert all("protocol" not in prediction for prediction in predictions)
        assert all(
            prediction["memory_internal_usage"]["available"] is True
            for prediction in predictions
        )
        assert all(
            prediction["retrieval"]["search_results"]
            for prediction in predictions
        )
        assert all(prediction["native"]["native_context"] for prediction in predictions)
        assert all(
            prediction["native"]["native_surface_diagnostics"]["surface_marker"]
            == "mem0_search_surface"
            for prediction in predictions
        )

        resumed_answerer = FakeAnswerer([])
        resumed_runner = (
            LoCoMoPredictOnlyRunner(
                artifact_store=store,
                framework=adapter,
                answerer=resumed_answerer,
            )
            if benchmark == "locomo"
            else LongMemEvalPredictOnlyRunner(
                artifact_store=store,
                framework=adapter,
                answerer=resumed_answerer,
            )
        )
        resumed = resumed_runner.run(config, resume=True)
        assert resumed.restarted_unit_ids == ()
        assert resumed_answerer.calls == []
        assert builder.build_count == 1

        finalized = OfflineLifecycleFinalizer(
            artifact_store=store,
            judge=FakeJudge(),
        ).finalize(config["experiment_id"])
        assert finalized.state == "COMPLETED"
        rigorous = read_json(run_dir / "evaluations" / "rigorous_report.json")
        ablation = read_json(run_dir / "evaluations" / "ablation_report.json")
        assert rigorous["status"] == "COMPLETED"
        assert rigorous["metrics"]["overall"]["recall_at_k"] > 0
        assert ablation["status"] == "NOT_APPLICABLE"
        assert (run_dir / "final" / "COMPLETED.json").is_file()
    finally:
        cleanup_committed_unit(
            adapter=adapter,
            unit=unit,
            item=item,
            config=config,
            run_dir=run_dir,
        )

    for collection in cleanup_manifest["collections"]:
        assert not adapter.qdrant_client.collection_exists(collection["name"])
    assert not history_path.exists()
