from __future__ import annotations

import os
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from dmf_bench.adapters.base import (
    AnswererRequest,
    FrameworkRunContext,
    JudgeRequest,
)
from dmf_bench.adapters.dmf import (
    DefaultDmfEngineBuilder,
    DmfQdrantFrameworkAdapter,
)
from dmf_bench.adapters.locomo import LoCoMoAdapter
from dmf_bench.adapters.longmemeval import LongMemEvalAdapter
from dmf_bench.artifacts import LocalArtifactStore
from dmf_bench.atomic_io import read_json
from dmf_bench.contracts import sha256_file
from dmf_bench.evaluation.finalizer import OfflineLifecycleFinalizer
from dmf_bench.fingerprints import judge_fingerprint
from dmf_bench.runner import LoCoMoPredictOnlyRunner, LongMemEvalPredictOnlyRunner
from dmf_bench.state import load_manifest


FIXTURE_DIR = Path(__file__).parents[1] / "fixtures"
CONFIG_DIR = Path(__file__).parents[2] / "config"


class DeterministicEmbedding:
    """Small offline embedder with the exact vector shape required by DMF."""

    def __init__(self, vector_config: Any) -> None:
        self.vector_dim = int(vector_config.vector_dim)

    def get_embedding(self, text: str) -> np.ndarray:
        del text
        vector = np.zeros(self.vector_dim, dtype=np.float32)
        vector[0] = 1.0
        return vector


class TrackingDmfEngineBuilder:
    def __init__(self) -> None:
        self.delegate = DefaultDmfEngineBuilder(
            embedding_cache_dir=Path("/tmp/dmf-bench-deterministic-embeddings"),
            embedding_factory=DeterministicEmbedding
        )
        self.build_count = 0

    def build(self, **kwargs: Any) -> Any:
        self.build_count += 1
        dmf_config = kwargs["dmf_config"]
        # The tiny fixtures do not naturally exceed the scientific token
        # budget. Force deterministic eviction here so the integration gate
        # exercises real Qdrant writes and reads, not only collection setup.
        kwargs["dmf_config"] = replace(
            dmf_config,
            decay=replace(dmf_config.decay, hard_kill_threshold=2.0),
            capacity=replace(dmf_config.capacity, pruning_frequency_x=1),
        )
        return self.delegate.build(**kwargs)


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
            "answerer_model": "deterministic-answerer",
            "answerer_usage": {
                "prompt_tokens_total": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "finish_reason": "stop",
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


def assert_evaluation_reports(
    *,
    store: LocalArtifactStore,
    run_id: str,
    run_dir: Path,
    expect_retrieval_hits: bool = True,
) -> None:
    result = OfflineLifecycleFinalizer(
        artifact_store=store,
        judge=FakeJudge(),
    ).finalize(run_id)

    assert result.state == "COMPLETED"
    rigorous = read_json(run_dir / "evaluations" / "rigorous_report.json")
    ablation = read_json(run_dir / "evaluations" / "ablation_report.json")
    assert rigorous["status"] == "COMPLETED"
    recall_at_k = rigorous["metrics"]["overall"]["recall_at_k"]
    assert isinstance(recall_at_k, (int, float))
    if expect_retrieval_hits:
        assert recall_at_k > 0
    assert ablation["status"] == "COMPLETED"
    assert ablation["metrics"]["stats"]["questions_without_diagnostics"] == 0
    assert (run_dir / "final" / "COMPLETED.json").is_file()


def experiment_config(
    tmp_path: Path,
    *,
    benchmark: str,
    protocol: str,
) -> dict[str, Any]:
    dataset_path = FIXTURE_DIR / f"{benchmark}-mini.json"
    framework_path = CONFIG_DIR / f"{benchmark}_dmf_qdrant_settings.toml"
    experiment_id = f"integration-{benchmark}-{protocol}-{uuid.uuid4().hex[:10]}"
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
        "schema_version": 1,
        "experiment_id": experiment_id,
        "benchmark": benchmark,
        "framework": "dmf",
        "protocol": protocol,
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
            "format": "toml",
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


def assert_owned_collections_committed(
    adapter: DmfQdrantFrameworkAdapter,
    cleanup_manifest: dict[str, Any],
) -> dict[str, int]:
    client = adapter.qdrant_client
    assert client is not None
    counts: dict[str, int] = {}
    for collection in cleanup_manifest["collections"]:
        name = collection["name"]
        assert client.collection_exists(name)
        counts[collection["role"]] = int(client.count(name, exact=True).count)
    assert set(counts) == {"primary", "cards"}
    return counts


def cleanup_committed_unit(
    *,
    adapter: DmfQdrantFrameworkAdapter,
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
def test_locomo_dmf_predict_only_on_qdrant(
    tmp_path: Path,
) -> None:
    require_qdrant()
    config = experiment_config(tmp_path, benchmark="locomo", protocol="native")
    benchmark = LoCoMoAdapter()
    unit = benchmark.enumerate_units(config)[0]
    _index, conversation, _questions = benchmark.selected_conversations_by_id(config)[
        unit.unit_id
    ]
    builder = TrackingDmfEngineBuilder()
    adapter = DmfQdrantFrameworkAdapter.from_experiment(
        config,
        engine_builder=builder,
    )
    answerer = FakeAnswerer(["Pixel", "near the kitchen window"])
    run_dir = Path(config["runtime"]["runs_dir"]) / config["experiment_id"]
    store = LocalArtifactStore(Path(config["runtime"]["runs_dir"]))

    try:
        result = LoCoMoPredictOnlyRunner(
            artifact_store=store,
            framework=adapter,
            answerer=answerer,
        ).run(config)

        assert result.committed_unit_ids == ("conversation-0001",)
        assert builder.build_count == 1
        cleanup_manifest = read_json(
            run_dir / "items" / "conversation-0001" / "cleanup-manifest.json"
        )
        counts = assert_owned_collections_committed(adapter, cleanup_manifest)
        assert counts["primary"] == 3
        predictions = read_json(
            run_dir / "items" / "conversation-0001" / "predictions.json"
        )["predictions"]
        assert [prediction["protocol"] for prediction in predictions] == [
            "native",
            "native",
        ]
        assert all(item["retrieval"]["search_results"] for item in predictions)
        assert all(
            item["retrieval"]["recall_diagnostics"][
                "final_candidates_canonical"
            ]
            for item in predictions
        )
        assert all(item["native"]["native_context"] for item in predictions)
        assert all(
            item["native"]["native_surface_diagnostics"][
                "raw_retrieval_outputs"
            ]["search_results"]
            for item in predictions
        )

        resumed_answerer = FakeAnswerer([])
        resumed = LoCoMoPredictOnlyRunner(
            artifact_store=store,
            framework=adapter,
            answerer=resumed_answerer,
        ).run(config, resume=True)
        assert resumed.restarted_unit_ids == ()
        assert resumed_answerer.calls == []
        assert builder.build_count == 1
        assert_evaluation_reports(
            store=store,
            run_id=config["experiment_id"],
            run_dir=run_dir,
        )
    finally:
        cleanup_committed_unit(
            adapter=adapter,
            unit=unit,
            item=conversation,
            config=config,
            run_dir=run_dir,
        )

    for collection in cleanup_manifest["collections"]:
        assert not adapter.qdrant_client.collection_exists(collection["name"])


@pytest.mark.integration
def test_longmemeval_dmf_predict_only_on_qdrant(
    tmp_path: Path,
) -> None:
    require_qdrant()
    config = experiment_config(tmp_path, benchmark="longmemeval", protocol="native")
    benchmark = LongMemEvalAdapter()
    unit = benchmark.enumerate_units(config)[0]
    question = benchmark.selected_questions_by_id(config)[unit.unit_id]
    builder = TrackingDmfEngineBuilder()
    adapter = DmfQdrantFrameworkAdapter.from_experiment(
        config,
        engine_builder=builder,
    )
    answerer = FakeAnswerer(["almonds"])
    run_dir = Path(config["runtime"]["runs_dir"]) / config["experiment_id"]
    store = LocalArtifactStore(Path(config["runtime"]["runs_dir"]))

    try:
        result = LongMemEvalPredictOnlyRunner(
            artifact_store=store,
            framework=adapter,
            answerer=answerer,
        ).run(config)

        assert result.committed_unit_ids == ("lme-001",)
        assert builder.build_count == 1
        cleanup_manifest = read_json(
            run_dir / "items" / "lme-001" / "cleanup-manifest.json"
        )
        counts = assert_owned_collections_committed(adapter, cleanup_manifest)
        assert counts["primary"] == 4
        prediction = read_json(run_dir / "items" / "lme-001" / "prediction.json")
        assert prediction["protocol"] == "native"
        assert prediction["retrieval"]["search_results"] == []
        assert prediction["retrieval"]["recall_diagnostics"][
            "diagnostics_available"
        ] is True
        assert prediction["retrieval"]["recall_diagnostics"][
            "final_candidates_canonical"
        ] == []
        assert prediction["native"]["native_context"]
        raw_outputs = prediction["native"]["native_surface_diagnostics"][
            "raw_retrieval_outputs"
        ]
        assert raw_outputs["retrieval_stack"] == "dmf_structured_native"
        assert isinstance(raw_outputs["search_results"], list)

        resumed_answerer = FakeAnswerer([])
        resumed = LongMemEvalPredictOnlyRunner(
            artifact_store=store,
            framework=adapter,
            answerer=resumed_answerer,
        ).run(config, resume=True)
        assert resumed.restarted_unit_ids == ()
        assert resumed_answerer.calls == []
        assert builder.build_count == 1
        assert_evaluation_reports(
            store=store,
            run_id=config["experiment_id"],
            run_dir=run_dir,
            expect_retrieval_hits=False,
        )
    finally:
        cleanup_committed_unit(
            adapter=adapter,
            unit=unit,
            item=question,
            config=config,
            run_dir=run_dir,
        )

    for collection in cleanup_manifest["collections"]:
        assert not adapter.qdrant_client.collection_exists(collection["name"])
