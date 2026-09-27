"""Four offline v3 micro-fixtures using the real Qdrant server boundary."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

from dmf_bench.container_fixture import fixture_application_builder
from dmf_bench.adapters.dmf import DmfQdrantFrameworkAdapter
from dmf_bench.adapters.mem0 import Mem0QdrantFrameworkAdapter
from dmf_bench.artifacts import LocalArtifactStore
from dmf_bench.atomic_io import read_json
from dmf_bench.benchmarks.locomo.adapter import LoCoMoAdapter
from dmf_bench.benchmarks.longmemeval.adapter import LongMemEvalAdapter
from dmf_bench.context import PACKING_ID, RENDERER_ID, TOKENIZER_ID
from dmf_bench.contracts import sha256_file
from dmf_bench.evaluation.finalizer import OfflineLifecycleFinalizer
from dmf_bench.fingerprints import judge_fingerprint
from dmf_bench.logging_config import JsonEventLogger
from dmf_bench.metrics import BenchmarkMetrics
from dmf_bench.retrieval_qa import RetrievalQAPredictOnlyRunner
from dmf_bench.state import plan_resume
from tests.integration.test_dmf_runtime_qdrant import TrackingDmfEngineBuilder
from tests.integration.test_mem0_runtime_qdrant import TrackingMem0EngineBuilder


FIXTURES = Path(__file__).parents[1] / "fixtures"
CONFIG_DIR = Path(__file__).parents[2] / "config"


def _config(tmp_path: Path, benchmark: str, framework: str) -> dict:
    dataset = tmp_path / f"{benchmark}.json"
    dataset.write_bytes((FIXTURES / f"{benchmark}-mini.json").read_bytes())
    suffix = "toml" if framework == "dmf" else "yaml"
    framework_file = tmp_path / f"{framework}.{suffix}"
    framework_file.write_text("fixture = true\n", encoding="utf-8")
    model = {
        "provider": "fixture", "requested_model": "deterministic-v1",
        "parameters": {"temperature": 0, "max_tokens": 128},
        "runtime": {"timeout_seconds": 30, "rpm": 1000, "max_retries": 0},
    }
    return {
        "schema_version": 3,
        "experiment_id": f"v3-container-{benchmark}-{framework}-{uuid.uuid4().hex[:8]}",
        "scientific_profile": "docker-qdrant-fixture-v1",
        "benchmark": benchmark, "framework": framework,
        "runtime": {
            "root": str(tmp_path), "runs_dir": str(tmp_path / "runs"),
            "cache_dir": str(tmp_path / "cache"), "metrics_port": 9464,
            "log_level": "INFO",
        },
        "framework_config": {
            "path": str(framework_file), "sha256": sha256_file(framework_file),
            "format": suffix, "profile": "fixture-v1",
        },
        "storage": {
            "kind": "qdrant-server", "profile": "qdrant-v1",
            "endpoint_env": "QDRANT_URL", "retention": "delete-on-success",
            "request_timeout_seconds": 10,
        },
        "dataset": {
            "name": benchmark, "path": str(dataset), "source": "fixture",
            "revision": "mini", "sha256": sha256_file(dataset),
        },
        "selection": {
            "ordered_item_ids": ["conversation-0001" if benchmark == "locomo" else "lme-001"],
            "filters": {}, "seed": 7,
        },
        "retrieval": {"max_results": 2},
        "context_budget": {
            "max_tokens": 512, "tokenizer": TOKENIZER_ID,
            "renderer": RENDERER_ID, "packing": PACKING_ID,
        },
        "models": {
            "answerer": model,
            "judges": [{"id": "primary", **model}],
        },
        "evaluation": {
            "primary_judge_id": "primary",
            "required": ["primary_judge_score", "rigorous_report", "analysis_rows"],
            "optional": ["retrieval_report", "judge_agreement", "ablation_report"],
        },
        "artifact_store": {"type": "local", "uri": str(tmp_path / "runs")},
    }


@pytest.mark.integration
@pytest.mark.parametrize("benchmark", ["locomo", "longmemeval"])
@pytest.mark.parametrize("framework", ["dmf", "mem0"])
def test_v3_qdrant_fixture_predicts_each_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    benchmark: str, framework: str,
) -> None:
    endpoint = os.getenv("DMF_BENCH_TEST_QDRANT_URL")
    if not endpoint:
        pytest.skip("Set DMF_BENCH_TEST_QDRANT_URL to run Qdrant integration.")
    monkeypatch.setenv("QDRANT_URL", endpoint)
    config = _config(tmp_path, benchmark, framework)
    application = fixture_application_builder(
        config, metrics=BenchmarkMetrics(), events=JsonEventLogger(),
    )
    result = application.run(config, predict_only=False)
    assert result.state == "COMPLETED"
    run_dir = tmp_path / "runs" / result.run_id
    assert plan_resume(run_dir).next_phase == "COMPLETED"
    assert (run_dir / "evaluations" / "analysis_rows.jsonl").is_file()
    for unit_id in application.components.benchmark.enumerate_units(config):
        unit_dir = run_dir / "items" / unit_id.unit_id
        assert (unit_dir / "commit-barrier.json").is_file()
        assert (unit_dir / "retention.json").is_file()
        assert (unit_dir / "predictions.json").is_file()


class _OfflineAnswerer:
    def generate(self, _request: object) -> dict:
        return {
            "generated_answer": "deterministic fixture answer",
            "answerer_usage": {"total_tokens": 0},
        }


class _OfflineJudge:
    def __init__(self, benchmark: str) -> None:
        self.judge_fingerprint = judge_fingerprint(benchmark)

    def judge(self, _request: object) -> dict:
        return {
            "judgment": "CORRECT", "score": 1.0,
            "reason": "deterministic fixture",
            "judge_provider": "fixture",
            "judge_requested_model": "deterministic-v1",
            "judge_model": "deterministic-v1",
            "judge_usage": {"total_tokens": 0},
            "judge_fingerprint": self.judge_fingerprint,
        }


@pytest.mark.integration
@pytest.mark.parametrize("benchmark", ["locomo", "longmemeval"])
@pytest.mark.parametrize("framework", ["dmf", "mem0"])
def test_v3_real_framework_adapter_on_qdrant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    benchmark: str, framework: str,
) -> None:
    endpoint = os.getenv("DMF_BENCH_TEST_QDRANT_URL")
    if not endpoint:
        pytest.skip("Set DMF_BENCH_TEST_QDRANT_URL to run Qdrant integration.")
    monkeypatch.setenv("QDRANT_URL", endpoint)
    config = _config(tmp_path, benchmark, framework)
    config["scientific_profile"] = "retrieval-controlled-v1"
    source = CONFIG_DIR / f"{benchmark}_{framework}_qdrant_settings.{config['framework_config']['format']}"
    target = Path(config["framework_config"]["path"])
    target.write_bytes(source.read_bytes())
    config["framework_config"]["sha256"] = sha256_file(target)
    benchmark_adapter = LoCoMoAdapter() if benchmark == "locomo" else LongMemEvalAdapter()
    if framework == "dmf":
        adapter = DmfQdrantFrameworkAdapter.from_experiment(
            config, engine_builder=TrackingDmfEngineBuilder(),
        )
    else:
        monkeypatch.setenv("MEM0_TELEMETRY", "false")
        from mem0.utils import spacy_models

        monkeypatch.setattr(spacy_models, "get_nlp_full", lambda: None)
        monkeypatch.setattr(spacy_models, "get_nlp_lemma", lambda: None)
        adapter = Mem0QdrantFrameworkAdapter.from_experiment(
            config, engine_builder=TrackingMem0EngineBuilder(),
        )
    runner = RetrievalQAPredictOnlyRunner(
        benchmark=benchmark_adapter, framework=adapter,
        answerer=_OfflineAnswerer(),
        artifact_store=LocalArtifactStore(tmp_path / "runs"),
    )
    result = runner.run(config)
    assert result.committed_unit_ids == result.expected_unit_ids
    run_dir = tmp_path / "runs" / result.run_id
    assert plan_resume(run_dir).next_phase == "JUDGING"
    completed = OfflineLifecycleFinalizer(
        artifact_store=LocalArtifactStore(tmp_path / "runs"),
        judges={"primary": _OfflineJudge(benchmark)},
        primary_judge_id="primary",
    ).finalize(result.run_id)
    assert completed.state == "COMPLETED"
    assert (run_dir / "evaluations" / "analysis_rows.jsonl").is_file()
    retrieval_report = read_json(run_dir / "evaluations" / "retrieval_report.json")
    assert retrieval_report["status"] == "COMPLETED"
    for unit_id in result.expected_unit_ids:
        assert (run_dir / "items" / unit_id / "commit-barrier.json").is_file()
        assert (run_dir / "items" / unit_id / "retention.json").is_file()
