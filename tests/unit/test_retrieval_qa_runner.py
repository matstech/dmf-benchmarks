"""Deterministic four-pair checks for the shared v3 prediction lifecycle."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from dmf_bench.adapters.base import (
    CanonicalRetrievalResult, OwnedResource, PreparedMemoryUnit, RetrievedMemory,
)
from dmf_bench.artifacts import LocalArtifactStore
from dmf_bench.benchmarks.locomo.adapter import LoCoMoAdapter
from dmf_bench.benchmarks.longmemeval.adapter import LongMemEvalAdapter
from dmf_bench.contracts import hash_canonical_json, sha256_file
from dmf_bench.fingerprints import build_v3_fingerprint_inputs
from dmf_bench.retrieval_qa import RetrievalQAPredictOnlyRunner
from dmf_bench.runner import InjectedInterrupt
from dmf_bench.runtime import RuntimeFactories, assemble_application
from dmf_bench.state import plan_resume


FIXTURES = Path(__file__).parents[1] / "fixtures"


def config_v3(tmp_path: Path, benchmark: str, framework: str) -> dict[str, Any]:
    dataset_path = tmp_path / f"{benchmark}.json"
    dataset_path.write_bytes((FIXTURES / f"{benchmark}-mini.json").read_bytes())
    suffix = "toml" if framework == "dmf" else "yaml"
    framework_path = tmp_path / f"framework.{suffix}"
    framework_path.write_text("fixture = true\n")
    return {
        "schema_version": 3,
        "experiment_id": f"{benchmark}-{framework}-fixture",
        "scientific_profile": "retrieval-controlled-v1",
        "benchmark": benchmark,
        "framework": framework,
        "runtime": {
            "root": str(tmp_path), "runs_dir": str(tmp_path / "runs"),
            "cache_dir": str(tmp_path / "cache"), "metrics_port": 9464,
            "log_level": "INFO",
        },
        "framework_config": {
            "path": str(framework_path), "sha256": sha256_file(framework_path),
            "format": suffix, "profile": "fixture-v1",
        },
        "storage": {
            "kind": "qdrant-server", "profile": "qdrant-v1",
            "endpoint_env": "QDRANT_URL", "retention": "keep",
            "request_timeout_seconds": 10,
        },
        "dataset": {
            "name": benchmark, "path": str(dataset_path),
            "source": "fixture", "revision": "fixture-v1",
            "sha256": sha256_file(dataset_path),
        },
        "selection": {"ordered_item_ids": ["*"], "filters": {}, "seed": 7},
        "retrieval": {"max_results": 2},
        "context_budget": {
            "max_tokens": 128,
            "tokenizer": "tiktoken-cl100k_base-v1",
            "renderer": "retrieved-memory-v1",
            "packing": "ranked-whole-items-v1",
        },
        "models": {
            "answerer": {
                "provider": "fixture", "requested_model": "echo-v1",
                "parameters": {"temperature": 0, "max_tokens": 64},
                "runtime": {"timeout_seconds": 1, "rpm": 100, "max_retries": 0},
            },
            "judges": [{
                "id": "primary", "provider": "fixture", "requested_model": "exact-v1",
                "parameters": {"temperature": 0, "max_tokens": 64},
                "runtime": {"timeout_seconds": 1, "rpm": 100, "max_retries": 0},
            }],
        },
        "evaluation": {
            "primary_judge_id": "primary",
            "required": ["primary_judge_score", "rigorous_report", "analysis_rows"],
            "optional": [],
        },
        "artifact_store": {"type": "local", "uri": str(tmp_path / "runs")},
    }


class FixtureFramework:
    def __init__(self, name: str, *, barrier_verified: bool = True) -> None:
        self.name = name
        self.barrier_verified = barrier_verified
        self.prepared = 0
        self.retrieved = 0
        self.cleaned: list[tuple[str, ...]] = []
        self.event_texts: list[str] = []

    def resources_for_unit_v3(self, unit: Any, config: dict, run_context: Any) -> tuple[OwnedResource, ...]:
        return (OwnedResource(
            resource_id=f"{run_context.run_id}:{unit.unit_id}",
            kind="fixture", role="primary",
            locator=f"{run_context.run_id}/{unit.unit_id}",
        ),)

    def cleanup_unit_v3(self, unit: Any, resources: tuple[OwnedResource, ...], config: dict, run_context: Any) -> dict:
        assert resources == self.resources_for_unit_v3(unit, config, run_context)
        self.cleaned.append(tuple(resource.resource_id for resource in resources))
        return {"verified": True, "deleted": list(self.cleaned[-1])}

    def prepare_unit_v3(self, unit: Any, events: tuple, config: dict, run_context: Any) -> PreparedMemoryUnit:
        self.prepared += 1
        self.event_texts.extend(event.content for event in events)
        return PreparedMemoryUnit(
            handle=events,
            resources=self.resources_for_unit_v3(unit, config, run_context),
            ingestion_usage={}, ingestion_timing={}, diagnostics={},
        )

    def verify_prepared_v3(self, unit: Any, prepared: PreparedMemoryUnit, config: dict, run_context: Any) -> dict:
        return {"verified": self.barrier_verified, "event_count": len(prepared.handle)}

    def retrieve_v3(self, unit: Any, query: Any, prepared: PreparedMemoryUnit, config: dict, run_context: Any) -> CanonicalRetrievalResult:
        self.retrieved += 1
        return CanonicalRetrievalResult((
            RetrievedMemory(
                memory_id=f"{query.query_id}:1",
                content=prepared.handle[0].content,
                rank=1,
                source_event_ids=(prepared.handle[0].event_id,),
            ),
        ), raw_payload={"private": "not-prompt"})


class FixtureAnswerer:
    def __init__(self) -> None:
        self.requests: list[Any] = []

    def generate(self, request: Any) -> dict:
        self.requests.append(request)
        return {"generated_answer": "fixture answer", "answerer_usage": {"total_tokens": 1}}


def runner_for(tmp_path: Path, benchmark: str, framework: str, *, barrier: bool = True):
    benchmark_adapter = LoCoMoAdapter() if benchmark == "locomo" else LongMemEvalAdapter()
    framework_adapter = FixtureFramework(framework, barrier_verified=barrier)
    answerer = FixtureAnswerer()
    runner = RetrievalQAPredictOnlyRunner(
        benchmark=benchmark_adapter,
        framework=framework_adapter,
        answerer=answerer,
        artifact_store=LocalArtifactStore(tmp_path / "runs"),
    )
    return runner, framework_adapter, answerer


@pytest.mark.parametrize("benchmark", ["locomo", "longmemeval"])
@pytest.mark.parametrize("framework", ["dmf", "mem0"])
def test_shared_runner_commits_four_pairs_with_canonical_artifacts(
    tmp_path: Path, benchmark: str, framework: str,
) -> None:
    config = config_v3(tmp_path, benchmark, framework)
    runner, memory, answerer = runner_for(tmp_path, benchmark, framework)
    result = runner.run(config)
    assert result.committed_unit_ids == result.expected_unit_ids
    assert memory.prepared == len(result.expected_unit_ids)
    assert memory.retrieved == len(answerer.requests)
    run_dir = tmp_path / "runs" / result.run_id
    assert plan_resume(run_dir).next_phase == "JUDGING"
    for unit_id in result.expected_unit_ids:
        unit_dir = run_dir / "items" / unit_id
        assert (unit_dir / "resources.json").exists()
        assert (unit_dir / "commit-barrier.json").exists()
        aggregate = json.loads((unit_dir / "predictions.json").read_text())
        assert all("packed_context" in row for row in aggregate["predictions"])
        assert all("retrieval" in row for row in aggregate["predictions"])
    assert all("not-prompt" not in request.user_prompt for request in answerer.requests)
    assert all("ground_truth_answer" not in json.dumps(request.metadata) for request in answerer.requests)


def test_barrier_failure_blocks_retrieval(tmp_path: Path) -> None:
    config = config_v3(tmp_path, "locomo", "dmf")
    runner, memory, answerer = runner_for(tmp_path, "locomo", "dmf", barrier=False)
    with pytest.raises(Exception, match="commit barrier"):
        runner.run(config)
    assert memory.retrieved == 0
    assert answerer.requests == []


def test_resume_restarts_incomplete_unit_and_reuses_committed_unit(tmp_path: Path) -> None:
    config = config_v3(tmp_path, "longmemeval", "mem0")
    config["selection"]["ordered_item_ids"] = ["lme-001", "lme-002"]
    runner, memory, _answerer = runner_for(tmp_path, "longmemeval", "mem0")
    with pytest.raises(InjectedInterrupt):
        runner.run(config, interrupt_at="after-commit")
    run_dir = tmp_path / "runs" / config["experiment_id"]
    first_prediction = run_dir / "items" / "lme-001" / "prediction.json"
    original_hash = sha256_file(first_prediction)
    result = runner.run(config, resume=True)
    assert result.committed_unit_ids == ("lme-001", "lme-002")
    assert result.restarted_unit_ids == ("lme-002",)
    assert sha256_file(first_prediction) == original_hash
    assert memory.prepared == 2


def test_v3_fingerprint_tracks_scientific_inputs_and_excludes_endpoint(tmp_path: Path) -> None:
    original = config_v3(tmp_path, "locomo", "mem0")
    question_ids = ("conv0_q0", "conv0_q1")
    fingerprint = lambda value: hash_canonical_json(
        build_v3_fingerprint_inputs(value, expected_question_ids=question_ids)
    )
    baseline = fingerprint(original)
    operational = deepcopy(original)
    operational["storage"]["request_timeout_seconds"] = 90
    operational["runtime"]["metrics_port"] = 1234
    assert fingerprint(operational) == baseline
    for field, value in (
        (("context_budget", "max_tokens"), 256),
        (("retrieval", "max_results"), 1),
        (("storage", "profile"), "qdrant-v2"),
        (("models", "judges", 0, "requested_model"), "other-model"),
    ):
        changed = deepcopy(original)
        target = changed
        for part in field[:-1]:
            target = target[part]
        target[field[-1]] = value
        assert fingerprint(changed) != baseline
    inputs = build_v3_fingerprint_inputs(original, expected_question_ids=question_ids)
    assert inputs["ingestion_policy"]["pairing"] == "adjacent-user-assistant-v1"
    assert inputs["ingestion_policy"]["observation_role"] == "observation-as-user-v1"


def test_runtime_selects_shared_v3_lifecycle_without_benchmark_branch(tmp_path: Path) -> None:
    config = config_v3(tmp_path, "locomo", "dmf")
    config["models"]["judges"].append({
        **config["models"]["judges"][0],
        "id": "secondary",
        "requested_model": "alternate-exact-v1",
    })
    framework = FixtureFramework("dmf")
    answerer = FixtureAnswerer()
    primary_judge = object()
    secondary_judge = object()
    factories = RuntimeFactories(
        benchmarks={"locomo": lambda _config: LoCoMoAdapter()},
        frameworks={"dmf": lambda _config: framework},
        answerers={"fixture": lambda _config: answerer},
        judges={("locomo", "fixture"): lambda judge_config: (
            primary_judge if judge_config["models"]["judge"]["id"] == "primary"
            else secondary_judge
        )},
    )
    application = assemble_application(config, factories=factories)
    assert isinstance(application.prediction_runner, RetrievalQAPredictOnlyRunner)
    assert application.components.judges == {
        "primary": primary_judge, "secondary": secondary_judge,
    }
    result = application.run(config, predict_only=True)
    assert result.committed_unit_ids == result.expected_unit_ids
