from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from dmf_bench.adapters.base import AnswererRequest, JudgeRequest, RetrievalResult
from dmf_bench.contracts import hash_canonical_json
from dmf_bench.fingerprints import build_scientific_fingerprint_inputs
from dmf_bench.registry import supported_combinations
from dmf_bench.runtime import (
    RuntimeAssemblyError,
    RuntimeFactories,
    assemble_runtime,
    benchmark_factories,
)


@dataclass(frozen=True)
class FixtureComponent:
    name: str


class FixtureAnswerer:
    name = "fixture-answerer"

    def generate(self, request: AnswererRequest) -> dict[str, Any]:
        return {"generated_answer": request.user_prompt}


class FixtureJudge:
    name = "fixture-judge"

    def judge(self, request: JudgeRequest) -> dict[str, Any]:
        return {"judgment": "CORRECT", "prediction": request.prediction}


def experiment_config(
    benchmark: str = "locomo",
    framework: str = "dmf",
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "experiment_id": "fixture-run",
        "benchmark": benchmark,
        "framework": framework,
        "runtime": {
            "root": "/bench",
            "runs_dir": "/bench/runs",
            "cache_dir": "/bench/cache",
            "metrics_port": 9464,
            "log_level": "INFO",
        },
        "framework_config": {
            "path": "/bench/config/framework.toml",
            "format": "toml" if framework == "dmf" else "yaml",
            "profile": "fixture",
            "sha256": "a" * 64,
        },
        "qdrant": {
            "endpoint_env": "QDRANT_URL",
            "retention": "keep",
            "request_timeout_seconds": 10,
        },
        "dataset": {
            "name": benchmark,
            "path": f"/bench/datasets/{benchmark}.json",
            "source": "fixture",
            "revision": "v1",
            "sha256": "b" * 64,
        },
        "selection": {"ordered_item_ids": ["item-001"], "seed": 7},
        "models": {
            "answerer": {
                "provider": "fixture",
                "endpoint_identity": "fixture://answerer",
                "requested_model": "fixture-answerer",
                "parameters": {"temperature": 0, "max_tokens": 128},
                "runtime": {"timeout_seconds": 30, "rpm": 100, "max_retries": 1},
            },
            "judge": {
                "provider": "fixture",
                "endpoint_identity": "fixture://judge",
                "requested_model": "fixture-judge",
                "parameters": {"temperature": 0, "max_tokens": 128},
                "runtime": {"timeout_seconds": 30, "rpm": 100, "max_retries": 1},
            },
        },
        "evaluation": {
            "required": ["primary_judge_score", "rigorous_report"],
            "optional": ["ablation_report"],
        },
        "artifact_store": {"type": "local", "uri": "/bench/runs"},
    }


def runtime_factories() -> RuntimeFactories:
    return RuntimeFactories(
        benchmarks=benchmark_factories(),
        frameworks={
            "dmf": lambda _config: FixtureComponent("dmf"),
            "mem0": lambda _config: FixtureComponent("mem0"),
        },
        answerers={"fixture": lambda _config: FixtureAnswerer()},
        judges={
            ("locomo", "fixture"): lambda _config: FixtureJudge(),
            ("longmemeval", "fixture"): lambda _config: FixtureJudge(),
        },
    )


def test_explicit_factories_assemble_every_supported_runtime_combination() -> None:
    for benchmark, framework in supported_combinations():
        components = assemble_runtime(
            experiment_config(benchmark, framework),
            factories=runtime_factories(),
        )

        assert components.request.benchmark == benchmark
        assert components.request.framework == framework
        assert components.benchmark.name == benchmark
        assert components.framework.name == framework


def test_runtime_assembly_refuses_missing_component_without_fallback() -> None:
    factories = runtime_factories()
    incomplete = RuntimeFactories(
        benchmarks=factories.benchmarks,
        frameworks={},
        answerers=factories.answerers,
        judges=factories.judges,
    )

    with pytest.raises(RuntimeAssemblyError, match="No framework factory"):
        assemble_runtime(experiment_config(), factories=incomplete)


def test_scientific_fingerprint_excludes_paths_and_runtime_tuning() -> None:
    first = experiment_config()
    second = experiment_config()
    second["experiment_id"] = "another-run"
    second["runtime"] = {
        "root": "/different",
        "runs_dir": "/different/runs",
        "cache_dir": "/different/cache",
        "metrics_port": 12345,
        "log_level": "DEBUG",
    }
    second["framework_config"]["path"] = "/different/framework.toml"
    second["dataset"]["path"] = "/different/dataset.json"
    second["qdrant"]["request_timeout_seconds"] = 60
    second["models"]["answerer"]["runtime"] = {
        "timeout_seconds": 300,
        "rpm": 1,
        "max_retries": 9,
    }
    second["artifact_store"]["uri"] = "/different/runs"

    assert hash_canonical_json(build_scientific_fingerprint_inputs(first)) == hash_canonical_json(
        build_scientific_fingerprint_inputs(second)
    )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("framework_config", "sha256"), "c" * 64),
        (("dataset", "sha256"), "d" * 64),
        (("models", "answerer", "parameters", "max_tokens"), 256),
        (("models", "judge", "requested_model"), "another-judge"),
        (("selection", "seed"), 8),
    ],
)
def test_scientific_fingerprint_changes_for_scientific_inputs(
    path: tuple[str, ...],
    value: Any,
) -> None:
    first = experiment_config()
    second = experiment_config()
    target = second
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    assert hash_canonical_json(build_scientific_fingerprint_inputs(first)) != hash_canonical_json(
        build_scientific_fingerprint_inputs(second)
    )


def test_retrieval_and_judge_contracts_keep_scientific_metadata() -> None:
    retrieval = RetrievalResult(
        cutoff_label="top_2",
        search_results=({"id": "hit-1"},),
        recall_diagnostics={"raw_candidates": [{"id": "raw-1"}]},
        memory_internal_usage={"total_tokens": 4},
        memories_evaluated=1,
        timing={"backend_search_ms": 2.5},
    ).to_dict()
    request = JudgeRequest(
        prediction={
            "question": "When?",
            "question_type": "temporal",
            "question_date": "2025-01-01",
            "ground_truth_answer": "Yesterday",
            "generated_answer": "Yesterday",
        },
        metadata={"benchmark": "longmemeval"},
    )

    assert retrieval["recall_diagnostics"]["raw_candidates"][0]["id"] == "raw-1"
    assert retrieval["memory_internal_usage"]["total_tokens"] == 4
    assert request.prediction["question_type"] == "temporal"
    assert request.metadata["benchmark"] == "longmemeval"
