"""A new deterministic framework completes v3 LoCoMo through registration alone."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import dmf_bench.registry as registry
import dmf_bench.runtime as runtime
from dmf_bench.adapters.base import (
    CanonicalRetrievalResult,
    OwnedResource,
    PreparedMemoryUnit,
    RetrievedMemory,
)
from dmf_bench.atomic_io import read_json
from dmf_bench.benchmarks.locomo.adapter import LoCoMoAdapter
from dmf_bench.contracts import hash_canonical_json, sha256_file
from dmf_bench.fingerprints import judge_contract_identity
from dmf_bench.registry import CompatibilityRecord, FrameworkDescriptor
from dmf_bench.runtime import RuntimeFactories, assemble_application
from dmf_bench.state import plan_resume

from test_retrieval_qa_runner import config_v3


FRAMEWORK_NAME = "deterministic-memory"


class DeterministicMemory:
    name = FRAMEWORK_NAME

    def __init__(self) -> None:
        self.prepared = 0
        self.retrieved = 0
        self.cleaned = 0

    def resources_for_unit_v3(self, unit: Any, config: dict, run_context: Any) -> tuple[OwnedResource, ...]:
        del config
        return (OwnedResource(
            resource_id=f"{run_context.run_id}:{unit.unit_id}",
            kind="fixture-local", role="primary",
            locator=f"{run_context.run_id}/{unit.unit_id}",
        ),)

    def cleanup_unit_v3(self, unit: Any, resources: tuple[OwnedResource, ...],
                        config: dict, run_context: Any) -> dict[str, Any]:
        assert resources == self.resources_for_unit_v3(unit, config, run_context)
        self.cleaned += 1
        return {"verified": True, "resource_ids": [resource.resource_id for resource in resources]}

    def prepare_unit_v3(self, unit: Any, events: tuple, config: dict,
                        run_context: Any) -> PreparedMemoryUnit:
        self.prepared += 1
        return PreparedMemoryUnit(
            handle=events,
            resources=self.resources_for_unit_v3(unit, config, run_context),
            ingestion_usage={"events": len(events)},
            ingestion_timing={"seconds": 0.0},
            diagnostics={"policy": "whole-events-v1"},
        )

    def verify_prepared_v3(self, unit: Any, prepared: PreparedMemoryUnit,
                           config: dict, run_context: Any) -> dict[str, Any]:
        assert prepared.resources == self.resources_for_unit_v3(unit, config, run_context)
        return {"verified": True, "event_count": len(prepared.handle)}

    def retrieve_v3(self, unit: Any, query: Any, prepared: PreparedMemoryUnit,
                    config: dict, run_context: Any) -> CanonicalRetrievalResult:
        del unit, config, run_context
        self.retrieved += 1
        selected = next(
            event for event in prepared.handle
            if ("window" in event.content.lower()) == ("bed" in query.text.lower())
        )
        return CanonicalRetrievalResult((RetrievedMemory(
            memory_id=f"fixture:{selected.event_id}", content=selected.content, rank=1,
            source_event_ids=(selected.event_id,),
            metadata={"source_refs": list(selected.source_refs)},
        ),), diagnostics={"fixture": {"selected_event_id": selected.event_id}})


class DeterministicAnswerer:
    name = "fixture"

    def generate(self, request: Any) -> dict[str, Any]:
        answer = "near the kitchen window" if "Where did Alice move" in request.user_prompt else "Pixel"
        return {"generated_answer": answer, "answerer_usage": {"total_tokens": 1}}


class DeterministicJudge:
    name = "fixture"
    judge_fingerprint = hash_canonical_json(judge_contract_identity("locomo"))

    def judge(self, request: Any) -> dict[str, Any]:
        prediction = request.prediction
        correct = str(prediction["generated_answer"]).casefold() == str(prediction["ground_truth_answer"]).casefold()
        return {
            "judgment": "CORRECT" if correct else "WRONG",
            "score": 1.0 if correct else 0.0,
            "reason": "deterministic exact match",
            "judge_provider": "fixture",
            "judge_requested_model": "exact-v1",
            "judge_model": "exact-v1",
            "judge_fingerprint": self.judge_fingerprint,
            "judge_usage": {"total_tokens": 1},
        }


def registered_config(tmp_path: Path) -> dict[str, Any]:
    config = config_v3(tmp_path, "locomo", "dmf")
    config["experiment_id"] = "locomo-deterministic-memory-fixture"
    config["framework"] = FRAMEWORK_NAME
    framework_config = tmp_path / "deterministic-memory.json"
    framework_config.write_text(json.dumps({"policy": "whole-events-v1"}), encoding="utf-8")
    config["framework_config"] = {
        "path": str(framework_config), "sha256": sha256_file(framework_config),
        "format": "json", "profile": "whole-events-v1",
    }
    config["storage"] = {"kind": "fixture-local", "profile": "fixture-v1", "retention": "keep"}
    return config


def register_framework(monkeypatch: pytest.MonkeyPatch, memory: DeterministicMemory) -> None:
    descriptor = FrameworkDescriptor(
        name=FRAMEWORK_NAME, adapter_version="fixture-v1",
        factory=lambda _config: memory,
        config_formats=frozenset({"json"}),
        storage_kinds=frozenset({"fixture-local"}),
        capabilities=frozenset({"deterministic-resource-naming", "cleanup-manifest"}),
        distribution="test-fixture",
    )
    monkeypatch.setitem(registry.FRAMEWORKS, FRAMEWORK_NAME, descriptor)
    monkeypatch.setitem(
        registry.COMPATIBILITY, ("locomo", FRAMEWORK_NAME),
        CompatibilityRecord("locomo", FRAMEWORK_NAME, "experimental", "locomo-v3"),
    )
    factories = RuntimeFactories(
        benchmarks={"locomo": lambda _config: LoCoMoAdapter()},
        frameworks={FRAMEWORK_NAME: lambda _config: memory},
        answerers={"fixture": lambda _config: DeterministicAnswerer()},
        judges={("locomo", "fixture"): lambda _config: DeterministicJudge()},
    )
    monkeypatch.setattr(runtime, "default_runtime_factories", lambda **_kwargs: factories)


def test_registered_framework_completes_full_v3_lifecycle_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = DeterministicMemory()
    register_framework(monkeypatch, memory)
    config = registered_config(tmp_path)
    application = assemble_application(config)

    result = application.run(config, predict_only=False)

    assert result.state == "COMPLETED"
    assert (memory.prepared, memory.retrieved, memory.cleaned) == (1, 2, 1)
    run_dir = tmp_path / "runs" / result.run_id
    assert plan_resume(run_dir).next_phase == "COMPLETED"
    assert application.artifact_store.verify_committed(result.run_id)["verified_artifact_count"] > 0
    evaluations = read_json(run_dir / "evaluations" / "evaluations.json")
    assert [item["score"] for item in evaluations] == [1.0, 1.0]
    assert read_json(run_dir / "evaluations" / "analysis_rows.json")["status"] == "COMPLETED"
    assert read_json(run_dir / "evaluations" / "retrieval_report.json")["status"] == "NOT_APPLICABLE"
    assert read_json(run_dir / "evaluations" / "ablation_report.json")["status"] == "NOT_APPLICABLE"


def test_missing_pair_is_rejected_even_with_descriptor_and_runtime_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = DeterministicMemory()
    register_framework(monkeypatch, memory)
    monkeypatch.delitem(registry.COMPATIBILITY, ("locomo", FRAMEWORK_NAME))
    config = registered_config(tmp_path)

    with pytest.raises(ValueError, match="Unsupported benchmark/framework pair"):
        assemble_application(config)
    assert not (tmp_path / "runs").exists()
