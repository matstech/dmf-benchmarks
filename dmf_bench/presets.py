"""Immutable benchmark presets separated from evolving defaults."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dmf_bench.contracts import hash_canonical_json
from dmf_bench.datasets import dataset_config_for_id


@dataclass(frozen=True)
class PresetRecord:
    preset_id: str
    profile: str
    benchmark: str
    framework: str
    protocol: str
    dataset_id: str
    selection: dict[str, Any]
    models: dict[str, Any]
    evaluation: dict[str, Any]

    def __post_init__(self) -> None:
        if self.profile not in {"paper", "default"}:
            raise ValueError("Preset profile must be paper or default.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "preset_id": self.preset_id,
            "profile": self.profile,
            "benchmark": self.benchmark,
            "framework": self.framework,
            "protocol": self.protocol,
            "dataset_id": self.dataset_id,
            "selection": self.selection,
            "models": self.models,
            "evaluation": self.evaluation,
        }

    def fingerprint(self) -> str:
        return hash_canonical_json(self.to_dict())


BUILTIN_PRESETS: dict[str, PresetRecord] = {
    "paper/locomo-dmf-strict-v1": PresetRecord(
        preset_id="paper/locomo-dmf-strict-v1",
        profile="paper",
        benchmark="locomo",
        framework="dmf",
        protocol="strict",
        dataset_id="locomo-official-unpinned",
        selection={"ordered_item_ids": ["*"], "filters": {}, "seed": 7},
        models={
            "answerer": {"provider": "openai", "requested_model": "gpt-4.1-mini", "parameters": {"temperature": 0}},
            "judge": {"provider": "openai", "requested_model": "gpt-5-mini", "parameters": {"temperature": 0}},
        },
        evaluation={"required": ["primary_judge_score", "rigorous_report"], "optional": ["ablation_report"]},
    ),
    "default/locomo-dmf-strict": PresetRecord(
        preset_id="default/locomo-dmf-strict",
        profile="default",
        benchmark="locomo",
        framework="dmf",
        protocol="strict",
        dataset_id="locomo-official-unpinned",
        selection={"ordered_item_ids": ["*"], "filters": {}, "seed": 7},
        models={
            "answerer": {"provider": "openai", "requested_model": "gpt-4.1-mini", "parameters": {"temperature": 0}},
            "judge": {"provider": "openai", "requested_model": "gpt-5-mini", "parameters": {"temperature": 0}},
        },
        evaluation={"required": ["primary_judge_score", "rigorous_report"], "optional": ["ablation_report"]},
    ),
    "paper/longmemeval-dmf-strict-v1": PresetRecord(
        preset_id="paper/longmemeval-dmf-strict-v1",
        profile="paper",
        benchmark="longmemeval",
        framework="dmf",
        protocol="strict",
        dataset_id="longmemeval-s-official-unpinned",
        selection={"ordered_item_ids": ["*"], "filters": {}, "seed": 7},
        models={
            "answerer": {"provider": "openai", "requested_model": "gpt-4.1-mini", "parameters": {"temperature": 0}},
            "judge": {"provider": "openai", "requested_model": "gpt-5-mini", "parameters": {"temperature": 0}},
        },
        evaluation={"required": ["primary_judge_score", "rigorous_report"], "optional": ["ablation_report"]},
    ),
    "default/longmemeval-dmf-strict": PresetRecord(
        preset_id="default/longmemeval-dmf-strict",
        profile="default",
        benchmark="longmemeval",
        framework="dmf",
        protocol="strict",
        dataset_id="longmemeval-s-official-unpinned",
        selection={"ordered_item_ids": ["*"], "filters": {}, "seed": 7},
        models={
            "answerer": {"provider": "openai", "requested_model": "gpt-4.1-mini", "parameters": {"temperature": 0}},
            "judge": {"provider": "openai", "requested_model": "gpt-5-mini", "parameters": {"temperature": 0}},
        },
        evaluation={"required": ["primary_judge_score", "rigorous_report"], "optional": ["ablation_report"]},
    ),
}


def resolve_preset(
    preset_id: str,
    *,
    dataset_path: str | Path,
    runtime_root: str | Path,
    registry_path: str | Path | None = None,
) -> dict[str, Any]:
    try:
        preset = BUILTIN_PRESETS[preset_id]
    except KeyError as exc:
        raise ValueError(f"Unknown preset_id: {preset_id}") from exc

    root = Path(runtime_root).resolve()
    return {
        "schema_version": 1,
        "experiment_id": preset_id.replace("/", "-"),
        "benchmark": preset.benchmark,
        "framework": preset.framework,
        "protocol": preset.protocol,
        "runtime": {
            "root": str(root),
            "runs_dir": str(root / "runs"),
            "cache_dir": str(root / "cache"),
        },
        "dataset": dataset_config_for_id(
            dataset_id=preset.dataset_id,
            path=dataset_path,
            registry_path=registry_path,
        ),
        "selection": preset.selection,
        "models": preset.models,
        "evaluation": preset.evaluation,
        "artifact_store": {"type": "local", "uri": str(root / "runs")},
        "preset": {
            "preset_id": preset.preset_id,
            "profile": preset.profile,
            "fingerprint": preset.fingerprint(),
        },
    }
