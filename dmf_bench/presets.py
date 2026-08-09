"""Immutable benchmark presets separated from evolving defaults."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dmf_bench.contracts import (
    EXPERIMENT_CONFIG_SCHEMA_VERSION,
    hash_canonical_json,
    sha256_file,
)
from dmf_bench.datasets import dataset_config_for_id


@dataclass(frozen=True)
class PresetRecord:
    preset_id: str
    profile: str
    benchmark: str
    framework: str
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
            "dataset_id": self.dataset_id,
            "selection": self.selection,
            "models": self.models,
            "evaluation": self.evaluation,
        }

    def fingerprint(self) -> str:
        return hash_canonical_json(self.to_dict())


def _model(
    provider: str,
    requested_model: str,
    *,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    parameters: dict[str, Any] = {"temperature": 0, "max_tokens": 4096}
    if reasoning_effort is not None:
        parameters["reasoning_effort"] = reasoning_effort
    return {
        "provider": provider,
        "requested_model": requested_model,
        "parameters": parameters,
        "runtime": {
            "timeout_seconds": 120,
            "rpm": 200,
            "max_retries": 5,
        },
    }


BUILTIN_PRESETS: dict[str, PresetRecord] = {
    "paper/locomo-dmf-v2": PresetRecord(
        preset_id="paper/locomo-dmf-v2",
        profile="paper",
        benchmark="locomo",
        framework="dmf",
        dataset_id="locomo-official-unpinned",
        selection={"ordered_item_ids": ["*"], "filters": {}, "seed": 7},
        models={
            "answerer": _model("openai", "gpt-4.1-mini"),
            "judge": _model("openai", "gpt-5-mini", reasoning_effort="low"),
        },
        evaluation={"required": ["primary_judge_score", "rigorous_report"], "optional": ["ablation_report"]},
    ),
    "default/locomo-dmf": PresetRecord(
        preset_id="default/locomo-dmf",
        profile="default",
        benchmark="locomo",
        framework="dmf",
        dataset_id="locomo-official-unpinned",
        selection={"ordered_item_ids": ["*"], "filters": {}, "seed": 7},
        models={
            "answerer": _model("openai", "gpt-4.1-mini"),
            "judge": _model("openai", "gpt-5-mini", reasoning_effort="low"),
        },
        evaluation={"required": ["primary_judge_score", "rigorous_report"], "optional": ["ablation_report"]},
    ),
    "paper/longmemeval-dmf-v2": PresetRecord(
        preset_id="paper/longmemeval-dmf-v2",
        profile="paper",
        benchmark="longmemeval",
        framework="dmf",
        dataset_id="longmemeval-s-official-unpinned",
        selection={"ordered_item_ids": ["*"], "filters": {}, "seed": 7},
        models={
            "answerer": _model("openai", "gpt-4.1-mini"),
            "judge": _model("openai", "gpt-5-mini", reasoning_effort="low"),
        },
        evaluation={"required": ["primary_judge_score", "rigorous_report"], "optional": ["ablation_report"]},
    ),
    "default/longmemeval-dmf": PresetRecord(
        preset_id="default/longmemeval-dmf",
        profile="default",
        benchmark="longmemeval",
        framework="dmf",
        dataset_id="longmemeval-s-official-unpinned",
        selection={"ordered_item_ids": ["*"], "filters": {}, "seed": 7},
        models={
            "answerer": _model("openai", "gpt-4.1-mini"),
            "judge": _model("openai", "gpt-5-mini", reasoning_effort="low"),
        },
        evaluation={"required": ["primary_judge_score", "rigorous_report"], "optional": ["ablation_report"]},
    ),
}


def resolve_preset(
    preset_id: str,
    *,
    dataset_path: str | Path,
    framework_config_path: str | Path,
    runtime_root: str | Path,
    registry_path: str | Path | None = None,
) -> dict[str, Any]:
    try:
        preset = BUILTIN_PRESETS[preset_id]
    except KeyError as exc:
        raise ValueError(f"Unknown preset_id: {preset_id}") from exc

    root = Path(runtime_root).resolve()
    framework_path = Path(framework_config_path).resolve()
    if not framework_path.is_file():
        raise ValueError(f"Framework config file not found: {framework_path}")
    return {
        "schema_version": EXPERIMENT_CONFIG_SCHEMA_VERSION,
        "experiment_id": preset_id.replace("/", "-"),
        "benchmark": preset.benchmark,
        "framework": preset.framework,
        "runtime": {
            "root": str(root),
            "runs_dir": str(root / "runs"),
            "cache_dir": str(root / "cache"),
            "metrics_port": 9464,
            "log_level": "INFO",
        },
        "framework_config": {
            "path": str(framework_path),
            "sha256": sha256_file(framework_path),
            "format": "toml" if preset.framework == "dmf" else "yaml",
            "profile": "paper" if preset.profile == "paper" else "default",
        },
        "qdrant": {
            "endpoint_env": "QDRANT_URL",
            "retention": "keep",
            "request_timeout_seconds": 10,
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
