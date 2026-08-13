"""Built-in experiment presets and portable user configuration bundles."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Preset:
    name: str
    benchmark: str
    framework: str
    experiment_path: str
    framework_path: str
    description: str


PRESETS = {
    preset.name: preset
    for preset in (
        Preset(
            name="locomo-dmf",
            benchmark="locomo",
            framework="dmf",
            experiment_path="smoke/config/experiment-locomo-dmf.json",
            framework_path="config/locomo_dmf_qdrant_settings.toml",
            description="LoCoMo official 0.5% conversation sample with DMF",
        ),
        Preset(
            name="locomo-mem0",
            benchmark="locomo",
            framework="mem0",
            experiment_path="smoke/config/experiment-locomo-mem0.json",
            framework_path="config/locomo_mem0_qdrant_settings.yaml",
            description="LoCoMo official 0.5% conversation sample with Mem0",
        ),
        Preset(
            name="longmemeval-dmf",
            benchmark="longmemeval",
            framework="dmf",
            experiment_path="smoke/config/experiment-longmemeval-dmf.json",
            framework_path="config/longmemeval_dmf_qdrant_settings.toml",
            description="LongMemEval official 0.5% record sample with DMF",
        ),
        Preset(
            name="longmemeval-mem0",
            benchmark="longmemeval",
            framework="mem0",
            experiment_path="smoke/config/experiment-longmemeval-mem0.json",
            framework_path="config/longmemeval_mem0_qdrant_settings.yaml",
            description="LongMemEval official 0.5% record sample with Mem0",
        ),
    )
}


def preset_catalog() -> list[dict[str, str]]:
    return [asdict(PRESETS[name]) for name in sorted(PRESETS)]


def export_preset(
    *,
    runtime_dir: Path,
    preset_name: str,
    output_dir: Path,
    force: bool = False,
) -> tuple[Path, Path]:
    try:
        preset = PRESETS[preset_name]
    except KeyError as exc:
        available = ", ".join(sorted(PRESETS))
        raise ValueError(
            f"unknown preset {preset_name!r}; available presets: {available}"
        ) from exc

    source_experiment = runtime_dir / preset.experiment_path
    source_framework = runtime_dir / preset.framework_path
    if not source_experiment.is_file() or not source_framework.is_file():
        raise ValueError(f"preset resources are incomplete: {preset_name}")

    experiment_target = output_dir / "experiment.json"
    framework_suffix = source_framework.suffix
    framework_target = output_dir / f"{preset.framework}-settings{framework_suffix}"
    conflicts = [path for path in (experiment_target, framework_target) if path.exists()]
    if conflicts and not force:
        raise ValueError(
            "configuration export would overwrite existing files; "
            "use --force to replace only the generated files"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_framework, framework_target)
    data = _read_json_object(source_experiment)
    framework_config = data.get("framework_config")
    if not isinstance(framework_config, dict):
        raise ValueError(f"preset has no framework_config object: {source_experiment}")
    framework_config["path"] = framework_target.name
    framework_config["sha256"] = _sha256_file(framework_target)
    _write_json_atomic(experiment_target, data)
    return experiment_target, framework_target


def pin_operator_config(config_path: Path, *, runtime_dir: Path) -> bool:
    """Refresh checksums for editable files referenced by an operator config."""

    config_path = config_path.resolve()
    try:
        config_path.relative_to(runtime_dir.resolve())
    except ValueError:
        pass
    else:
        return False

    data = _read_json_object(config_path)
    changed = False
    framework_config = data.get("framework_config")
    if isinstance(framework_config, dict):
        framework_path = _host_resource_path(
            framework_config.get("path"),
            config_path=config_path,
            runtime_dir=runtime_dir,
        )
        if framework_path is not None:
            observed = _sha256_file(framework_path)
            if framework_config.get("sha256") != observed:
                framework_config["sha256"] = observed
                changed = True

    dataset = data.get("dataset")
    if isinstance(dataset, dict):
        dataset_path = _host_resource_path(
            dataset.get("path"),
            config_path=config_path,
            runtime_dir=runtime_dir,
        )
        if dataset_path is not None:
            observed = _sha256_file(dataset_path)
            if dataset.get("sha256") != observed:
                dataset["sha256"] = observed
                changed = True

    if changed:
        _write_json_atomic(config_path, data)
    return changed


def _host_resource_path(
    raw_path: Any,
    *,
    config_path: Path,
    runtime_dir: Path,
) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        candidate = (config_path.parent / path).resolve()
        _require_within(candidate, config_path.parent.resolve())
    else:
        mappings = (
            (Path("/bench/operator"), config_path.parent.resolve()),
            (Path("/bench/config"), (runtime_dir / "config").resolve()),
            (Path("/bench/datasets"), (runtime_dir / "datasets").resolve()),
            (Path("/bench/smoke"), (runtime_dir / "smoke").resolve()),
        )
        candidate = None
        for container_root, host_root in mappings:
            try:
                relative = path.relative_to(container_root)
            except ValueError:
                continue
            candidate = (host_root / relative).resolve()
            _require_within(candidate, host_root)
            break
        if candidate is None:
            return None
    if not candidate.is_file():
        raise ValueError(f"referenced configuration resource does not exist: {candidate}")
    return candidate


def _require_within(path: Path, root: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"configuration resource escapes its bundle: {path}") from exc


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read experiment config {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"experiment config must contain a JSON object: {path}")
    return data


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
