"""Persistent preparation context for reproducible benchmark batches."""

from __future__ import annotations

import json
import os
import subprocess
from hashlib import sha256
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol


PREPARATION_SCHEMA_VERSION = 2
SUPPORTED_PREPARATION_SCHEMA_VERSIONS = (1, PREPARATION_SCHEMA_VERSION)
DEFAULT_PREPARATION_PATH = Path(".dmf-bench/prepared.json")
SMOKE_CONFIG_PATHS = (
    "smoke/config/experiment-locomo-dmf.json",
    "smoke/config/experiment-locomo-mem0.json",
    "smoke/config/experiment-longmemeval-dmf.json",
    "smoke/config/experiment-longmemeval-mem0.json",
)


@dataclass(frozen=True, slots=True)
class CommandOutput:
    returncode: int
    stdout: str
    stderr: str


class OutputRunner(Protocol):
    def __call__(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> CommandOutput: ...


def run_output_command(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
) -> CommandOutput:
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        env=dict(env),
        check=False,
        capture_output=True,
        text=True,
    )
    return CommandOutput(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def default_batch_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def inspect_image(
    image_ref: str,
    *,
    project_dir: Path,
    environment: Mapping[str, str],
    output_runner: OutputRunner,
) -> dict[str, str]:
    image_result = output_runner(
        ["docker", "image", "inspect", image_ref],
        cwd=project_dir,
        env=environment,
    )
    if image_result.returncode != 0:
        detail = image_result.stderr.strip() or "docker image inspect failed"
        raise ValueError(detail)
    try:
        images = json.loads(image_result.stdout)
        image = images[0]
    except (IndexError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("docker returned invalid image inspection data") from exc

    repo_digests = image.get("RepoDigests") or []
    if not repo_digests or not isinstance(repo_digests[0], str):
        raise ValueError(
            "the selected image has no registry digest; prepare requires a published image"
        )
    resolved_ref = _matching_repo_digest(image_ref, repo_digests)
    digest = resolved_ref.rsplit("@", 1)[-1]
    architecture = _normalise_architecture(str(image.get("Architecture", "")))
    operating_system = str(image.get("Os", ""))
    labels = (image.get("Config") or {}).get("Labels") or {}
    revision = str(labels.get("org.opencontainers.image.revision", "unknown"))

    engine_result = output_runner(
        ["docker", "info", "--format", "{{json .}}"],
        cwd=project_dir,
        env=environment,
    )
    if engine_result.returncode != 0:
        detail = engine_result.stderr.strip() or "docker info failed"
        raise ValueError(detail)
    try:
        engine = json.loads(engine_result.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("docker returned invalid engine information") from exc
    engine_os = str(engine.get("OSType", ""))
    engine_architecture = _normalise_architecture(
        str(engine.get("Architecture", ""))
    )
    if (operating_system, architecture) != (engine_os, engine_architecture):
        raise ValueError(
            "image platform does not match the Docker engine: "
            f"image={operating_system}/{architecture}, "
            f"engine={engine_os}/{engine_architecture}"
        )

    return {
        "requested_ref": image_ref,
        "resolved_ref": resolved_ref,
        "digest": digest,
        "revision": revision,
        "platform": f"{operating_system}/{architecture}",
    }


def inspect_dataset_registry(
    image_ref: str,
    *,
    project_dir: Path,
    environment: Mapping[str, str],
    output_runner: OutputRunner,
) -> dict[str, dict[str, Any]]:
    """Read the immutable dataset registry embedded in the selected image."""

    result = output_runner(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "dmf-bench",
            image_ref,
            "list-datasets",
        ],
        cwd=project_dir,
        env=environment,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or "cannot inspect image dataset registry"
        raise ValueError(detail)
    try:
        payload = json.loads(result.stdout)
        datasets = payload["datasets"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("image returned an invalid dataset registry") from exc
    if not isinstance(datasets, list):
        raise ValueError("image dataset registry must contain a datasets array")
    registry: dict[str, dict[str, Any]] = {}
    for raw_record in datasets:
        if not isinstance(raw_record, dict):
            raise ValueError("image dataset registry contains a non-object entry")
        dataset_id = raw_record.get("dataset_id")
        if not isinstance(dataset_id, str) or not dataset_id:
            raise ValueError("image dataset registry contains an invalid dataset_id")
        registry[dataset_id] = dict(raw_record)
    return registry


def load_experiment_metadata(config_paths: Sequence[Path]) -> list[dict[str, Any]]:
    metadata: list[dict[str, Any]] = []
    datasets_by_benchmark: dict[str, dict[str, Any]] = {}
    paths_by_combination: dict[tuple[str, str], Path] = {}
    for path in config_paths:
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read experiment config {path}: {exc}") from exc
        if not isinstance(config, dict):
            raise ValueError(f"experiment config must contain a JSON object: {path}")
        benchmark = _required_string(config, "benchmark", path)
        framework = _required_string(config, "framework", path)
        combination = (benchmark, framework)
        previous_path = paths_by_combination.get(combination)
        if previous_path is not None:
            raise ValueError(
                "prepared batches accept one config per benchmark/framework; "
                f"duplicate {benchmark}/{framework} configs: "
                f"{previous_path} and {path}"
            )
        paths_by_combination[combination] = path
        dataset = config.get("dataset")
        if not isinstance(dataset, dict):
            raise ValueError(f"experiment config has no dataset object: {path}")
        previous_dataset = datasets_by_benchmark.setdefault(benchmark, dataset)
        if previous_dataset != dataset:
            raise ValueError(
                f"prepared configs for benchmark {benchmark!r} do not declare "
                "the same dataset and sampling policy"
            )
        requested_models = _requested_models(config)
        # Config exports intentionally use the portable filename
        # ``experiment.json``. Run identity must therefore come from the
        # scientific matrix, never from the host filename.
        run_suffix = f"{benchmark}-{framework}"
        metadata.append(
            {
                "host_path": str(path.resolve()),
                "config_sha256": _sha256_file(path),
                "benchmark": benchmark,
                "framework": framework,
                "run_suffix": run_suffix,
                "dataset_request": json.loads(json.dumps(dataset)),
                "requested_models": requested_models,
            }
        )
    return metadata


def lock_experiment_datasets(
    experiments: Sequence[Mapping[str, Any]],
    registry: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Attach immutable dataset identity from the selected runtime image."""

    locked: list[dict[str, Any]] = []
    for experiment in experiments:
        item = dict(experiment)
        request = item.get("dataset_request")
        if not isinstance(request, dict):
            raise ValueError("prepared experiment has no dataset request")
        dataset_id = request.get("registry_id")
        if not isinstance(dataset_id, str) or not dataset_id:
            raise ValueError("prepare requires dataset.registry_id")
        try:
            record = dict(registry[dataset_id])
        except KeyError as exc:
            raise ValueError(
                f"dataset {dataset_id!r} is not present in the selected image"
            ) from exc
        if not bool(record.get("pinned")):
            raise ValueError(
                f"prepare requires a pinned dataset; {dataset_id!r} is unpinned"
            )
        expected_benchmark = str(item.get("benchmark", ""))
        if record.get("benchmark") != expected_benchmark:
            raise ValueError(
                f"dataset {dataset_id!r} belongs to {record.get('benchmark')!r}, "
                f"not {expected_benchmark!r}"
            )
        for field_name in ("source_url", "revision", "sha256", "expected_schema"):
            value = record.get(field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"pinned dataset {dataset_id!r} has no valid {field_name}"
                )
        item["dataset_lock"] = {
            "dataset_id": dataset_id,
            "benchmark": record["benchmark"],
            "source_url": record["source_url"],
            "revision": record["revision"],
            "sha256": record["sha256"],
            "expected_schema": record["expected_schema"],
            "sampling": request.get("sampling"),
        }
        locked.append(item)
    return locked


def require_provider_credentials(
    experiments: Sequence[Mapping[str, Any]],
    environment: Mapping[str, str],
) -> None:
    uses_openai = any(
        any(model.get("provider") == "openai" for model in experiment["requested_models"])
        for experiment in experiments
    )
    if uses_openai and not environment.get("OPENAI_API_KEY", "").strip():
        raise ValueError(
            "OPENAI_API_KEY is required by the prepared experiment configs"
        )


def build_preparation_state(
    *,
    project_dir: Path,
    image: Mapping[str, str],
    experiments: Sequence[Mapping[str, Any]],
    batch_id: str,
    observability: bool,
    allow_downloads: bool,
) -> dict[str, Any]:
    prepared_at = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    prepared_experiments = []
    run_ids: set[str] = set()
    for experiment in experiments:
        item = dict(experiment)
        run_id = f"{batch_id}-{experiment['run_suffix']}"
        if run_id in run_ids:
            raise ValueError(f"prepared batch would generate duplicate run_id: {run_id}")
        run_ids.add(run_id)
        item["run_id"] = run_id
        prepared_experiments.append(item)
    return {
        "schema_version": PREPARATION_SCHEMA_VERSION,
        "prepared_at": prepared_at,
        "batch_id": batch_id,
        "project_dir": str(project_dir.resolve()),
        "image": dict(image),
        "observability": observability,
        "allow_downloads": allow_downloads,
        "experiments": prepared_experiments,
    }


def write_preparation_state(path: Path, state: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_preparation_state(path: Path) -> dict[str, Any]:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read prepared context {path}: {exc}") from exc
    if (
        not isinstance(state, dict)
        or state.get("schema_version") not in SUPPORTED_PREPARATION_SCHEMA_VERSIONS
    ):
        raise ValueError(f"unsupported prepared context: {path}")
    image = state.get("image")
    experiments = state.get("experiments")
    if not isinstance(image, dict) or not isinstance(image.get("resolved_ref"), str):
        raise ValueError(f"prepared context has no resolved image: {path}")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError(f"prepared context has no experiments: {path}")
    return state


def verify_preparation_inputs(state: Mapping[str, Any]) -> None:
    """Refuse config drift for v2 preparation contexts before any run starts."""

    if state.get("schema_version") != PREPARATION_SCHEMA_VERSION:
        return
    experiments = state.get("experiments")
    if not isinstance(experiments, list):
        raise ValueError("prepared context has no experiments")
    for experiment in experiments:
        if not isinstance(experiment, dict):
            raise ValueError("prepared context contains an invalid experiment")
        path = Path(str(experiment.get("host_path", "")))
        expected = experiment.get("config_sha256")
        if not path.is_file() or not isinstance(expected, str):
            raise ValueError(f"prepared config is unavailable: {path}")
        observed = _sha256_file(path)
        if observed != expected:
            raise ValueError(
                f"prepared config changed after preparation: {path} "
                f"(expected {expected}, got {observed})"
            )
        if not isinstance(experiment.get("dataset_lock"), dict):
            raise ValueError(f"prepared config has no dataset lock: {path}")


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matching_repo_digest(image_ref: str, repo_digests: Sequence[str]) -> str:
    repository = image_ref.split("@", 1)[0]
    final_slash = repository.rfind("/")
    final_colon = repository.rfind(":")
    if final_colon > final_slash:
        repository = repository[:final_colon]
    prefix = f"{repository}@"
    return next((item for item in repo_digests if item.startswith(prefix)), repo_digests[0])


def _normalise_architecture(value: str) -> str:
    return {"aarch64": "arm64", "x86_64": "amd64"}.get(value, value)


def _required_string(config: Mapping[str, Any], key: str, path: Path) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"experiment config has no valid {key}: {path}")
    return value


def _requested_models(config: Mapping[str, Any]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    models = config.get("models")
    if not isinstance(models, dict):
        return result
    for role, declaration in models.items():
        if not isinstance(declaration, dict):
            continue
        provider = declaration.get("provider")
        model = declaration.get("requested_model")
        if isinstance(provider, str) and isinstance(model, str):
            result.append({"role": str(role), "provider": provider, "model": model})
    return result
