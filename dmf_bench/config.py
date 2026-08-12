"""Experiment config loading, validation, and secret redaction."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .atomic_io import write_json_atomic
from .contracts import EXPERIMENT_CONFIG_SCHEMA_VERSION, sha256_file
from .registry import validate_combination


SECRET_FIELD_NAMES = {
    "api_key",
    "apikey",
    "authorization",
    "password",
    "secret",
    "token",
}

ENV_SECRET_NAMES = (
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "QDRANT_API_KEY",
)

ENV_ENDPOINT_NAMES = (
    "OPENAI_BASE_URL",
    "OPENROUTER_BASE_URL",
    "OLLAMA_BASE_URL",
    "QDRANT_URL",
)

EXPERIMENT_CONFIG_FIELDS = frozenset(
    {
        "schema_version",
        "experiment_id",
        "benchmark",
        "framework",
        "runtime",
        "framework_config",
        "qdrant",
        "dataset",
        "selection",
        "models",
        "evaluation",
        "artifact_store",
    }
)


@dataclass(frozen=True)
class ResolvedConfig:
    source_path: Path
    data: dict[str, Any]

    def redacted(self) -> dict[str, Any]:
        redacted_data = redact_secrets(self.data)
        runtime = redacted_data.setdefault("runtime", {})
        if isinstance(runtime, dict):
            env = runtime.setdefault("environment", {})
            if isinstance(env, dict):
                env["secrets_present"] = {
                    name: bool(os.getenv(name))
                    for name in ENV_SECRET_NAMES
                }
                env["endpoints"] = {
                    name: sanitize_endpoint(os.environ[name])
                    for name in ENV_ENDPOINT_NAMES
                    if os.getenv(name)
                }
        return redacted_data

    def persist_redacted(self, run_dir: str | Path) -> Path:
        """Persist the resolved non-secret config for a future run preflight."""
        target = Path(run_dir) / "resolved-config.json"
        write_json_atomic(target, self.redacted())
        return target


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise ValueError(f"Config file not found: {config_path}")
    if not config_path.is_file():
        raise ValueError(f"Config path is not a file: {config_path}")
    if config_path.suffix.lower() != ".json":
        raise ValueError(
            f"Experiment config must be JSON in this phase: {config_path}"
        )

    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON config {config_path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("Experiment config root must be a JSON object.")
    return data


def resolve_config(
    path: str | Path,
    *,
    benchmark: str | None = None,
    framework: str | None = None,
    materialize_datasets: bool = False,
    dataset_registry_path: str | Path | None = None,
) -> ResolvedConfig:
    source_path = Path(path).resolve()
    data = load_experiment_config(source_path)

    if benchmark is not None:
        data["benchmark"] = benchmark
    if framework is not None:
        data["framework"] = framework
    if materialize_datasets:
        from .datasets import materialize_dataset_for_config

        data = materialize_dataset_for_config(
            data,
            registry_path=dataset_registry_path,
        )
    _resolve_relative_resource_paths(data, source_path=source_path)
    validate_config(data, source_path=source_path)
    data["source_path"] = str(source_path)
    return ResolvedConfig(source_path=source_path, data=data)


def _resolve_relative_resource_paths(
    data: dict[str, Any],
    *,
    source_path: Path,
) -> None:
    """Make exported configuration bundles portable inside their mounted directory."""

    for section_name in ("framework_config", "dataset"):
        section = data.get(section_name)
        if not isinstance(section, dict):
            continue
        raw_path = section.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            continue
        path = Path(raw_path)
        if not path.is_absolute():
            section["path"] = str((source_path.parent / path).resolve())


def validate_config(data: dict[str, Any], *, source_path: Path) -> None:
    if data.get("schema_version") != EXPERIMENT_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            "Experiment config must declare "
            f"schema_version={EXPERIMENT_CONFIG_SCHEMA_VERSION}; "
            f"v{data.get('schema_version')} configs are not supported."
        )
    unsupported_fields = sorted(set(data) - EXPERIMENT_CONFIG_FIELDS)
    if unsupported_fields:
        raise ValueError(
            f"Experiment config contains unsupported fields: {unsupported_fields}."
        )
    reject_inline_secrets(data)

    benchmark = required_string(data, "benchmark")
    framework = required_string(data, "framework")
    validate_combination(benchmark, framework)

    runtime = required_mapping(data, "runtime")
    for key in ("root", "runs_dir", "cache_dir"):
        validate_absolute_path(runtime, key, source_path=source_path)
    runtime_root = Path(str(runtime["root"])).resolve()
    for key in ("runs_dir", "cache_dir"):
        validate_path_within_root(
            Path(str(runtime[key])),
            root=runtime_root,
            field_name=f"runtime.{key}",
        )
    validate_port(runtime, "metrics_port")
    log_level = required_string(runtime, "log_level").upper()
    if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        raise ValueError("runtime.log_level must be DEBUG, INFO, WARNING, or ERROR.")

    framework_config = required_mapping(data, "framework_config")
    validate_absolute_path(framework_config, "path", source_path=source_path)
    validate_path_within_root(
        Path(str(framework_config["path"])),
        root=runtime_root,
        field_name="framework_config.path",
    )
    require_sha256(framework_config, "sha256")
    framework_format = required_string(framework_config, "format").lower()
    expected_format = "toml" if framework == "dmf" else "yaml"
    if framework_format != expected_format:
        raise ValueError(
            f"framework_config.format must be {expected_format!r} for framework {framework!r}."
        )
    verify_pinned_file(
        framework_config,
        field_name="framework_config",
    )

    qdrant = required_mapping(data, "qdrant")
    if required_string(qdrant, "endpoint_env") != "QDRANT_URL":
        raise ValueError("qdrant.endpoint_env must be 'QDRANT_URL'.")
    if required_string(qdrant, "retention") not in {"keep", "delete-on-success"}:
        raise ValueError("qdrant.retention must be keep or delete-on-success.")
    validate_positive_number(qdrant, "request_timeout_seconds")

    dataset = required_mapping(data, "dataset")
    if dataset.get("name") != benchmark:
        raise ValueError("dataset.name must match benchmark.")
    validate_absolute_path(dataset, "path", source_path=source_path)
    validate_path_within_root(
        Path(str(dataset["path"])),
        root=runtime_root,
        field_name="dataset.path",
    )
    require_sha256(dataset, "sha256")
    verify_dataset_file(dataset)

    selection = required_mapping(data, "selection")
    ordered_item_ids = selection.get("ordered_item_ids")
    if not isinstance(ordered_item_ids, list) or not ordered_item_ids:
        raise ValueError("selection.ordered_item_ids must be a non-empty list.")
    if not all(isinstance(item, str) and item for item in ordered_item_ids):
        raise ValueError("selection.ordered_item_ids must contain non-empty strings.")

    models = required_mapping(data, "models")
    for role in ("answerer", "judge"):
        model = required_mapping(models, role)
        required_string(model, "provider")
        required_string(model, "requested_model")
        parameters = required_mapping(model, "parameters")
        unsupported_parameters = sorted(
            set(parameters) - {"temperature", "max_tokens", "reasoning_effort"}
        )
        if unsupported_parameters:
            raise ValueError(
                f"models.{role}.parameters contains unsupported fields: "
                f"{unsupported_parameters}."
            )
        validate_non_negative_number(parameters, "temperature")
        validate_positive_integer(parameters, "max_tokens")
        reasoning_effort = parameters.get("reasoning_effort")
        if reasoning_effort is not None and (
            not isinstance(reasoning_effort, str) or not reasoning_effort.strip()
        ):
            raise ValueError(
                f"models.{role}.parameters.reasoning_effort must be a non-empty string or null."
            )
        model_runtime = required_mapping(model, "runtime")
        validate_positive_number(model_runtime, "timeout_seconds")
        validate_positive_integer(model_runtime, "rpm")
        validate_non_negative_integer(model_runtime, "max_retries")
        if "response_max_retries" in model_runtime:
            validate_non_negative_integer(model_runtime, "response_max_retries")

    evaluation = required_mapping(data, "evaluation")
    for key in ("required", "optional"):
        value = evaluation.get(key)
        if not isinstance(value, list):
            raise ValueError(f"evaluation.{key} must be a list.")
        if not all(isinstance(item, str) and item for item in value):
            raise ValueError(f"evaluation.{key} must contain non-empty strings.")

    artifact_store = required_mapping(data, "artifact_store")
    store_type = artifact_store.get("type")
    if store_type not in {"local", "s3-compatible"}:
        raise ValueError("artifact_store.type must be local or s3-compatible.")
    if store_type == "local":
        store_uri = Path(required_string(artifact_store, "uri"))
        if not store_uri.is_absolute():
            raise ValueError("artifact_store.uri must be absolute for a local store.")
        validate_path_within_root(
            store_uri,
            root=runtime_root,
            field_name="artifact_store.uri",
        )
        if store_uri.resolve() != Path(str(runtime["runs_dir"])).resolve():
            raise ValueError("artifact_store.uri must match runtime.runs_dir.")


def required_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object.")
    return value


def required_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string.")
    return value.strip()


def validate_absolute_path(
    data: dict[str, Any],
    key: str,
    *,
    source_path: Path,
) -> None:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty absolute path.")
    if not Path(value).is_absolute():
        raise ValueError(
            f"{key} must be absolute in {source_path}; got {value!r}."
        )


def validate_path_within_root(
    path: Path,
    *,
    root: Path,
    field_name: str,
) -> None:
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(
            f"{field_name} must be contained by runtime.root {root}; got {resolved}."
        )


def require_sha256(data: dict[str, Any], key: str) -> None:
    value = data.get(key)
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{key} must be a 64-character lowercase SHA-256 hex digest.")
    if any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{key} must be a lowercase SHA-256 hex digest.")


def validate_port(data: dict[str, Any], key: str) -> None:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 65535:
        raise ValueError(f"{key} must be an integer between 1 and 65535.")


def validate_positive_number(data: dict[str, Any], key: str) -> None:
    value = data.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{key} must be a positive number.")


def validate_non_negative_number(data: dict[str, Any], key: str) -> None:
    value = data.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{key} must be a non-negative number.")


def validate_positive_integer(data: dict[str, Any], key: str) -> None:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{key} must be a positive integer.")


def validate_non_negative_integer(data: dict[str, Any], key: str) -> None:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{key} must be a non-negative integer.")


def verify_dataset_file(dataset: dict[str, Any]) -> None:
    dataset_path = Path(str(dataset["path"]))
    if not dataset_path.is_file():
        raise ValueError(f"dataset.path must point to a materialized file: {dataset_path}")
    expected_sha256 = str(dataset["sha256"])
    observed_sha256 = sha256_file(dataset_path)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            "dataset SHA-256 mismatch: "
            f"expected {expected_sha256}, got {observed_sha256}"
        )


def verify_pinned_file(data: dict[str, Any], *, field_name: str) -> None:
    path = Path(str(data["path"]))
    if not path.is_file():
        raise ValueError(f"{field_name}.path must point to a materialized file: {path}")
    expected_sha256 = str(data["sha256"])
    observed_sha256 = sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            f"{field_name} SHA-256 mismatch: expected {expected_sha256}, got {observed_sha256}"
        )


def reject_inline_secrets(value: Any, *, path: str = "config") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            key_path = f"{path}.{key}"
            if is_secret_field_name(str(key)):
                raise ValueError(
                    f"Inline secret field {key_path} is forbidden; use environment variables."
                )
            reject_inline_secrets(item, path=key_path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            reject_inline_secrets(item, path=f"{path}[{index}]")


def redact_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = str(key).lower().replace("-", "_")
            if is_secret_field_name(normalized_key):
                redacted[key] = "<redacted>"
            else:
                redacted[key] = redact_secrets(item)
        return redacted
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    return value


def sanitize_endpoint(value: str) -> str:
    """Keep only a credential-free endpoint origin for persisted metadata."""
    try:
        parsed = urlsplit(value.strip())
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return "<redacted-endpoint>"
    if not parsed.scheme or not hostname:
        return "<redacted-endpoint>"
    rendered_host = f"[{hostname}]" if ":" in hostname else hostname
    netloc = f"{rendered_host}:{port}" if port is not None else rendered_host
    return urlunsplit((parsed.scheme, netloc, "", "", ""))


def is_secret_field_name(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return (
        normalized in SECRET_FIELD_NAMES
        or "api_key" in normalized
        or normalized.endswith(("_password", "_secret", "_token"))
    )
