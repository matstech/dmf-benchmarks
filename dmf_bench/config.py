"""Experiment config loading, validation, and secret redaction."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
                    name: os.getenv(name)
                    for name in ENV_ENDPOINT_NAMES
                    if os.getenv(name)
                }
        return redacted_data


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
    protocol: str | None = None,
) -> ResolvedConfig:
    source_path = Path(path).resolve()
    data = load_experiment_config(source_path)

    if benchmark is not None:
        data["benchmark"] = benchmark
    if framework is not None:
        data["framework"] = framework
    if protocol is not None:
        data["protocol"] = protocol

    validate_config(data, source_path=source_path)
    data["source_path"] = str(source_path)
    return ResolvedConfig(source_path=source_path, data=data)


def validate_config(data: dict[str, Any], *, source_path: Path) -> None:
    if data.get("schema_version") != 1:
        raise ValueError("Experiment config must declare schema_version=1.")

    benchmark = required_string(data, "benchmark")
    framework = required_string(data, "framework")
    protocol = required_string(data, "protocol")
    validate_combination(benchmark, framework, protocol)

    runtime = required_mapping(data, "runtime")
    for key in ("root", "runs_dir", "cache_dir"):
        validate_absolute_path(runtime, key, source_path=source_path)

    dataset = required_mapping(data, "dataset")
    if dataset.get("name") != benchmark:
        raise ValueError("dataset.name must match benchmark.")
    validate_absolute_path(dataset, "path", source_path=source_path)
    require_sha256(dataset, "sha256")

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
        parameters = model.get("parameters")
        if parameters is not None and not isinstance(parameters, dict):
            raise ValueError(f"models.{role}.parameters must be an object.")

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


def require_sha256(data: dict[str, Any], key: str) -> None:
    value = data.get(key)
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{key} must be a 64-character lowercase SHA-256 hex digest.")
    if any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{key} must be a lowercase SHA-256 hex digest.")


def redact_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = str(key).lower().replace("-", "_")
            if normalized_key in SECRET_FIELD_NAMES or any(
                part in normalized_key for part in ("api_key", "password", "secret", "token")
            ):
                redacted[key] = "<redacted>"
            else:
                redacted[key] = redact_secrets(item)
        return redacted
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    return value
