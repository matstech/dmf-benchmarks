"""Experiment config loading, validation, and secret redaction."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .atomic_io import write_json_atomic
from .context import PACKING_ID, RENDERER_ID, TOKENIZER_ID
from .contracts import EXPERIMENT_CONFIG_SCHEMA_VERSION, sha256_file
from .registry import FRAMEWORKS, validate_combination


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
    allow_dataset_downloads: bool = False,
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
            allow_downloads=allow_dataset_downloads,
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

    for section_name in ("framework_config", "dataset", "ablation"):
        section = data.get(section_name)
        if not isinstance(section, dict):
            continue
        path_key = "full_config_path" if section_name == "ablation" else "path"
        raw_path = section.get(path_key)
        if not isinstance(raw_path, str) or not raw_path.strip():
            continue
        path = Path(raw_path)
        if not path.is_absolute():
            section[path_key] = str((source_path.parent / path).resolve())


def validate_config(data: dict[str, Any], *, source_path: Path) -> None:
    if data.get("schema_version") != EXPERIMENT_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            "Experiment config must declare "
            f"schema_version={EXPERIMENT_CONFIG_SCHEMA_VERSION}; "
            f"v{data.get('schema_version')} configs are not supported."
        )
    validate_v3_config(data, source_path=source_path)


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


V3_CONFIG_FIELDS = frozenset(
    {
        "schema_version", "experiment_id", "scientific_profile", "benchmark",
        "framework", "runtime", "framework_config", "storage", "dataset",
        "selection", "retrieval", "context_budget", "models", "evaluation",
        "artifact_store", "preset", "ablation",
    }
)
V3_EVALUATORS = frozenset(
    {
        "primary_judge_score", "rigorous_report", "analysis_rows",
        "retrieval_report", "judge_agreement", "ablation_report",
    }
)
V3_TOKENIZERS = frozenset({TOKENIZER_ID})
V3_RENDERERS = frozenset({RENDERER_ID})
V3_PACKING_POLICIES = frozenset({PACKING_ID})


def _reject_unknown_fields(data: dict[str, Any], allowed: frozenset[str], name: str) -> None:
    unsupported = sorted(set(data) - allowed)
    if unsupported:
        raise ValueError(f"{name} contains unsupported fields: {unsupported}.")


def validate_v3_config(data: dict[str, Any], *, source_path: Path) -> None:
    """Validate the isolated v3 format before activating the v3 runtime."""
    if data.get("schema_version") != 3:
        raise ValueError("Experiment config must declare schema_version=3.")
    unsupported = sorted(set(data) - V3_CONFIG_FIELDS)
    if unsupported:
        raise ValueError(f"V3 config contains unsupported fields: {unsupported}.")
    reject_inline_secrets(data)
    required_string(data, "experiment_id")
    required_string(data, "scientific_profile")
    benchmark = required_string(data, "benchmark")
    framework = required_string(data, "framework")
    validate_combination(benchmark, framework)
    descriptor = FRAMEWORKS[framework]

    runtime = required_mapping(data, "runtime")
    _reject_unknown_fields(
        runtime, frozenset({"root", "runs_dir", "cache_dir", "metrics_port", "log_level"}),
        "runtime",
    )
    for key in ("root", "runs_dir", "cache_dir"):
        validate_absolute_path(runtime, key, source_path=source_path)
    runtime_root = Path(str(runtime["root"])).resolve()
    for key in ("runs_dir", "cache_dir"):
        validate_path_within_root(
            Path(str(runtime[key])), root=runtime_root, field_name=f"runtime.{key}"
        )
    validate_port(runtime, "metrics_port")
    if required_string(runtime, "log_level").upper() not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        raise ValueError("runtime.log_level must be DEBUG, INFO, WARNING, or ERROR.")

    framework_config = required_mapping(data, "framework_config")
    _reject_unknown_fields(
        framework_config, frozenset({"path", "sha256", "format", "profile"}),
        "framework_config",
    )
    validate_absolute_path(framework_config, "path", source_path=source_path)
    validate_path_within_root(
        Path(str(framework_config["path"])), root=runtime_root,
        field_name="framework_config.path",
    )
    require_sha256(framework_config, "sha256")
    if required_string(framework_config, "format") not in descriptor.config_formats:
        raise ValueError(
            f"framework_config.format must be one of {sorted(descriptor.config_formats)!r}."
        )
    required_string(framework_config, "profile")
    verify_pinned_file(framework_config, field_name="framework_config")
    from .ablations import PROFILE_CHANGES

    if framework_config["profile"] in PROFILE_CHANGES and "ablation" not in data:
        raise ValueError("DMF ablation profile requires a pinned full configuration.")
    if "ablation" in data:
        from .ablations import ablation_identity

        ablation = required_mapping(data, "ablation")
        validate_absolute_path(
            {"full_config_path": ablation.get("full_config_path")},
            "full_config_path", source_path=source_path,
        )
        validate_path_within_root(
            Path(str(ablation["full_config_path"])), root=runtime_root,
            field_name="ablation.full_config_path",
        )
        ablation_identity(data)

    storage = required_mapping(data, "storage")
    _reject_unknown_fields(
        storage,
        frozenset({"kind", "profile", "retention", "endpoint_env", "request_timeout_seconds", "root"}),
        "storage",
    )
    storage_kind = required_string(storage, "kind")
    if storage_kind not in descriptor.storage_kinds:
        raise ValueError(
            f"storage.kind {storage_kind!r} is unsupported by framework {framework!r}."
        )
    required_string(storage, "profile")
    if required_string(storage, "retention") not in {"keep", "delete-on-success"}:
        raise ValueError("storage.retention must be keep or delete-on-success.")
    if storage_kind == "qdrant-server":
        if required_string(storage, "endpoint_env") != "QDRANT_URL":
            raise ValueError("storage.endpoint_env must be QDRANT_URL.")
        validate_positive_number(storage, "request_timeout_seconds")
    elif storage_kind in {"none", "embedded-local"}:
        if any(key in storage for key in ("endpoint_env", "request_timeout_seconds", "root")):
            raise ValueError(f"storage.kind {storage_kind!r} does not use endpoint or root fields.")

    dataset = required_mapping(data, "dataset")
    if required_string(dataset, "name") != benchmark:
        raise ValueError("dataset.name must match benchmark.")
    validate_absolute_path(dataset, "path", source_path=source_path)
    validate_path_within_root(
        Path(str(dataset["path"])), root=runtime_root, field_name="dataset.path"
    )
    require_sha256(dataset, "sha256")
    verify_dataset_file(dataset)

    selection = required_mapping(data, "selection")
    _reject_unknown_fields(
        selection, frozenset({"ordered_item_ids", "filters", "seed"}), "selection",
    )
    item_ids = selection.get("ordered_item_ids")
    if not isinstance(item_ids, list) or not item_ids:
        raise ValueError("selection.ordered_item_ids must be a non-empty list.")
    if any(not isinstance(item, str) or not item.strip() for item in item_ids):
        raise ValueError("selection.ordered_item_ids must contain non-empty strings.")
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("selection.ordered_item_ids contains duplicate IDs.")
    if not isinstance(selection.get("filters"), dict):
        raise ValueError("selection.filters must be an object.")

    retrieval = required_mapping(data, "retrieval")
    _reject_unknown_fields(retrieval, frozenset({"max_results"}), "retrieval")
    validate_positive_integer(retrieval, "max_results")
    budget = required_mapping(data, "context_budget")
    _reject_unknown_fields(
        budget, frozenset({"max_tokens", "tokenizer", "renderer", "packing"}),
        "context_budget",
    )
    validate_positive_integer(budget, "max_tokens")
    for key, registry in (
        ("tokenizer", V3_TOKENIZERS), ("renderer", V3_RENDERERS),
        ("packing", V3_PACKING_POLICIES),
    ):
        value = required_string(budget, key)
        if value not in registry:
            raise ValueError(f"context_budget.{key} is not registered: {value!r}.")

    models = required_mapping(data, "models")
    if set(models) != {"answerer", "judges"}:
        raise ValueError("models must contain exactly answerer and judges.")
    _validate_v3_model(required_mapping(models, "answerer"), "models.answerer")
    judges = models["judges"]
    if not isinstance(judges, list) or not judges:
        raise ValueError("models.judges must be a non-empty list.")
    judge_ids: set[str] = set()
    for index, judge in enumerate(judges):
        if not isinstance(judge, dict):
            raise ValueError(f"models.judges[{index}] must be an object.")
        judge_id = required_string(judge, "id")
        if judge_id in judge_ids:
            raise ValueError(f"Duplicate judge ID: {judge_id!r}.")
        judge_ids.add(judge_id)
        _validate_v3_model(judge, f"models.judges[{index}]")

    evaluation = required_mapping(data, "evaluation")
    _reject_unknown_fields(
        evaluation, frozenset({"primary_judge_id", "required", "optional"}),
        "evaluation",
    )
    primary_judge_id = required_string(evaluation, "primary_judge_id")
    if primary_judge_id not in judge_ids:
        raise ValueError("evaluation.primary_judge_id must resolve to a configured judge.")
    for key in ("required", "optional"):
        names = evaluation.get(key)
        if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
            raise ValueError(f"evaluation.{key} must be a list of names.")
        if len(names) != len(set(names)):
            raise ValueError(f"evaluation.{key} must contain unique names.")
        if any(name not in V3_EVALUATORS for name in names):
            raise ValueError(f"evaluation.{key} contains an unregistered evaluator.")
    if set(evaluation["required"]) & set(evaluation["optional"]):
        raise ValueError("An evaluator cannot be both required and optional.")
    mandatory_evaluators = {"primary_judge_score", "rigorous_report", "analysis_rows"}
    if not mandatory_evaluators.issubset(evaluation["required"]):
        raise ValueError(
            "evaluation.required must include primary_judge_score, rigorous_report, and analysis_rows."
        )
    from .evaluation.registry import evaluation_plan_v3

    plan = {
        requirement.name: requirement
        for requirement in evaluation_plan_v3(
            benchmark, framework, tuple(judge["id"] for judge in judges)
        )
    }
    for name in evaluation["required"]:
        requirement = plan[name]
        if requirement.not_applicable_reason is not None:
            raise ValueError(
                f"Required evaluator {name!r} is NOT_APPLICABLE: "
                f"{requirement.not_applicable_reason}"
            )

    artifact_store = required_mapping(data, "artifact_store")
    _reject_unknown_fields(
        artifact_store, frozenset({"type", "uri"}), "artifact_store",
    )
    if artifact_store.get("type") != "local":
        raise ValueError("V3 currently supports only local artifact storage.")
    uri = Path(required_string(artifact_store, "uri"))
    if not uri.is_absolute():
        raise ValueError("artifact_store.uri must be absolute.")
    validate_path_within_root(uri, root=runtime_root, field_name="artifact_store.uri")
    if uri.resolve() != Path(str(runtime["runs_dir"])).resolve():
        raise ValueError("artifact_store.uri must match runtime.runs_dir.")

    if "preset" in data:
        preset = required_mapping(data, "preset")
        _reject_unknown_fields(
            preset, frozenset({"preset_id", "profile", "fingerprint"}), "preset",
        )
        required_string(preset, "preset_id")
        required_string(preset, "profile")
        require_sha256(preset, "fingerprint")


def _validate_v3_model(model: dict[str, Any], name: str) -> None:
    _reject_unknown_fields(
        model,
        frozenset({"id", "provider", "endpoint_identity", "requested_model", "parameters", "runtime"}),
        name,
    )
    required_string(model, "provider")
    required_string(model, "requested_model")
    parameters = required_mapping(model, "parameters")
    if set(parameters) - {"temperature", "max_tokens", "reasoning_effort"}:
        raise ValueError(f"{name}.parameters contains unsupported fields.")
    validate_non_negative_number(parameters, "temperature")
    validate_positive_integer(parameters, "max_tokens")
    effort = parameters.get("reasoning_effort")
    if effort is not None and (not isinstance(effort, str) or not effort.strip()):
        raise ValueError(f"{name}.parameters.reasoning_effort must be a non-empty string.")
    runtime = required_mapping(model, "runtime")
    _reject_unknown_fields(
        runtime,
        frozenset({"timeout_seconds", "rpm", "max_retries", "response_max_retries"}),
        f"{name}.runtime",
    )
    validate_positive_number(runtime, "timeout_seconds")
    validate_positive_integer(runtime, "rpm")
    validate_non_negative_integer(runtime, "max_retries")
    if "response_max_retries" in runtime:
        validate_non_negative_integer(runtime, "response_max_retries")
