"""Scientific and operational provenance for run manifests."""

from __future__ import annotations

import os
import platform
from importlib import metadata
from pathlib import Path
from typing import Any

from dmf_bench.config import redact_secrets
from dmf_bench.contracts import hash_canonical_json, sha256_file


def build_run_provenance(
    config: dict[str, Any],
    *,
    fingerprint_inputs: dict[str, Any],
    source_path: str | Path | None = None,
) -> dict[str, Any]:
    sanitized = redact_secrets(config)
    source = Path(source_path or sanitized.get("source_path", "")).resolve() if source_path or sanitized.get("source_path") else None
    dataset = _mapping(sanitized.get("dataset"))
    models = _mapping(sanitized.get("models"))

    return {
        "schema_version": 1,
        "scientific": {
            "resolved_config_sha256": hash_canonical_json(_scientific_config(sanitized)),
            "fingerprint_inputs_sha256": hash_canonical_json(fingerprint_inputs),
            "dataset": {
                "name": dataset.get("name", ""),
                "source": dataset.get("source", ""),
                "revision": dataset.get("revision", ""),
                "sha256": dataset.get("sha256", ""),
                "registry_id": dataset.get("registry_id", ""),
                "expected_schema": dataset.get("expected_schema", ""),
            },
            "selection": sanitized.get("selection", {}),
            "models": {
                role: _model_request(_mapping(models.get(role)))
                for role in ("answerer", "judge")
                if role in models
            },
            "evaluation": {
                "config_sha256": hash_canonical_json(sanitized.get("evaluation", {})),
                "rubric_hash": hash_canonical_json(sanitized.get("evaluation", {})),
            },
            "prompt_surface": {
                "policy": "adapter-owned; concrete prompts are captured in prediction artifacts, not manifest",
                "config_hash": hash_canonical_json(
                    {
                        "benchmark": sanitized.get("benchmark"),
                        "framework": sanitized.get("framework"),
                        "protocol": sanitized.get("protocol"),
                    }
                ),
            },
        },
        "operational": {
            "harness": {
                "package_version": _package_version(),
                "commit": os.getenv("DMF_BENCH_HARNESS_COMMIT", "unknown"),
                "source_config_path": str(source) if source else "",
                "source_config_sha256": sha256_file(source) if source and source.is_file() else "",
            },
            "image": {
                "digest": os.getenv("DMF_BENCH_IMAGE_DIGEST", "unknown"),
            },
            "framework": {
                "name": sanitized.get("framework", ""),
                "version": os.getenv("DMF_BENCH_FRAMEWORK_VERSION", "unknown"),
                "commit": os.getenv("DMF_BENCH_FRAMEWORK_COMMIT", "unknown"),
            },
            "qdrant": {
                "url": os.getenv("QDRANT_URL", ""),
                "version": os.getenv("QDRANT_VERSION", "unknown"),
                "image_digest": os.getenv("QDRANT_IMAGE_DIGEST", "unknown"),
                "config_hash": os.getenv("QDRANT_CONFIG_SHA256", ""),
            },
            "platform": {
                "system": platform.system(),
                "machine": platform.machine(),
                "python": platform.python_version(),
                "resources": {
                    "cpu_limit": os.getenv("DMF_BENCH_CPU_LIMIT", ""),
                    "memory_limit": os.getenv("DMF_BENCH_MEMORY_LIMIT", ""),
                },
            },
            "resume_lineage": sanitized.get("resume_lineage", []),
        },
        "limits": {
            "remote_provider_bitwise_reproducibility": "not guaranteed unless the provider explicitly guarantees it",
            "returned_model_id": "captured in prediction/judge artifacts when providers return it",
        },
    }


def _scientific_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: config.get(key)
        for key in (
            "schema_version",
            "experiment_id",
            "benchmark",
            "framework",
            "protocol",
            "dataset",
            "selection",
            "models",
            "evaluation",
            "artifact_store",
        )
    }


def _model_request(model: dict[str, Any]) -> dict[str, Any]:
    return {
        "provider": model.get("provider", ""),
        "requested_model": model.get("requested_model", ""),
        "parameters_sha256": hash_canonical_json(model.get("parameters", {})),
    }


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _package_version() -> str:
    try:
        return metadata.version("dmf-bench")
    except metadata.PackageNotFoundError:
        return "0.1.0"
