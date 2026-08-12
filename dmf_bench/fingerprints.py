"""Canonical scientific fingerprint inputs for benchmark runs."""

from __future__ import annotations

from typing import Any

from dmf_bench.providers.judge_prompts import (
    JUDGE_RETRY_INSTRUCTION,
    JUDGE_SYSTEM_PROMPT,
    JUDGE_USER_PROMPT_TEMPLATE,
)
from dmf_bench.contracts import SCIENTIFIC_FINGERPRINT_SCHEMA_VERSION, hash_canonical_json


JUDGE_ADAPTER_VERSION = 2
JUDGE_PARSER_VERSIONS = {
    "locomo": "locomo-judge-parser-v1",
    "longmemeval": "longmemeval-judge-parser-v1",
}


def build_scientific_fingerprint_inputs(
    config: dict[str, Any],
    *,
    expected_question_ids: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Return only inputs that can change scientific benchmark results."""
    payload = {
        "schema_version": SCIENTIFIC_FINGERPRINT_SCHEMA_VERSION,
        "benchmark": config.get("benchmark"),
        "framework": config.get("framework"),
        "framework_config": framework_config_identity(config.get("framework_config")),
        "dataset": dataset_identity(config.get("dataset")),
        "selection": _mapping(config.get("selection")),
        "models": {
            role: model_scientific_identity(_mapping(_mapping(config.get("models")).get(role)))
            for role in ("answerer", "judge")
        },
        "judge_contract": judge_contract_identity(str(config.get("benchmark", ""))),
        "evaluation": _mapping(config.get("evaluation")),
    }
    if expected_question_ids is not None:
        payload["expected_question_ids"] = list(expected_question_ids)
    return payload


def scientific_config_payload(config: dict[str, Any]) -> dict[str, Any]:
    """Return the scientific portion of a resolved experiment config."""
    return build_scientific_fingerprint_inputs(config)


def dataset_identity(value: Any) -> dict[str, Any]:
    dataset = _mapping(value)
    return {
        key: dataset.get(key)
        for key in (
            "name",
            "source",
            "revision",
            "sha256",
            "registry_id",
            "expected_schema",
        )
        if dataset.get(key) is not None
    }


def framework_config_identity(value: Any) -> dict[str, Any]:
    framework_config = _mapping(value)
    return {
        key: framework_config.get(key)
        for key in ("format", "profile", "sha256")
        if framework_config.get(key) is not None
    }


def model_scientific_identity(model: dict[str, Any]) -> dict[str, Any]:
    return {
        key: model.get(key)
        for key in ("provider", "endpoint_identity", "requested_model", "parameters")
        if model.get(key) is not None
    }


def judge_contract_identity(benchmark: str) -> dict[str, Any]:
    """Return the versioned rubric/prompt/parser identity for one benchmark."""
    try:
        parser_version = JUDGE_PARSER_VERSIONS[benchmark]
    except KeyError as exc:
        raise ValueError(f"Unsupported judge benchmark: {benchmark!r}.") from exc
    return {
        "schema_version": 1,
        "adapter_version": JUDGE_ADAPTER_VERSION,
        "benchmark": benchmark,
        "system_prompt": JUDGE_SYSTEM_PROMPT,
        "user_prompt_template": JUDGE_USER_PROMPT_TEMPLATE,
        "response_retry_instruction": JUDGE_RETRY_INSTRUCTION,
        "optional_context_fields": (
            ["question_type", "question_date"]
            if benchmark == "longmemeval"
            else []
        ),
        "parser_version": parser_version,
    }


def judge_fingerprint(benchmark: str) -> str:
    """Hash the complete scientific judge surface used by the adapter."""
    return hash_canonical_json(judge_contract_identity(benchmark))


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}
