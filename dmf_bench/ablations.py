"""Pinned DMF component ablations and their declared scientific diffs."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from dmf_bench.contracts import hash_canonical_json, sha256_file


PROFILE_CHANGES: dict[str, dict[str, Any]] = {
    "dmf-full": {},
    "dmf-no-temporal-decay": {
        "temporal_decay.lambda_base": 0.0,
        "temporal_decay.inertia_strength": 0.0,
        "temporal_decay.hard_kill_threshold": 0.0,
    },
    "dmf-no-salience-scoring": {},
    "dmf-no-structured-cards": {
        "ltm.cards_enabled": False,
        "retrieval.enable_card_semantic": False,
        "retrieval.enable_card_symbolic": False,
    },
    "dmf-semantic-only": {
        "ltm.cards_enabled": False,
        "retrieval.enable_card_semantic": False,
        "retrieval.enable_card_symbolic": False,
        "retrieval.enable_raw_lexical": False,
        "retrieval.include_neighbor_turns": False,
        "retrieval.include_superseded_when_historical": False,
    },
}

PROFILE_POLICY = {
    "dmf-full": "pinned-dmf-0.3.0",
    "dmf-no-temporal-decay": "zero-decay-inertia-and-hard-kill-v1",
    "dmf-no-salience-scoring": "constant-0.5-unstable-for-every-event-v1",
    "dmf-no-structured-cards": "no-card-construction-or-retrieval-v1",
    "dmf-semantic-only": "raw-semantic-only-no-card-lexical-or-evidence-expansion-v1",
}

FULL_CONFIG_SHA256 = {
    "locomo": "54c99182ca2e5506b12a0480559a026f5cb445587832a4eb1a0653c3fbc5a890",
    "longmemeval": "5ce36630b0d6d186efc2fdbc9d2e75af8f7c7de1789e23779c5216ef0192c637",
}


def _flatten(value: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for section, fields in value.items():
        if not isinstance(fields, dict):
            raise ValueError(f"DMF TOML section {section!r} must be a table.")
        for field, item in fields.items():
            flattened[f"{section}.{field}"] = item
    return flattened


def scientific_controls_sha256(config: dict[str, Any]) -> str:
    """Hash experiment controls stable across dataset materialization."""
    dataset = config["dataset"]
    storage = config["storage"]
    payload = {
        key: config[key] for key in (
            "scientific_profile", "benchmark", "framework", "selection",
            "retrieval", "context_budget", "models", "evaluation",
        )
    }
    payload["dataset"] = {
        key: dataset[key] for key in (
            "name", "source", "revision", "registry_id", "sampling",
        ) if key in dataset
    }
    payload["storage"] = {key: storage[key] for key in ("kind", "profile")}
    return hash_canonical_json(payload)


def ablation_identity(config: dict[str, Any]) -> dict[str, Any] | None:
    """Validate a variant against its pinned full config before runtime startup."""
    declared = config.get("ablation")
    if declared is None:
        return None
    if config.get("framework") != "dmf" or not isinstance(declared, dict):
        raise ValueError("Ablation requires a DMF experiment and an ablation object.")
    if set(declared) != {"full_config_path", "full_config_sha256", "scientific_controls_sha256"}:
        raise ValueError("Ablation must pin the full TOML and scientific controls.")
    if declared["scientific_controls_sha256"] != scientific_controls_sha256(config):
        raise ValueError("Undeclared experiment scientific control change.")
    profile = config.get("framework_config", {}).get("profile")
    if profile not in PROFILE_CHANGES:
        raise ValueError(f"Unknown DMF ablation profile: {profile!r}.")
    if config.get("scientific_profile") != "dmf-components-v1":
        raise ValueError("DMF ablation requires scientific_profile='dmf-components-v1'.")
    full_path = Path(declared["full_config_path"])
    if declared["full_config_sha256"] != FULL_CONFIG_SHA256.get(config.get("benchmark")):
        raise ValueError("Ablation full config does not match the frozen benchmark control.")
    if not full_path.is_file() or sha256_file(full_path) != declared["full_config_sha256"]:
        raise ValueError("Pinned full DMF configuration is absent or changed.")
    variant_path = Path(config["framework_config"]["path"])
    full = _flatten(tomllib.loads(full_path.read_text(encoding="utf-8")))
    variant = _flatten(tomllib.loads(variant_path.read_text(encoding="utf-8")))
    if full.keys() != variant.keys():
        raise ValueError("DMF ablation changes the set of scientific fields.")
    actual = {key: {"full": full[key], "variant": variant[key]}
              for key in sorted(full) if full[key] != variant[key]}
    expected = PROFILE_CHANGES[profile]
    if set(actual) != set(expected) or any(
        type(actual[key]["variant"]) is not type(wanted)
        or actual[key]["variant"] != wanted for key, wanted in expected.items()
    ):
        raise ValueError(f"Undeclared DMF scientific diff for {profile}: {actual!r}.")
    return {
        "profile": profile,
        "runtime_policy": PROFILE_POLICY[profile],
        "full_config_sha256": declared["full_config_sha256"],
        "scientific_controls_sha256": declared["scientific_controls_sha256"],
        "scientific_diff": actual,
    }


class NeutralScoringEngine:
    """Give every interaction the same admission score and unstable tier."""

    def calculate_score(self, report: Any, text: str = "") -> float:
        from dmf.models.status import SurvivalStatus

        del text
        report.survival_score = 0.5
        report.status = SurvivalStatus.UNSTABLE
        report.raw_metadata["ablation_scoring_policy"] = PROFILE_POLICY["dmf-no-salience-scoring"]
        return 0.5
