"""Isolated v3 config validation during the runtime cutover."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dmf_bench.config import validate_v3_config
from dmf_bench.contracts import sha256_file


FIXTURES = Path(__file__).parents[1] / "fixtures"


def config_v3(tmp_path: Path) -> dict:
    data = json.loads((FIXTURES / "experiment-valid.json").read_text())
    dataset_path = tmp_path / "locomo.json"
    dataset_path.write_bytes((FIXTURES / "locomo-mini.json").read_bytes())
    framework_path = tmp_path / "framework.toml"
    framework_path.write_text("[ltm]\nstorage_type = 'qdrant'\n")
    data["schema_version"] = 3
    data["scientific_profile"] = "retrieval-controlled-v1"
    data["runtime"].update(
        root=str(tmp_path), runs_dir=str(tmp_path / "runs"),
        cache_dir=str(tmp_path / "cache"),
    )
    data["framework_config"].update(
        path=str(framework_path), sha256=sha256_file(framework_path),
        profile="fixture-v1",
    )
    data["dataset"].update(
        path=str(dataset_path), sha256=sha256_file(dataset_path),
    )
    data["storage"] = {
        "kind": "qdrant-server", "profile": "qdrant-v1",
        **data.pop("qdrant"),
    }
    data["retrieval"] = {"max_results": 20}
    data["context_budget"] = {
        "max_tokens": 128, "tokenizer": "tiktoken-cl100k_base-v1",
        "renderer": "retrieved-memory-v1",
        "packing": "ranked-whole-items-v1",
    }
    data["models"]["judges"] = [{"id": "primary", **data["models"].pop("judge")}]
    data["evaluation"] = {
        "primary_judge_id": "primary",
        "required": ["primary_judge_score", "rigorous_report", "analysis_rows"],
        "optional": ["retrieval_report", "judge_agreement", "ablation_report"],
    }
    data["artifact_store"]["uri"] = str(tmp_path / "runs")
    return data


def test_valid_v3_config_passes_descriptor_checks(tmp_path: Path) -> None:
    validate_v3_config(config_v3(tmp_path), source_path=tmp_path / "experiment.json")


@pytest.mark.parametrize("benchmark", ["locomo", "longmemeval"])
@pytest.mark.parametrize("framework", ["dmf", "mem0"])
def test_v3_config_accepts_each_declared_pair(
    tmp_path: Path, benchmark: str, framework: str,
) -> None:
    data = config_v3(tmp_path)
    data["benchmark"] = benchmark
    data["framework"] = framework
    if benchmark == "longmemeval":
        dataset_path = tmp_path / "longmemeval.json"
        dataset_path.write_bytes((FIXTURES / "longmemeval-mini.json").read_bytes())
        data["dataset"].update(
            name=benchmark, path=str(dataset_path), sha256=sha256_file(dataset_path),
        )
    if framework == "mem0":
        framework_path = tmp_path / "framework.yaml"
        framework_path.write_text("memory: fixture\n")
        data["framework_config"].update(
            path=str(framework_path), sha256=sha256_file(framework_path),
            format="yaml",
        )
    validate_v3_config(data, source_path=tmp_path / "experiment.json")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", 2, "schema_version=3"),
        ("qdrant", {}, "unsupported fields"),
        ("context_budget.tokenizer", "missing", "not registered"),
        ("context_budget.packing", "missing", "not registered"),
        ("storage.kind", "chroma-embedded", "unsupported by framework"),
        ("framework_config.format", "yaml", "framework_config.format"),
        ("evaluation.primary_judge_id", "missing", "must resolve"),
        ("retrieval.unknown", 1, "unsupported fields"),
        ("models.answerer.api_key", "SECRET", "Inline secret field"),
    ],
)
def test_v3_config_rejects_incompatible_or_unknown_values(
    tmp_path: Path, field: str, value: object, message: str,
) -> None:
    data = config_v3(tmp_path)
    target = data
    parts = field.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value
    with pytest.raises(ValueError, match=message):
        validate_v3_config(data, source_path=tmp_path / "experiment.json")


def test_v3_config_rejects_duplicate_judges_and_inline_secrets(tmp_path: Path) -> None:
    data = config_v3(tmp_path)
    data["models"]["judges"].append(dict(data["models"]["judges"][0]))
    with pytest.raises(ValueError, match="Duplicate judge ID"):
        validate_v3_config(data, source_path=tmp_path / "experiment.json")
    data = config_v3(tmp_path)
    data["models"]["answerer"]["api_key"] = "SECRET"
    with pytest.raises(ValueError, match="Inline secret field"):
        validate_v3_config(data, source_path=tmp_path / "experiment.json")
