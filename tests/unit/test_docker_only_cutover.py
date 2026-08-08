from __future__ import annotations

import re
import tomllib
from importlib import metadata
from pathlib import Path

import dmf_bench
from dmf_bench import models, provenance
from dmf_bench.frameworks import mem0_runtime
from dmf_bench.models import IngestedConversationBundle, IngestedQuestionBundle
from dmf_bench.reporting import results


ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_CONFIG_FILES = {
    "experiment-locomo-dmf.json",
    "experiment-locomo-mem0.json",
    "experiment-longmemeval-dmf.json",
    "experiment-longmemeval-mem0.json",
    "locomo_dmf_qdrant_settings.toml",
    "locomo_mem0_qdrant_settings.yaml",
    "longmemeval_dmf_qdrant_settings.toml",
    "longmemeval_mem0_qdrant_settings.yaml",
}
LOCAL_SMOKE_CONFIG_FILES = {"experiment.json", "experiment-mem0.json"}


def test_distribution_contains_only_the_v2_package() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert project["project"]["version"] == "0.2.0"
    assert project["tool"]["poetry"]["packages"] == [{"include": "dmf_bench"}]
    dependency_names = {
        dependency.split(maxsplit=1)[0].split("@")[0].split("[")[0].lower()
        for dependency in project["project"]["dependencies"]
    }
    assert dependency_names.isdisjoint({"requests", "rich", "python-dotenv"})


def test_config_directory_contains_only_qdrant_v2_surfaces() -> None:
    visible_files = {
        path.name for path in (ROOT / "config").iterdir() if path.is_file()
    }

    assert visible_files - LOCAL_SMOKE_CONFIG_FILES == OFFICIAL_CONFIG_FILES
    assert visible_files <= OFFICIAL_CONFIG_FILES | LOCAL_SMOKE_CONFIG_FILES


def test_runtime_version_metadata_uses_the_distribution_name(monkeypatch) -> None:
    requested: list[str] = []

    def installed_version(distribution: str) -> str:
        requested.append(distribution)
        return "0.2.0"

    monkeypatch.setattr(metadata, "version", installed_version)
    assert dmf_bench.__version__ == "0.2.0"
    assert provenance._package_version() == "0.2.0"
    assert requested == ["dmf-benchmarks"]

    def missing_version(_distribution: str) -> str:
        raise metadata.PackageNotFoundError

    monkeypatch.setattr(metadata, "version", missing_version)
    assert provenance._package_version() == "0.2.0"


def test_docker_context_and_runtime_exclude_repository_surfaces() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")

    assert "COPY dmf_bench ./dmf_bench" in dockerfile
    assert "COPY --from=builder --chown=dmfbench:dmfbench /opt/venv /opt/venv" in dockerfile
    assert "COPY --from=builder --chown=dmfbench:dmfbench /app /app" not in dockerfile
    assert "/opt/poetry /opt/poetry" not in dockerfile
    assert 'org.opencontainers.image.version="0.2.0"' in dockerfile
    for legacy_package in ("common", "locomo", "longmemeval"):
        assert f"COPY {legacy_package}" not in dockerfile

    assert dockerignore.splitlines() == [
        "**",
        "!pyproject.toml",
        "!poetry.lock",
        "!LICENSE",
        "!dmf_bench/",
        "!dmf_bench/**",
    ]


def test_makefile_exposes_only_docker_operational_targets() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    targets = {
        match.group(1)
        for match in re.finditer(r"^([a-z][a-z0-9-]*):", makefile, re.MULTILINE)
    }

    assert targets == {"help", "stack-up", "stack-down", "run", "resume", "status", "verify"}
    assert "docker compose" in makefile
    assert "poetry" not in makefile.lower()
    assert "python -m" not in makefile
    assert "legacy" not in makefile.lower()
    assert "judge:" not in makefile
    assert "rigorous:" not in makefile


def test_readme_has_one_docker_surface_and_four_official_configs() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    configs = {
        "experiment-locomo-dmf.json",
        "experiment-locomo-mem0.json",
        "experiment-longmemeval-dmf.json",
        "experiment-longmemeval-mem0.json",
    }

    assert readme.count("## Docker quickstart") == 1
    assert "No host-side benchmark execution is supported." in readme
    assert "poetry run" not in readme
    assert "python -m locomo" not in readme
    assert "python -m longmemeval" not in readme
    assert "Legacy compatibility" not in readme
    assert "-native.json" not in readme
    for config in configs:
        assert f"/bench/config/{config}" in readme


def test_workflows_never_invoke_removed_top_level_packages() -> None:
    workflows = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((ROOT / ".github" / "workflows").glob("*.yml"))
    )

    assert "python -m locomo" not in workflows
    assert "python -m longmemeval" not in workflows
    assert "python -m common" not in workflows
    assert "-legacy" not in workflows


def test_removed_v1_runtime_aliases_do_not_exist() -> None:
    assert not hasattr(models, "TemporalMemory")
    assert not hasattr(IngestedConversationBundle, "memory_engine")
    assert not hasattr(IngestedQuestionBundle, "memory_engine")
    assert not hasattr(mem0_runtime, "LocalMem0ConversationBackend")
    assert "search_latency_ms" not in results.METRIC_DESCRIPTIONS


def test_native_evaluation_emits_only_canonical_retrieval_latency() -> None:
    item = results.build_native_evaluation_item(
        base_result={"framework": "dmf"},
        native_context="context",
        task_prompt="prompt",
        surface_marker="dmf_render_context",
        raw_retrieval_outputs=[],
        answerer_usage=None,
        latency_breakdown={"framework_retrieval_ms": 12.5},
    )

    assert item["retrieval"]["retrieval_pipeline_latency_ms"] == 12.5
    assert "search_latency_ms" not in item["retrieval"]
