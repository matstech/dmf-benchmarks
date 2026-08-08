from __future__ import annotations

import re
import tomllib
from importlib import metadata
from pathlib import Path

import dmf_bench
from dmf_bench import provenance


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


def test_distribution_contains_the_runtime_package_and_entrypoint() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert project["project"]["version"] == "0.2.0"
    assert project["tool"]["poetry"]["packages"] == [{"include": "dmf_bench"}]
    assert project["project"]["scripts"] == {"dmf-bench": "dmf_bench.cli:main"}


def test_config_directory_contains_the_supported_qdrant_surfaces() -> None:
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
    compose = (ROOT / "deploy" / "compose.yaml").read_text(encoding="utf-8")

    assert "COPY dmf_bench ./dmf_bench" in dockerfile
    assert "COPY --from=builder --chown=dmfbench:dmfbench /opt/venv /opt/venv" in dockerfile
    assert "COPY --from=builder --chown=dmfbench:dmfbench /app /app" not in dockerfile
    assert "/opt/poetry /opt/poetry" not in dockerfile
    assert 'org.opencontainers.image.version="0.2.0"' in dockerfile
    assert "image: ${DMF_BENCH_IMAGE:-dmf-benchmarks:local}" in compose

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
    for config in configs:
        assert f"/bench/config/{config}" in readme
