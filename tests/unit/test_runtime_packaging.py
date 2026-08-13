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
    assert project["project"]["dependencies"] == []
    assert "runtime" in project["project"]["optional-dependencies"]
    assert project["tool"]["poetry"]["packages"] == [
        {"include": "dmf_bench"},
        {"include": "dmf_benchctl"},
    ]
    assert project["project"]["scripts"] == {
        "dmf-bench": "dmf_bench.cli:main",
        "dmf-benchctl": "dmf_benchctl.cli:main",
    }


def test_controller_install_is_lightweight_and_runtime_extra_is_explicit() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    runtime_dependencies = project["project"]["optional-dependencies"]["runtime"]
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    workflow = (ROOT / ".github" / "workflows" / "pr-fast.yml").read_text(
        encoding="utf-8"
    )

    assert any(dependency.startswith("dmf-memory") for dependency in runtime_dependencies)
    assert any(dependency.startswith("mem0ai") for dependency in runtime_dependencies)
    assert "--extras runtime" in dockerfile
    assert "poetry install --all-extras" in workflow


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
    assert "COPY dmf_benchctl ./dmf_benchctl" in dockerfile
    assert "COPY --from=builder --chown=dmfbench:dmfbench /opt/venv /opt/venv" in dockerfile
    assert "COPY --from=builder --chown=dmfbench:dmfbench /app /app" not in dockerfile
    assert "/opt/poetry /opt/poetry" not in dockerfile
    assert 'org.opencontainers.image.version="0.2.0"' in dockerfile
    assert "image: ${DMF_BENCH_IMAGE:-dmf-benchmarks:local}" in compose
    assert "OPENAI_BASE_URL: ${OPENAI_BASE_URL:-}" not in compose
    assert "OPENROUTER_BASE_URL: ${OPENROUTER_BASE_URL:-}" not in compose
    assert "OLLAMA_BASE_URL: ${OLLAMA_BASE_URL:-}" not in compose

    assert dockerignore.splitlines() == [
        "**",
        "!pyproject.toml",
        "!poetry.lock",
        "!LICENSE",
        "!dmf_bench/",
        "!dmf_bench/**",
        "!dmf_benchctl/",
        "!dmf_benchctl/**",
    ]


def test_makefile_exposes_maintainer_targets() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    targets = {
        match.group(1)
        for match in re.finditer(r"^([a-z][a-z0-9-]*):", makefile, re.MULTILINE)
    }

    expected = {
        "help",
        "install",
        "compile",
        "test",
        "check",
        "test-integration",
        "compose-config",
        "prometheus-check",
        "stack-up",
        "stack-down",
        "image-build",
        "image-build-ghcr",
        "image-inspect",
        "image-smoke",
        "ghcr-login",
        "image-push",
        "image-buildx-push",
        "image-manifest-create",
        "run-oci-dry-run",
        "run-oci-push",
        "gh-workflows",
        "gh-workflow-run",
        "gh-workflow-watch",
        "gh-release-dry-run",
        "gh-container-smoke",
        "gh-scientific-canary",
    }
    assert expected <= targets
    assert "-m dmf_benchctl config" in makefile
    assert "-m dmf_benchctl stack up --observability" in makefile
    assert "$(DOCKER) compose" not in makefile
    assert "poetry" in makefile.lower()
    assert "publish-run-oci" in makefile
    assert "$(GH) workflow run" in makefile
    assert "PUBLISH_LATEST ?= 0" in makefile
    assert "IMAGE_PLATFORMS ?=" in makefile
    assert '--platform "$(IMAGE_PLATFORMS)"' in makefile
    assert "$(DOCKER) buildx imagetools create" in makefile
    assert "run-oci-push: require-RUN_ID require-GHCR_OWNER require-RUN_SUBJECT" in makefile


def test_publish_image_workflow_pushes_rc_and_tagged_release_to_ghcr() -> None:
    workflow = (ROOT / ".github" / "workflows" / "publish-image.yml").read_text(
        encoding="utf-8"
    )

    assert "packages: write" in workflow
    assert "branches:\n      - main" in workflow
    assert 'tags:\n      - "*"' in workflow
    assert "prepare:" in workflow
    assert "build-platform:" in workflow
    assert "publish-manifest:" in workflow
    assert "date -u +%Y%m%dT%H%M%SZ)-RC" in workflow
    assert 'tag="$GITHUB_REF_NAME"' in workflow
    assert 'git merge-base --is-ancestor "$GITHUB_SHA" origin/main' in workflow
    assert 'docker login ghcr.io -u "$GITHUB_ACTOR" --password-stdin' in workflow
    assert "runner: ubuntu-24.04-arm" in workflow
    assert "platform: linux/amd64" in workflow
    assert "platform: linux/arm64" in workflow
    assert "IMAGE_PLATFORMS=\"$IMAGE_PLATFORMS\"" in workflow
    assert "make image-manifest-create" in workflow
    assert "Require both published platforms" in workflow
    assert "tonistiigi/binfmt" not in workflow


def test_container_fixture_still_enters_through_dmf_bench() -> None:
    fixture = (ROOT / "deploy" / "compose.fixture.yaml").read_text(
        encoding="utf-8"
    )

    assert "entrypoint:" not in fixture
    assert 'DMF_BENCH_INTERNAL_APPLICATION: "container-fixture"' in fixture
