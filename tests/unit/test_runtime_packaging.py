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
    assert project["tool"]["poetry"]["packages"] == [
        {"include": "dmf_bench"},
        {"include": "dmf_benchctl"},
    ]
    assert project["project"]["scripts"] == {
        "dmf-bench": "dmf_bench.cli:main",
        "dmf-benchctl": "dmf_benchctl.cli:main",
    }


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
        "docs-build",
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
    assert "run-oci-push: require-RUN_ID require-GHCR_OWNER require-RUN_SUBJECT" in makefile


def test_publish_image_workflow_pushes_rc_and_tagged_release_to_ghcr() -> None:
    workflow = (ROOT / ".github" / "workflows" / "publish-image.yml").read_text(
        encoding="utf-8"
    )

    assert "packages: write" in workflow
    assert "branches:\n      - main" in workflow
    assert 'tags:\n      - "*"' in workflow
    assert "publish-main-rc:" in workflow
    assert "publish-tagged-release:" in workflow
    assert "date -u +%Y%m%dT%H%M%SZ)-RC" in workflow
    assert "IMAGE_TAG=$GITHUB_REF_NAME" in workflow
    assert 'git merge-base --is-ancestor "$GITHUB_SHA" origin/main' in workflow
    assert 'docker login ghcr.io -u "$GITHUB_ACTOR" --password-stdin' in workflow
    assert workflow.count("--driver docker-container --use") == 2
    assert workflow.count("docker buildx inspect --bootstrap") == 2
    assert 'make image-buildx-push GHCR_OWNER="$GHCR_OWNER" IMAGE_NAME="$IMAGE_NAME" IMAGE_TAG="$IMAGE_TAG" PUBLISH_LATEST=0' in workflow


def test_readme_documents_cli_surface_and_user_manual() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert readme.count("## Table of contents") == 1
    assert "## User manual" in readme
    assert "## CLI quickstart" in readme
    assert "## Maintainer Makefile" in readme
    assert "manual/user-manual.md" in readme
    assert "manual/user-manual.pdf" in readme
    assert "User operations go through `dmf-benchctl`." in readme
    assert "poetry run" not in readme
    assert "The Makefile is a maintainer surface" in readme
    assert "The checked-in `smoke/` directory contains ready-to-use" in readme


def test_container_fixture_still_enters_through_dmf_bench() -> None:
    fixture = (ROOT / "deploy" / "compose.fixture.yaml").read_text(
        encoding="utf-8"
    )

    assert "entrypoint:" not in fixture
    assert 'DMF_BENCH_INTERNAL_APPLICATION: "container-fixture"' in fixture
