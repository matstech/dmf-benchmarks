from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


WORKFLOW_DIR = Path(__file__).parents[2] / ".github" / "workflows"
EXPECTED_WORKFLOWS = {
    "pr-fast.yml",
    "container-smoke.yml",
    "scheduled-integration.yml",
    "release-dry-run.yml",
    "publish-image.yml",
    "scientific-canary.yml",
}


def load_workflow(path: Path) -> dict[str, Any]:
    payload = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert isinstance(payload, dict)
    return payload


def walk(value: Any) -> list[Any]:
    items = [value]
    if isinstance(value, dict):
        for key, item in value.items():
            items.extend(walk(key))
            items.extend(walk(item))
    elif isinstance(value, list):
        for item in value:
            items.extend(walk(item))
    return items


def test_expected_workflows_exist_and_parse() -> None:
    workflows = {path.name for path in WORKFLOW_DIR.glob("*.yml")}

    assert workflows == EXPECTED_WORKFLOWS
    for workflow_path in WORKFLOW_DIR.glob("*.yml"):
        payload = load_workflow(workflow_path)
        assert payload["name"]
        assert payload["on"]
        assert payload["permissions"]["contents"] == "read"
        assert payload["jobs"]


def test_workflows_do_not_use_unpinned_external_actions() -> None:
    for workflow_path in WORKFLOW_DIR.glob("*.yml"):
        payload = load_workflow(workflow_path)
        uses_values = [
            item
            for item in walk(payload)
            if isinstance(item, dict) and "uses" in item
        ]
        assert uses_values == []


def test_pr_workflows_do_not_access_secrets_or_trusted_target_event() -> None:
    for workflow_name in ("pr-fast.yml", "container-smoke.yml"):
        payload = load_workflow(WORKFLOW_DIR / workflow_name)
        serialized = (WORKFLOW_DIR / workflow_name).read_text(encoding="utf-8")

        assert "pull_request_target" not in payload["on"]
        assert "secrets." not in serialized
        assert "OPENAI_API_KEY" not in serialized
        assert "OPENROUTER_API_KEY" not in serialized
        assert "QDRANT_API_KEY" not in serialized


def test_release_workflow_is_manual_and_does_not_publish() -> None:
    payload = load_workflow(WORKFLOW_DIR / "release-dry-run.yml")
    serialized = (WORKFLOW_DIR / "release-dry-run.yml").read_text(encoding="utf-8")

    assert set(payload["on"]) == {"workflow_dispatch"}
    assert "docker push" not in serialized
    assert "docker login" not in serialized
    assert "packages: write" not in serialized
    assert "docker build" in serialized
    assert "docker run" in serialized


def test_publish_image_workflow_publishes_only_on_main_or_tag() -> None:
    payload = load_workflow(WORKFLOW_DIR / "publish-image.yml")
    serialized = (WORKFLOW_DIR / "publish-image.yml").read_text(encoding="utf-8")

    assert payload["on"]["push"]["branches"] == ["main"]
    assert payload["on"]["push"]["tags"] == ["*"]
    assert payload["permissions"] == {"contents": "read", "packages": "write"}
    assert "pull_request" not in payload["on"]
    assert "workflow_dispatch" not in payload["on"]
    assert set(payload["jobs"]) == {
        "prepare",
        "build-platform",
        "publish-manifest",
    }
    assert "docker login ghcr.io" in serialized
    assert "ubuntu-24.04-arm" in serialized
    assert "linux/amd64" in serialized
    assert "linux/arm64" in serialized
    assert "IMAGE_PLATFORMS" in serialized
    assert "image-manifest-create" in serialized
    assert "imagetools inspect --raw" in serialized
    assert "PUBLISH_LATEST=0" in serialized
    assert "merge-base --is-ancestor" in serialized


def test_scheduled_integration_is_not_part_of_pr_fast_path() -> None:
    payload = load_workflow(WORKFLOW_DIR / "scheduled-integration.yml")
    serialized = (WORKFLOW_DIR / "scheduled-integration.yml").read_text(
        encoding="utf-8"
    )

    assert set(payload["on"]) == {"schedule", "workflow_dispatch"}
    assert "pull_request" not in payload["on"]
    assert "poetry install" not in serialized
    assert "poetry run" not in serialized
    assert "deploy/compose.fixture.yaml" in serialized
    assert "dmf_benchctl verify" in serialized
    assert "docker compose" not in serialized
