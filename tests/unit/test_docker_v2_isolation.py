from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
COMPOSE_PATH = ROOT / "deploy" / "compose.yaml"


def load_compose() -> dict[str, Any]:
    payload = yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def mounts_by_target(service: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(mount["target"]): mount
        for mount in service.get("volumes", [])
        if isinstance(mount, dict)
    }


def test_compose_uses_stable_v2_project_and_explicit_v2_volumes() -> None:
    compose = load_compose()
    canary_workflow = (
        ROOT / ".github" / "workflows" / "scientific-canary.yml"
    ).read_text(encoding="utf-8")

    assert compose["name"] == "dmf-benchmarks-v2"
    assert "COMPOSE_PROJECT_NAME" not in canary_workflow
    assert compose["volumes"] == {
        "qdrant-storage-v2": {"name": "dmf-benchmarks-v2-qdrant"},
        "benchmark-runs-v2": {"name": "dmf-benchmarks-v2-runs"},
        "benchmark-cache-v2": {"name": "dmf-benchmarks-v2-cache"},
        "prometheus-data-v2": {"name": "dmf-benchmarks-v2-prometheus"},
        "grafana-data-v2": {"name": "dmf-benchmarks-v2-grafana"},
    }

    services = compose["services"]
    assert services["qdrant"]["environment"] == {
        "QDRANT__TELEMETRY_DISABLED": "true"
    }
    assert mounts_by_target(services["qdrant"])["/qdrant/storage"]["source"] == (
        "qdrant-storage-v2"
    )
    assert mounts_by_target(services["prometheus"])["/prometheus"]["source"] == (
        "prometheus-data-v2"
    )
    assert mounts_by_target(services["grafana"])["/var/lib/grafana"]["source"] == (
        "grafana-data-v2"
    )


def test_api_and_job_share_only_read_only_v2_runs_volume() -> None:
    services = load_compose()["services"]
    api = services["artifact-api"]
    benchmark = services["benchmark"]
    api_mounts = mounts_by_target(api)
    benchmark_mounts = mounts_by_target(benchmark)

    assert api["read_only"] is True
    assert benchmark["read_only"] is True
    assert set(api_mounts) == {"/bench/runs"}
    assert api_mounts["/bench/runs"] == {
        "type": "volume",
        "source": "benchmark-runs-v2",
        "target": "/bench/runs",
        "read_only": True,
    }
    assert benchmark_mounts["/bench/runs"]["source"] == "benchmark-runs-v2"
    assert benchmark_mounts["/bench/cache"]["source"] == "benchmark-cache-v2"

    api_named_sources = {
        mount["source"]
        for mount in api_mounts.values()
        if mount.get("type") == "volume"
    }
    job_named_sources = {
        mount["source"]
        for mount in benchmark_mounts.values()
        if mount.get("type") == "volume"
    }
    assert api_named_sources & job_named_sources == {"benchmark-runs-v2"}


def test_standard_commands_never_expose_volume_deletion() -> None:
    command_surfaces = [
        ROOT / "Makefile",
        ROOT / "README.md",
        *sorted((ROOT / ".github" / "workflows").glob("*.yml")),
        *sorted((ROOT / ".github" / "workflows").glob("*.yaml")),
    ]

    for path in command_surfaces:
        text = path.read_text(encoding="utf-8")
        assert "down --volumes" not in text, path
        assert "down -v" not in text, path

    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "stack-down:\n\t$(COMPOSE) down\n" in makefile


def test_readme_points_to_historical_tag_without_migration_command() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "git worktree add ../dmf-benchmarks-v0.1.0 v0.1.0" in readme
    assert "No v0.1 run, volume or Qdrant collection is read or migrated by v2" in readme
