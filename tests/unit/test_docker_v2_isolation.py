import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
COMPOSE_PATH = ROOT / "deploy" / "compose.yaml"
DASHBOARD_PROVIDER_PATH = (
    ROOT / "deploy" / "grafana" / "provisioning" / "dashboards" / "dashboards.yml"
)
OPERATIONS_DASHBOARD_PATH = (
    ROOT / "deploy" / "grafana" / "dashboards" / "dmf-benchmarks.json"
)
EVALUATION_DASHBOARD_PATH = (
    ROOT / "deploy" / "grafana" / "dashboards" / "dmf-benchmarks-evaluation.json"
)


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
    assert "OPENAI_API_KEY" not in benchmark["environment"]
    assert "OPENROUTER_API_KEY" not in benchmark["environment"]

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


def test_qdrant_health_gates_dependent_services_and_job_resources_are_bounded() -> None:
    services = load_compose()["services"]
    qdrant = services["qdrant"]
    benchmark = services["benchmark"]

    assert qdrant["healthcheck"]["test"][0] == "CMD-SHELL"
    assert "/dev/tcp/127.0.0.1/6333" in qdrant["healthcheck"]["test"][1]
    for service_name in ("artifact-api", "prometheus", "benchmark"):
        assert services[service_name]["depends_on"]["qdrant"]["condition"] == (
            "service_healthy"
        )

    assert benchmark["cpus"] == "${DMF_BENCH_CPUS:-4.0}"
    environment = benchmark["environment"]
    assert environment["DMF_BENCH_THREADS"] == "${DMF_BENCH_THREADS:-4}"
    for name in (
        "OMP_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "RAYON_NUM_THREADS",
    ):
        assert environment[name] == "${DMF_BENCH_THREADS:-4}"
    assert environment["TOKENIZERS_PARALLELISM"] == "false"


def test_grafana_dashboards_are_dark_editable_and_persisted() -> None:
    services = load_compose()["services"]
    grafana = services["grafana"]
    environment = grafana["environment"]
    provider = yaml.safe_load(DASHBOARD_PROVIDER_PATH.read_text(encoding="utf-8"))

    assert environment["GF_USERS_DEFAULT_THEME"] == "dark"
    assert environment["GF_AUTH_ANONYMOUS_ORG_ROLE"] == "Editor"
    assert mounts_by_target(grafana)["/var/lib/grafana"] == {
        "type": "volume",
        "source": "grafana-data-v2",
        "target": "/var/lib/grafana",
    }
    assert provider["providers"][0]["allowUiUpdates"] is True
    assert provider["providers"][0]["disableDeletion"] is True


def test_operations_and_evaluation_dashboards_are_separate() -> None:
    operations = json.loads(OPERATIONS_DASHBOARD_PATH.read_text(encoding="utf-8"))
    evaluation = json.loads(EVALUATION_DASHBOARD_PATH.read_text(encoding="utf-8"))

    assert operations["editable"] is True
    assert evaluation["editable"] is True
    assert operations["uid"] == "dmf-benchmarks"
    assert evaluation["uid"] == "dmf-benchmarks-evaluation"

    operations_queries = " ".join(
        target.get("expr", "")
        for panel in operations["panels"]
        for target in panel.get("targets", [])
    )
    evaluation_queries = " ".join(
        target.get("expr", "")
        for panel in evaluation["panels"]
        for target in panel.get("targets", [])
    )
    assert "dmf_bench_persisted_run_result" not in operations_queries
    assert "dmf_bench_persisted_run_result" in evaluation_queries
    assert "process_cpu_seconds_total" in operations_queries
    assert "process_resident_memory_bytes" in operations_queries

    for dashboard in (operations, evaluation):
        variables = dashboard["templating"]["list"]
        assert [variable["name"] for variable in variables] == [
            "benchmark_framework",
            "memory_system",
        ]
        assert all(variable["multi"] is False for variable in variables)
        assert all(variable["includeAll"] is False for variable in variables)
        for panel in dashboard["panels"]:
            for target in panel.get("targets", []):
                assert "$benchmark_framework" in target["expr"]
                assert "$memory_system" in target["expr"]

    for panel in operations["panels"]:
        if panel["type"] == "timeseries":
            assert panel["options"]["legend"]["placement"] == "right"


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
    assert (
        "stack-down:\n\t$(POETRY) run $(PYTHON) -m dmf_benchctl stack down\n"
        in makefile
    )


def test_readme_points_to_historical_tag_without_migration_command() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "historical Git tag" in readme
    assert "`v0.1.0`" in readme
    assert "does not provide compatibility commands" in readme
    assert "git worktree" not in readme
