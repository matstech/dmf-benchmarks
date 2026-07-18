from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from dmf_bench.api import create_default_app


ROOT = Path(__file__).parents[2]


def test_dockerfile_defines_single_non_root_benchmark_runner_image() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "python:3.12-slim-bookworm@sha256:" in dockerfile
    assert "FROM ${PYTHON_IMAGE} AS base" in dockerfile
    assert "FROM base AS builder" in dockerfile
    assert "FROM base AS runtime" in dockerfile
    assert "apt-get install -y --no-install-recommends build-essential ca-certificates git" in dockerfile
    assert "COPY --from=builder" in dockerfile
    assert "USER dmfbench" in dockerfile
    assert 'STOPSIGNAL SIGTERM' in dockerfile
    assert 'HEALTHCHECK' in dockerfile
    assert '["dmf-bench", "list"]' in dockerfile
    assert 'ENTRYPOINT ["dmf-bench"]' in dockerfile
    assert 'org.opencontainers.image.licenses="MIT"' in dockerfile
    assert "COPY . ." not in dockerfile


def test_dockerignore_excludes_local_data_secrets_and_plans() -> None:
    ignored = set(
        line.strip()
        for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    )

    assert ".git" in ignored
    assert ".env" in ignored
    assert ".env.*" in ignored
    assert "data" in ignored
    assert "datasets" in ignored
    assert "results" in ignored
    assert "docs" in ignored
    assert "plans" in ignored
    assert "schemas" in ignored
    assert "deploy/secrets/*" in ignored


def test_compose_starts_qdrant_and_artifact_api_independent_from_benchmark_job() -> None:
    compose = yaml.safe_load((ROOT / "deploy" / "compose.yaml").read_text(encoding="utf-8"))
    services = compose["services"]

    assert set(services) == {"qdrant", "artifact-api", "benchmark"}
    assert services["qdrant"]["image"].startswith("qdrant/qdrant:v1.18.2@sha256:")
    assert services["qdrant"]["ports"] == ["127.0.0.1:${QDRANT_HTTP_PORT:-6333}:6333"]
    assert services["qdrant"]["expose"] == ["6334"]
    assert "qdrant-storage" in {
        volume["source"] for volume in services["qdrant"]["volumes"]
    }
    assert "environment" not in services["qdrant"]
    assert "secrets" not in services["qdrant"]
    assert "command" not in services["qdrant"]

    assert "profiles" not in services["artifact-api"]
    assert services["artifact-api"]["entrypoint"] == ["python", "-m", "uvicorn"]
    assert "dmf_bench.api:create_default_app" in services["artifact-api"]["command"]
    assert "--factory" in services["artifact-api"]["command"]
    assert services["artifact-api"]["read_only"] is True

    assert services["benchmark"]["profiles"] == ["job"]
    assert services["benchmark"]["read_only"] is True
    assert services["benchmark"]["depends_on"]["artifact-api"]["condition"] == "service_healthy"
    assert "QDRANT_API_KEY_FILE" not in services["benchmark"]["environment"]


def test_compose_mounts_config_and_dataset_read_only_but_runs_and_cache_writable() -> None:
    compose = yaml.safe_load((ROOT / "deploy" / "compose.yaml").read_text(encoding="utf-8"))
    benchmark_volumes = compose["services"]["benchmark"]["volumes"]
    artifact_api_volumes = compose["services"]["artifact-api"]["volumes"]

    for volumes in (benchmark_volumes, artifact_api_volumes):
        by_target = {volume["target"]: volume for volume in volumes}
        assert by_target["/bench/config"]["read_only"] is True
        assert by_target["/bench/datasets"]["read_only"] is True
        assert "read_only" not in by_target["/bench/runs"]
        assert "read_only" not in by_target["/bench/cache"]


def test_default_artifact_api_factory_uses_runs_dir_environment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("DMF_BENCH_RUNS_DIR", str(tmp_path / "runs"))

    client = TestClient(create_default_app())

    assert client.get("/health").json()["write_contract"] == "json-artifact-batch"
