# DMF Benchmarks

[![arXiv](https://img.shields.io/badge/arXiv-2606.03463-b31b1b.svg)](https://arxiv.org/abs/2606.03463)

DMF Benchmarks compares [DMF](https://github.com/matstech/dmf) and
[Mem0](https://github.com/mem0ai/mem0) on LoCoMo and LongMemEval. Version 0.2
has one supported runtime: the native `dmf-bench` application, normally run
inside the locked Docker image with Qdrant Server. It publishes verified JSON
artifacts to a run volume and exposes committed results through a
loopback-only, read-only FastAPI service.

The framework-owned retrieval path is product behavior, not a selectable
protocol. DMF and Mem0 are installed from the exact Git commits recorded in
`pyproject.toml` and `poetry.lock`; changing either commit is a scientific
change and requires rerunning the certification gates.

## Table of contents

- [User manual](#user-manual)
- [CLI quickstart](#cli-quickstart)
- [Configuration model](#configuration-model)
- [Results and observability](#results-and-observability)
- [Official run OCI artifacts](#official-run-oci-artifacts)
- [Maintainer Makefile](#maintainer-makefile)
- [Dataset registry](#dataset-registry)
- [Pipeline and artifacts](#pipeline-and-artifacts)
- [Historical version](#historical-version)

## User manual

The operational user guide is available in:

- [`manual/user-manual.md`](manual/user-manual.md)
- [`manual/user-manual.pdf`](manual/user-manual.pdf)

The manual is the canonical user-facing documentation for `dmf-bench`
configuration, dataset materialization, benchmark execution, result inspection,
and OCI publication of official run bundles.

## CLI quickstart

All user commands go through `dmf-bench`. The paths in an experiment config must
be valid in the environment where the CLI runs. In the official image, paths are
typically under `/bench`; on the host, use host-local paths.

Check the CLI surface:

```bash
dmf-bench --help
dmf-bench list
dmf-bench list-datasets
```

Prepare provider and service endpoints through environment variables, never in
JSON config files:

```bash
export OPENAI_API_KEY=your_openai_api_key_here
export QDRANT_URL=http://127.0.0.1:6333
```

Validate a config without calling the provider:

```bash
dmf-bench validate --materialize-datasets --config config/experiment.json
```

Run a benchmark with a fresh run ID:

```bash
dmf-bench run --config config/experiment.json --run-id local-locomo-dmf-001
```

Inspect, resume, and verify committed local state:

```bash
dmf-bench status --run-id local-locomo-dmf-001 --runs-dir runs
dmf-bench resume --run-id local-locomo-dmf-001 --runs-dir runs
dmf-bench verify --run-id local-locomo-dmf-001 --runs-dir runs
```

The checked-in `smoke/` directory contains ready-to-use restricted configs for
LoCoMo and LongMemEval across DMF and Mem0. They are substantial smoke checks,
not paper-scale reproduction configs.

## Configuration model

Experiment configs use `schema_version: 2` and define benchmark, framework,
runtime paths, framework config, Qdrant settings, dataset, selection, models,
evaluation reports, and artifact store.

Two dataset modes are supported:

- already materialized datasets with explicit `dataset.path` and
  `dataset.sha256`;
- registry-backed datasets with `dataset.registry_id`, optionally with
  `dataset.sampling`.

When registry materialization is used, `dmf-bench` writes a dataset
`manifest.json` next to the materialized file and copies it into the run
directory as:

```text
datasets/materialization-manifest.json
```

Unpinned official registry entries fail closed unless the config or command
explicitly sets `allow_unpinned`.

## Results and observability

The Artifact API is read-only and intended for loopback/local use. When running
locally, the standard endpoint is:

```text
http://127.0.0.1:8000
```

Useful endpoints:

```bash
RUN_ID=local-locomo-dmf-001
curl -fsS "http://127.0.0.1:8000/runs/$RUN_ID" | jq
curl -fsS "http://127.0.0.1:8000/runs/$RUN_ID/artifacts" | jq
curl -fsS "http://127.0.0.1:8000/runs/$RUN_ID/artifacts/reports/usage.json" | jq
curl -fsS "http://127.0.0.1:8000/runs/$RUN_ID/artifacts/reports/timing.json" | jq
curl -fsS "http://127.0.0.1:8000/runs/$RUN_ID/verify" | jq
```

`usage.json` separates framework-internal, answerer, and judge calls and
tokens. `timing.json` aggregates measured lifecycle work for the active
attempt. Scientific quality remains in `rigorous_report.json` and
`ablation_report.json`.

The Artifact API is unauthenticated only because it is read-only and bound to
loopback. Do not expose it directly to an untrusted network.

## Official run OCI artifacts

A completed run can be packaged as a GHCR/OCI artifact containing the complete
`runs/<RUN_ID>/` directory. The command verifies the committed run, writes
`OFFICIAL-RUN.json` and `SHA256SUMS` into the run directory, creates a tar.gz
snapshot, and pushes it with ORAS:

```bash
dmf-bench publish-run-oci \
  --run-id "$RUN_ID" \
  --runs-dir runs \
  --ref "ghcr.io/ORG/dmf-benchmarks-runs:$RUN_ID" \
  --subject "ghcr.io/ORG/dmf-benchmarks@sha256:IMAGE_DIGEST"
```

Use `--dry-run --output-dir bundles` to create and inspect the local bundle
without pushing. The OCI artifact type is:

```text
application/vnd.dmf-bench.run.v1+tar
```

## Maintainer Makefile

The Makefile is a maintainer surface for repository management, CI parity,
Docker image publication, GitHub Actions, and official run OCI artifacts. It is
not the user-facing benchmark interface.

Common maintainer targets:

| Target | Purpose |
|---|---|
| `make check` | Run package checks, compile, and unit tests. |
| `make compose-config` | Validate the Docker Compose configuration. |
| `make prometheus-check` | Validate Prometheus configuration with `promtool`. |
| `make image-build` | Build the local benchmark image. |
| `make image-buildx-push GHCR_OWNER=org` | Publish the benchmark image to GHCR with SBOM/provenance. |
| `make run-oci-dry-run RUN_ID=id GHCR_OWNER=org` | Build a local official run bundle without pushing. |
| `make run-oci-push RUN_ID=id GHCR_OWNER=org RUN_SUBJECT=image@sha256:...` | Publish an official run OCI artifact to GHCR. |
| `make gh-container-smoke` | Trigger the container smoke GitHub Actions workflow. |
| `make gh-scientific-canary CANARY_CONFIRM=APPROVED` | Trigger the protected provider-backed canary. |

Run `make help` for the full list.

## Dataset registry

List registry entries:

```bash
dmf-bench list-datasets
```

Materialize a dataset manually:

```bash
dmf-bench materialize-dataset \
  --dataset-id locomo-official-unpinned \
  --allow-unpinned \
  --sample-fraction 0.05 \
  --sample-unit qa \
  --sample-stratify-by category \
  --sample-seed 7 \
  --sample-rounding ceil \
  --output-dir .cache/datasets/manual-locomo-smoke
```

Materialization always writes `manifest.json`; when sampling options are
provided, the manifest records the sample policy and resulting counts.

Dataset pinning, embedding model materialization, provider-backed canaries, and
paper-scale runs require separate review because they can introduce downloads
or cost.

## Pipeline and artifacts

Each run performs sequential ingestion, framework-owned retrieval, answer
generation, LLM judging, deterministic rigorous/ablation evaluation, atomic
JSON publication, and digest verification. Prompt/completion token usage and
wall-clock timings are first-class reports.

The manual `Scientific canary` workflow is cost-gated by the protected
`scientific-canary` environment and requires the exact
`confirm_costs=APPROVED` input. Do not trigger it without an approved dataset,
model scope, call/token estimate, and maximum cost.

## Historical version

The previous architecture is available exclusively from the historical Git tag
`v0.1.0`. Version 0.2 does not provide compatibility commands, state or
artifact migration, or fallback execution paths.
