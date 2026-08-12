# DMF Benchmarks

[![arXiv](https://img.shields.io/badge/arXiv-2606.03463-b31b1b.svg)](https://arxiv.org/abs/2606.03463)

DMF Benchmarks compares [DMF](https://github.com/matstech/dmf) and
[Mem0](https://github.com/mem0ai/mem0) on LoCoMo and LongMemEval. Version 0.2
has one supported runtime: the native `dmf-bench` application inside the
locked Docker image with Qdrant Server. Host-side operations go through
`dmf-benchctl`, which owns orchestration and delegates scientific work to
`dmf-bench`. Runs publish verified JSON artifacts to a shared volume and expose
committed results through a loopback-only, read-only FastAPI service.

Official RC and tagged images are published as a single multi-platform GHCR
reference containing native `linux/amd64` and `linux/arm64` variants. Docker
selects the matching variant automatically.

The framework-owned retrieval path is product behavior, not a selectable
protocol. Framework dependency versions are locked by `pyproject.toml` and
`poetry.lock`; changing them is a scientific change and requires rerunning the
certification gates.

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

User operations go through `dmf-benchctl`. It locates the repository Compose
definition, starts the required services, translates known host paths to
container paths, and invokes `dmf-bench` explicitly inside the locked image.
Docker Compose is an implementation detail of the current orchestrator adapter.

Install the lightweight host controller in an isolated environment; Poetry is
not required for user operations:

```bash
pipx install .
dmf-benchctl --help
```

The default install has no scientific runtime dependencies. Maintainers and the
Docker image install the explicit `runtime` extra.

Check the CLI surface:

```bash
dmf-benchctl --help
dmf-benchctl runtime -- list
dmf-benchctl runtime -- list-datasets
```

Prepare provider and service endpoints through environment variables, never in
JSON config files:

```bash
export OPENAI_API_KEY=your_openai_api_key_here
export DMF_BENCH_IMAGE=ghcr.io/ORG/dmf-benchmarks:TAG
```

The controller reads provider credentials from the shell or the repository
`.env`, writes them to a private ephemeral file for the duration of the job,
and mounts that file read-only. The key is not placed in Compose service
environment metadata or command arguments.

Prepare the complete checked-in smoke suite without calling the provider:

```bash
dmf-benchctl --image "$DMF_BENCH_IMAGE" prepare \
  --suite smoke \
  --allow-downloads \
  --observability
```

`prepare` checks Docker and Compose, pulls and locks the native image by digest,
checks required provider credentials, starts the services, materializes and
validates every dataset, and writes `.dmf-bench/prepared.json`. The local state
contains no secret and is ignored by Git. Version 2 contexts also lock every
configuration SHA-256 and the pinned dataset identity embedded in the selected
image; execution fails before starting services if a configuration changes.

Execute the prepared experiments sequentially:

```bash
dmf-benchctl run --prepared
```

The controller launches a detached host worker and returns immediately with the
job PID, run IDs, worker log, and exact `status`/`verify` commands. Benchmark
output is retained under `.dmf-bench/logs/<RUN_ID>.log`; orchestration output is
retained in the reported worker log. Add `--follow` only to run in the
foreground and stream container logs for interactive diagnostics.

Rerunning `dmf-benchctl run --prepared` is safe after an interruption: the
controller inspects each run ID, resumes existing incomplete/failed runs, and
starts only runs whose state does not exist yet.

Resume a failed member with the same locked image and quiet logging:

```bash
dmf-benchctl resume --prepared --run-id RUN_ID
```

For a single unprepared experiment, pass its config and a fresh run ID:

```bash
dmf-benchctl run --config config/experiment.json --run-id local-locomo-dmf-001
```

Inspect, resume, and verify committed local state:

```bash
dmf-benchctl status --run-id local-locomo-dmf-001
dmf-benchctl resume --run-id local-locomo-dmf-001
dmf-benchctl verify --run-id local-locomo-dmf-001
dmf-benchctl stack down
```

`status` is human-oriented by default. It shows the current lifecycle phase, an
ASCII progress bar, completed evaluation items (questions or scored records),
and processed input records (source conversations or records). Automation can
request the stable renamed schema with:

```bash
dmf-benchctl status --run-id local-locomo-dmf-001 --json
```

The controller status schema exposes `evaluation_items` and `input_records`;
the internal checkpoint term `units` is not part of this user-facing format.

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

Official registry entries use immutable upstream revisions and verified
SHA-256 digests. Unpinned custom entries remain available for explicitly
exploratory direct runs and are rejected by `dmf-benchctl prepare`.

## Results and observability

The Artifact API is read-only and intended for loopback/local use. When running
locally, the standard endpoint is:

```text
http://127.0.0.1:8000
```

Open `http://127.0.0.1:8000` for the run and artifact browser. Select a
committed run to view or download any artifact without constructing an API
URL. The read-only API remains available for automation:

```bash
RUN_ID=local-locomo-dmf-001
curl -fsS "http://127.0.0.1:8000/runs/$RUN_ID" | jq
curl -fsS "http://127.0.0.1:8000/runs/$RUN_ID/artifacts" | jq
curl -fsS "http://127.0.0.1:8000/runs/$RUN_ID/artifacts/reports/usage.json" | jq
curl -fsS "http://127.0.0.1:8000/runs/$RUN_ID/artifacts/reports/timing.json" | jq
curl -fsS "http://127.0.0.1:8000/runs/$RUN_ID/artifacts/reports/resources.json" | jq
curl -fsS "http://127.0.0.1:8000/runs/$RUN_ID/verify" | jq
```

`usage.json` separates framework-internal, answerer, and judge calls and
tokens. `timing.json` aggregates measured lifecycle work for the active
attempt. `resources.json` records process CPU, average CPU utilization, RSS,
available cgroup measurements, and configured CPU/thread limits. Scientific
quality remains in `rigorous_report.json` and
`ablation_report.json`.

When observability is enabled, Grafana provides two dashboards:

- Operations at `http://127.0.0.1:3000/d/dmf-benchmarks/dmf-benchmarks` for
  progress, state, token usage, timings, CPU, memory, and Qdrant activity.
- Evaluation at
  `http://127.0.0.1:3000/d/dmf-benchmarks-evaluation/dmf-benchmarks-evaluation`
  for final scientific metrics from completed evaluations. It intentionally
  does not show partial judging values.

Both dashboards require one Benchmark framework and one Memory system
selection, so panels never mix LoCoMo with LongMemEval or DMF with Mem0.

Grafana uses a dark theme. Local anonymous users are Editors, and dashboard
changes are stored in the persistent Grafana volume. The Artifact API projects
the latest persisted run per benchmark/framework into Prometheus, so final
values remain visible after the one-shot benchmark container exits.

The benchmark job defaults to four CPUs and four worker threads. Override the
limits deliberately with `DMF_BENCH_CPUS` and `DMF_BENCH_THREADS`; keep the
same values when comparing DMF and Mem0 timing.

Deterministic LoCoMo reports can be re-derived from an immutable committed run
without provider calls:

```bash
dmf-benchctl evaluate --run-id "$RUN_ID"
```

The source run remains unchanged. A provenance-linked bundle is written under
`.derived-evaluations/<RUN_ID>/<EVALUATOR_VERSION>/` in the shared run volume.

The Artifact API is unauthenticated only because it is read-only and bound to
loopback. Do not expose it directly to an untrusted network.

## Official run OCI artifacts

A completed run can be packaged as a GHCR/OCI artifact containing the complete
`runs/<RUN_ID>/` directory. The command verifies the committed run, writes
`OFFICIAL-RUN.json` and `SHA256SUMS` into the run directory, creates a tar.gz
snapshot, and pushes it with ORAS:

```bash
dmf-benchctl publish-run \
  --run-id "$RUN_ID" \
  --ref "ghcr.io/ORG/dmf-benchmarks-runs:$RUN_ID" \
  --subject "ghcr.io/ORG/dmf-benchmarks@sha256:IMAGE_DIGEST"
```

The controller first asks `dmf-bench` to verify and package the complete run
volume, then invokes the host `oras` client. Use `--dry-run --output-dir
bundles` to create and inspect the local bundle without pushing. The OCI
artifact type is:

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
| `make image-buildx-push GHCR_OWNER=org IMAGE_PLATFORMS=linux/arm64` | Publish one native platform image with SBOM/provenance. |
| `make image-manifest-create GHCR_OWNER=org IMAGE_SOURCE_REFS="..."` | Combine native platform images under one multi-platform tag. |
| `make run-oci-dry-run RUN_ID=id GHCR_OWNER=org` | Build a local official run bundle without pushing. |
| `make run-oci-push RUN_ID=id GHCR_OWNER=org RUN_SUBJECT=image@sha256:...` | Publish an official run OCI artifact to GHCR. |
| `make gh-container-smoke` | Trigger the container smoke GitHub Actions workflow. |
| `make gh-scientific-canary CANARY_CONFIRM=APPROVED` | Trigger the protected provider-backed canary. |

Run `make help` for the full list.

## Dataset registry

List registry entries:

```bash
dmf-benchctl runtime -- list-datasets
```

Materialize a dataset manually:

```bash
dmf-benchctl runtime -- materialize-dataset \
  --dataset-id locomo-official-v1 \
  --sample-fraction 0.05 \
  --sample-unit conversation \
  --sample-seed 7 \
  --sample-rounding ceil \
  --output-dir /bench/cache/datasets/manual-locomo-smoke
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
wall-clock timings are first-class reports. Official configs can use Qdrant
`delete-on-success` retention to remove each unit's collections only after its
predictions are complete; a cleanup failure prevents the atomic unit commit.

The manual `Scientific canary` workflow is cost-gated by the protected
`scientific-canary` environment and requires the exact
`confirm_costs=APPROVED` input. Do not trigger it without an approved dataset,
model scope, call/token estimate, and maximum cost.

## Historical version

The previous architecture is available exclusively from the historical Git tag
`v0.1.0`. Version 0.2 does not provide compatibility commands, state or
artifact migration, or fallback execution paths.
