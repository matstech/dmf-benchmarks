# DMF Benchmarks

[![arXiv](https://img.shields.io/badge/arXiv-2606.03463-b31b1b.svg)](https://arxiv.org/abs/2606.03463)

DMF Benchmarks compares [DMF](https://github.com/matstech/dmf) and
[Mem0](https://github.com/mem0ai/mem0) on LoCoMo and LongMemEval.

Version 0.2 uses a Docker-native architecture. The host-side controller and
the benchmark runtime are distributed together, while benchmark executions
remain isolated in versioned container images published through GHCR.

## Contents

- [User documentation](#user-documentation)
- [Repository development](#repository-development)
- [Continuous integration and distribution](#continuous-integration-and-distribution)
- [Historical version](#historical-version)

## User documentation

The [DMF Benchmarks operator manual](manual/user-manual.md) is the canonical
reference for installing and using `dmf-benchctl`, configuring experiments,
sampling datasets, running benchmarks, monitoring progress, interpreting
results, recovering executions, and publishing run artifacts.

The installed CLI also provides contextual help:

```text
dmf-benchctl --help
dmf-benchctl COMMAND --help
```

`dmf-benchctl` is the public operator interface. The bundled `dmf-bench`
executable is the container runtime invoked by the controller and is not a
separate user workflow.

## Repository development

The Makefile is the maintainer interface for local quality checks, packaging,
container builds, integration infrastructure, and GitHub Actions parity. It is
not the user interface for benchmark execution.

Common targets are:

| Target | Purpose |
|---|---|
| `make install` | Install development dependencies with Poetry |
| `make check` | Validate the package, compile sources, and run unit tests |
| `make test-integration` | Run integration tests against the local stack |
| `make package-build` | Build the controller wheel and source distribution |
| `make package-smoke` | Test the built wheel outside the repository |
| `make compose-config` | Validate the orchestration configuration |
| `make prometheus-check` | Validate the Prometheus configuration |
| `make image-build` | Build the local benchmark image |
| `make image-buildx-push` | Publish a native image with attestations |
| `make image-manifest-create` | Assemble the multi-platform image reference |
| `make run-oci-dry-run` | Build a local OCI run bundle without publishing it |
| `make run-oci-push` | Publish an official run artifact |
| `make release` | Tag `VERSION` and create the GitHub Release after prompting for notes |

Run `make help` for the complete target list and required variables.

## Continuous integration and distribution

GitHub Actions provides fast pull-request checks, container smoke tests,
scheduled integration tests, scientific canaries, release validation, and
image publication. A push to `main` publishes a timestamped release-candidate
image and a temporary controller wheel. A tag reachable from `main` publishes
the matching release image and controller package.

Published benchmark images are multi-platform GHCR references with native
`linux/amd64` and `linux/arm64` variants.

## Historical version

The previous architecture remains available at Git tag `v0.1.0`. Version 0.2
does not support legacy protocols, fallback pipelines, state migration, or
artifact migration.
