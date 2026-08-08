# DMF Benchmarks

[![arXiv](https://img.shields.io/badge/arXiv-2606.03463-b31b1b.svg)](https://arxiv.org/abs/2606.03463)

DMF Benchmarks compares [DMF](https://github.com/matstech/dmf) and
[Mem0](https://github.com/mem0ai/mem0) on LoCoMo and LongMemEval. Version 0.2
has one supported runtime: the `dmf-bench` application in Docker Compose,
backed by Qdrant Server. It publishes verified JSON artifacts to a named volume
and exposes committed results through a loopback-only, read-only FastAPI
service.

The framework-owned retrieval path is product behavior, not a selectable
protocol. DMF and Mem0 are installed from the exact Git commits recorded in
`pyproject.toml` and `poetry.lock`; changing either commit is a scientific
change and requires rerunning the certification gates.

## Docker quickstart

Requirements: Docker with the Compose plugin and GNU Make. Create a root
`.env` file for the provider selected by the experiment, for example:

```env
OPENAI_API_KEY=your_openai_api_key_here
# OPENAI_BASE_URL=https://api.openai.com/v1
```

OpenRouter credentials and an Ollama endpoint can instead be supplied through
`OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`, and `OLLAMA_BASE_URL`. Provider and
model selection remain scientific inputs in the experiment JSON; environment
variables supply only credentials and endpoints.

Build the locked image and start the durable services:

```bash
docker compose -f deploy/compose.yaml build artifact-api benchmark
make stack-up
```

Choose one of the four checked-in experiment configs:

| Benchmark | Framework | Container config |
|---|---|---|
| LoCoMo | DMF | `/bench/config/experiment-locomo-dmf.json` |
| LoCoMo | Mem0 | `/bench/config/experiment-locomo-mem0.json` |
| LongMemEval | DMF | `/bench/config/experiment-longmemeval-dmf.json` |
| LongMemEval | Mem0 | `/bench/config/experiment-longmemeval-mem0.json` |

Run exactly one configuration with a fresh run ID:

```bash
make run \
  CONFIG=/bench/config/experiment-locomo-dmf.json \
  RUN_ID=local-locomo-dmf-001
```

The same wrapper applies to every row in the table. The checked-in configs use
pinned mini datasets for smoke and development runs; they are not paper-scale
reproduction configs.

The benchmark container has a read-only root filesystem. Run state, Qdrant
data, model cache, Prometheus data, and Grafana data use isolated v2 named
volumes. `HF_HUB_OFFLINE=1` is the default, so a missing or incomplete model
cache fails closed instead of downloading material during a benchmark.

## Operations

Resume and inspect a persisted run through the same image:

```bash
make resume RUN_ID=local-locomo-dmf-001
make status RUN_ID=local-locomo-dmf-001
make verify RUN_ID=local-locomo-dmf-001
```

Stop services without deleting named volumes:

```bash
make stack-down
```

The supported Make surface is deliberately small:

| Command | Container operation |
|---|---|
| `make stack-up` | Start Qdrant, Artifact API, Prometheus, and Grafana |
| `make run CONFIG=… RUN_ID=…` | Execute one benchmark lifecycle |
| `make resume RUN_ID=…` | Continue from persisted state |
| `make status RUN_ID=…` | Read persisted lifecycle state |
| `make verify RUN_ID=…` | Verify committed artifact digests |
| `make stack-down` | Stop containers while preserving volumes |

No host-side benchmark execution is supported.

## Results and observability

Open <http://127.0.0.1:8000> for the operational landing page. It links all
local service interfaces and the main reports for an exact run ID:

- Artifact API and landing page: <http://127.0.0.1:8000>
- Artifact API documentation: <http://127.0.0.1:8000/docs>
- Qdrant dashboard: <http://127.0.0.1:6333/dashboard>
- Prometheus: <http://127.0.0.1:9090>
- Grafana: <http://127.0.0.1:3000>

Committed JSON artifacts are available through the read-only API:

```bash
RUN_ID=local-locomo-dmf-001
curl -fsS "http://127.0.0.1:8000/runs/$RUN_ID" | jq
curl -fsS "http://127.0.0.1:8000/runs/$RUN_ID/artifacts/reports/summary.json" | jq
curl -fsS "http://127.0.0.1:8000/runs/$RUN_ID/artifacts/reports/usage.json" | jq
curl -fsS "http://127.0.0.1:8000/runs/$RUN_ID/artifacts/reports/timing.json" | jq
curl -fsS "http://127.0.0.1:8000/runs/$RUN_ID/artifacts/evaluations/rigorous_report.json" | jq
curl -fsS "http://127.0.0.1:8000/runs/$RUN_ID/artifacts/evaluations/ablation_report.json" | jq
curl -fsS "http://127.0.0.1:8000/runs/$RUN_ID/verify" | jq
```

`usage.json` separates framework-internal, answerer, and judge calls and
tokens. `timing.json` aggregates measured lifecycle work for the active
attempt. Scientific quality remains in `rigorous_report.json` and
`ablation_report.json`.

The Artifact API is unauthenticated only because it is read-only and bound to
loopback. Do not expose it directly to an untrusted network.

## Dataset registry

Dataset registry operations also run in the image:

```bash
docker compose --profile job -f deploy/compose.yaml run --rm --no-deps \
  benchmark list-datasets
```

Materialization is explicit. The output bind is made writable only for this
operation:

```bash
docker compose --profile job -f deploy/compose.yaml run --rm --no-deps \
  --volume "$PWD/datasets:/bench/datasets" \
  benchmark materialize-dataset \
  --dataset-id <pinned-dataset-id> \
  --output-dir /bench/datasets/<benchmark>
```

Unpinned official registry entries fail closed. Dataset pinning, embedding
model materialization, provider-backed canaries, and paper-scale runs require
separate review because they can introduce downloads or cost.

## Pipeline and artifacts

Each run performs sequential ingestion, framework-owned retrieval, answer
generation, LLM judging, deterministic rigorous/ablation evaluation, atomic
JSON publication, and digest verification. Prompt/completion token usage and
wall-clock timings are first-class reports. The Artifact API sees only the
runs volume; it does not share benchmark cache, Qdrant, Prometheus, or Grafana
state.

The manual `Scientific canary` workflow is cost-gated by the protected
`scientific-canary` environment and requires the exact
`confirm_costs=APPROVED` input. Do not trigger it without an approved dataset,
model scope, call/token estimate, and maximum cost.

## Historical version

The previous architecture is available exclusively from the historical Git
tag `v0.1.0`. Version 0.2 does not provide compatibility commands, state or
artifact migration, or fallback execution paths.
