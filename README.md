# DMF Benchmarks

[![arXiv](https://img.shields.io/badge/arXiv-2606.03463-b31b1b.svg)](https://arxiv.org/abs/2606.03463)

This repository contains the benchmark evaluation suite for the [dmf](https://github.com/matstech/dmf), a specialized memory system for conversational AI, comparing its performance and resource footprint against [Mem0](https://github.com/mem0ai/mem0).

It implements isolated end-to-end evaluation pipelines for two standard long-term memory benchmarks: [LoCoMo (Long-Context Memory Benchmark)](https://github.com/snap-research/locomo) and [LongMemEval](https://github.com/xiaowu0162/LongMemEval/tree/main).

## Framework-owned retrieval approach

The benchmark directly queries each memory framework's natural storage surface. This is the product behavior in the v2 runtime and is no longer a selectable configuration dimension.

Specifically, during question answering, the framework-native context is retrieved and formatted into a minimal context injection prompt for the answerer model (following the official guidelines and examples of each benchmark framework). This provides a realistic assessment of each framework's retrieval quality.

## Dependencies

This benchmark requires the following key memory frameworks:

- **dmf**: The benchmark installs [dmf-memory](https://github.com/matstech/dmf) from the exact Git commit recorded in `pyproject.toml` and `poetry.lock`.
- **mem0**: The benchmark installs a commit-pinned [Mem0 fork](https://github.com/matstech/mem0). The fork provides the internal telemetry required to account for prompt/completion tokens and API calls made by Mem0 itself.

Upgrading either commit is a scientific change and requires rerunning the runtime and parity gates.

## Environment

The evaluation pipelines rely on LLM providers for generating answers and executing LLM-as-a-judge scoring. By default, the benchmark is configured to use **OpenAI** (defaulting to `gpt-4.1-mini` for the answerer and `gpt-5-mini` for the judge).

OpenRouter and Ollama are also supported. Answerer and judge selection are
scientific inputs stored in the experiment JSON; framework-internal models
(such as Mem0's extraction LLM) live in the hash-pinned framework config.
Environment variables provide only credentials and endpoints. Do not switch
provider or model by editing the Makefile.

To configure your environment, create a `.env` file in the root of the project (which will be loaded automatically) or export the variables in your shell:

### 1. OpenAI (Default)

To use the default OpenAI models:

```env
OPENAI_API_KEY=your_openai_api_key_here
# Optional:
# OPENAI_BASE_URL=https://api.openai.com/v1
```

### 2. OpenRouter (Alternative)

To route calls through OpenRouter:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

### 3. Ollama (Alternative)

For an Ollama server reachable from a Docker Desktop container:

```env
OLLAMA_BASE_URL=http://host.docker.internal:11434
```

The selected experiment must also set `models.answerer.provider` and/or
`models.judge.provider` to `ollama` (or `openrouter`) and declare the requested
model. A Mem0 experiment must separately use a framework config whose internal
LLM provider is compatible with that environment. The four checked-in
smoke configs and their Mem0 framework configs currently select OpenAI.

## Implemented Pipeline

The complete evaluation and publication lifecycle is summarized below:

```
           +---------------------------------------------+
           |              Benchmark Dataset              |
           +----------------------+----------------------+
                                  |
                                  | (sequential turns)
                                  v
           +---------------------------------------------+
           |                 Ingestion                   |
           |  - DMF: TemporalMemory engine direct writes |
           |  - Mem0: sequential profile additions       |
           +----------------------+----------------------+
                                  |
                                  | (memory state)
                                  v
           +---------------------------------------------+
           |               Native Context                |
           |  - Extract native memory surface from store |
           |  - Build minimal context-injection prompt   |
           +----------------------+----------------------+
                                  |
                                  | (context & question)
                                  v
           +---------------------------------------------+
           |                Answer Phase                 |
           |  - Query LLM (gpt-4.1-mini) for predictions |
           |  - Tracks inference token usage & latency   |
           +----------------------+----------------------+
                                  |
                                  | (predictions)
                                  v
           +---------------------------------------------+
           |                 Evaluation                  |
           |  - LLM Judge (gpt-5-mini) grades predictions|
           |  - Rigorous and ablation quality metrics    |
           +----------------------+----------------------+
                                  |
                                  v
           +---------------------------------------------+
           |             Report / Publish / Verify       |
           |  - Quality summary, token and timing reports|
           |  - Atomic publication to the shared volume  |
           +---------------------------------------------+
```

1. **Ingestion**: Dialog turns or interaction logs are sequentially loaded into the memory backend. For the _dmf_, ingestion interacts directly with the _TemporalMemory_ storage layer. For _Mem0_, turns are written sequentially using its standard user memory interface.
2. **Native Context Construction**: The pipeline queries the memory backend to fetch the native memory context surface relevant to the current user state. This context is embedded directly in a minimal, natural QA prompt.
3. **Answer Generation**: The prompt is sent to the answerer model (defaulting to gpt-4.1-mini) to produce candidate answers. Prompt tokens, and completion tokens are recorded.
4. **Evaluation and publication**: A stronger LLM judge (defaulting to gpt-5-mini) compares generated answers against ground truth. The rigorous evaluator computes deterministic answer and retrieval metrics; DMF also produces an ablation report, while Mem0 records ablation as `NOT_APPLICABLE`. Token usage and wall-clock timings are separate operational reports. The lifecycle publishes all committed artifacts atomically and verifies their digests before declaring the run complete.

## Containerized benchmark runner

The authoritative local path is the `dmf-bench` lifecycle running in Docker
against Qdrant Server. It writes verified JSON artifacts to a shared named
volume; the independent FastAPI service exposes the committed copy read-only.

Build the image and start the durable services:

```bash
docker compose -f deploy/compose.yaml build benchmark artifact-api
make stack-up
```

The benchmark job uses a read-only root filesystem. Runs, Qdrant state and
model caches live in named volumes. Hugging Face access is disabled by default
with `HF_HUB_OFFLINE=1`: an empty or incomplete embedding cache fails closed
instead of downloading an unpinned model during a benchmark. The pinned model
materialization procedure will be defined as part of the separately approved
provider-backed canary; do not set `HF_HUB_OFFLINE=0` for a scientific run.

Models, dataset selection, filters and evaluator requirements live in the local
experiment JSON files,
not in the Makefile:

- `config/experiment-locomo-dmf.json`
- `config/experiment-locomo-mem0.json`
- `config/experiment-longmemeval-dmf.json`
- `config/experiment-longmemeval-mem0.json`

They currently reference the pinned local mini datasets under `datasets/` and
are smoke/development experiments, not paper-scale reproductions.

Run a benchmark with a fresh operational run id:

```bash
make locomo LOCOMO_FRAMEWORK=dmf RUN_ID=local-locomo-dmf-001
make locomo LOCOMO_FRAMEWORK=mem0 RUN_ID=local-locomo-mem0-001

make longmemeval LONGMEMEVAL_FRAMEWORK=dmf RUN_ID=local-longmemeval-dmf-001
make longmemeval LONGMEMEVAL_FRAMEWORK=mem0 RUN_ID=local-longmemeval-mem0-001
```

The equivalent direct Docker command is:

```bash
docker compose --profile job -f deploy/compose.yaml run --rm benchmark \
  run \
  --config /bench/config/experiment-locomo-dmf.json \
  --run-id local-locomo-dmf-001
```

### Resume and status

Resume always uses the persisted resolved config and does not accept scientific
overrides:

```bash
make resume RUN_ID=local-locomo-dmf-001
make status RUN_ID=local-locomo-dmf-001
```

### Results and services

Open the local service landing page at <http://127.0.0.1:8000>. It links the
Grafana, Prometheus and Qdrant interfaces, the Artifact API documentation and
the principal reports for any exact run ID. Links follow the published ports
configured in Compose, so overrides such as `GRAFANA_PORT` keep working.

Committed artifacts are also available directly through the read-only API:

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

`usage.json` separates memory-internal, answerer and judge calls/tokens and
reports telemetry completeness. `timing.json` aggregates measured lifecycle
work for the active attempt. Neither report is a substitute for the scientific
quality metrics in `rigorous_report.json` and `ablation_report.json`.

- Service landing page: <http://127.0.0.1:8000>
- Artifact API (OpenAPI): <http://127.0.0.1:8000/docs>
- Qdrant Web UI: <http://127.0.0.1:6333/dashboard>
- Prometheus: <http://127.0.0.1:9090>
- Grafana: <http://127.0.0.1:3000>

Grafana shows live unit/item progress, lifecycle state, provider activity and
Qdrant operations while the benchmark container is running. The Artifact API
is unauthenticated by design only because it is read-only and bound to
loopback; do not expose it directly to an untrusted network.

Stop services without deleting named volumes:

```bash
make stack-down
```

Do not use `docker compose down --volumes` unless the persisted benchmark,
Qdrant and dashboard state should be intentionally deleted.

### Dataset materialization

Inspect the versioned registry with:

```bash
poetry run dmf-bench list-datasets
```

Materialization is explicit and fail-closed:

```bash
poetry run dmf-bench materialize-dataset \
  --dataset-id <pinned-dataset-id> \
  --output-dir datasets/<benchmark>
```

The built-in official dataset entries remain intentionally unpinned and cannot
be materialized until an approved revision and SHA-256 are recorded. Official
dataset pinning and provider-backed canaries are separate, cost-gated work.

### Provider-backed scientific canary

The deterministic container matrix certifies infrastructure and lifecycle
behavior without remote LLM calls. It does not establish provider-backed or
paper-scale equivalence. That requires a separately approved provider-backed canary with:

- immutable dataset and embedding-model revisions;
- an agreed item/question scope;
- an advance estimate of answerer/judge calls, tokens and maximum cost;
- verification of returned model IDs, usage, timing, reports and resume;
- a separate decision before removing any legacy pipeline.

The `Scientific canary` GitHub workflow is manual-only, requires the exact
`confirm_costs=APPROVED` input and uses the protected `scientific-canary`
environment. Do not trigger it until the remaining revisions and cost envelope
have been reviewed.

## Legacy compatibility path

The historical Python pipelines remain available for comparison and recovery,
but they are no longer the primary local entrypoints:

```bash
make locomo-legacy LEGACY_LOCOMO_ARGS='--framework dmf --config config/locomo_dmf_settings.toml'
make longmemeval-legacy LEGACY_LONGMEMEVAL_ARGS='--framework dmf --config config/longmemeval_dmf_settings.toml'
```

Judge-only and rigorous-only compatibility targets use the explicit
`*-judge-legacy` and `*-rigorous-legacy` suffixes. The previous unsuffixed
maintenance names remain deprecated aliases for one compatibility window.
They must not be removed solely on the basis of the deterministic container
gate; removal requires the provider-backed canary and a separate review.
