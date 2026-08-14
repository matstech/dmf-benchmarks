# DMF Benchmarks Operator Manual

Complete guide to `dmf-benchctl`, experiment configuration, execution, and
results.

# 1. Start Here

This manual is the complete user reference for `dmf-benchctl`. It explains not
only which commands exist, but when to use each command, what it changes, where
its output goes, and how to recover when a run fails.

Use this document in one of three ways:

1. Follow **First Reproducible Run** if this is your first execution.
2. Follow **Custom Experiments** when changing models, datasets, sampling, or
   memory-system settings.
3. Use **Command Reference** and **Troubleshooting** while operating or
   automating existing runs.

All normal user operations go through `dmf-benchctl`. Do not invoke Docker
Compose directly. The controller is the stable external interface; Docker
Compose is only its current orchestration implementation.

Do not invoke the bundled `dmf-bench` executable directly for normal
operations. It is the internal runtime executed inside the benchmark image.
The controller exposes supported runtime diagnostics through its advanced
`runtime` command.

## What do you want to do?

| Goal | Command |
|---|---|
| See available starting configurations | `dmf-benchctl config list` |
| Copy a configuration for editing | `dmf-benchctl config export PRESET DIR` |
| Check one experiment config | `dmf-benchctl validate --config FILE` |
| Lock image, config, dataset, and run IDs | `dmf-benchctl prepare ...` |
| Start a run or prepared batch | `dmf-benchctl run ...` |
| See progress | `dmf-benchctl status --run-id RUN_ID` |
| Continue interrupted work | `dmf-benchctl resume --run-id RUN_ID` |
| See run metadata and artifact inventory | `dmf-benchctl inspect --run-id RUN_ID` |
| Verify committed artifact integrity | `dmf-benchctl verify --run-id RUN_ID` |
| Recompute deterministic LoCoMo reports | `dmf-benchctl evaluate --run-id RUN_ID` |
| Browse and download results | open `http://127.0.0.1:8000` |
| Start or stop local services | `dmf-benchctl stack ...` |
| Package or publish a completed run | `dmf-benchctl publish-run ...` |

The recommended reproducible path is:

```text
choose/export config -> prepare -> review prepared.json -> run -> status
    -> inspect results -> verify -> optionally evaluate -> optionally publish
```

`prepare` does not call the answerer or judge provider. Paid provider calls
begin at `run`.

# 2. Installation and Operator Workspace

## Host requirements

You need:

- Docker with the Compose plugin;
- Python 3.12 or newer for the lightweight controller;
- access to the selected benchmark image;
- provider credentials required by the selected models;
- ORAS only when publishing a run OCI artifact.

You do not need Poetry or a repository clone. The wheel contains the
orchestration definition, dashboards, framework settings, dataset registry,
and built-in experiment presets.

## Install the controller

Install a release wheel with `pipx`:

```text
python3 -m pip install --user pipx
python3 -m pipx ensurepath
pipx install ./dmf_benchmarks-VERSION-py3-none-any.whl
dmf-benchctl --help
```

Replace an installed version with another wheel:

```text
pipx install --force ./dmf_benchmarks-VERSION-py3-none-any.whl
```

Every successful `main` image-publication workflow also stores an RC wheel for
14 days. From a checkout of this repository, download the latest artifact with:

```text
RUN_ID="$(gh run list --repo matstech/dmf-benchmarks \
  --workflow publish-image.yml --branch main --status success --limit 1 \
  --json databaseId --jq '.[0].databaseId')"
gh run download --repo matstech/dmf-benchmarks "$RUN_ID" \
  --pattern 'dmf-benchctl-*-RC'
pipx install --force dmf-benchctl-*-RC/*.whl
```

Tagged releases retain their wheel permanently. RC workflow artifacts are
temporary.

### Install a tagged release

For a reproducible installation, download the wheel attached to the GitHub
release that matches the benchmark image. For example, for `v0.2.0`:

```text
VERSION=v0.2.0
mkdir -p dmf-bench-$VERSION
cd dmf-bench-$VERSION

gh release download "$VERSION" \
  --repo matstech/dmf-benchmarks \
  --pattern 'dmf_benchmarks-*.whl' \
  --pattern 'dmf_benchmarks-*.tar.gz'

pipx install --force dmf_benchmarks-*.whl
```

The release wheel contains the controller and its Docker/Compose resources.
The source archive is optional and is useful only when inspecting the release
contents. No repository clone is required for normal operation.

Download the matching multi-platform benchmark image by tag:

```text
docker pull ghcr.io/matstech/dmf-benchmarks:$VERSION
```

For an official reproducible run, resolve the tag to its immutable registry
digest and use that digest in the workspace configuration. The digest can be
read with:

```text
docker buildx imagetools inspect ghcr.io/matstech/dmf-benchmarks:$VERSION
```

Then configure, for example:

```dotenv
DMF_BENCH_IMAGE=ghcr.io/matstech/dmf-benchmarks@sha256:IMAGE_MANIFEST_DIGEST
```

The image manifest selects `linux/amd64` or `linux/arm64` for the local Docker
engine. Do not commit the provider key or local `.env` file. Release images do
not contain private API keys or the complete real benchmark datasets.

## Choose a workspace

Run controller commands from a dedicated operator directory:

```text
mkdir my-benchmark-workspace
cd my-benchmark-workspace
```

The current directory is the default workspace. It contains user-owned state:

```text
.env
.dmf-bench/
|-- prepared.json
|-- jobs/
|-- logs/
`-- secrets/       ephemeral files only
my-locomo-dmf/
|-- experiment.json
`-- dmf-settings.toml
```

Use the global `--work-dir PATH` option to operate another directory without
changing the shell directory. A prepared context belongs to its recorded
workspace and is rejected from a different workspace.

`--project-dir PATH` has a different purpose: it tells the controller to use
orchestration assets from a source checkout. Normal wheel users should not use
it.

## Configure image, secrets, and limits

Create `.env` in the workspace:

```dotenv
DMF_BENCH_IMAGE=ghcr.io/ORG/dmf-benchmarks:TAG
OPENAI_API_KEY=paste_the_key_here
HF_HUB_OFFLINE=1
DMF_BENCH_CPUS=4
DMF_BENCH_THREADS=4
```

You may export the same variables instead. Exported shell variables take
precedence over `.env`. The global `--image IMAGE` option takes precedence over
`DMF_BENCH_IMAGE` for that command.

Provider secrets must never be placed in experiment JSON, framework TOML/YAML,
or command arguments. For `run` and `resume`, the controller writes each secret
to a temporary private host file, mounts it read-only in the job, and deletes
the file when that job command ends. Secrets are removed from the environment
passed to Docker Compose.

Supported provider credential variables currently include:

- `OPENAI_API_KEY`;
- `OPENROUTER_API_KEY`.

Optional endpoint and network variables passed to the runtime include
`OPENAI_BASE_URL`, `OPENROUTER_BASE_URL`, `OLLAMA_BASE_URL`, `HTTP_PROXY`,
`HTTPS_PROXY`, `ALL_PROXY`, and `NO_PROXY`.

Local Qdrant does not require an API key.

## Image platform

Official image references contain native `linux/amd64` and `linux/arm64`
variants. `prepare` verifies that the selected variant matches the Docker
engine. Do not compare official timings from an emulated architecture.

# 3. First Reproducible Run

This tutorial executes the complete built-in smoke suite: LoCoMo and
LongMemEval with both DMF and Mem0.

## Step 1: confirm configuration and credentials

```text
dmf-benchctl config list
dmf-benchctl stack status
```

Ensure `.env` or the shell contains `DMF_BENCH_IMAGE` and `OPENAI_API_KEY`.
The stack may be stopped at this point.

## Step 2: prepare the batch

```text
dmf-benchctl prepare \
  --suite smoke \
  --allow-downloads \
  --observability
```

Preparation performs deterministic checks before paid calls:

1. verifies Docker and the Compose plugin;
2. requires provider credentials declared by the configs, without using them;
3. pulls the requested image;
4. resolves and records its registry digest, revision, and native platform;
5. reads the pinned dataset registry embedded in that image;
6. checks that each prepared dataset is immutable and belongs to the expected
   benchmark;
7. validates orchestration and starts the requested services;
8. materializes and validates datasets;
9. records every config SHA-256 and generated run ID;
10. writes `.dmf-bench/prepared.json` atomically.

`--allow-downloads` permits controlled dataset and model downloads during
preparation and records that policy for later runs. Omit it only when all
required bytes are already cached.

## Step 3: review the preparation receipt

```text
python3 -m json.tool .dmf-bench/prepared.json
```

Before spending provider tokens, confirm:

- `image.resolved_ref` contains the expected registry digest;
- `image.platform` is native for the machine;
- `image.revision` is the intended harness revision;
- every `experiments[].benchmark` and `framework` is expected;
- every requested answerer and judge model is correct;
- dataset locks and sampling policies match between DMF and Mem0 for the same
  benchmark;
- generated run IDs are unique;
- `observability` and `allow_downloads` match the intended policy.

Do not edit a config after preparation. Version-2 prepared contexts verify
each config hash before execution and reject drift.

## Step 4: launch the batch

```text
dmf-benchctl run --prepared
```

Default execution is detached. The command starts a background host worker,
prints a JSON receipt, and returns to the shell. Save the receipt. It contains:

- the worker PID;
- the local job-receipt path;
- the worker-log path;
- all generated run IDs;
- exact `status` and `verify` commands;
- the locked image reference.

The worker executes experiments sequentially in prepared order and stops at
the first failure. Provider calls begin here.

Use `--follow` only for interactive diagnostics:

```text
dmf-benchctl run --prepared --follow
```

With `--follow`, the command stays in the foreground and streams job output.
It does not detach.

## Step 5: monitor each run

Use a run ID printed by the launch receipt:

```text
dmf-benchctl status --run-id RUN_ID
```

The output distinguishes:

- lifecycle state and phase;
- overall progress across the run;
- completed evaluation items;
- completed input records;
- current partial activity and progress inside the active record;
- whether verification is the next valid action.

For agents and scripts:

```text
dmf-benchctl status --run-id RUN_ID --json
```

Re-run `status`; it is read-only. Do not infer a hang from zero overall
progress while partial ingestion or answer generation is advancing.

## Step 6: inspect and verify completion

```text
dmf-benchctl inspect --run-id RUN_ID
dmf-benchctl verify --run-id RUN_ID
```

`inspect` answers “what is this run and which artifacts exist?” It returns the
manifest, completion state, artifact inventory, and verification details for a
committed run.

`verify` answers “are all committed files present and byte-for-byte identical
to the final manifest?” It verifies checksums and sizes without changing the
run.

## Step 7: read results

Open:

```text
http://127.0.0.1:8000
```

Select a committed run, browse its artifact list, and open or download files.
The main result files are explained in **Results and Artifacts**.

If observability was enabled, open:

```text
Operations: http://127.0.0.1:3000/d/dmf-benchmarks/dmf-benchmarks
Evaluation: http://127.0.0.1:3000/d/dmf-benchmarks-evaluation/dmf-benchmarks-evaluation
```

Select one benchmark and one memory system in each dashboard.

## Step 8: stop services

```text
dmf-benchctl stack down
```

This stops and removes service containers but preserves named volumes,
including run artifacts, model/dataset cache, Grafana data, Prometheus data,
and Qdrant storage. The controller has no user command that deletes these
volumes.

# 4. Custom Experiments

## Export an editable bundle

List built-in presets:

```text
dmf-benchctl config list
```

Current presets are:

- `locomo-dmf`;
- `locomo-mem0`;
- `longmemeval-dmf`;
- `longmemeval-mem0`.

Export one:

```text
dmf-benchctl config export locomo-mem0 ./my-locomo-mem0
```

The bundle contains portable filenames:

```text
my-locomo-mem0/
|-- experiment.json
`-- mem0-settings.yaml
```

A DMF export contains `dmf-settings.toml`. Different bundles may all use the
filename `experiment.json`; run identity is derived from benchmark/framework,
not from the filename.

Use `--force` only when intentionally replacing files previously generated by
the controller:

```text
dmf-benchctl config export locomo-mem0 ./my-locomo-mem0 --force
```

## Understand where models are configured

There are two configuration layers:

1. `experiment.json` configures the benchmark **answerer** and **judge** under
   `models.answerer` and `models.judge`.
2. The framework settings file configures memory-system internals.

For Mem0, `mem0-settings.yaml` contains its internal `llm`, `embedder`, search,
and vector-store declarations. Mem0 can therefore make internal LLM calls in
addition to the benchmark answerer and judge calls. These categories are
reported separately in `reports/usage.json`.

For DMF, `dmf-settings.toml` contains embedding, scoring, temporal-decay,
capacity, long-term-memory, and retrieval settings.

Changing either layer changes the experiment. Exported bundles are designed
for editing before preparation. The controller refreshes the declared hash of
an adjacent edited framework settings file before validation/preparation, then
locks the resulting experiment JSON hash in the prepared context.

## Validate before preparing

Check a config whose dataset is already available:

```text
dmf-benchctl validate --config ./my-locomo-mem0/experiment.json
```

Allow registry materialization during validation:

```text
dmf-benchctl validate \
  --config ./my-locomo-mem0/experiment.json \
  --materialize-datasets \
  --allow-downloads
```

`validate` checks one current config. It does not create a reproducibility
receipt, generate locked run IDs, or execute the benchmark lifecycle. Use it
while editing.

## Prepare one experiment

```text
dmf-benchctl prepare \
  --config ./my-locomo-mem0/experiment.json \
  --state-file .dmf-bench/prepared-locomo-mem0.json \
  --batch-id 20260813T120000Z-locomo-mem0 \
  --allow-downloads \
  --observability
```

Run that exact context:

```text
dmf-benchctl run --prepared .dmf-bench/prepared-locomo-mem0.json
```

## Prepare a comparison batch

Repeat `--config` to place comparable experiments in one sequential batch:

```text
dmf-benchctl prepare \
  --config ./my-locomo-dmf/experiment.json \
  --config ./my-locomo-mem0/experiment.json \
  --state-file .dmf-bench/prepared-locomo-comparison.json \
  --batch-id 20260813T120000Z-locomo \
  --allow-downloads \
  --observability
```

A prepared batch accepts at most one config for each benchmark/framework
combination. Configs for the same benchmark must declare the same dataset and
sampling policy. This prevents an accidental DMF/Mem0 comparison over
different samples.

# 5. Execution Modes

## Prepared batch: recommended for reproducible runs

```text
dmf-benchctl run --prepared [PREPARED_FILE]
```

If the path is omitted, the default is `.dmf-bench/prepared.json`.

Characteristics:

- uses the image digest and policies recorded by `prepare`;
- rejects changed or missing config files before starting;
- uses generated, unique run IDs;
- executes entries sequentially;
- inspects each run before execution;
- resumes an existing incomplete/failed run;
- starts only a run whose state is absent;
- stops the batch on the first non-zero run result;
- detaches by default.

Re-running the same command is safe after an interruption. It does not silently
replace a committed run.

## Direct run: exploratory or individually managed

```text
dmf-benchctl run \
  --config ./my-locomo-dmf/experiment.json \
  --run-id exploratory-locomo-dmf-001 \
  --allow-downloads \
  --observability
```

The caller chooses the run ID. A direct run does not create a prepared context
and does not lock the image by registry digest on the user's behalf. It is
useful for development and diagnosis, but `prepare` is safer for official work.

A run ID is immutable. Use a new ID for a different experiment or a clean
restart. Use `resume` for the same persisted run.

## Detached versus foreground

`run` detaches by default. It prints a job receipt and returns immediately.

Add `--follow` to run in the foreground and stream logs. `--follow` changes
only operator interaction; it does not change scientific behavior.

Detached logs are stored under `.dmf-bench/logs/`. The launch receipt identifies
the exact worker log and run IDs.

## Plan-only

```text
dmf-benchctl run --config FILE --run-id RUN_ID --plan-only
dmf-benchctl resume --run-id RUN_ID --plan-only
```

For a new run, plan-only resolves the dataset and prints expected input/evaluation
IDs and final mode without creating run state. For resume, it prints the
persisted resume plan. It does not detach.

## Predict-only

```text
dmf-benchctl run --config FILE --run-id RUN_ID --predict-only
```

Predict-only stops after predictions and produces a `PARTIAL` run. It is not a
completed official result. A later full `resume` can continue the persisted
lifecycle.

## Use an already running stack

```text
dmf-benchctl stack up --observability
dmf-benchctl run --prepared PREPARED_FILE --no-stack-up
```

`--no-stack-up` skips service startup checks. Use it only when the matching
stack is already healthy.

## Run DMF and Mem0 in parallel

One prepared context is intentionally sequential. To run two experiments in
parallel, create two prepared contexts with different batch IDs and therefore
different detached job IDs:

```text
STAMP=20260813T120000Z

dmf-benchctl prepare \
  --config ./my-locomo-dmf/experiment.json \
  --state-file .dmf-bench/prepared-dmf.json \
  --batch-id "${STAMP}-dmf" \
  --allow-downloads --observability

dmf-benchctl prepare \
  --config ./my-locomo-mem0/experiment.json \
  --state-file .dmf-bench/prepared-mem0.json \
  --batch-id "${STAMP}-mem0" \
  --allow-downloads --observability

dmf-benchctl stack up --observability
dmf-benchctl run --prepared .dmf-bench/prepared-dmf.json --no-stack-up
dmf-benchctl run --prepared .dmf-bench/prepared-mem0.json --no-stack-up
```

Both `run` commands detach; shell `&` is unnecessary.

Parallel execution shares CPU, memory, network, Qdrant, cache, and provider
limits. It is appropriate for throughput or functional checks, but not for an
official timing comparison. Live process metrics may also be ambiguous when
multiple one-shot benchmark containers exist simultaneously. For fair timing
and resource measurements, run DMF and Mem0 sequentially with identical CPU
and thread limits.

# 6. Command Reference

Global options must appear before the command:

```text
dmf-benchctl [GLOBAL OPTIONS] COMMAND [COMMAND OPTIONS]
```

Example:

```text
dmf-benchctl --work-dir ./operator --image ghcr.io/ORG/IMAGE:TAG \
  prepare --suite smoke --allow-downloads
```

## Global options

`--work-dir PATH`

: Select writable operator state and `.env`. Defaults to the current directory
  for wheel users.

`--project-dir PATH`

: Use deploy/config resources from a source checkout. Intended for controller
  development, not normal operation.

`--image REF`

: Select the benchmark image for this command. Overrides `DMF_BENCH_IMAGE`.

`--pull always|missing|never`

: Select service-image pull behavior when bringing up the stack. Default:
  `missing`.

`--compose-override FILE`

: Add an orchestrator override file. Repeatable and intended for advanced
  deployment/development use.

`--dry-run`

: Print orchestrator commands without executing them. It cannot be combined
  with `prepare`, because preparation must inspect and persist real state.

## `config`

```text
dmf-benchctl config
dmf-benchctl config list
dmf-benchctl config export PRESET OUTPUT_DIR [--force]
```

- `config` checks the resolved orchestration definition.
- `config list` prints built-in presets as JSON.
- `config export` copies one preset and its framework settings into an editable
  portable directory.
- `--force` replaces files generated in the target directory.

No service is started and no provider call occurs.

## `prepare`

```text
dmf-benchctl prepare \
  (--config FILE [--config FILE ...] | --suite smoke) \
  [--state-file FILE] [--batch-id ID] \
  [--observability] [--allow-downloads]
```

Use it to create the reproducibility boundary before execution. It requires a
published image with a registry digest; a purely local unnamed image is not
sufficient. It starts services needed for validation but makes no answerer or
judge request.

Outputs:

- prepared context JSON;
- preparation logs under `.dmf-bench/logs/`;
- materialized datasets/model cache in persistent volumes;
- running core/observability services.

## `image build`

```text
dmf-benchctl image build
```

Builds the local benchmark and Artifact API services from source runtime
assets. This is primarily a repository-development command. Reproducible
`prepare` still requires a published image with a registry digest.

## `stack`

```text
dmf-benchctl stack up [--observability]
dmf-benchctl stack status
dmf-benchctl stack stop [--observability]
dmf-benchctl stack down
```

- `up` starts and waits for core services; `--observability` adds Prometheus
  and Grafana.
- `status` prints current service/container state.
- `stop` stops selected services without removing their containers.
- `down` removes service containers and network but preserves named volumes.

`stop --observability` includes observability services in the stop set.

## `validate`

```text
dmf-benchctl validate --config FILE \
  [--materialize-datasets] [--allow-downloads]
```

Checks one experiment through the selected benchmark image. It prints the
resolved redacted config. `--materialize-datasets` resolves a registry-backed
dataset; `--allow-downloads` changes the offline policy for this operation.

Validation does not generate a prepared context, run ID, predictions, scores,
or committed run.

## `run`

```text
dmf-benchctl run --config FILE --run-id ID [OPTIONS]
dmf-benchctl run --prepared [PREPARED_FILE] [OPTIONS]
```

Options:

- `--predict-only`: stop after prediction with a `PARTIAL` state;
- `--plan-only`: print the execution plan without run mutation;
- `--observability`: start Prometheus and Grafana;
- `--no-stack-up`: assume services already run;
- `--allow-downloads`: permit Hugging Face downloads for this operation;
- `--follow`: foreground execution with streamed logs.

Without `--follow`, `--plan-only`, or global `--dry-run`, execution detaches.

## `status`

```text
dmf-benchctl status --run-id RUN_ID [--json]
```

Read-only progress view. Human output is the default. `--json` emits the stable
operator schema with `evaluation_items`, `input_records`, and optional
`current_activity`. It deliberately hides the internal checkpoint term
`units`.

Status is the primary monitoring command. A completed status is integrity
checked before it is reported as verified.

## `inspect`

```text
dmf-benchctl inspect --run-id RUN_ID
```

Returns raw structured run identity, manifest, paths, completion marker,
artifact inventory, and verification data. Use it to discover artifacts or
audit provenance. It is read-only.

## `verify`

```text
dmf-benchctl verify --run-id RUN_ID
```

Requires a committed run. Recomputes final-manifest and artifact checksums and
sizes. A successful result proves local integrity, not scientific quality.
It is read-only.

## `resume`

```text
dmf-benchctl resume --run-id RUN_ID [OPTIONS]
dmf-benchctl resume --prepared [PREPARED_FILE] --run-id RUN_ID [OPTIONS]
```

Resume loads the resolved config persisted inside the run; no `--config` is
accepted. With `--prepared`, it also enforces the locked image/policies and
requires the run ID to belong to that prepared context.

Options:

- `--plan-only`: show the resume plan without execution;
- `--no-stack-up`: use an already running core stack;
- `--follow`: stream foreground logs.

Resume restarts incomplete atomic records rather than trusting partial
framework state.

## `evaluate`

```text
dmf-benchctl evaluate --run-id RUN_ID
```

Recomputes supported deterministic LoCoMo rigorous/ablation reports from an
immutable committed source run. It performs no ingestion, answerer call, or
judge call. Derived files are written separately under:

```text
/bench/runs/.derived-evaluations/RUN_ID/EVALUATOR_VERSION/
```

It is not a command for completing an interrupted judging phase; use `resume`
for that.

## `publish-run`

```text
dmf-benchctl publish-run \
  --run-id RUN_ID \
  --ref OCI_REFERENCE \
  [--subject IMAGE_DIGEST_REFERENCE] \
  [--output-dir DIR] [--oras-bin PATH] [--dry-run]
```

The command verifies the run, writes official bundle metadata, creates
`RUN_ID.run.tar.gz` from the complete run directory, retains it on the host,
and, unless `--dry-run` is present, pushes the archive with host ORAS.

- `--ref` is the destination, for example
  `ghcr.io/ORG/dmf-benchmarks-runs:RUN_ID`.
- `--subject` links the run artifact to the exact producing image digest.
- `--output-dir` defaults to `bundles` in the workspace.
- `--dry-run` packages and verifies but does not push and does not require
  ORAS.

## `runtime`

```text
dmf-benchctl runtime -- DMF_BENCH_ARGUMENTS...
```

Advanced escape hatch that invokes `dmf-bench` inside the selected image while
still using controller-managed orchestration and volumes. Common read-only
examples:

```text
dmf-benchctl runtime -- list
dmf-benchctl runtime -- list-datasets
```

Manual dataset materialization example:

```text
dmf-benchctl runtime -- materialize-dataset \
  --dataset-id locomo-official-v1 \
  --sample-fraction 0.05 \
  --sample-unit conversation \
  --sample-seed 7 \
  --sample-rounding ceil \
  --output-dir /bench/cache/datasets/manual-locomo
```

Prefer dedicated controller commands when one exists. `runtime` is intended
for advanced runtime capabilities and diagnosis.

# 7. Experiment Configuration

An experiment JSON uses `schema_version: 2`. It is the scientific contract for
one benchmark and one memory system.

## Top-level fields

`schema_version`

: Must be `2`.

`experiment_id`

: Human-readable experiment name. A prepared run ID is generated independently
  as `BATCH_ID-BENCHMARK-FRAMEWORK`.

`benchmark`

: Built-in values are `locomo` and `longmemeval`.

`framework`

: Built-in memory systems are `dmf` and `mem0`.

`runtime`

: Container paths, metrics port, and log level. Standard writable locations
  are `/bench/runs` and `/bench/cache`.

`framework_config`

: Path, format, profile, and SHA-256 of DMF TOML or Mem0 YAML settings.

`qdrant`

: Endpoint environment-variable name, request timeout, and retention policy.

`dataset`

: Local pinned dataset or registry ID, optionally with deterministic sampling.

`selection`

: Ordered source IDs and benchmark-specific filters applied after dataset
  materialization.

`models`

: Benchmark answerer and judge provider/model declarations, parameters,
  throttling, timeout, and retries.

`evaluation`

: Required and optional reports. A missing required report prevents successful
  completion.

`artifact_store`

: Local artifact target, normally `/bench/runs`.

## Standard runtime and artifact paths

```json
{
  "runtime": {
    "root": "/bench",
    "runs_dir": "/bench/runs",
    "cache_dir": "/bench/cache",
    "metrics_port": 9464,
    "log_level": "INFO"
  },
  "artifact_store": {
    "type": "local",
    "uri": "/bench/runs"
  }
}
```

Paths in experiment configs describe the container, not arbitrary host paths.
External exported bundles are mounted read-only under `/bench/operator`, and
adjacent relative framework/dataset resources are translated by the
controller. Writable work belongs under `/bench/cache` and `/bench/runs`.

## Models and retries

Example:

```json
{
  "models": {
    "answerer": {
      "provider": "openai",
      "requested_model": "gpt-4.1-mini",
      "parameters": {"temperature": 0, "max_tokens": 4096},
      "runtime": {
        "timeout_seconds": 120,
        "rpm": 200,
        "max_retries": 5
      }
    },
    "judge": {
      "provider": "openai",
      "requested_model": "gpt-5-mini",
      "parameters": {
        "temperature": 0,
        "max_tokens": 4096,
        "reasoning_effort": "low"
      },
      "runtime": {
        "timeout_seconds": 120,
        "rpm": 200,
        "max_retries": 5,
        "response_max_retries": 1
      }
    }
  }
}
```

`max_retries` handles retryable provider/transport failures.
`response_max_retries` handles malformed or truncated judge responses. All
attempts contribute to token usage. Exhaustion fails closed; the runner does
not invent a score.

## Qdrant retention

`delete-on-success` is recommended for official batches. A record's owned
collections are deleted only after predictions are complete and before the
record is committed. Cleanup failure leaves the record resumable and prevents
a false commit.

Use `keep` only for intentional post-run memory-system debugging. Retained
collections consume storage and can make operational inspection noisier,
although run namespaces remain isolated.

## Scientific versus operational changes

Scientific identity includes benchmark, framework, framework settings hash,
dataset and sampling, selection, answerer/judge models and parameters, and
evaluation requirements. Changing any of these defines a different
experiment.

Operational settings include cache/run paths, metrics port, log level,
request timeout, and artifact location. They do not normally define quality,
but must still be recorded when comparing timing or resource use.

# 8. Datasets, Sampling, and Selection

## Registry-backed dataset

Recommended official form:

```json
{
  "dataset": {
    "name": "locomo",
    "registry_id": "locomo-official-v1"
  }
}
```

`prepare` requires the registry entry embedded in the image to be pinned by
immutable source revision and SHA-256. Materialization writes `manifest.json`
under the dataset cache and copies it into the run as:

```text
datasets/materialization-manifest.json
```

The manifest records source identity, source/materialized hashes and sizes,
schema, counts, and optional sampling policy.

## Already materialized local dataset

Exploratory/direct configurations can declare:

```json
{
  "dataset": {
    "name": "locomo",
    "source": "local-approved-dataset",
    "revision": "v1",
    "path": "/bench/operator/locomo.json",
    "sha256": "64_CHARACTER_SHA256",
    "expected_schema": "locomo10-json-array-v1"
  }
}
```

`prepare` is intentionally stricter and requires `dataset.registry_id` with a
pinned registry entry.

## Sampling complete records

To reduce both ingestion and answering, sample complete top-level records.

LoCoMo, 5% of conversations:

```json
{
  "dataset": {
    "name": "locomo",
    "registry_id": "locomo-official-v1",
    "sampling": {
      "fraction": 0.05,
      "unit": "conversation",
      "seed": 7,
      "rounding": "ceil"
    }
  }
}
```

LoCoMo has only ten top-level conversations. Five percent rounded upward is one
complete conversation, so the effective top-level sample is ten percent. The
manifest records actual counts.

LongMemEval, 5% of records:

```json
{
  "dataset": {
    "name": "longmemeval",
    "registry_id": "longmemeval-s-official-v1",
    "sampling": {
      "fraction": 0.05,
      "unit": "record",
      "seed": 7,
      "rounding": "ceil"
    }
  }
}
```

LoCoMo also supports question-level sampling (`unit: "qa"`) and category
stratification, but that may retain most conversations and therefore does not
reduce ingestion proportionally.

If sampling is omitted, materialization still generates a manifest without a
sampling block.

## Apply the same sample to DMF and Mem0

Use this procedure when comparing two memory systems over a reduced dataset.
The example uses LoCoMo; for LongMemEval, export the two `longmemeval-*`
presets and use `unit: "record"`.

### Step 1: export one bundle per memory system

```text
dmf-benchctl config export locomo-dmf ./sample/locomo-dmf
dmf-benchctl config export locomo-mem0 ./sample/locomo-mem0
```

The two files may both be named `experiment.json` because they live in
different directories.

### Step 2: declare the identical sampling policy

In both `experiment.json` files, make the complete `dataset` object identical:

```json
{
  "dataset": {
    "name": "locomo",
    "registry_id": "locomo-official-v1",
    "sampling": {
      "fraction": 0.05,
      "unit": "conversation",
      "seed": 7,
      "rounding": "ceil"
    }
  }
}
```

The fields have these effects:

- `fraction`: requested portion of the population, greater than zero and at
  most one;
- `unit`: record type being sampled; use `conversation` for complete LoCoMo
  conversations and `record` for LongMemEval;
- `seed`: deterministic random selection; identical seed and source bytes
  produce the same selected records;
- `rounding`: converts a fractional record count using `ceil`, `floor`, or
  `round`. `ceil` avoids an empty sample for small populations.

Do not add `stratify_by` to LoCoMo conversation sampling. Question-level
LoCoMo sampling can instead use `unit: "qa"` and
`stratify_by: "category"`, while LongMemEval record sampling can optionally
use `stratify_by: "question_type"`.

### Step 3: make selection consume the materialized sample

To execute every record selected by sampling, use this in both configs:

```json
{
  "selection": {
    "ordered_item_ids": ["*"],
    "filters": {},
    "seed": 7
  }
}
```

For LoCoMo, retain identical category filters in both configs when categories
are part of the experiment. If explicit LongMemEval IDs are left in
`ordered_item_ids`, every ID must exist in the new sample; changing the sample
seed or fraction can otherwise make validation fail. `"*"` is the simplest way
to run the complete sampled dataset.

### Step 4: prepare both configs together

```text
dmf-benchctl prepare \
  --config ./sample/locomo-dmf/experiment.json \
  --config ./sample/locomo-mem0/experiment.json \
  --state-file .dmf-bench/prepared-locomo-sample.json \
  --batch-id 20260813T120000Z-locomo-sample \
  --allow-downloads \
  --observability
```

Preparation rejects the batch if DMF and Mem0 declare different dataset or
sampling objects for the same benchmark. It then downloads or reuses the
pinned source, samples it deterministically, writes `manifest.json`, validates
the materialized bytes, and locks the dataset identity in the prepared context.
No separate sampling command is required.

Review the receipt before running:

```text
python3 -m json.tool .dmf-bench/prepared-locomo-sample.json
```

Confirm that both experiment entries contain the same
`dataset_lock.sampling`, source revision, and source SHA-256.

### Step 5: run and verify the materialization manifests

```text
dmf-benchctl run --prepared .dmf-bench/prepared-locomo-sample.json
```

After both runs complete, open `http://127.0.0.1:8000`, select each run, and
open:

```text
datasets/materialization-manifest.json
```

The manifest is generated automatically whenever a registry dataset is
materialized and copied into every run. Compare these fields between DMF and
Mem0:

- `dataset.dataset_id` and `dataset.revision`;
- `source.sha256`;
- `materialized.sha256`;
- the complete `sampling` object, especially `unit`, `fraction`, `seed`,
  `rounding`, `population_count`, and `sample_count`;
- benchmark-specific population/sample counts when present.

The two runs are using the same sample when source identity,
`materialized.sha256`, and the complete sampling policy/counts match. Paths and
`generated_at` are operational metadata and do not establish sample equality.

For automation, download the two manifests from:

```text
GET /runs/RUN_ID/artifacts/datasets/materialization-manifest.json
```

If there is no `sampling` object in a manifest, the full materialized dataset
was used.

## Selection after materialization

`selection.ordered_item_ids` determines which materialized records execute and
their order. `"*"` means all records remaining after materialization.

LoCoMo example:

```json
{
  "selection": {
    "ordered_item_ids": ["*"],
    "filters": {"categories": [1, 2, 3, 4, 5]},
    "seed": 7
  }
}
```

LongMemEval may use `"*"` or explicit question IDs. Explicit IDs must exist in
the materialized sample. Sampling happens before selection.

# 9. Monitoring and Recovery

## Status states

Common states include:

- `RUNNING`: prediction work is active;
- `JUDGING`: generated answers are being judged;
- `EVALUATING`: deterministic metrics are being calculated;
- `REPORTING`, `PUBLISHING`, `VERIFYING`: final artifacts are being committed;
- `COMPLETED`: the final marker exists and verifies;
- `PARTIAL`: predict-only execution ended before terminal phases;
- `INTERRUPTING` or `INTERRUPTED`: safe cancellation occurred;
- `FAILED_*`: the suffix identifies the lifecycle phase that failed.

Judging intentionally has no fabricated item-by-item quality metric. Progress
shows lifecycle state; final evaluation values appear only after committed
reports exist.

## Resume behavior

Resume uses persisted `resolved-config.json`; it never asks the operator to
reconstruct the original config. Completed atomic records are reused.
Incomplete records are cleaned and restarted from their safe boundary because
memory-system side effects inside a record are not assumed trustworthy.

For a prepared run, prefer:

```text
dmf-benchctl resume --prepared PREPARED_FILE --run-id RUN_ID
```

This restores the image digest and download policy as well as the run config.

## Logs

Detached execution produces:

- `.dmf-bench/logs/BATCH_ID.worker.log` for prepared-batch orchestration;
- `.dmf-bench/logs/RUN_ID.log` for a new prepared run;
- `.dmf-bench/logs/RUN_ID.resume.log` for prepared resume work;
- operation-specific status and preparation logs.

Use the exact paths printed in the launch receipt. Logs can contain prompts or
dataset-derived text; handle them as experiment data even though provider keys
are not included.

# 10. Results and Artifacts

## Which files contain official results?

`final/COMPLETED.json` proves completion; it is not the quality score.

Read these files for outcomes:

| Artifact | Meaning |
|---|---|
| `evaluations/primary_judge_score.json` | Aggregate LLM-judge result |
| `evaluations/rigorous_report.json` | Deterministic benchmark metrics |
| `evaluations/ablation_report.json` | Retrieval ablation metrics, when requested |
| `reports/usage.json` | Calls and tokens by answerer, judge, and memory internals |
| `reports/timing.json` | Lifecycle and pipeline timing |
| `reports/resources.json` | CPU, memory, cgroup observations, and limits |

For per-item diagnosis, inspect predictions, judgments, and item timing under
`items/` and `evaluations/`.

## Run directory map

```text
runs/RUN_ID/
|-- run-manifest.json
|-- resolved-config.json
|-- run-status.json
|-- datasets/
|   `-- materialization-manifest.json
|-- items/
|-- checkpoints/
|-- attempts/
|-- evaluations/
|   |-- primary_judge_score.json
|   |-- rigorous_report.json
|   `-- ablation_report.json
|-- reports/
|   |-- usage.json
|   |-- timing.json
|   `-- resources.json
|-- metrics/
|-- logs/
`-- final/
    |-- manifest.json
    `-- COMPLETED.json
```

The final manifest lists committed artifacts and hashes. `verify` checks the
files against it.

## Artifact browser and API

The browser is the easiest interface:

```text
http://127.0.0.1:8000
```

It lists committed runs and lets users browse/download artifacts without
constructing URLs. The FastAPI documentation is available at `/docs`.

Automation endpoints include:

```text
GET /runs
GET /runs/RUN_ID
GET /runs/RUN_ID/artifacts
GET /runs/RUN_ID/artifacts/ARTIFACT_PATH
GET /runs/RUN_ID/verify
GET /runs/RUN_ID/derived-evaluations
```

Example:

```text
curl -fsS http://127.0.0.1:8000/runs/RUN_ID/artifacts/reports/usage.json
```

The API is read-only, unauthenticated, and bound to loopback. Do not expose it
directly to an untrusted network.

## Deterministic re-evaluation

For a committed LoCoMo run:

```text
dmf-benchctl evaluate --run-id RUN_ID
```

The source remains unchanged. `derivation.json` links derived report hashes,
evaluator version, source final-manifest hash, and derivation fingerprint.

# 11. Observability and Fair Comparisons

Enable observability in `prepare`, `run`, or `stack up`.

The **Operations** dashboard shows overall/partial progress, run state, tokens,
pipeline time, CPU, memory, and Qdrant activity.

The **Evaluation** dashboard shows final quality metrics from committed
reports. It is intentionally separate and does not display provisional judging
values.

Both dashboards provide selectors for:

- benchmark framework (`locomo` or `longmemeval`);
- memory system (`dmf`, `mem0`, or future adapters).

Select one value for each to avoid mixing series.

Grafana uses a dark theme. Anonymous local users are Editors. Dashboard changes
are stored in the persistent Grafana volume. Provisioned source dashboards are
read-only files, but user-edited dashboard state persists in Grafana's data
volume.

The Artifact API projects the latest persisted run for each benchmark/memory
system into Prometheus, so final values remain visible after a benchmark job
exits.

For timing comparisons:

1. use the same image digest and native platform;
2. use identical dataset materialization manifests and selections;
3. use identical answerer/judge models and parameters;
4. use identical `DMF_BENCH_CPUS` and `DMF_BENCH_THREADS`;
5. avoid simultaneous benchmark jobs;
6. account for first-run model/dataset downloads separately;
7. compare `timing.json` and `resources.json`, not only wall-clock observation.

CPU percentages can exceed 100% because 100% represents one fully used logical
CPU. A 400% limit allows approximately four logical CPUs.

# 12. Publishing an Official Run

Before publication:

```text
dmf-benchctl status --run-id RUN_ID
dmf-benchctl inspect --run-id RUN_ID
dmf-benchctl verify --run-id RUN_ID
```

Create a local bundle first:

```text
dmf-benchctl publish-run \
  --run-id RUN_ID \
  --ref ghcr.io/ORG/dmf-benchmarks-runs:RUN_ID \
  --subject ghcr.io/ORG/dmf-benchmarks@sha256:IMAGE_DIGEST \
  --output-dir bundles \
  --dry-run
```

The archive contains every file under the run directory at packaging time,
including `OFFICIAL-RUN.json` and `SHA256SUMS` added by the packaging step.

After inspecting the archive, authenticate ORAS and remove `--dry-run`:

```text
dmf-benchctl publish-run \
  --run-id RUN_ID \
  --ref ghcr.io/ORG/dmf-benchmarks-runs:RUN_ID \
  --subject ghcr.io/ORG/dmf-benchmarks@sha256:IMAGE_DIGEST
```

Artifact type:

```text
application/vnd.dmf-bench.run.v1+tar
```

The published run artifact is separate from the executable benchmark image.
The optional OCI subject creates an auditable link between them.

# 13. Exit Codes and Automation

Controller commands print structured runtime output where appropriate and
return non-zero on failure. Important codes are:

| Code | Meaning |
|---:|---|
| `0` | Command succeeded; status may still be actively `RUNNING` |
| `1` | Host/orchestrator operation failed |
| `2` | Invalid controller arguments or experiment/config error |
| `3` | Runtime state, infrastructure, inspect, or verification error |
| `4` | Failed benchmark/scientific lifecycle state |
| `5` | Partial or interrupted run state |
| `127` | Required host executable was not found |

Runtime codes are passed through by the controller for runtime operations.
Automation should inspect both the exit code and JSON state rather than
assuming that all non-completed states share one code.

For monitoring agents:

```text
dmf-benchctl status --run-id RUN_ID --json
```

Treat these public fields as the operator contract:

- `run_id`, `state`, `phase`, and `phase_label`;
- `progress.percentage` for overall progress;
- `evaluation_items.completed|total|failed`;
- `input_records.completed|total|failed`;
- optional `current_activity`;
- `ready_for_verification`.

The detached launch receipt under `.dmf-bench/jobs/` is also JSON and contains
the worker PID, run IDs, logs, image, and suggested commands. It is a launch
receipt, not proof that the worker later completed successfully.

# 14. Troubleshooting

## `dmf-benchctl` command is not found

Run `pipx ensurepath`, open a new shell, and verify `pipx list`. If a newer wheel
was installed, use `pipx install --force WHEEL`.

## The wrong image is used

Check the exported shell value, workspace `.env`, global `--image`, and the
prepared context. Exported variables override `.env`; `--image` overrides both.
A prepared context then enforces its locked digest and rejects a conflicting
`--image`.

## `prepare` says the image has no registry digest

Preparation requires a published image such as `ghcr.io/...:TAG`. Build-only
local images do not have a registry digest. Push the image or use a published
RC/tag.

## Image platform does not match Docker engine

Pull a multi-platform/native image. Do not force emulation for official timing
results.

## `OPENAI_API_KEY` is missing

Set it in the shell or `.env` of the selected workspace. Do not add it to JSON,
TOML, YAML, or command arguments.

## Provider requests fail with `APIConnectionError`

Check host internet access, proxy variables, custom `OPENAI_BASE_URL`, Docker
network access, and provider status. The secret presence check cannot prove
that the provider endpoint is reachable.

## A prepared config changed

Do not edit prepared inputs. Export/edit again, run `prepare` again with a new
batch ID, review the new receipt, and launch the new run IDs.

## Run ID already exists

Use `resume` for the same experiment state. Use a new run ID or new prepared
batch ID for a distinct run. The controller does not overwrite existing run
directories.

## Status stays at zero percent

Read `Current activity` and `Partial progress`. Overall progress advances only
after an atomic input record commits. A large LoCoMo conversation can spend a
long time ingesting and answering before the first overall increment.

## The detached run appears silent

This is expected. Use the `status` command printed in the launch receipt and
inspect the reported `.dmf-bench/logs/...worker.log` or run log.

## Read-only filesystem error

The benchmark container is intentionally read-only. Frameworks and libraries
must write under `/bench/cache`, `/bench/runs`, or `/tmp`. Verify that framework
cache paths use the standard config and that the current image includes the
required writable-volume fix.

## Hugging Face download is blocked or a model file is missing

The default stack sets `HF_HUB_OFFLINE=1`. Run a controlled preparation with
`--allow-downloads` to fill the persistent cache, then keep the same cache for
offline execution.

## Qdrant is not reachable

```text
dmf-benchctl stack status
dmf-benchctl stack up
```

The standard stack injects `QDRANT_URL=http://qdrant:6333`; users do not need a
local Qdrant API key.

## The batch stopped before later experiments

Prepared batches stop on the first failing run. Check that run's status/log,
fix only external causes, and rerun the same `run --prepared` command. Existing
state is inspected and resumed automatically.

## `verify` fails

Do not publish the run. Use `inspect` and the error to identify a missing or
changed artifact. Verification failure means the local committed directory no
longer matches its final manifest.

## Grafana is empty

Confirm the stack was started with `--observability`, select the correct
benchmark and memory system, and check `dmf-benchctl stack status`. Evaluation
panels populate from completed reports; they do not show provisional judge
scores.

## OCI publication fails

First run a local `publish-run --dry-run`. For a real push, verify ORAS is
installed, registry authentication is valid, destination permissions exist,
and `verify` succeeds.

# 15. Official Run Checklist

Before execution:

1. Install the controller version matching the intended image release.
2. Use a dedicated workspace and protected `.env`.
3. Export and review experiment/framework configs.
4. Confirm dataset, sampling, selection, answerer, judge, memory-internal
   models, retries, CPU, threads, and expected cost.
5. Run `prepare` with a unique batch ID.
6. Review image digest, platform, revision, config hashes, dataset locks,
   model names, run IDs, and download policy in the prepared context.

During execution:

7. Launch `run --prepared` and save the detached receipt.
8. Monitor every run with `status`; use logs only for diagnosis.
9. Resume using the same prepared context when needed.
10. Run fair DMF/Mem0 timing comparisons sequentially.

After execution:

11. Require `COMPLETED` and successful `verify`.
12. Read rigorous/judge metrics, usage, timing, resources, resolved config, and
    dataset materialization manifest.
13. Confirm compared frameworks used identical dataset manifests and limits.
14. Optionally derive updated deterministic LoCoMo reports with `evaluate`.
15. Build and inspect a local official bundle with `publish-run --dry-run`.
16. Publish the run artifact with an image-digest subject when approved.
17. Stop services with `stack down`; retain or back up volumes according to the
    experiment's retention policy.

# 16. Historical Version

The previous benchmark architecture is available from Git tag `v0.1.0`.
Version 0.2 does not provide compatibility protocols, legacy execution paths,
state migration, or artifact migration. Current runs are Docker-native and use
only the framework-native pipeline.
