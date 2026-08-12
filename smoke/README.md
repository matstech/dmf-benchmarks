# Official restricted smoke resources

This directory contains four native Docker run configurations used for an
official smoke pass:

| Dataset | DMF | Mem0 |
|---|---|---|
| LoCoMo | `config/experiment-locomo-dmf.json` | `config/experiment-locomo-mem0.json` |
| LongMemEval | `config/experiment-longmemeval-dmf.json` | `config/experiment-longmemeval-mem0.json` |

The configs declare deterministic dataset materialization (seed 7). During
`dmf-bench run`, the runner downloads or reuses the registered real dataset,
writes the materialized file under `/bench/cache/datasets/...`, and generates
`manifest.json` next to it.

- LoCoMo samples 5% of the complete conversation population, rounded up. Since
  the official dataset contains 10 conversations, the smoke sample contains
  one complete conversation and all of its QA items. Both ingestion and
  answering therefore operate only on the materialized sample.
- LongMemEval samples 5% of the 500 real records, rounded up: 25 records.
  Each selected record includes the complete memory and question used by both
  ingestion and answering.

They validate a substantial end-to-end lifecycle, but they are not paper-scale
reproduction datasets. The official source revisions and SHA-256 digests are
pinned in `dmf_bench/datasets.py`. Materialization refuses changed bytes, and
the generated `manifest.json` records the verified source, materialized sample,
sample policy, and sample counts.

Prepare all four configurations through the host-side control plane without
making provider calls:

```text
dmf-benchctl --image ghcr.io/ORG/dmf-benchmarks:TAG prepare \
  --suite smoke \
  --allow-downloads
```

Then execute the locked batch sequentially:

```text
dmf-benchctl run --prepared
```

You can still run one configuration directly, for example:

```text
dmf-benchctl run \
  --config smoke/config/experiment-locomo-dmf.json \
  --run-id smoke-locomo-dmf
```

`dmf-benchctl` detects that the configuration is under `smoke/` and applies the
required read-only resource mount automatically. The local Compose override is
an orchestrator implementation detail and is not part of the user command.

The configs use the OpenAI provider. A run requires `OPENAI_API_KEY` in the
operator's environment or repository `.env`; the controller mounts it as an
ephemeral read-only secret file and no key is stored in this directory or in
Compose service metadata. Smoke configs delete their per-unit Qdrant resources
after successful prediction so the four runs do not accumulate stale
collections.
