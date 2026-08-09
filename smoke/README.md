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

- LoCoMo keeps all 10 real conversations and samples 5% of the QA population
  independently in each category, rounded up: 15/282, 17/321, 5/96, 43/841
  and 23/446, for 103 QA items total.
- LongMemEval samples 5% of the 500 real records, rounded up: 25 records.

They validate a substantial end-to-end lifecycle, but they are not paper-scale
reproduction datasets. The source URLs are the official sources registered in
`dmf_bench/datasets.py`. The built-in official entries are marked unpinned, so
the smoke configs explicitly set `allow_unpinned: true`; the generated
`manifest.json` records the observed source SHA-256 and, when sampling is
configured, the sample policy and sample counts.

Run any configuration through the host-side control plane, for example:

```text
dmf-benchctl run \
  --config smoke/config/experiment-locomo-dmf.json \
  --run-id smoke-locomo-dmf
```

`dmf-benchctl` detects that the configuration is under `smoke/` and applies the
required read-only resource mount automatically. The local Compose override is
an orchestrator implementation detail and is not part of the user command.

The configs use the OpenAI provider. A run requires `OPENAI_API_KEY` in the
operator's environment; no key is stored in this directory.
