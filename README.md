# DMF Benchmarks

This repository contains the benchmark evaluation suite for the [dmf](https://github.com/matstech/dmf), a specialized memory system for conversational AI, comparing its performance and resource footprint against [Mem0](https://github.com/mem0ai/mem0).

It implements isolated end-to-end evaluation pipelines for two standard long-term memory benchmarks: [LoCoMo (Long-Context Memory Benchmark)](https://github.com/snap-research/locomo) and [LongMemEval](https://github.com/xiaowu0162/LongMemEval/tree/main).

## Native Approach

The pipelines in this repository default to a native approach. Unlike strict benchmark setups that reconstruct rigid session contexts or historical dialogue transcripts, the native approach directly queries the memory framework's natural storage surface.

Specifically, during question answering, the framework-native context is retrieved and formatted into a minimal context injection prompt for the answerer model (fllowing the official guidelines and examples of each benchmark framework). This provides a realistic assessment of each framework's retrieval quality.

## Implemented Pipeline

The evaluation pipeline runs through four major stages:

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
           |  - Rigorous resource & quality report logs  |
           +---------------------------------------------+
```

1. **Ingestion**: Dialog turns or interaction logs are sequentially loaded into the memory backend. For the _dmf_, ingestion interacts directly with the _TemporalMemory_ storage layer. For _Mem0_, turns are written sequentially using its standard user memory interface.
2. **Native Context Construction**: The pipeline queries the memory backend to fetch the native memory context surface relevant to the current user state. This context is embedded directly in a minimal, natural QA prompt.
3. **Answer Generation**: The prompt is sent to the answerer model (defaulting to gpt-4.1-mini) to produce candidate answers. Prompt tokens, and completion tokens are recorded.
4. **Evaluation**: A stronger LLM judge (defaulting to gpt-5-mini) compares the generated answers against ground-truth solutions. In addition to accuracy, a rigorous evaluation module computes system metrics and generates structured quality and resource reports.

## Makefile Commands

A Makefile is provided to run different stages of the benchmarks. The commands use [Poetry](https://python-poetry.org/) for environment and dependency isolation.

### Project Setup

- `make lock`: Regenerate the poetry.lock file to update dependencies.
- `make install`: Install the project and all framework dependencies in the local virtual environment.

### Running LoCoMo

- `make locomo`: Run the entire end-to-end native LoCoMo benchmark including ingestion, prediction, and judge evaluation.
- `make locomo-judge`: Run only the LLM judge scoring on already saved predictions, bypassing ingestion and answer generation.
- `make locomo-rigorous`: Run only the rigorous secondary metric reports on existing outputs to compile final comparison summaries.

### Running LongMemEval

- `make longmemeval`: Run the entire end-to-end native LongMemEval benchmark.
- `make longmemeval-judge`: Run only the LLM judge evaluation on existing LongMemEval predictions.
- `make longmemeval-rigorous`: Run only the rigorous evaluation and reporting scripts on existing LongMemEval outputs.

## Configuration Variables

You can customize the execution by overriding variables on the command line when running any make target.

### General Variables

- `POETRY`: Command to invoke poetry. Defaults to poetry.
- `BENCHMARKS_DIR`: Target benchmarks root path. Defaults to .
- `ANSWERER_PROVIDER`: LLM API provider for generating candidate answers. Defaults to openai.
- `ANSWERER_MODEL`: LLM model name for generating answers. Defaults to gpt-4.1-mini.
- `JUDGE_PROVIDER`: LLM API provider for scoring. Defaults to openai.
- `JUDGE_MODEL`: LLM model name for scoring candidate answers. Defaults to gpt-5-mini.

### LoCoMo Variables

- `LOCOMO_PROJECT`: Isolated project directory name for saved results. Defaults to locomo_dmf.
- `LOCOMO_FRAMEWORK`: Memory framework under evaluation (dmf or mem0). Defaults to dmf.
- `LOCOMO_CONFIG`: Settings configuration path. Defaults to config/locomo_dmf_settings.toml.
- `LOCOMO_CONVERSATION_IDS`: Comma-separated list of conversation indices to process. Defaults to running all.
- `LOCOMO_CATEGORIES`: Question categories to evaluate. Defaults to 1,2,3,4.

### LongMemEval Variables

- `LONGMEMEVAL_PROJECT`: Isolated project directory name for saved results. Defaults to longmemeval_dmf.
- `LONGMEMEVAL_FRAMEWORK`: Memory framework under evaluation (dmf or mem0). Defaults to dmf.
- `LONGMEMEVAL_CONFIG`: Settings configuration path. Defaults to config/longmemeval_dmf_settings.toml.
- `LONGMEMEVAL_PER_TYPE`: Maximum questions to run per question type. Defaults to 10.

### Execution Examples

Run LoCoMo with Mem0:

```bash
make locomo LOCOMO_FRAMEWORK=mem0 LOCOMO_CONFIG=config/locomo_mem0_settings.toml
```

Run LongMemEval with DMF using custom models:

```bash
make longmemeval LONGMEMEVAL_FRAMEWORK=dmf ANSWERER_MODEL=gpt-4o-mini JUDGE_MODEL=gpt-4o
```

Only rerun the judge on existing saved outputs:

```bash
make locomo-judge LOCOMO_PROJECT=locomo_dmf
```
