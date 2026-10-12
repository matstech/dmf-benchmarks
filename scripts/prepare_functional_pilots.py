"""Create deterministic, evidence-preserving local benchmark micro-pilots."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dmf_bench.ablations import scientific_controls_sha256

DATA_DIR = ROOT / "datasets" / "pilots"
CONFIG_DIR = ROOT / "config" / "pilots"
SOURCE_FILES = {
    "locomo": (
        ROOT / "smoke/datasets/locomo/locomo-official-smoke.json",
        "dd809a8f06d95bea7100805f6e37a1408f019947726a8feca80cb409bb2ed9d5",
        "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376",
    ),
    "longmemeval": (
        ROOT / "smoke/datasets/longmemeval/longmemeval-official-smoke.json",
        "f86481384b3d35ba8d6838056b80ecc9e56289cf7d42c60b84d0abd999c4264f",
        "98d7416c24c778c2fee6e6f3006e7a073259d48f",
    ),
}
LOCOMO_SAMPLE_ID = "conv-41"
LOCOMO_QUESTION_INDEXES = (0, 1, 3, 5, 10)
LONGMEMEVAL_QUESTION_IDS = (
    "0e5e2d1a",
    "0edc2aef",
    "10d9b85a",
    "8c18457d",
    "945e3d21",
)
MODEL = "qwen2.5:0.5b"
MODEL_DIGEST = "a8b0c51577010a279d933d14c2a8ab4b268079d44c5c8830c0a93900f1827c67"
MEM0_INTERNAL_MODEL = "qwen2.5:1.5b"
MEM0_INTERNAL_MODEL_DIGEST = "65ec06548149b04c096a120e4a6da9d4017ea809c91734ea5631e89f96ddc57b"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return sha256(path)


def read_source(benchmark: str) -> list[dict]:
    path, expected_hash, _revision = SOURCE_FILES[benchmark]
    if sha256(path) != expected_hash:
        raise ValueError(f"Source hash changed: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON array: {path}")
    return value


def build_locomo() -> tuple[list[dict], dict]:
    source = read_source("locomo")
    original = next(item for item in source if item.get("sample_id") == LOCOMO_SAMPLE_ID)
    questions = [original["qa"][index] for index in LOCOMO_QUESTION_INDEXES]
    categories = {int(item["category"]) for item in questions}
    if categories != {1, 2, 3, 4, 5}:
        raise ValueError("LoCoMo pilot no longer covers all five categories")
    session_numbers = sorted({
        int(match.group(1))
        for item in questions
        for evidence in item["evidence"]
        if (match := re.fullmatch(r"D(\d+):\d+", evidence))
    })
    if len(session_numbers) != 4:
        raise ValueError("LoCoMo evidence session selection changed")
    conversation = original["conversation"]
    selected_conversation = {
        "speaker_a": conversation["speaker_a"],
        "speaker_b": conversation["speaker_b"],
    }
    for number in session_numbers:
        for suffix in ("_date_time", ""):
            key = f"session_{number}{suffix}"
            selected_conversation[key] = conversation[key]
    item = {
        "sample_id": original["sample_id"],
        "conversation": selected_conversation,
        "qa": questions,
    }
    detail = {
        "source_sample_id": LOCOMO_SAMPLE_ID,
        "source_qa_indexes": list(LOCOMO_QUESTION_INDEXES),
        "pilot_query_ids": [f"conv0_q{index}" for index in range(len(questions))],
        "categories": sorted(categories),
        "evidence_sessions": [f"session_{number}" for number in session_numbers],
        "memory_events": sum(len(selected_conversation[f"session_{number}"]) for number in session_numbers),
    }
    return [item], detail


def build_longmemeval() -> tuple[list[dict], dict]:
    source = {str(item["question_id"]): item for item in read_source("longmemeval")}
    selected = []
    detail = []
    types = set()
    for question_id in LONGMEMEVAL_QUESTION_IDS:
        original = source[question_id]
        types.add(str(original["question_type"]))
        answer_ids = set(original["answer_session_ids"])
        distractor = next(identifier for identifier in original["haystack_session_ids"] if identifier not in answer_ids)
        keep = answer_ids | {distractor}
        indexes = [
            index for index, identifier in enumerate(original["haystack_session_ids"])
            if identifier in keep
        ]
        item = copy.deepcopy(original)
        for field in ("haystack_session_ids", "haystack_dates", "haystack_sessions"):
            item[field] = [original[field][index] for index in indexes]
        if not answer_ids.issubset(item["haystack_session_ids"]):
            raise ValueError(f"Missing answer evidence for {question_id}")
        selected.append(item)
        detail.append({
            "question_id": question_id,
            "question_type": item["question_type"],
            "answer_session_ids": sorted(answer_ids),
            "retained_session_ids": item["haystack_session_ids"],
            "memory_events": sum(len(session) for session in item["haystack_sessions"]),
        })
    if len(types) != 5:
        raise ValueError("LongMemEval pilot no longer covers five question types")
    return selected, {"records": detail, "question_types": sorted(types)}


def template_path(benchmark: str, framework: str) -> Path:
    if framework in {"dmf", "mem0"}:
        return ROOT / "smoke/config" / f"experiment-{benchmark}-{framework}.json"
    if framework in {"lightmem", "amem"}:
        return ROOT / "smoke" / framework / f"experiment-{benchmark}-{framework}.json"
    if framework.startswith("dmf-"):
        return ROOT / "smoke/ablations" / f"experiment-{benchmark}-{framework}.json"
    return ROOT / "smoke/baselines" / f"experiment-{benchmark}-{framework}.json"


def write_mem0_profile(benchmark: str) -> tuple[Path, str]:
    source = ROOT / "config" / f"{benchmark}_mem0_qdrant_settings.yaml"
    original = source.read_text(encoding="utf-8")
    replacement = (
        f'model: "{MEM0_INTERNAL_MODEL}"\n'
        '    openai_base_url: "http://host.docker.internal:11434/v1"'
    )
    if (original.count('model: "gpt-4o-mini"') != 1
            or original.count('temperature: 0') != 1
            or original.count('version: "v1.1"') != 1):
        raise ValueError(f"Expected pinned Mem0 model settings were not found in {source}")
    updated = original.replace('model: "gpt-4o-mini"', replacement)
    updated = updated.replace('temperature: 0', 'temperature: 0\n    max_tokens: 8192')
    updated = updated.replace(
        'version: "v1.1"',
        'custom_instructions: >-\n'
        '  Extract at most six distinct memories from the new messages.\n'
        '  Keep each memory text to one sentence of at most 35 words.\n'
        '  Return only a complete JSON object with a memory array; use sequential IDs.\n'
        '  Do not repeat existing or recently extracted memories.\n\n'
        'version: "v1.1"',
    )
    path = CONFIG_DIR / f"mem0-{benchmark}-ollama.yaml"
    path.write_text(updated, encoding="utf-8")
    return path, sha256(path)


def build_config(benchmark: str, framework: str, dataset_hash: str, ids: list[str], mem0_profile: tuple[Path, str] | None) -> dict:
    config = json.loads(template_path(benchmark, framework).read_text(encoding="utf-8"))
    revision = "v8"
    config["experiment_id"] = f"functional-pilot-{benchmark}-{framework}-{revision}"
    config["dataset"] = {
        "name": benchmark,
        "path": f"/bench/datasets/pilots/{benchmark}.json",
        "sha256": dataset_hash,
        "source": "pinned-official-smoke-derived-functional-pilot",
        "revision": SOURCE_FILES[benchmark][2],
        "expected_schema": "locomo10-json-array-v1" if benchmark == "locomo" else "longmemeval-cleaned-json-array-v1",
    }
    config["selection"] = {
        "ordered_item_ids": ids,
        "filters": {"categories": [1, 2, 3, 4, 5]} if benchmark == "locomo" else {},
        "seed": 7,
    }
    if mem0_profile is not None and framework == "mem0":
        path, digest = mem0_profile
        config["framework_config"] = {
            "path": f"/bench/config/pilots/{path.name}",
            "sha256": digest,
            "format": "yaml",
            "profile": "mem0-ollama-functional-v4",
        }
    if framework == "amem":
        profile = ROOT / "config/amem/amem-local-v3.json"
        config["framework_config"] = {
            "path": "/bench/config/amem/amem-local-v3.json",
            "sha256": sha256(profile),
            "format": "json",
            "profile": "amem-local-v3",
        }
    common_model = {
        "provider": "ollama",
        "requested_model": MODEL,
        "parameters": {"temperature": 0, "max_tokens": 256},
        "runtime": {"timeout_seconds": 120, "rpm": 120, "max_retries": 0},
    }
    config["models"] = {
        "answerer": common_model,
        "judges": [{"id": "primary", **copy.deepcopy(common_model)}],
    }
    config["models"]["judges"][0]["runtime"]["response_max_retries"] = 1
    config["retrieval"]["max_results"] = 20
    config["context_budget"]["max_tokens"] = 1024
    config["storage"]["retention"] = "delete-on-success"
    if "ablation" in config:
        config["ablation"]["scientific_controls_sha256"] = scientific_controls_sha256(config)
    return config


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    datasets = {}
    ids = {
        "locomo": ["conversation-0001"],
        "longmemeval": list(LONGMEMEVAL_QUESTION_IDS),
    }
    for benchmark, build in (("locomo", build_locomo), ("longmemeval", build_longmemeval)):
        subset, detail = build()
        datasets[benchmark] = {
            "path": f"datasets/pilots/{benchmark}.json",
            "sha256": write_json(DATA_DIR / f"{benchmark}.json", subset),
            "source_path": str(SOURCE_FILES[benchmark][0].relative_to(ROOT)),
            "source_sha256": SOURCE_FILES[benchmark][1],
            "source_revision": SOURCE_FILES[benchmark][2],
            "selection": detail,
        }
    frameworks = (
        "model-only", "full-context", "vector-rag", "mem0", "dmf", "lightmem", "amem",
        "dmf-full", "dmf-no-salience-scoring", "dmf-no-structured-cards",
        "dmf-no-temporal-decay", "dmf-semantic-only",
    )
    configs = {}
    for benchmark in datasets:
        mem0_profile = write_mem0_profile(benchmark)
        for framework in frameworks:
            config = build_config(benchmark, framework, datasets[benchmark]["sha256"], ids[benchmark], mem0_profile)
            filename = f"{benchmark}-{framework}.json"
            configs[f"{benchmark}/{framework}"] = {
                "path": f"config/pilots/{filename}",
                "sha256": write_json(CONFIG_DIR / filename, config),
            }
    manifest = {
        "schema_version": 1,
        "purpose": "local functional verification only; reduced evidence-preserving haystacks",
        "seed": 7,
        "answerer_and_judge_model": MODEL,
        "model_digest": MODEL_DIGEST,
        "mem0_internal_model": MEM0_INTERNAL_MODEL,
        "mem0_internal_model_digest": MEM0_INTERNAL_MODEL_DIGEST,
        "answerer_prompt_and_parser": "repository commit and benchmark adapters",
        "datasets": datasets,
        "configurations": configs,
    }
    write_json(DATA_DIR / "manifest.json", manifest)
    print(json.dumps({"dataset_hashes": {key: value["sha256"] for key, value in datasets.items()}, "configs": len(configs)}, sort_keys=True))


if __name__ == "__main__":
    main()
