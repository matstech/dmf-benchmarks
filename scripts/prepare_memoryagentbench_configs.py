"""Generate pinned MemoryAgentBench profile configs from local materialization."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from dmf_bench.contracts import sha256_file


ROOT = Path(__file__).resolve().parents[1]
PROFILES = (
    "memoryagentbench-ar-pilot-v1",
    "memoryagentbench-core-v1",
    "memoryagentbench-full-v1",
)
FRAMEWORKS = ("model-only", "full-context", "vector-rag", "dmf", "mem0", "lightmem", "amem")


def selected_records(records: list[dict], profile: str, *, smoke: bool) -> list[dict]:
    supported = [record for record in records if record["source"] != "infbench_sum_eng_shots2"]
    if profile == PROFILES[0]:
        return [next(record for record in supported if record["source"] == "eventqa_65536")]
    core_sources = (
        "eventqa_65536", "icl_banking77_5900shot_balance",
        "detective_qa", "factconsolidation_sh_6k",
    )
    if profile == PROFILES[1]:
        return [next(record for record in supported if record["source"] == source) for source in core_sources]
    if profile == PROFILES[2] and smoke:
        sources = core_sources + ("ruler_qa1_197K", "longmemeval_s*", "recsys_redial_full")
        return [next(record for record in supported if record["source"] == source) for source in sources]
    if profile == PROFILES[2]:
        return supported
    raise ValueError(f"Unknown profile: {profile}")


def build_config(*, profile: str, framework: str, smoke: bool, runtime_root: Path,
                 dataset_manifest: Path) -> dict:
    template = (
        ROOT / "smoke" / "baselines" / f"experiment-locomo-{framework}.json"
        if framework in {"model-only", "full-context", "vector-rag"} else
        ROOT / "smoke" / framework / f"experiment-locomo-{framework}.json"
        if framework in {"lightmem", "amem"} else
        ROOT / "smoke" / "config" / f"experiment-locomo-{framework}.json"
    )
    config = json.loads(template.read_text(encoding="utf-8"))
    manifest = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    records = selected_records(manifest["records"], profile, smoke=smoke)
    root = runtime_root.resolve()
    config["experiment_id"] = f"{profile}-{framework}{'-smoke' if smoke else ''}"
    config["scientific_profile"] = profile
    config["benchmark"] = "memoryagentbench"
    runs_dir = root / "runs" if root == Path("/bench") else root / ".dmf-bench" / "memoryagentbench-runs"
    cache_dir = root / "cache" if root == Path("/bench") else root / ".dmf-bench" / "cache"
    config["runtime"] = {
        "root": str(root),
        "runs_dir": str(runs_dir),
        "cache_dir": str(cache_dir),
        "metrics_port": 9464,
        "log_level": "INFO",
    }
    framework_relative = Path(config["framework_config"]["path"]).relative_to("/bench")
    config["framework_config"]["path"] = str(root / framework_relative)
    config["framework_config"]["sha256"] = sha256_file(ROOT / framework_relative)
    materialized_path = root / dataset_manifest.resolve().relative_to(ROOT)
    config["dataset"] = {
        "name": "memoryagentbench",
        "path": str(materialized_path),
        "sha256": sha256_file(dataset_manifest),
        "source": "ai-hyz/MemoryAgentBench",
        "revision": manifest["dataset_revision"],
        "expected_schema": "memoryagentbench-units-v1",
    }
    config["selection"] = {
        "ordered_item_ids": [record["unit_id"] for record in records],
        "filters": {"max_queries_per_unit": 1 if smoke else 5}
        if smoke or profile != PROFILES[2] else {},
        "seed": 7,
    }
    common_model = {
        "provider": "ollama",
        "requested_model": "qwen2.5:0.5b",
        "parameters": {"temperature": 0, "max_tokens": 256},
        "runtime": {"timeout_seconds": 120, "rpm": 120, "max_retries": 0},
    }
    config["models"] = {
        "answerer": common_model,
        "judges": [{"id": "primary", **copy.deepcopy(common_model)}],
    }
    config["models"]["judges"][0]["runtime"]["response_max_retries"] = 1
    config["evaluation"] = {
        "primary_judge_id": "primary",
        "required": ["primary_judge_score", "rigorous_report", "analysis_rows"],
        "optional": ["retrieval_report", "ablation_report", "judge_agreement"],
    }
    config["retrieval"]["max_results"] = 20
    config["context_budget"]["max_tokens"] = 1024
    config["storage"]["retention"] = "delete-on-success"
    config["artifact_store"] = {"type": "local", "uri": str(runs_dir)}
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--framework", choices=FRAMEWORKS, required=True)
    parser.add_argument("--smoke", action="store_true", help="One query per selected context")
    parser.add_argument("--runtime-root", type=Path, default=Path("/bench"))
    parser.add_argument("--dataset-manifest", type=Path, default=ROOT / "datasets/memoryagentbench/manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    config = build_config(
        profile=arguments.profile, framework=arguments.framework,
        smoke=arguments.smoke, runtime_root=arguments.runtime_root,
        dataset_manifest=arguments.dataset_manifest,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(arguments.output), "units": len(config["selection"]["ordered_item_ids"]),
                      "dataset_sha256": config["dataset"]["sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
