"""Generate immutable BEAM profile configs from a local materialized release."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from dmf_bench.benchmarks.beam.spec import PROFILES
from dmf_bench.contracts import sha256_file


ROOT = Path(__file__).resolve().parents[1]
FRAMEWORKS = ("model-only", "full-context", "vector-rag", "dmf", "mem0", "lightmem", "amem")


def build_config(*, profile: str, framework: str, smoke: bool,
                 runtime_root: Path, dataset_manifest: Path) -> dict:
    template = (
        ROOT / "smoke" / "baselines" / f"experiment-locomo-{framework}.json"
        if framework in {"model-only", "full-context", "vector-rag"} else
        ROOT / "smoke" / framework / f"experiment-locomo-{framework}.json"
        if framework in {"lightmem", "amem"} else
        ROOT / "smoke" / "config" / f"experiment-locomo-{framework}.json"
    )
    config = json.loads(template.read_text(encoding="utf-8"))
    manifest = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    scale = PROFILES[profile]
    records = [record for record in manifest["records"] if record["scale"] == scale]
    if smoke:
        records = records[:1]
    root = runtime_root.resolve()
    runs_dir = root / "runs" if root == Path("/bench") else root / ".dmf-bench" / "beam-runs"
    cache_dir = root / "cache" if root == Path("/bench") else root / ".dmf-bench" / "cache"
    config["experiment_id"] = f"{profile}-{framework}{'-smoke' if smoke else ''}"
    config["scientific_profile"] = profile
    config["benchmark"] = "beam"
    config["runtime"] = {
        "root": str(root), "runs_dir": str(runs_dir), "cache_dir": str(cache_dir),
        "metrics_port": 9464, "log_level": "INFO",
    }
    framework_relative = Path(config["framework_config"]["path"]).relative_to("/bench")
    config["framework_config"]["path"] = str(root / framework_relative)
    config["framework_config"]["sha256"] = sha256_file(ROOT / framework_relative)
    config["dataset"] = {
        "name": "beam", "path": str(root / dataset_manifest.resolve().relative_to(ROOT)),
        "sha256": sha256_file(dataset_manifest), "source": "Mohammadta/BEAM + Mohammadta/BEAM-10M",
        "revision": "+".join(manifest["dataset_revisions"].values()),
        "expected_schema": "beam-conversations-v1",
    }
    config["selection"] = {
        "ordered_item_ids": [record["unit_id"] for record in records],
        "filters": {"max_queries_per_unit": 1} if smoke else {}, "seed": 7,
    }
    model = {
        "provider": "ollama", "requested_model": "qwen2.5:0.5b",
        "parameters": {"temperature": 0, "max_tokens": 256},
        "runtime": {"timeout_seconds": 120, "rpm": 120, "max_retries": 0},
    }
    config["models"] = {"answerer": model, "judges": [{"id": "primary", **copy.deepcopy(model)}]}
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
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--runtime-root", type=Path, default=Path("/bench"))
    parser.add_argument("--dataset-manifest", type=Path, default=ROOT / "datasets/beam/manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = build_config(profile=args.profile, framework=args.framework,
                          smoke=args.smoke, runtime_root=args.runtime_root,
                          dataset_manifest=args.dataset_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "units": len(config["selection"]["ordered_item_ids"]),
                      "dataset_sha256": config["dataset"]["sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
