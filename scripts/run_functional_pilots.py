"""Run the pinned local micro-pilot matrix sequentially with isolated logs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / ".dmf-bench" / "pilot-runs"
LOGS_DIR = ROOT / ".dmf-bench" / "pilot-logs"
FRAMEWORKS = (
    "model-only", "full-context", "vector-rag", "mem0", "dmf", "lightmem", "amem",
    "dmf-full", "dmf-no-salience-scoring", "dmf-no-structured-cards",
    "dmf-no-temporal-decay", "dmf-semantic-only",
)
BENCHMARKS = ("locomo", "longmemeval")
IMAGES = {
    "base": "dmf-benchmarks:pilot-base",
    "lightmem": "dmf-benchmarks:lightmem-local",
    "amem": "dmf-benchmarks:amem-local",
}


def image_for(framework: str) -> str:
    return IMAGES[framework] if framework in {"lightmem", "amem"} else IMAGES["base"]


def run_id_for(benchmark: str, framework: str) -> str:
    return f"pilot-v8-{benchmark}-{framework}"


def verify_ollama_model(frameworks: tuple[str, ...] | list[str]) -> None:
    manifest = json.loads((ROOT / "datasets" / "pilots" / "manifest.json").read_text(encoding="utf-8"))
    with urlopen("http://127.0.0.1:11434/api/tags", timeout=10) as response:
        models = json.load(response)["models"]
    required = [(manifest["answerer_and_judge_model"], manifest["model_digest"])]
    if "mem0" in frameworks:
        required.append((manifest["mem0_internal_model"], manifest["mem0_internal_model_digest"]))
    for expected_name, expected_digest in required:
        if not any(row["name"] == expected_name and row["digest"] == expected_digest for row in models):
            raise RuntimeError(f"Pinned local Ollama model {expected_name} has a different digest or is missing.")


def docker_arguments(image: str, models_dir: Path | None, framework: str) -> list[str]:
    harness_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()
    image_digest = subprocess.check_output(
        ["docker", "image", "inspect", image, "--format", "{{.Id}}"], text=True,
    ).strip()
    arguments = [
        "docker", "run", "--rm", "--read-only", "--tmpfs", "/tmp", "--cpus", "4.0",
        "--mount", f"type=bind,source={ROOT / 'config'},target=/bench/config,readonly",
        "--mount", f"type=bind,source={ROOT / 'datasets'},target=/bench/datasets,readonly",
        "--mount", f"type=bind,source={RUNS_DIR},target=/bench/runs",
        "--mount", "type=volume,source=dmf-benchmarks-v3-cache,target=/bench/cache",
    ]
    if image != IMAGES["base"]:
        if models_dir is None or not models_dir.is_dir():
            raise ValueError("PILOT_MODELS_DIR must point to the provisioned LightMem/A-MEM models")
        arguments += ["--mount", f"type=bind,source={models_dir},target=/bench/models,readonly"]
    variables = {
        "OLLAMA_BASE_URL": "http://host.docker.internal:11434/v1",
        "QDRANT_URL": "http://host.docker.internal:6333",
        "MEM0_TELEMETRY": "false",
        "MEM0_DIR": "/bench/cache/mem0",
        "ANONYMIZED_TELEMETRY": "false",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "FASTEMBED_CACHE_PATH": "/bench/cache/models/embeddings",
        "HF_HOME": "/bench/cache/huggingface",
        "XDG_CACHE_HOME": "/bench/cache/xdg",
        "DMF_BENCH_HARNESS_COMMIT": harness_commit,
        "DMF_BENCH_IMAGE_DIGEST": image_digest,
    }
    for name, value in variables.items():
        arguments += ["-e", f"{name}={value}"]
    if framework == "mem0":
        arguments += ["-e", "OPENAI_API_KEY=ollama"]
    return arguments + [image]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=BENCHMARKS, action="append")
    parser.add_argument("--framework", choices=FRAMEWORKS, action="append")
    args = parser.parse_args()
    benchmarks = args.benchmark or BENCHMARKS
    frameworks = args.framework or FRAMEWORKS
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    verify_ollama_model(frameworks)
    models_dir_raw = os.environ.get("PILOT_MODELS_DIR")
    models_dir = Path(models_dir_raw).expanduser().resolve() if models_dir_raw else None
    outcomes: list[dict] = []
    for benchmark in benchmarks:
        for framework in frameworks:
            run_id = run_id_for(benchmark, framework)
            run_dir = RUNS_DIR / run_id
            image = image_for(framework)
            config = ROOT / "config" / "pilots" / f"{benchmark}-{framework}.json"
            if not config.is_file():
                raise FileNotFoundError(f"Prepare the pilot configurations first: {config}")
            if (run_dir / "final" / "COMPLETED.json").is_file():
                outcomes.append({"run_id": run_id, "state": "ALREADY_COMPLETED", "image": image})
                print(f"{run_id}: already completed", flush=True)
                continue
            command = docker_arguments(image, models_dir, framework)
            if run_dir.exists():
                command += ["resume", "--run-id", run_id]
            else:
                command += ["run", "--config", f"/bench/config/pilots/{config.name}", "--run-id", run_id]
            log = LOGS_DIR / f"{run_id}.log"
            print(f"{run_id}: starting", flush=True)
            started = time.monotonic()
            with log.open("w", encoding="utf-8") as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
            elapsed = round(time.monotonic() - started, 2)
            state = "COMPLETED" if result.returncode == 0 and (run_dir / "final" / "COMPLETED.json").is_file() else "FAILED"
            outcomes.append({
                "run_id": run_id,
                "state": state,
                "exit_code": result.returncode,
                "elapsed_seconds": elapsed,
                "image": image,
                "log": str(log.relative_to(ROOT)),
            })
            print(f"{run_id}: {state} ({elapsed}s); {log.relative_to(ROOT)}", flush=True)
            (LOGS_DIR / "matrix-status.json").write_text(json.dumps(outcomes, indent=2) + "\n", encoding="utf-8")
    return 0 if all(item["state"] != "FAILED" for item in outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
