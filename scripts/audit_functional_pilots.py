"""Check local pilot artifacts, item coverage, provenance, and cleanup."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

from run_functional_pilots import BENCHMARKS, FRAMEWORKS, ROOT, RUNS_DIR, run_id_for


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def expected_sources(benchmark: str) -> dict[str, set[str]]:
    rows = read_json(ROOT / "datasets" / "pilots" / f"{benchmark}.json")
    if benchmark == "locomo":
        conversation = rows[0]["conversation"]
        return {"conversation-0001": {
            turn["dia_id"] for name, turns in conversation.items()
            if name.startswith("session_") and not name.endswith("_date_time")
            for turn in turns
        }}
    return {
        row["question_id"]: {
            f"{row['question_id']}:{session_id}:{index}"
            for session_id, turns in zip(row["haystack_session_ids"], row["haystack_sessions"])
            for index, turn in enumerate(turns) if str(turn.get("content", "")).strip()
        }
        for row in rows
    }


def qdrant_collections() -> set[str]:
    with urlopen("http://127.0.0.1:6333/collections", timeout=10) as response:
        payload = json.load(response)
    return {row["name"] for row in payload["result"]["collections"]}


def audit_inputs() -> dict:
    manifest = read_json(ROOT / "datasets" / "pilots" / "manifest.json")
    checked = 0
    for group in ("datasets", "configurations"):
        for entry in manifest[group].values():
            path = ROOT / entry["path"]
            assert path.is_file() and digest(path) == entry["sha256"], path
            checked += 1
            if group == "datasets":
                source = ROOT / entry["source_path"]
                assert source.is_file() and digest(source) == entry["source_sha256"], source
                checked += 1
            else:
                config = read_json(path)
                profile = config["framework_config"]
                profile_path = ROOT / profile["path"].removeprefix("/bench/")
                assert profile_path.is_file() and digest(profile_path) == profile["sha256"], profile_path
                checked += 1
    return {"input_manifest_sha256": digest(ROOT / "datasets" / "pilots" / "manifest.json"),
            "checked_inputs": checked}


def audit_run(benchmark: str, framework: str, collections: set[str]) -> dict:
    run_id = run_id_for(benchmark, framework)
    root = RUNS_DIR / run_id
    marker = root / "final" / "COMPLETED.json"
    if not marker.is_file():
        return {"run_id": run_id, "state": "MISSING_OR_FAILED"}
    completed = read_json(marker)
    pilot_inputs = read_json(ROOT / "datasets" / "pilots" / "manifest.json")
    pilot_config = read_json(ROOT / pilot_inputs["configurations"][f"{benchmark}/{framework}"]["path"])
    run_config = read_json(root / "final" / "resolved-config.json")
    run_manifest = read_json(root / "final" / "run-manifest.json")
    fingerprint_inputs = run_manifest["fingerprint_inputs"]
    assert run_config["experiment_id"] == pilot_config["experiment_id"], run_id
    assert fingerprint_inputs["dataset"]["sha256"] == pilot_inputs["datasets"][benchmark]["sha256"], run_id
    assert fingerprint_inputs["framework_config"]["sha256"] == pilot_config["framework_config"]["sha256"], run_id
    assert len(fingerprint_inputs["harness_commit"]) == 40, run_id
    assert fingerprint_inputs["image_digest"].startswith("sha256:"), run_id
    manifest_path = root / "final" / "manifest.json"
    manifest = read_json(manifest_path)
    assert completed["run_id"] == manifest["run_id"] == run_id
    assert completed["state"] == "COMPLETED"
    assert completed["final_manifest_sha256"] == digest(manifest_path)
    for artifact in manifest["artifacts"]:
        path = root / "final" / artifact["path"]
        assert path.is_file() and path.stat().st_size == artifact["bytes"], (run_id, artifact["path"])
        assert digest(path) == artifact["sha256"], (run_id, artifact["path"])
    summary = read_json(root / "reports" / "summary.json")
    expected = 5
    assert summary["counts"] == {"expected": expected, "evaluated": expected, "failed": 0, "excluded": 0}, run_id
    evaluations = read_json(root / "evaluations" / "evaluations.json")
    assert len(evaluations) == expected, run_id
    ids = {row["question_id"] for row in evaluations}
    assert ids == set(fingerprint_inputs["expected_question_ids"]), run_id
    sources = expected_sources(benchmark)
    questions = sorted(root.glob("items/*/questions/*.json"))
    predictions = [path for path in questions if not path.name.endswith((".context.json", ".retrieval.json"))]
    assert len(predictions) == expected, run_id
    provenance_count = 0
    for path in root.glob("items/*/questions/*.retrieval.json"):
        valid = sources[path.parent.parent.name]
        for item in read_json(path)["items"]:
            for source in item["source_event_ids"]:
                assert source in valid, (run_id, path.name, source)
                provenance_count += 1
    cleanup_files = sorted(root.glob("attempts/*/cleanup/*.json"))
    assert len(cleanup_files) >= len(sources), run_id
    for path in cleanup_files:
        evidence = read_json(path)
        assert evidence["verified"] is True, path
    for path in root.glob("items/*/resources.json"):
        for resource in read_json(path)["resources"]:
            locator = resource["locator"]
            if resource["kind"] == "qdrant-collection":
                assert locator not in collections, (run_id, locator)
            elif resource["kind"] == "embedded-path":
                assert not (root / locator).exists(), (run_id, locator)
            elif resource["kind"] == "local-file":
                # The container's /bench/cache path is checked through its cleanup receipt.
                assert cleanup_files
    usage = read_json(root / "reports" / "usage.json")
    timing = read_json(root / "reports" / "timing.json")
    resources = read_json(root / "reports" / "resources.json")
    assert usage["item_count"] == expected and timing["attempt"]["execution_seconds"] >= 0
    assert resources["container"]["memory_peak_bytes"] > 0
    ingestion_totals = None
    if framework in {"mem0", "lightmem", "amem"}:
        fields = ("calls", "prompt_tokens", "completion_tokens", "total_tokens")
        ingestion_totals = {field: 0 for field in fields}
        for path in root.glob("items/*/prepared.json"):
            raw = read_json(path)["ingestion_usage"]
            if framework == "mem0":
                source = raw
            elif framework == "lightmem":
                source = {field: sum(int(part[field]) for part in raw["lightmem"]["llm"].values())
                          for field in fields}
            else:
                source = raw["amem"]
                source = {**source, "total_tokens": source["prompt_tokens"] + source["completion_tokens"]}
            for field in fields:
                ingestion_totals[field] += int(source[field])
        reported = usage["components"]["memory_internal"]
        for field in fields:
            assert reported[field] >= ingestion_totals[field], (run_id, field)
        assert reported["frameworks"] == [framework], run_id
    events = [json.loads(line) for line in (root / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    error_events = [event for event in events if event["level"] == "ERROR"]
    parser_errors = [event for event in error_events if "pars" in str(event).lower()]
    upstream_log = ROOT / ".dmf-bench" / "pilot-logs" / f"{run_id}.log"
    upstream_parser_errors = sum(
        "Error parsing extraction response" in line or "LLM extraction failed" in line
        for line in upstream_log.read_text(encoding="utf-8").splitlines()
    )
    return {"run_id": run_id, "state": "VERIFIED", "artifacts": len(manifest["artifacts"]),
            "items": expected, "source_links": provenance_count,
            "final_artifact_bytes": sum(artifact["bytes"] for artifact in manifest["artifacts"]),
            "logical_run_bytes": sum(path.stat().st_size for path in root.rglob("*") if path.is_file()),
            "error_events": len(error_events), "parser_error_events": len(parser_errors),
            "upstream_parser_errors": upstream_parser_errors,
            "answerer_calls": usage["components"]["answerer"]["calls"],
            "judge_calls": usage["components"]["judge"]["calls"],
            "memory_internal_calls": usage["components"]["memory_internal"]["calls"],
            "ingestion_llm_calls": ingestion_totals["calls"] if ingestion_totals else 0,
            "memory_peak_bytes": resources["container"]["memory_peak_bytes"],
            "execution_seconds": timing["attempt"]["execution_seconds"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=BENCHMARKS, action="append")
    parser.add_argument("--framework", choices=FRAMEWORKS, action="append")
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    inputs = audit_inputs()
    collections = qdrant_collections()
    rows = [audit_run(benchmark, framework, collections)
            for benchmark in (args.benchmark or BENCHMARKS)
            for framework in (args.framework or FRAMEWORKS)]
    print(json.dumps({"inputs": inputs, "runs": rows}, indent=2))
    return int(args.require_all and any(row["state"] != "VERIFIED" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
