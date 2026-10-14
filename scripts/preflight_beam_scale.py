"""Estimate BEAM scale requirements from a pinned materialized manifest."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from dmf_bench.contracts import sha256_file


SCALES = ("100K", "500K", "1M", "10M")


def summarize(manifest_path: Path, *, internal_calls_per_event: float | None,
              events_per_second: float | None, seconds_per_answer: float | None,
              storage_multiplier: float, framework: str | None = None) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("benchmark") != "beam" or manifest.get("schema_version") != 1:
        raise ValueError("Expected a materialized BEAM manifest.")
    rows = defaultdict(lambda: {"conversations": 0, "events": 0, "questions": 0,
                                "input_chars": 0, "input_tokens_estimate": 0,
                                "event_storage_bytes": 0})
    for record in manifest["records"]:
        row = rows[record["scale"]]
        row["conversations"] += 1
        row["events"] += record["event_count"]
        row["questions"] += record["question_count"]
        row["input_chars"] += record["context_chars"]
        row["input_tokens_estimate"] += record["estimated_tokens_chars_div_4"]
        row["event_storage_bytes"] += record["events_bytes"]
    if set(rows) != set(SCALES):
        raise ValueError("BEAM manifest does not contain all four scales.")
    if framework in {"model-only", "full-context", "vector-rag"}:
        internal_calls_per_event = 0.0
    result = {}
    for scale in SCALES:
        row = rows[scale]
        row["projected_storage_bytes"] = round(row["event_storage_bytes"] * storage_multiplier)
        row["internal_llm_calls_estimate"] = (
            round(row["events"] * internal_calls_per_event)
            if internal_calls_per_event is not None else None
        )
        row["duration_seconds_estimate"] = (
            round(row["events"] / events_per_second
                  + row["questions"] * seconds_per_answer, 1)
            if events_per_second is not None and seconds_per_answer is not None else None
        )
        result[scale] = row
    return {"manifest_sha256": sha256_file(manifest_path),
            "framework": framework,
            "token_estimate_method": "sum ceil(content characters / 4) per conversation; not tokenizer-accurate",
            "storage_multiplier": storage_multiplier,
            "internal_calls_per_event": internal_calls_per_event,
            "events_per_second": events_per_second,
            "seconds_per_answer": seconds_per_answer,
            "scales": result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("datasets/beam/manifest.json"))
    parser.add_argument("--framework", choices=("model-only", "full-context", "vector-rag",
                                                "dmf", "mem0", "lightmem", "amem"))
    parser.add_argument("--internal-calls-per-event", type=float)
    parser.add_argument("--events-per-second", type=float)
    parser.add_argument("--seconds-per-answer", type=float)
    parser.add_argument("--storage-multiplier", type=float, default=1)
    args = parser.parse_args()
    if args.internal_calls_per_event is not None and args.internal_calls_per_event < 0:
        parser.error("Internal calls per event must be non-negative.")
    if (args.framework in {"model-only", "full-context", "vector-rag"}
            and args.internal_calls_per_event not in (None, 0, 0.0)):
        parser.error("The controlled baselines have zero internal LLM calls per event.")
    for value in (args.events_per_second, args.seconds_per_answer, args.storage_multiplier):
        if value is not None and value <= 0:
            parser.error("Rates and storage multiplier must be positive.")
    if (args.events_per_second is None) != (args.seconds_per_answer is None):
        parser.error("Supply both duration rates or neither.")
    print(json.dumps(summarize(
        args.manifest, internal_calls_per_event=args.internal_calls_per_event,
        events_per_second=args.events_per_second,
        seconds_per_answer=args.seconds_per_answer,
        storage_multiplier=args.storage_multiplier, framework=args.framework,
    ), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
