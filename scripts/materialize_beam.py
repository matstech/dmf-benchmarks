"""Materialize pinned BEAM Parquet releases into isolated conversation files."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterator


CODE_REVISION = "b2da22eac88bb0874c64665f13457eb99835774a"
DATASET_REVISIONS = {
    "BEAM": "3205395e897e7318c7b094ef4e6047b9b82dbb03",
    "BEAM-10M": "9b2096193fe74e2837e4713e483351e19817773c",
}
SOURCE_HASHES = {
    "BEAM/data/100K-00000-of-00001.parquet": "c0519be25907005ba873c927c50877471d550873039d96c041554d0075a78ace",
    "BEAM/data/500K-00000-of-00001.parquet": "af05921c979355038e1761b7cde3d2dd713200dd3071b278de0200f6c7f30122",
    "BEAM/data/1M-00000-of-00001.parquet": "41b5acbbb55a586b1305514ef9d9fb03365d9b3331b598a1c2dd7603d93ef533",
    "BEAM-10M/data/10M-00000-of-00002.parquet": "31d96fd47ec56221d202e68792f26c00e49467dd4b36ee105c36ebd19ef78ad5",
    "BEAM-10M/data/10M-00001-of-00002.parquet": "a4f13fe25af51d57405ae41008689c31d1421377f3efde56a024b441deb2ee65",
}
ABILITIES = (
    "abstention", "contradiction_resolution", "event_ordering",
    "information_extraction", "instruction_following", "knowledge_update",
    "multi_session_reasoning", "preference_following", "summarization",
    "temporal_reasoning",
)
EXPECTED_UNITS = {"100K": 20, "500K": 35, "1M": 35, "10M": 10}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return sha256(path)


def source_files(source: Path) -> Iterator[tuple[str, Path]]:
    for relative in SOURCE_HASHES:
        repository, filename = relative.split("/", maxsplit=1)
        yield relative, source / f"{repository}-dataset" / filename


def iter_messages(row: dict[str, Any], scale: str) -> Iterator[tuple[int, int, dict[str, Any]]]:
    if scale != "10M":
        for batch_index, batch in enumerate(row["chat"]):
            for message in batch:
                yield 0, batch_index, message
        return
    for plan_index, plan in enumerate(row["chat"]):
        active = [(key, value) for key, value in plan.items() if value is not None]
        if len(active) != 1 or active[0][0] != f"plan-{plan_index + 1}":
            raise ValueError("BEAM 10M plan order differs from the pinned release.")
        for batch_index, batch in enumerate(active[0][1]):
            for turn in batch["turns"]:
                for message in turn:
                    yield plan_index, batch_index, message


def materialize(source: Path, output: Path) -> dict[str, Any]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("BEAM materialization requires pyarrow.") from exc
    for relative, path in source_files(source):
        if not path.is_file() or sha256(path) != SOURCE_HASHES[relative]:
            raise ValueError(f"Pinned BEAM source file is absent or has a SHA-256 mismatch: {path}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    ability_counts: Counter[str] = Counter()
    for relative, path in source_files(source):
        scale = path.name.split("-")[0]
        file_rows = 0
        for batch in parquet.ParquetFile(path).iter_batches(batch_size=1):
            row = batch.to_pylist()[0]
            file_rows += 1
            conversation_id = str(row["conversation_id"])
            unit_id = f"beam-{scale.lower()}-{int(conversation_id):03d}"
            if any(record["unit_id"] == unit_id for record in records):
                raise ValueError(f"Duplicate BEAM conversation ID: {unit_id}")
            events = []
            for plan_index, batch_index, message in iter_messages(row, scale):
                event_id = f"{unit_id}--e{len(events):07d}"
                role = str(message["role"]).lower()
                content = str(message["content"])
                if role not in {"user", "assistant", "system"} or not content.strip():
                    raise ValueError(f"Invalid BEAM message in {unit_id}")
                events.append({
                    "event_id": event_id, "session_id": f"{unit_id}--p{plan_index:02d}-b{batch_index:04d}",
                    "sequence": len(events), "role": role, "content": content,
                    "occurred_at": None, "source_refs": [event_id],
                    "metadata": {"plan_index": plan_index, "batch_index": batch_index,
                                 "upstream_message_id": message.get("id"),
                                 "time_anchor_raw": message.get("time_anchor")},
                })
            questions = ast.literal_eval(row["probing_questions"])
            if not isinstance(questions, dict) or set(questions) != set(ABILITIES):
                raise ValueError(f"BEAM ability labels differ in {unit_id}")
            public_queries = []
            references = []
            for ability in ABILITIES:
                items = questions[ability]
                if not isinstance(items, list) or len(items) != 2:
                    raise ValueError(f"BEAM requires two questions for {ability} in {unit_id}")
                for index, item in enumerate(items):
                    query_id = f"{unit_id}--q-{ability}-{index + 1:02d}"
                    rubric = item.get("rubric")
                    if not isinstance(rubric, list) or not rubric or any(not isinstance(x, str) or not x for x in rubric):
                        raise ValueError(f"Invalid BEAM rubric in {query_id}")
                    question = item.get("question")
                    if not isinstance(question, str) or not question.strip():
                        raise ValueError(f"Invalid BEAM question in {query_id}")
                    public_queries.append({"query_id": query_id, "text": question,
                                           "ability": ability, "difficulty": item.get("difficulty")})
                    references.append({"query_id": query_id, "expected_answer": next(
                        (item[key] for key in ("ideal_response", "ideal_answer", "answer",
                                                     "ideal_summary", "expected_compliance") if key in item), None),
                                       "rubric": rubric, "source_chat_ids": item.get("source_chat_ids"),
                                       "upstream_question": item})
                    ability_counts[ability] += 1
            unit_dir = output / "units" / unit_id
            event_path = unit_dir / "events.json"
            query_path = unit_dir / "queries.json"
            reference_path = unit_dir / "references.json"
            event_sha = write_json(event_path, events)
            query_sha = write_json(query_path, public_queries)
            reference_sha = write_json(reference_path, references)
            records.append({
                "unit_id": unit_id, "scale": scale, "conversation_id": conversation_id,
                "category": row["conversation_seed"]["category"],
                "event_count": len(events), "question_count": len(public_queries),
                "context_chars": sum(len(event["content"]) for event in events),
                "estimated_tokens_chars_div_4": (sum(len(event["content"]) for event in events) + 3) // 4,
                "events_path": event_path.relative_to(output).as_posix(), "events_sha256": event_sha,
                "events_bytes": event_path.stat().st_size,
                "queries_path": query_path.relative_to(output).as_posix(), "queries_sha256": query_sha,
                "references_path": reference_path.relative_to(output).as_posix(), "references_sha256": reference_sha,
            })
        if file_rows != (5 if scale == "10M" else EXPECTED_UNITS[scale]):
            raise ValueError(f"Unexpected BEAM row count in {relative}: {file_rows}")
    counts = Counter(record["scale"] for record in records)
    if dict(counts) != EXPECTED_UNITS or any(ability_counts[ability] != 200 for ability in ABILITIES):
        raise ValueError("BEAM conversation or ability counts differ from the pinned release.")
    manifest = {"schema_version": 1, "benchmark": "beam", "code_revision": CODE_REVISION,
                "dataset_revisions": DATASET_REVISIONS, "source_sha256": SOURCE_HASHES,
                "license_code": "MIT", "license_data": "CC-BY-SA-4.0", "abilities": list(ABILITIES),
                "records": records}
    manifest_sha = write_json(output / "manifest.json", manifest)
    return {"units": len(records), "questions": sum(record["question_count"] for record in records),
            "events": sum(record["event_count"] for record in records), "manifest_sha256": manifest_sha}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(args.source, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
