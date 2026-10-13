"""Convert a pinned local MemoryAgentBench release into immutable unit files."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path


DATASET_REVISION = "7ea066982b140a19337e17e60d45d4076e042faf"
CODE_REVISION = "538026089d1a8a8eff05121d0db89b388f360eba"
SOURCE_HASHES = {
    "README.md": "5a9fd0dc2fbab5beb081eae40f24964ad1e47da75b9a7a01f0007791845b1580",
    "entity2id.json": "63353aca481bc9558b502f91cb98f6fa26438796fdd7e0bc06b5a1532126e8b5",
    "data/Accurate_Retrieval-00000-of-00001.parquet": "56c3cd80fb6731a3e53cd1a6be3148f54df60ff2d290ee50e28f8acebf9655c1",
    "data/Conflict_Resolution-00000-of-00001.parquet": "24d5c3f09ce0ce15625cb9f8a98f44f0d864ca6c94d7b4ad04eb697ca3a5ff45",
    "data/Long_Range_Understanding-00000-of-00001.parquet": "5ab175461954db67770d4a4cb69e569b513ebb96aceb9ee79b57f67488bcd539",
    "data/Test_Time_Learning-00000-of-00001.parquet": "5338753be48f925d03318eed66117286e3489025fabe050a547bd086cd7d79c0",
}
SPLITS = (
    "Accurate_Retrieval", "Test_Time_Learning", "Long_Range_Understanding",
    "Conflict_Resolution",
)
EXPECTED_COUNTS = {
    "Accurate_Retrieval": (22, 2000),
    "Test_Time_Learning": (6, 700),
    "Long_Range_Understanding": (110, 171),
    "Conflict_Resolution": (8, 800),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return sha256(path)


def materialize(source: Path, output: Path) -> dict:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("Offline materialization requires pyarrow; install it before running this script.") from exc

    for relative, expected in SOURCE_HASHES.items():
        path = source / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Pinned MemoryAgentBench file is absent or has the wrong SHA-256: {path}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for split in SPLITS:
        path = source / "data" / f"{split}-00000-of-00001.parquet"
        rows = parquet.read_table(path).to_pylist()
        if (len(rows), sum(len(row.get("questions") or []) for row in rows)) != EXPECTED_COUNTS[split]:
            raise ValueError(f"Pinned MemoryAgentBench split count differs: {split}")
        for row_index, row in enumerate(rows):
            metadata = row.get("metadata") or {}
            name = metadata.get("source")
            if not isinstance(name, str) or not name:
                raise ValueError(f"Missing source in {split} row {row_index}")
            questions, answers = row.get("questions"), row.get("answers")
            qa_ids = metadata.get("qa_pair_ids")
            if not isinstance(questions, list) or not questions or not isinstance(answers, list) or len(questions) != len(answers):
                raise ValueError(f"Invalid question/answer alignment in {split} row {row_index}")
            if not isinstance(qa_ids, list) or len(qa_ids) != len(questions) or len(set(qa_ids)) != len(qa_ids):
                raise ValueError(f"Invalid QA IDs in {split} row {row_index}")
            context = row.get("context")
            if not isinstance(context, str) or not context.strip():
                raise ValueError(f"Empty context in {split} row {row_index}")
            unit_id = f"{re.sub(r'[^a-zA-Z0-9_-]', '_', name)}--{row_index:03d}"
            relative = f"units/{unit_id}.json"
            digest = write_json(output / relative, row)
            records.append({
                "unit_id": unit_id,
                "split": split,
                "source": name,
                "row_index": row_index,
                "path": relative,
                "sha256": digest,
                "context_chars": len(context),
                "query_count": len(questions),
            })
    shutil.copyfile(source / "entity2id.json", output / "entity2id.json")
    manifest = {
        "schema_version": 1,
        "benchmark": "memoryagentbench",
        "dataset_revision": DATASET_REVISION,
        "code_revision": CODE_REVISION,
        "source_sha256": SOURCE_HASHES,
        "entity2id_sha256": SOURCE_HASHES["entity2id.json"],
        "records": records,
    }
    write_json(output / "manifest.json", manifest)
    return {"units": len(records), "queries": sum(record["query_count"] for record in records),
            "manifest_sha256": sha256(output / "manifest.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Directory of the six pinned release files")
    parser.add_argument("--output", type=Path, required=True, help="Empty output directory")
    arguments = parser.parse_args()
    print(json.dumps(materialize(arguments.source, arguments.output), sort_keys=True))


if __name__ == "__main__":
    main()
