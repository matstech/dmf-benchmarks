"""Versioned dataset registry and explicit materialization."""

from __future__ import annotations

import json
import os
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dmf_bench.atomic_io import read_json
from dmf_bench.contracts import assert_sha256, sha256_file


REGISTRY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DatasetRecord:
    dataset_id: str
    benchmark: str
    source_url: str
    revision: str
    sha256: str
    filename: str
    expected_schema: str
    registry_schema_version: int = REGISTRY_SCHEMA_VERSION
    pinned: bool = True

    def __post_init__(self) -> None:
        if self.registry_schema_version != REGISTRY_SCHEMA_VERSION:
            raise ValueError("Dataset registry schema_version must be 1.")
        for field_name in ("dataset_id", "benchmark", "source_url", "revision", "filename", "expected_schema"):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} cannot be empty.")
        assert_sha256(self.sha256, field_name="dataset sha256")
        if Path(self.filename).name != self.filename:
            raise ValueError("dataset filename must be a single file name.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.registry_schema_version,
            "dataset_id": self.dataset_id,
            "benchmark": self.benchmark,
            "source_url": self.source_url,
            "revision": self.revision,
            "sha256": self.sha256,
            "filename": self.filename,
            "expected_schema": self.expected_schema,
            "pinned": self.pinned,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetRecord":
        return cls(
            registry_schema_version=int(data.get("schema_version", REGISTRY_SCHEMA_VERSION)),
            dataset_id=str(data["dataset_id"]),
            benchmark=str(data["benchmark"]),
            source_url=str(data["source_url"]),
            revision=str(data["revision"]),
            sha256=str(data["sha256"]),
            filename=str(data["filename"]),
            expected_schema=str(data["expected_schema"]),
            pinned=bool(data.get("pinned", True)),
        )


BUILTIN_DATASETS: dict[str, DatasetRecord] = {
    "locomo-official-unpinned": DatasetRecord(
        dataset_id="locomo-official-unpinned",
        benchmark="locomo",
        source_url="https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json",
        revision="main",
        sha256="0" * 64,
        filename="locomo10.json",
        expected_schema="locomo10-json-array-v1",
        pinned=False,
    ),
    "longmemeval-s-official-unpinned": DatasetRecord(
        dataset_id="longmemeval-s-official-unpinned",
        benchmark="longmemeval",
        source_url=(
            "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/"
            "resolve/main/longmemeval_s_cleaned.json"
        ),
        revision="main",
        sha256="0" * 64,
        filename="longmemeval_s_cleaned.json",
        expected_schema="longmemeval-cleaned-json-array-v1",
        pinned=False,
    ),
}


def load_dataset_registry(path: str | Path | None = None) -> dict[str, DatasetRecord]:
    if path is None:
        return dict(BUILTIN_DATASETS)

    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError("Dataset registry root must be a JSON object.")
    if int(payload.get("schema_version", REGISTRY_SCHEMA_VERSION)) != REGISTRY_SCHEMA_VERSION:
        raise ValueError("Dataset registry schema_version must be 1.")
    records = payload.get("datasets")
    if not isinstance(records, list):
        raise ValueError("Dataset registry datasets must be a list.")

    registry: dict[str, DatasetRecord] = {}
    for raw_record in records:
        if not isinstance(raw_record, dict):
            raise ValueError("Dataset registry entries must be objects.")
        record = DatasetRecord.from_dict(raw_record)
        if record.dataset_id in registry:
            raise ValueError(f"Duplicate dataset_id in registry: {record.dataset_id}")
        registry[record.dataset_id] = record
    return registry


def materialize_dataset(
    *,
    dataset_id: str,
    output_dir: str | Path,
    registry_path: str | Path | None = None,
) -> Path:
    registry = load_dataset_registry(registry_path)
    try:
        record = registry[dataset_id]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset_id: {dataset_id}") from exc
    if not record.pinned:
        raise ValueError(
            f"Dataset {dataset_id!r} is not pinned; add a registry entry with the approved SHA-256."
        )

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / record.filename
    if target.exists():
        observed = sha256_file(target)
        if observed != record.sha256:
            raise ValueError(
                f"Refusing to overwrite existing dataset with SHA-256 {observed}; "
                f"expected {record.sha256}."
            )
        return target

    temp_path: Path | None = None
    fd = -1
    try:
        fd, raw_temp_path = tempfile.mkstemp(
            prefix=f".{record.filename}.",
            suffix=".tmp",
            dir=str(target_dir),
        )
        temp_path = Path(raw_temp_path)
        with os.fdopen(fd, "wb") as file:
            fd = -1
            _copy_url_to_file(record.source_url, file)
            file.flush()
            os.fsync(file.fileno())
        observed = sha256_file(temp_path)
        if observed != record.sha256:
            raise ValueError(
                f"Downloaded dataset SHA-256 mismatch: expected {record.sha256}, got {observed}."
            )
        os.replace(temp_path, target)
        temp_path = None
        _fsync_directory(target_dir)
        return target
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        if temp_path is not None:
            try:
                temp_path.unlink()
            except OSError:
                pass


def dataset_config_for(record: DatasetRecord, path: str | Path) -> dict[str, Any]:
    return {
        "name": record.benchmark,
        "source": record.source_url,
        "revision": record.revision,
        "sha256": record.sha256,
        "path": str(Path(path).resolve()),
        "registry_id": record.dataset_id,
        "expected_schema": record.expected_schema,
    }


def dataset_config_for_id(
    *,
    dataset_id: str,
    path: str | Path,
    registry_path: str | Path | None = None,
) -> dict[str, Any]:
    registry = load_dataset_registry(registry_path)
    try:
        record = registry[dataset_id]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset_id: {dataset_id}") from exc
    return dataset_config_for(record, path)


def registry_as_dict(registry: dict[str, DatasetRecord] | None = None) -> dict[str, Any]:
    selected = registry or BUILTIN_DATASETS
    return {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "datasets": [record.to_dict() for record in sorted(selected.values(), key=lambda item: item.dataset_id)],
    }


def _copy_url_to_file(source_url: str, file: Any) -> None:
    parsed = urllib.parse.urlparse(source_url)
    if parsed.scheme == "file":
        source_path = Path(urllib.request.url2pathname(parsed.path))
        with source_path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                file.write(chunk)
        return
    if parsed.scheme in {"http", "https"}:
        with urllib.request.urlopen(source_url, timeout=120) as response:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                file.write(chunk)
        return
    raise ValueError(f"Unsupported dataset source URL scheme: {parsed.scheme!r}")


def _fsync_directory(path: str | Path) -> None:
    fd = os.open(Path(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
