"""Versioned dataset registry and explicit materialization."""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dmf_bench.atomic_io import read_json, write_json_atomic
from dmf_bench.contracts import assert_sha256, hash_canonical_json, sha256_file


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


@dataclass(frozen=True)
class MaterializedDataset:
    path: Path
    manifest_path: Path
    dataset_config: dict[str, Any]
    manifest: dict[str, Any]


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
    sampling: dict[str, Any] | None = None,
    allow_unpinned: bool = False,
) -> Path:
    return materialize_dataset_record(
        dataset_id=dataset_id,
        output_dir=output_dir,
        registry_path=registry_path,
        sampling=sampling,
        allow_unpinned=allow_unpinned,
    ).path


def materialize_dataset_record(
    *,
    dataset_id: str,
    output_dir: str | Path,
    registry_path: str | Path | None = None,
    sampling: dict[str, Any] | None = None,
    allow_unpinned: bool = False,
) -> MaterializedDataset:
    registry = load_dataset_registry(registry_path)
    try:
        record = registry[dataset_id]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset_id: {dataset_id}") from exc
    if not record.pinned and not allow_unpinned:
        raise ValueError(
            f"Dataset {dataset_id!r} is not pinned; pass allow_unpinned=true only for exploratory runs."
        )
    sampling_spec = _normalize_sampling_spec(sampling)

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    source_path = target_dir / record.filename
    source_sha256 = _ensure_source_dataset(record, source_path)

    materialized_path = source_path
    sampling_manifest: dict[str, Any] | None = None
    if sampling_spec is not None:
        materialized_path, sampling_manifest = _materialize_sample(
            record=record,
            source_path=source_path,
            output_dir=target_dir,
            sampling=sampling_spec,
        )

    materialized_sha256 = sha256_file(materialized_path)
    manifest_path = target_dir / "manifest.json"
    manifest = _build_materialization_manifest(
        record=record,
        source_path=source_path,
        source_sha256=source_sha256,
        materialized_path=materialized_path,
        materialized_sha256=materialized_sha256,
        sampling=sampling_manifest,
    )
    write_json_atomic(manifest_path, manifest)
    dataset_config = dataset_config_for(
        record,
        materialized_path,
        sha256=materialized_sha256,
        manifest_path=manifest_path,
    )
    return MaterializedDataset(
        path=materialized_path,
        manifest_path=manifest_path,
        dataset_config=dataset_config,
        manifest=manifest,
    )


def materialize_dataset_for_config(
    data: dict[str, Any],
    *,
    registry_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return a config copy with source-based dataset entries materialized."""
    dataset = data.get("dataset")
    runtime = data.get("runtime")
    if not isinstance(dataset, dict) or not isinstance(runtime, dict):
        return data
    if dataset.get("path") and dataset.get("sha256"):
        return data
    registry_id = dataset.get("registry_id")
    if not isinstance(registry_id, str) or not registry_id.strip():
        return data
    cache_dir = runtime.get("cache_dir")
    if not isinstance(cache_dir, str) or not Path(cache_dir).is_absolute():
        raise ValueError("runtime.cache_dir must be an absolute path before dataset materialization.")
    output_dir = Path(cache_dir) / "datasets" / _dataset_cache_key(dataset)
    materialized = materialize_dataset_record(
        dataset_id=registry_id,
        output_dir=output_dir,
        registry_path=registry_path,
        sampling=_dataset_sampling(dataset),
        allow_unpinned=bool(dataset.get("allow_unpinned", False)),
    )
    updated = json.loads(json.dumps(data))
    updated_dataset = dict(updated["dataset"])
    updated_dataset.update(materialized.dataset_config)
    updated["dataset"] = updated_dataset
    return updated


def _ensure_source_dataset(record: DatasetRecord, target: Path) -> str:
    target_dir = target.parent
    if target.exists():
        observed = sha256_file(target)
        if record.pinned and observed != record.sha256:
            raise ValueError(
                f"Refusing to overwrite existing dataset with SHA-256 {observed}; "
                f"expected {record.sha256}."
            )
        return observed

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
        if record.pinned and observed != record.sha256:
            raise ValueError(
                f"Downloaded dataset SHA-256 mismatch: expected {record.sha256}, got {observed}."
            )
        os.replace(temp_path, target)
        temp_path = None
        _fsync_directory(target_dir)
        return observed
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


def dataset_config_for(
    record: DatasetRecord,
    path: str | Path,
    *,
    sha256: str | None = None,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    dataset_sha256 = sha256 or record.sha256
    config = {
        "name": record.benchmark,
        "source": record.source_url,
        "revision": record.revision,
        "sha256": dataset_sha256,
        "path": str(Path(path).resolve()),
        "registry_id": record.dataset_id,
        "expected_schema": record.expected_schema,
    }
    if manifest_path is not None:
        config["materialization_manifest"] = str(Path(manifest_path).resolve())
    return config


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


def _dataset_sampling(dataset: dict[str, Any]) -> dict[str, Any] | None:
    sampling = dataset.get("sampling", dataset.get("sample"))
    if sampling is None:
        return None
    if not isinstance(sampling, dict):
        raise ValueError("dataset.sampling must be an object when provided.")
    return sampling


def _dataset_cache_key(dataset: dict[str, Any]) -> str:
    payload = {
        "registry_id": dataset.get("registry_id"),
        "sampling": _dataset_sampling(dataset),
        "allow_unpinned": bool(dataset.get("allow_unpinned", False)),
    }
    return hash_canonical_json(payload)[:24]


def _normalize_sampling_spec(sampling: dict[str, Any] | None) -> dict[str, Any] | None:
    if sampling is None:
        return None
    if not isinstance(sampling, dict):
        raise ValueError("sampling must be an object.")
    fraction = sampling.get("fraction")
    if not isinstance(fraction, (int, float)) or isinstance(fraction, bool):
        raise ValueError("sampling.fraction must be a number.")
    if fraction <= 0 or fraction > 1:
        raise ValueError("sampling.fraction must be greater than 0 and at most 1.")
    seed = sampling.get("seed", 42)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("sampling.seed must be an integer.")
    rounding = str(sampling.get("rounding", "ceil")).strip().lower()
    if rounding not in {"ceil", "floor", "round"}:
        raise ValueError("sampling.rounding must be ceil, floor, or round.")
    unit = str(sampling.get("unit", "") or "").strip()
    stratify_by = sampling.get("stratify_by")
    if stratify_by is not None:
        stratify_by = str(stratify_by).strip()
        if not stratify_by:
            raise ValueError("sampling.stratify_by must be a non-empty string when provided.")
    return {
        "fraction": float(fraction),
        "seed": seed,
        "rounding": rounding,
        "unit": unit or None,
        "stratify_by": stratify_by,
    }


def _materialize_sample(
    *,
    record: DatasetRecord,
    source_path: Path,
    output_dir: Path,
    sampling: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    payload = read_json(source_path)
    if not isinstance(payload, list):
        raise ValueError("Sampling currently requires a JSON array dataset root.")
    if record.benchmark == "locomo":
        sampled, manifest = _sample_locomo(payload, sampling)
    elif record.benchmark == "longmemeval":
        sampled, manifest = _sample_longmemeval(payload, sampling)
    else:
        raise ValueError(f"Sampling is not implemented for benchmark {record.benchmark!r}.")
    target = output_dir / f"{Path(record.filename).stem}.sample.json"
    write_json_atomic(target, sampled)
    return target, manifest


def _sample_locomo(payload: list[Any], sampling: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
    unit = sampling["unit"] or "qa"
    if unit != "qa":
        raise ValueError("LoCoMo sampling.unit must be 'qa'.")
    stratify_by = sampling["stratify_by"]
    if stratify_by not in {None, "category"}:
        raise ValueError("LoCoMo sampling.stratify_by must be 'category' when provided.")
    candidates: list[tuple[int, int, str, dict[str, Any]]] = []
    for conversation_idx, item in enumerate(payload):
        if not isinstance(item, dict):
            continue
        qa_items = item.get("qa")
        if not isinstance(qa_items, list):
            continue
        for question_idx, qa_item in enumerate(qa_items):
            if isinstance(qa_item, dict):
                group = str(int(qa_item.get("category", 0) or 0)) if stratify_by == "category" else "all"
                candidates.append((conversation_idx, question_idx, group, qa_item))
    selected_keys, population_by_group, sample_by_group = _select_sample_keys(
        [(conversation_idx, question_idx, group) for conversation_idx, question_idx, group, _ in candidates],
        sampling=sampling,
    )
    selected_by_conversation: dict[int, set[int]] = {}
    for conversation_idx, question_idx in selected_keys:
        selected_by_conversation.setdefault(conversation_idx, set()).add(question_idx)
    sampled: list[Any] = []
    for conversation_idx, item in enumerate(payload):
        question_indices = selected_by_conversation.get(conversation_idx)
        if not question_indices or not isinstance(item, dict):
            continue
        cloned = json.loads(json.dumps(item))
        cloned["qa"] = [
            qa_item
            for question_idx, qa_item in enumerate(cloned.get("qa", []))
            if question_idx in question_indices
        ]
        sampled.append(cloned)
    manifest = _sampling_manifest(
        sampling=sampling,
        unit=unit,
        population_count=len(candidates),
        sample_count=sum(len(indices) for indices in selected_by_conversation.values()),
        population_by_group=population_by_group,
        sample_by_group=sample_by_group,
    )
    manifest["sample_conversation_count"] = len(sampled)
    manifest["population_conversation_count"] = len(payload)
    return sampled, manifest


def _sample_longmemeval(payload: list[Any], sampling: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
    unit = sampling["unit"] or "record"
    if unit != "record":
        raise ValueError("LongMemEval sampling.unit must be 'record'.")
    stratify_by = sampling["stratify_by"]
    if stratify_by not in {None, "question_type"}:
        raise ValueError("LongMemEval sampling.stratify_by must be 'question_type' when provided.")
    candidates: list[tuple[str, int, str]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            continue
        group = str(item.get("question_type", "unknown")) if stratify_by == "question_type" else "all"
        candidates.append((str(item.get("question_id", index)), index, group))
    selected_indices, population_by_group, sample_by_group = _select_longmemeval_indices(
        candidates,
        sampling=sampling,
    )
    sampled = [item for index, item in enumerate(payload) if index in selected_indices]
    manifest = _sampling_manifest(
        sampling=sampling,
        unit=unit,
        population_count=len(candidates),
        sample_count=len(sampled),
        population_by_group=population_by_group,
        sample_by_group=sample_by_group,
    )
    return sampled, manifest


def _select_longmemeval_indices(
    candidates: list[tuple[str, int, str]],
    *,
    sampling: dict[str, Any],
) -> tuple[set[int], dict[str, int], dict[str, int]]:
    grouped: dict[str, list[tuple[str, int]]] = {}
    for question_id, index, group in candidates:
        grouped.setdefault(group, []).append((question_id, index))
    rng = random.Random(sampling["seed"])
    selected: set[int] = set()
    population_by_group: dict[str, int] = {}
    sample_by_group: dict[str, int] = {}
    for group in sorted(grouped):
        items = sorted(grouped[group])
        population_by_group[group] = len(items)
        sample_size = _sample_size(
            len(items),
            fraction=float(sampling["fraction"]),
            rounding=str(sampling["rounding"]),
        )
        selected_items = rng.sample(items, sample_size)
        selected.update(index for _, index in selected_items)
        sample_by_group[group] = len(selected_items)
    return selected, population_by_group, sample_by_group


def _select_sample_keys(
    candidates: list[tuple[int, int, str]],
    *,
    sampling: dict[str, Any],
) -> tuple[set[tuple[int, int]], dict[str, int], dict[str, int]]:
    grouped: dict[str, list[tuple[int, int]]] = {}
    for first, second, group in candidates:
        grouped.setdefault(group, []).append((first, second))
    rng = random.Random(sampling["seed"])
    selected: set[tuple[int, int]] = set()
    population_by_group: dict[str, int] = {}
    sample_by_group: dict[str, int] = {}
    for group in sorted(grouped):
        items = sorted(grouped[group])
        population_by_group[group] = len(items)
        sample_size = _sample_size(
            len(items),
            fraction=float(sampling["fraction"]),
            rounding=str(sampling["rounding"]),
        )
        selected_items = rng.sample(items, sample_size)
        selected.update(selected_items)
        sample_by_group[group] = len(selected_items)
    return selected, population_by_group, sample_by_group


def _sample_size(population: int, *, fraction: float, rounding: str) -> int:
    if population <= 0:
        return 0
    raw = population * fraction
    if rounding == "ceil":
        value = math.ceil(raw)
    elif rounding == "floor":
        value = math.floor(raw)
    else:
        value = math.floor(raw + 0.5)
    return min(population, max(1, value))


def _sampling_manifest(
    *,
    sampling: dict[str, Any],
    unit: str,
    population_count: int,
    sample_count: int,
    population_by_group: dict[str, int],
    sample_by_group: dict[str, int],
) -> dict[str, Any]:
    manifest = {
        "fraction": sampling["fraction"],
        "seed": sampling["seed"],
        "rounding": sampling["rounding"],
        "unit": unit,
        "population_count": population_count,
        "sample_count": sample_count,
    }
    if sampling["stratify_by"] is not None:
        manifest["stratify_by"] = sampling["stratify_by"]
        manifest["population_by_group"] = population_by_group
        manifest["sample_by_group"] = sample_by_group
    return manifest


def _build_materialization_manifest(
    *,
    record: DatasetRecord,
    source_path: Path,
    source_sha256: str,
    materialized_path: Path,
    materialized_sha256: str,
    sampling: dict[str, Any] | None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "dataset-materialization",
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "dataset": {
            "dataset_id": record.dataset_id,
            "benchmark": record.benchmark,
            "source_url": record.source_url,
            "revision": record.revision,
            "expected_schema": record.expected_schema,
            "pinned": record.pinned,
        },
        "source": {
            "path": str(source_path.resolve()),
            "sha256": source_sha256,
            "bytes": source_path.stat().st_size,
        },
        "materialized": {
            "path": str(materialized_path.resolve()),
            "sha256": materialized_sha256,
            "bytes": materialized_path.stat().st_size,
        },
    }
    if sampling is not None:
        manifest["sampling"] = sampling
    return manifest


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
