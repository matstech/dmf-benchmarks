"""Qdrant Server lifecycle helpers shared by framework adapters."""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol


MAX_COLLECTION_NAME_LENGTH = 63
COLLECTION_PREFIX = "bench"


class QdrantLifecycleError(RuntimeError):
    """Raised when Qdrant lifecycle validation fails."""


class CollectionRole(str, Enum):
    PRIMARY = "primary"
    CARDS = "cards"
    ENTITIES = "entities"


@dataclass(frozen=True)
class QdrantCollectionResource:
    name: str
    role: CollectionRole
    vector_size: int
    distance: str = "Cosine"


@dataclass(frozen=True)
class CleanupManifest:
    run_hash: str
    framework: str
    unit_hash: str
    collections: tuple[QdrantCollectionResource, ...]
    local_paths: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_hash": self.run_hash,
            "framework": self.framework,
            "unit_hash": self.unit_hash,
            "collections": [
                {
                    "name": collection.name,
                    "role": collection.role.value,
                    "vector_size": collection.vector_size,
                    "distance": collection.distance,
                }
                for collection in self.collections
            ],
            "local_paths": list(self.local_paths),
        }


class QdrantClientProtocol(Protocol):
    def collection_exists(self, collection_name: str) -> bool: ...

    def create_collection(self, collection_name: str, vectors_config: Any) -> Any: ...

    def delete_collection(self, collection_name: str) -> Any: ...

    def count(self, collection_name: str, exact: bool = True) -> Any: ...


def stable_hash(value: str, *, length: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def qdrant_collection_name(
    *,
    run_hash: str,
    framework: str,
    unit_id: str,
    role: CollectionRole,
) -> str:
    safe_framework = normalize_name_part(framework)
    unit_hash = stable_hash(unit_id)
    raw = f"{COLLECTION_PREFIX}_{run_hash[:16]}_{safe_framework}_{unit_hash}_{role.value}"
    if len(raw) <= MAX_COLLECTION_NAME_LENGTH:
        return raw
    suffix = stable_hash(raw, length=12)
    keep = MAX_COLLECTION_NAME_LENGTH - len(suffix) - 1
    return f"{raw[:keep]}_{suffix}"


def normalize_name_part(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9_]+", "_", value.strip().lower())
    normalized = normalized.strip("_")
    if not normalized:
        raise ValueError("Qdrant collection name component cannot be empty.")
    return normalized


def build_cleanup_manifest(
    *,
    run_hash: str,
    framework: str,
    unit_id: str,
    roles: tuple[CollectionRole, ...],
    vector_size: int,
    local_paths: tuple[str, ...] = (),
) -> CleanupManifest:
    return CleanupManifest(
        run_hash=run_hash[:16],
        framework=normalize_name_part(framework),
        unit_hash=stable_hash(unit_id),
        collections=tuple(
            QdrantCollectionResource(
                name=qdrant_collection_name(
                    run_hash=run_hash,
                    framework=framework,
                    unit_id=unit_id,
                    role=role,
                ),
                role=role,
                vector_size=vector_size,
            )
            for role in roles
        ),
        local_paths=local_paths,
    )


class QdrantLifecycleManager:
    def __init__(self, client: QdrantClientProtocol) -> None:
        self.client = client

    def check_ready(self) -> None:
        try:
            self.client.collection_exists("__dmf_bench_readiness_probe__")
        except Exception as exc:
            raise QdrantLifecycleError(f"Qdrant is not ready: {exc}") from exc

    def assert_absent(self, manifest: CleanupManifest) -> None:
        existing = [
            collection.name
            for collection in manifest.collections
            if self.client.collection_exists(collection.name)
        ]
        if existing:
            raise QdrantLifecycleError(
                f"Qdrant collections already exist for unit: {', '.join(existing)}"
            )

    def create_collections(self, manifest: CleanupManifest) -> None:
        for collection in manifest.collections:
            self.client.create_collection(
                collection_name=collection.name,
                vectors_config=build_vector_params(collection),
            )

    def verify_counts(
        self,
        manifest: CleanupManifest,
        *,
        minimum_count_by_role: dict[CollectionRole, int],
    ) -> None:
        for collection in manifest.collections:
            minimum = minimum_count_by_role.get(collection.role, 0)
            observed = collection_count(self.client.count(collection.name, exact=True))
            if observed < minimum:
                raise QdrantLifecycleError(
                    f"Qdrant collection {collection.name!r} has {observed} points; expected at least {minimum}."
                )

    def collection_counts(
        self,
        manifest: CleanupManifest,
    ) -> dict[CollectionRole, int]:
        """Read an exact count barrier for every owned collection."""
        counts: dict[CollectionRole, int] = {}
        for collection in manifest.collections:
            if not self.client.collection_exists(collection.name):
                raise QdrantLifecycleError(
                    f"Qdrant collection {collection.name!r} is missing at the count barrier."
                )
            counts[collection.role] = collection_count(
                self.client.count(collection.name, exact=True)
            )
        return counts

    def delete_and_wait(
        self,
        manifest: CleanupManifest,
        *,
        max_attempts: int = 5,
        poll_interval_seconds: float = 0.0,
    ) -> None:
        for collection in manifest.collections:
            if self.client.collection_exists(collection.name):
                self.client.delete_collection(collection.name)

        for _attempt in range(max_attempts):
            remaining = [
                collection.name
                for collection in manifest.collections
                if self.client.collection_exists(collection.name)
            ]
            if not remaining:
                return
            if poll_interval_seconds:
                time.sleep(poll_interval_seconds)
        raise QdrantLifecycleError(
            f"Qdrant collections still exist after deletion: {', '.join(remaining)}"
        )


def build_vector_params(collection: QdrantCollectionResource) -> Any:
    try:
        from qdrant_client import models
    except ModuleNotFoundError:
        return {
            "size": collection.vector_size,
            "distance": collection.distance,
        }
    distance = getattr(models.Distance, collection.distance.upper())
    return models.VectorParams(size=collection.vector_size, distance=distance)


def collection_count(response: Any) -> int:
    if isinstance(response, int):
        return response
    count = getattr(response, "count", None)
    if count is not None:
        return int(count)
    if isinstance(response, dict) and "count" in response:
        return int(response["count"])
    raise QdrantLifecycleError(f"Unsupported Qdrant count response: {response!r}")
