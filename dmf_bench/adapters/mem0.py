"""Mem0 framework adapter resource contract for Qdrant Server mode."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .base import FrameworkCapability, ResumeCapability
from .qdrant_lifecycle import CleanupManifest, CollectionRole, build_cleanup_manifest


@dataclass(frozen=True)
class Mem0QdrantFrameworkAdapter:
    vector_size: int
    sqlite_root: Path
    name: str = "mem0"
    resume_capability: ResumeCapability = ResumeCapability.RESTART_UNIT
    capabilities: frozenset[FrameworkCapability] = frozenset(
        {
            FrameworkCapability.NATIVE_SURFACE,
            FrameworkCapability.USAGE,
            FrameworkCapability.QDRANT_SERVER,
            FrameworkCapability.CLEANUP_MANIFEST,
        }
    )

    def resources_for_unit(self, run_hash: str, unit_id: str) -> CleanupManifest:
        unit_hash = unit_id.replace("/", "_")
        return build_cleanup_manifest(
            run_hash=run_hash,
            framework=self.name,
            unit_id=unit_id,
            roles=(CollectionRole.PRIMARY, CollectionRole.ENTITIES),
            vector_size=self.vector_size,
            local_paths=(str(self.sqlite_root / f"{unit_hash}.sqlite"),),
        )

    def validate_runtime(self) -> None:
        try:
            import mem0
        except ModuleNotFoundError as exc:
            raise RuntimeError("Mem0 runtime is unavailable.") from exc
        if not hasattr(mem0, "Memory"):
            raise RuntimeError("Mem0 runtime does not expose Memory.")
