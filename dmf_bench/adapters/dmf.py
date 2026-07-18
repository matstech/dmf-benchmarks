"""DMF framework adapter for Qdrant Server mode."""

from __future__ import annotations

from dataclasses import dataclass

from .base import FrameworkCapability, ResumeCapability
from .qdrant_lifecycle import CleanupManifest, CollectionRole, build_cleanup_manifest


REQUIRED_DMF_QDRANT_COMMIT = "5c4318e36120c60c1a4f4322b999f964cefa42d4"


@dataclass(frozen=True)
class DmfQdrantFrameworkAdapter:
    vector_size: int
    name: str = "dmf"
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
        return build_cleanup_manifest(
            run_hash=run_hash,
            framework=self.name,
            unit_id=unit_id,
            roles=(CollectionRole.PRIMARY, CollectionRole.CARDS),
            vector_size=self.vector_size,
        )

    def validate_runtime(self) -> None:
        try:
            from dmf.memory.ltm_hooks import QdrantLTMHook
            from dmf.memory.ltm_hooks.qdrant_client import QdrantConnectionMode
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "DMF Qdrant Server mode is unavailable. Pin dmf-memory[qdrant] "
                f"to commit {REQUIRED_DMF_QDRANT_COMMIT} or a later approved release."
            ) from exc
        if QdrantLTMHook is None or QdrantConnectionMode.SERVER.value != "server":
            raise RuntimeError("DMF Qdrant Server mode has an incompatible API.")
