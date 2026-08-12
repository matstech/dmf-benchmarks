"""Locate immutable controller resources and the operator workspace."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RuntimeLayout:
    """Separate installed orchestration assets from writable operator state."""

    runtime_dir: Path
    workspace_dir: Path
    source_checkout: bool


def resolve_runtime_layout(
    project_dir: str | None,
    work_dir: str | None,
) -> RuntimeLayout:
    """Resolve a source checkout override or the resources shipped in the wheel."""

    if project_dir:
        runtime_dir = _validated_runtime_dir(
            Path(project_dir).expanduser(),
            require_complete=False,
        )
        workspace_dir = (
            Path(work_dir).expanduser().resolve()
            if work_dir
            else runtime_dir
        )
        return RuntimeLayout(runtime_dir, workspace_dir, True)

    for candidate in (Path.cwd(), *Path.cwd().parents):
        if _has_runtime_assets(candidate):
            runtime_dir = candidate.resolve()
            workspace_dir = (
                Path(work_dir).expanduser().resolve()
                if work_dir
                else runtime_dir
            )
            return RuntimeLayout(runtime_dir, workspace_dir, True)

    runtime_dir = _validated_runtime_dir(Path(__file__).resolve().parent.parent)
    workspace_dir = (
        Path(work_dir).expanduser().resolve()
        if work_dir
        else Path.cwd().resolve()
    )
    return RuntimeLayout(runtime_dir, workspace_dir, False)


def _validated_runtime_dir(path: Path, *, require_complete: bool = True) -> Path:
    candidate = path.resolve()
    valid = (
        _has_runtime_assets(candidate)
        if require_complete
        else (candidate / "deploy" / "compose.yaml").is_file()
    )
    if not valid:
        raise ValueError(
            "runtime directory does not contain the packaged deploy/config resources: "
            f"{candidate}"
        )
    return candidate


def _has_runtime_assets(path: Path) -> bool:
    return all(
        candidate.is_file()
        for candidate in (
            path / "deploy" / "compose.yaml",
            path / "config" / "locomo_mem0_qdrant_settings.yaml",
            path / "smoke" / "config" / "experiment-locomo-mem0.json",
        )
    )
