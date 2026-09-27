"""Root-scoped cleanup for future embedded storage adapters."""

from __future__ import annotations

import shutil
from pathlib import Path

from .base import OwnedResource


def cleanup_owned_embedded_resource(
    resource: OwnedResource,
    *,
    root: Path,
    owned_resources: tuple[OwnedResource, ...],
) -> dict[str, object]:
    """Delete one declared embedded resource strictly inside its configured root."""
    if resource not in owned_resources or resource.kind != "embedded-path":
        raise ValueError("Embedded resource is absent from the owned manifest.")
    relative = Path(resource.locator)
    if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
        raise ValueError("Embedded resource locator must be a child path.")
    canonical_root = root.resolve(strict=True)
    target = (canonical_root / relative).resolve(strict=False)
    if target == canonical_root or not target.is_relative_to(canonical_root):
        raise ValueError("Embedded resource escapes the configured root.")
    if target.is_dir():
        shutil.rmtree(target)
    elif target.exists():
        target.unlink()
    if target.exists():
        raise RuntimeError("Embedded resource remains after cleanup.")
    return {"verified": True, "resource_id": resource.resource_id}
