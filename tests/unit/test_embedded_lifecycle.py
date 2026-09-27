from __future__ import annotations

from pathlib import Path

import pytest

from dmf_bench.adapters.base import OwnedResource
from dmf_bench.adapters.embedded_lifecycle import cleanup_owned_embedded_resource


def test_cleanup_stays_within_declared_embedded_root(tmp_path: Path) -> None:
    root = tmp_path / "embedded"
    owned = root / "unit-1"
    owned.mkdir(parents=True)
    (owned / "index.bin").write_bytes(b"fixture")
    sibling = root / "unit-2"
    sibling.mkdir()
    resource = OwnedResource("unit-1", "embedded-path", "primary", "unit-1")
    assert cleanup_owned_embedded_resource(
        resource, root=root, owned_resources=(resource,),
    ) == {"verified": True, "resource_id": "unit-1"}
    assert not owned.exists()
    assert sibling.exists()


def test_cleanup_rejects_unowned_and_escaping_embedded_paths(tmp_path: Path) -> None:
    root = tmp_path / "embedded"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep")
    link = root / "escape"
    link.symlink_to(outside, target_is_directory=True)
    manifest = (
        OwnedResource("escape", "embedded-path", "primary", "escape"),
        OwnedResource("traversal", "embedded-path", "primary", "../outside"),
    )
    for resource in manifest:
        with pytest.raises(ValueError, match="escapes|child path"):
            cleanup_owned_embedded_resource(
                resource, root=root, owned_resources=manifest,
            )
    with pytest.raises(ValueError, match="absent from the owned manifest"):
        cleanup_owned_embedded_resource(
            OwnedResource("unknown", "embedded-path", "primary", "unit-2"),
            root=root, owned_resources=manifest,
        )
    assert (outside / "keep.txt").read_text() == "keep"
