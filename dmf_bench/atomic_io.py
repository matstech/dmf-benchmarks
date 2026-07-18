"""Atomic filesystem primitives for local benchmark state."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


class AtomicWriteError(RuntimeError):
    """Raised when an atomic write cannot be completed."""


def canonical_json_bytes(payload: Any) -> bytes:
    """Return stable UTF-8 JSON bytes without a trailing newline."""
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def read_json(path: str | Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file {path}: {exc}") from exc


def write_json_atomic(path: str | Path, payload: Any) -> None:
    """Write canonical JSON using temp-file, fsync, rename, and directory fsync."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    content = canonical_json_bytes(payload) + b"\n"
    temp_path: Path | None = None
    fd = -1

    try:
        fd, raw_temp_path = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=str(target.parent),
        )
        temp_path = Path(raw_temp_path)
        with os.fdopen(fd, "wb") as file:
            fd = -1
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_path, target)
        temp_path = None
        fsync_directory(target.parent)
    except Exception as exc:
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
        if isinstance(exc, AtomicWriteError):
            raise
        raise AtomicWriteError(f"Failed to atomically write {target}: {exc}") from exc


def fsync_directory(path: str | Path) -> None:
    directory = Path(path)
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
