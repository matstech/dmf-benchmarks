"""Contracts shared by orchestration adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol


class CommandRunner(Protocol):
    """Execute one external command and return its process exit code."""

    def __call__(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> int: ...
