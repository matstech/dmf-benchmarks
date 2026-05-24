# Copyright (c) 2026-present matstech
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT

"""Mem0 OSS benchmark configuration loader.

The YAML file follows Mem0's native MemoryConfig schema for the
memory-engine sections (vector_store, llm, embedder, version).
The benchmark adds a top-level ``search`` section with ``top_k``
to control the retrieval operating point.

Runtime-only fields (collection_name, storage path, history_db_path)
are injected by the benchmark harness and must NOT appear in the
config file.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Mem0Config:
    """Parsed Mem0 benchmark configuration."""

    top_k: int
    memory_config: dict[str, Any]

    @property
    def schema_version(self) -> str:
        return str(self.memory_config.get("version", "v1.1"))


def load_mem0_config(path: str | Path) -> Mem0Config:
    """Load and validate a Mem0 benchmark YAML configuration file."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Mem0 config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Mem0 config must be a YAML mapping, got {type(raw).__name__}")

    # --- benchmark-specific section ---
    search = raw.get("search")
    if not isinstance(search, dict) or "top_k" not in search:
        raise ValueError("Mem0 config must contain a 'search' section with 'top_k'")

    top_k = int(search["top_k"])
    if top_k <= 0:
        raise ValueError(f"search.top_k must be a positive integer, got {top_k}")

    # --- native Mem0 MemoryConfig sections ---
    # Everything except 'search' is passed through to Memory.from_config().
    memory_config = {k: v for k, v in raw.items() if k != "search"}

    return Mem0Config(
        top_k=top_k,
        memory_config=memory_config,
    )


def build_mem0_runtime_config(
    config: Mem0Config,
    *,
    collection_name: str,
    storage_path: str,
    history_db_path: str,
) -> dict[str, Any]:
    """Merge the static Mem0 config with runtime-only fields.

    Returns a dict ready for ``Memory.from_config()``.
    """
    merged = copy.deepcopy(config.memory_config)

    # Inject runtime vector_store paths
    vs = merged.setdefault("vector_store", {})
    vs_config = vs.setdefault("config", {})
    vs_config["collection_name"] = collection_name
    vs_config["path"] = storage_path

    # Inject runtime history db path
    merged["history_db_path"] = history_db_path

    return merged
