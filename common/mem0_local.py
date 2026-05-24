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

"""Local in-process Mem0 integration pinned to the repo-local fork."""

from __future__ import annotations

import logging
import re
import sys
from hashlib import sha1
from datetime import datetime, timezone
from inspect import Parameter, signature
from pathlib import Path
from typing import Any

from .mem0_config import Mem0Config, build_mem0_runtime_config

logger = logging.getLogger(__name__)


def _timestamp_to_created_at(timestamp: int | None) -> str | None:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(int(timestamp), tz=timezone.utc).isoformat()


def empty_memory_internal_usage(
    available: bool = False,
    framework: str | None = None,
) -> dict[str, Any]:
    """Init memory_usage: DMF will not populate any of these counters"""
    return {
        "framework": framework,
        "available": available,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "calls": 0,
    }


def normalize_memory_internal_usage(data: dict[str, Any] | None) -> dict[str, Any]:
    usage = empty_memory_internal_usage()
    if not isinstance(data, dict):
        return usage

    raw_framework = data.get("framework")
    usage["framework"] = str(raw_framework) if raw_framework is not None else None
    usage["available"] = bool(data.get("available", True))
    usage["prompt_tokens"] = int(data.get("prompt_tokens", 0) or 0)
    usage["completion_tokens"] = int(data.get("completion_tokens", 0) or 0)
    usage["total_tokens"] = int(data.get("total_tokens", 0) or 0)
    usage["calls"] = int(data.get("calls", 0) or 0)
    return usage


def add_memory_internal_usage(*usages: dict[str, Any] | None) -> dict[str, Any]:
    total = empty_memory_internal_usage()
    for usage in usages:
        normalized = normalize_memory_internal_usage(usage)
        if total["framework"] is None and normalized["framework"] is not None:
            total["framework"] = normalized["framework"]
        total["available"] = total["available"] or normalized["available"]
        total["prompt_tokens"] += normalized["prompt_tokens"]
        total["completion_tokens"] += normalized["completion_tokens"]
        total["total_tokens"] += normalized["total_tokens"]
        total["calls"] += normalized["calls"]
    return total


def subtract_memory_internal_usage(
    usage: dict[str, Any] | None,
    baseline: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return one normalized non-negative delta between cumulative Mem0 snapshots."""
    current = normalize_memory_internal_usage(usage)
    previous = normalize_memory_internal_usage(baseline)
    framework = current["framework"] or previous["framework"]
    available = current["available"] or previous["available"]
    delta = empty_memory_internal_usage(
        available=available,
        framework=framework,
    )
    for field in ("prompt_tokens", "completion_tokens", "total_tokens", "calls"):
        delta[field] = max(0, int(current[field]) - int(previous[field]))
    return delta


def _minimal_search_metadata(metadata: Any) -> dict[str, Any] | None:
    if not isinstance(metadata, dict):
        return None

    allowed_keys = (
        "benchmark",
        "conversation_idx",
        "question_id",
        "source_unit_type",
        "source_unit_id",
        "source_unit_ids",
        "session_id",
        "session_date_raw",
    )
    minimal: dict[str, Any] = {}
    for key in allowed_keys:
        value = metadata.get(key)
        if value is None:
            continue
        if key == "source_unit_ids" and isinstance(value, list):
            filtered = [item for item in value if item is not None]
            if filtered:
                minimal[key] = filtered
            continue
        minimal[key] = value
    return minimal or None


def _safe_storage_fragment(value: Any) -> str:
    text = str(value).strip().lower()
    safe = re.sub(r"[^a-z0-9._-]+", "_", text)
    safe = safe.strip("._-")
    return safe or "item"


def _build_collection_name(
    *,
    benchmark_name: str,
    project_name: str,
    storage_label: str,
    max_length: int = 54,
) -> str:
    """Build a Chroma-safe Mem0 collection name.

    Mem0 creates derived collections such as ``<collection>_entities``.  Keep
    the base name short enough for those derived names to remain Chroma-valid.
    """
    parts = (
        "mem0",
        _safe_storage_fragment(benchmark_name),
        _safe_storage_fragment(project_name),
        _safe_storage_fragment(storage_label),
    )
    collection_name = "_".join(part for part in parts if part)
    if len(collection_name) > max_length:
        digest = sha1(collection_name.encode("utf-8")).hexdigest()[:8]
        prefix = collection_name[: max(1, max_length - len(digest) - 1)]
        collection_name = f"{prefix.rstrip('_-')}_{digest}"
    return collection_name


def _mem0_search_limit_kwargs(memory: Any, top_k: int) -> dict[str, int]:
    """Return search limit kwargs for the installed Mem0 API variant."""
    try:
        parameters = signature(memory.search).parameters
    except (TypeError, ValueError):
        return {"limit": top_k}

    accepts_top_k = "top_k" in parameters
    accepts_limit = "limit" in parameters
    accepts_arbitrary_kwargs = any(
        parameter.kind is Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    if accepts_top_k and not accepts_limit and not accepts_arbitrary_kwargs:
        return {"top_k": top_k}
    return {"limit": top_k}


class LocalMem0BenchmarkItemBackend:
    """One local Mem0 OSS memory instance isolated per benchmark item."""

    def __init__(
        self,
        *,
        benchmark_name: str,
        project_name: str,
        item_kind: str,
        item_id: str | int,
        config: Mem0Config,
        storage_label: str | None = None,
    ):
        self.benchmark_name = benchmark_name
        self.project_name = project_name
        self.item_kind = item_kind
        self.item_id = str(item_id)
        self.config = config
        self.storage_label = (
            storage_label
            if storage_label is not None
            else f"{_safe_storage_fragment(item_kind)}_{_safe_storage_fragment(item_id)}"
        )
        self.storage_root = (
            Path("results")
            / benchmark_name
            / ".mem0_local"
            / project_name
            / self.storage_label
        )
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.memory = self._build_memory()
        self.memory.reset()
        self.memory.reset_llm_usage()
        logger.info(
            (
                "Mem0 local backend initialized | benchmark=%s item_kind=%s item_id=%s "
                "store=%s config_schema=%s"
            ),
            benchmark_name,
            item_kind,
            self.item_id,
            self.storage_root,
            self.config.schema_version,
        )

    def _build_memory(self) -> Any:
        import mem0
        from mem0 import Memory

        runtime_config = build_mem0_runtime_config(
            self.config,
            collection_name=_build_collection_name(
                benchmark_name=self.benchmark_name,
                project_name=self.project_name,
                storage_label=self.storage_label,
            ),
            storage_path=str(self.storage_root / "chroma"),
            history_db_path=str(self.storage_root / "history.db"),
        )
        try:
            return Memory.from_config(runtime_config)
        except Exception as exc:
            embedder = runtime_config.get("embedder") or {}
            provider = embedder.get("provider")
            if (
                provider == "fastembed"
                and "Unsupported embedding provider: fastembed" in str(exc)
            ):
                raise RuntimeError(
                    "The active Python interpreter is importing a Mem0 package "
                    "that does not support embedder.provider='fastembed'. "
                    "This benchmark expects the repo Poetry environment with the "
                    "pinned Mem0 fork. Re-run with `poetry run python -m "
                    "longmemeval.pipeline ...`. "
                    f"Interpreter: {sys.executable}. "
                    f"Imported mem0 module: {getattr(mem0, '__file__', '<unknown>')}."
                ) from exc
            raise

    def add(
        self,
        messages: list[dict[str, str]],
        *,
        user_id: str,
        timestamp: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        effective_metadata = dict(metadata) if isinstance(metadata, dict) else {}
        created_at = _timestamp_to_created_at(timestamp)
        if created_at is not None:
            effective_metadata.setdefault("created_at", created_at)

        kwargs: dict[str, Any] = {
            "user_id": user_id,
            "metadata": effective_metadata or None,
        }
        return self.memory.add(messages, **kwargs)

    def search(
        self,
        query: str,
        *,
        user_id: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Return canonical retrieval provenance for benchmark-side strict reconstruction.

        The `memory` field preserves Mem0's native surface for audit/debug only.
        Benchmark strict readers should rebuild their final context from
        `metadata.source_unit_id` / `metadata.source_unit_ids`, not from the raw
        text returned here.
        """
        response = self.memory.search(
            query,
            **_mem0_search_limit_kwargs(self.memory, top_k),
            filters={"user_id": user_id},
        )
        results = response.get("results", response) if isinstance(response, dict) else response
        if not isinstance(results, list):
            return []

        normalized: list[dict[str, Any]] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            entry: dict[str, Any] = {
                "memory": str(item.get("memory", "") or ""),
                "score": float(item.get("score", 0.0) or 0.0),
                "id": str(item.get("id", "") or ""),
            }
            created_at = item.get("created_at")
            if created_at:
                entry["created_at"] = created_at
            updated_at = item.get("updated_at")
            if updated_at:
                entry["updated_at"] = updated_at
            metadata = _minimal_search_metadata(item.get("metadata"))
            if metadata:
                entry["metadata"] = metadata
            normalized.append(entry)

        normalized.sort(key=lambda result: float(result.get("score", 0.0) or 0.0), reverse=True)
        return normalized

    def search_raw(
        self,
        query: str,
        *,
        user_id: str,
        top_k: int,
    ) -> Any:
        """Return the raw Mem0 search response for native end-to-end evaluation."""
        return self.memory.search(
            query,
            **_mem0_search_limit_kwargs(self.memory, top_k),
            filters={"user_id": user_id},
        )

    def get_usage(self) -> dict[str, Any]:
        return normalize_memory_internal_usage(
            {
                "framework": "mem0",
                "available": True,
                **self.memory.get_llm_usage(),
            }
        )


class LocalMem0ConversationBackend(LocalMem0BenchmarkItemBackend):
    """Backward-compatible Mem0 backend isolated per LoCoMo conversation."""

    def __init__(
        self,
        *,
        project_name: str,
        conversation_idx: int,
        config: Mem0Config,
    ):
        self.conversation_idx = conversation_idx
        super().__init__(
            benchmark_name="locomo",
            project_name=project_name,
            item_kind="conv",
            item_id=conversation_idx,
            config=config,
            storage_label=f"conv_{conversation_idx}",
        )
