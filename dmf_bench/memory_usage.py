"""Shared accounting for internal memory-model calls."""

from __future__ import annotations

from typing import Any


def apportion_memory_internal_usage(
    usage: dict[str, Any], item_ids: tuple[str, ...], query_id: str,
) -> dict[str, Any]:
    """Assign one unit's ingestion cost to its questions without rounding loss."""
    if query_id not in item_ids or len(set(item_ids)) != len(item_ids):
        raise ValueError("Usage allocation requires a unique question in the unit.")
    share = dict(usage)
    index = item_ids.index(query_id)
    for field in ("prompt_tokens", "completion_tokens", "total_tokens", "calls"):
        value = share[field]
        if type(value) is not int or value < 0:
            raise ValueError("Usage counters must be non-negative integers.")
        base, remainder = divmod(value, len(item_ids))
        share[field] = base + (index < remainder)
    return share
