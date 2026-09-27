"""Deterministic rendering and token budgeting for v3 retrieval context."""

from __future__ import annotations

import hashlib
import codecs
from dataclasses import dataclass
from typing import Any

from .adapters.base import CanonicalRetrievalResult, RetrievedMemory


TOKENIZER_ID = "tiktoken-cl100k_base-v1"
RENDERER_ID = "retrieved-memory-v1"
PACKING_ID = "ranked-whole-items-v1"
_SEPARATOR = "\n\n"


@dataclass(frozen=True)
class PackedContext:
    """The exact text sent to the answerer and its auditable packing record."""

    text: str
    returned_ids: tuple[str, ...]
    included_ids: tuple[str, ...]
    excluded_ids: tuple[str, ...]
    original_tokens: int
    included_tokens: int
    cuts: tuple[dict[str, Any], ...]
    tokenizer_id: str
    renderer_id: str
    packing_id: str
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "returned_ids": list(self.returned_ids),
            "included_ids": list(self.included_ids),
            "excluded_ids": list(self.excluded_ids),
            "original_tokens": self.original_tokens,
            "included_tokens": self.included_tokens,
            "cuts": [dict(cut) for cut in self.cuts],
            "tokenizer_id": self.tokenizer_id,
            "renderer_id": self.renderer_id,
            "packing_id": self.packing_id,
            "sha256": self.sha256,
        }


def render_memory(item: RetrievedMemory, *, content: str | None = None) -> str:
    """Render one whole item with stable, framework-neutral labels."""
    lines = [f"[Memory {item.rank}]", f"ID: {item.memory_id}"]
    if item.occurred_at is not None:
        lines.append(f"Occurred at: {item.occurred_at}")
    if item.source_event_ids:
        lines.append(f"Source events: {', '.join(item.source_event_ids)}")
    lines.extend(("Content:", item.content if content is None else content))
    return "\n".join(lines)


def _encoding(tokenizer_id: str) -> Any:
    if tokenizer_id != TOKENIZER_ID:
        raise ValueError(f"Unsupported tokenizer: {tokenizer_id!r}.")
    try:
        import tiktoken
    except ImportError as exc:
        raise RuntimeError("tiktoken is required for v3 context packing.") from exc
    return tiktoken.get_encoding("cl100k_base")


def _token_count(encoding: Any, text: str) -> int:
    return len(encoding.encode(text, disallowed_special=()))


def _truncate_first(item: RetrievedMemory, budget: int, encoding: Any) -> tuple[str, dict[str, Any]]:
    """Keep the largest tokenizer prefix of the first item's content that fits."""
    header = render_memory(item, content="")
    if _token_count(encoding, header) > budget:
        raise ValueError("Context budget cannot fit the first memory's labels and provenance.")
    content_tokens = encoding.encode(item.content, disallowed_special=())
    utf8_decoder = codecs.getincrementaldecoder("utf-8")("strict")
    valid_prefixes = [0]
    for length, token in enumerate(content_tokens, start=1):
        utf8_decoder.decode(encoding.decode_single_token_bytes(token))
        if not utf8_decoder.getstate()[0]:
            valid_prefixes.append(length)
    low, high = 0, len(valid_prefixes) - 1
    while low < high:
        middle = (low + high + 1) // 2
        candidate = encoding.decode(content_tokens[:valid_prefixes[middle]], errors="strict")
        if _token_count(encoding, render_memory(item, content=candidate)) <= budget:
            low = middle
        else:
            high = middle - 1
    kept_tokens = valid_prefixes[low]
    shortened = encoding.decode(content_tokens[:kept_tokens], errors="strict")
    rendered = render_memory(item, content=shortened)
    if _token_count(encoding, rendered) > budget:
        raise ValueError("Tokenizer truncation exceeded the context budget.")
    return rendered, {
        "memory_id": item.memory_id,
        "original_content_tokens": len(content_tokens),
        "included_content_tokens": kept_tokens,
        "removed_content_tokens": len(content_tokens) - kept_tokens,
    }


def pack_context(
    retrieval: CanonicalRetrievalResult,
    max_tokens: int,
    *,
    tokenizer_id: str = TOKENIZER_ID,
    renderer_id: str = RENDERER_ID,
    packing_id: str = PACKING_ID,
) -> PackedContext:
    """Include ranked whole items until the next fails; truncate only item one."""
    if not isinstance(retrieval, CanonicalRetrievalResult):
        raise ValueError("retrieval must be a CanonicalRetrievalResult.")
    if type(max_tokens) is not int or max_tokens < 0:
        raise ValueError("max_tokens must be a non-negative integer.")
    if renderer_id != RENDERER_ID or packing_id != PACKING_ID:
        raise ValueError("Unsupported context renderer or packing policy.")
    encoding = _encoding(tokenizer_id)
    items = retrieval.items
    rendered_all = _SEPARATOR.join(render_memory(item) for item in items)
    original_tokens = _token_count(encoding, rendered_all)
    included: list[str] = []
    text = ""
    cuts: list[dict[str, Any]] = []

    for index, item in enumerate(items):
        rendered = render_memory(item)
        candidate = _SEPARATOR.join((text, rendered)) if included else rendered
        if _token_count(encoding, candidate) <= max_tokens:
            text = candidate
            included.append(item.memory_id)
            continue
        if index == 0:
            text, cut = _truncate_first(item, max_tokens, encoding)
            included.append(item.memory_id)
            cuts.append(cut)
        break

    returned_ids = tuple(item.memory_id for item in items)
    included_ids = tuple(included)
    return PackedContext(
        text=text,
        returned_ids=returned_ids,
        included_ids=included_ids,
        excluded_ids=returned_ids[len(included_ids):],
        original_tokens=original_tokens,
        included_tokens=_token_count(encoding, text),
        cuts=tuple(cuts),
        tokenizer_id=tokenizer_id,
        renderer_id=renderer_id,
        packing_id=packing_id,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )
