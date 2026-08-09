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

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MEM0_NATIVE_SURFACE_MARKER = "mem0_search_surface"


@dataclass(frozen=True)
class Mem0NativeContextSurface:
    """Mem0 native context built from the memory surface returned by search()."""

    native_context: list[Any]
    raw_search_output: Any
    surface_marker: str
    search_kwargs: dict[str, Any]
    result_count: int


def extract_mem0_search_surface(raw_search_output: Any) -> list[Any]:
    """Extract Mem0's search result surface without benchmark-side rebuilding."""
    if isinstance(raw_search_output, dict):
        surface = raw_search_output.get("results", [])
    else:
        surface = raw_search_output

    if surface is None:
        return []
    if not isinstance(surface, list):
        raise ValueError("Mem0 search surface must be a list of search results.")
    return list(surface)


def build_mem0_native_context_surface_from_search_output(
    raw_search_output: Any,
    *,
    search_kwargs: dict[str, Any] | None = None,
) -> Mem0NativeContextSurface:
    """Build the canonical Mem0 native answerer surface from raw search output.

    The native benchmark path keeps the `search()` results as the answerer
    memory surface. It does not sort by score, map provenance to dataset
    records, or render a synthetic benchmark-side context.
    """
    native_context = extract_mem0_search_surface(raw_search_output)
    return Mem0NativeContextSurface(
        native_context=native_context,
        raw_search_output=raw_search_output,
        surface_marker=MEM0_NATIVE_SURFACE_MARKER,
        search_kwargs=dict(search_kwargs or {}),
        result_count=len(native_context),
    )


def build_mem0_native_context_surface(
    *,
    mem0_backend: Any,
    query_text: str,
    user_id: str,
    top_k: int,
) -> Mem0NativeContextSurface:
    """Invoke Mem0 search and keep its native result surface for the answerer."""
    search_kwargs = {
        "query": query_text,
        "user_id": user_id,
        "top_k": top_k,
    }

    if hasattr(mem0_backend, "search_raw"):
        raw_search_output = mem0_backend.search_raw(
            query_text,
            user_id=user_id,
            top_k=top_k,
        )
    else:
        raw_search_output = mem0_backend.search(
            query_text,
            user_id=user_id,
            top_k=top_k,
        )

    return build_mem0_native_context_surface_from_search_output(
        raw_search_output,
        search_kwargs=search_kwargs,
    )
