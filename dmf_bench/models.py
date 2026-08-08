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
from typing import Any, Literal

MemoryFramework = Literal["dmf", "mem0"]


@dataclass
class IngestedConversationBundle:
    """Conversation-level ingestion output reused by later benchmark phases."""

    conversation_idx: int
    framework: MemoryFramework
    backend_state: dict[str, Any]
    record_index: dict[str, dict[str, Any]]


@dataclass
class IngestedQuestionBundle:
    """Per-question ingestion output for LongMemEval."""

    question_id: str
    framework: MemoryFramework
    backend_state: dict[str, Any]
    record_index: dict[str, dict[str, Any]]


@dataclass
class TokenUsage:
    """Normalized token accounting for one LLM call."""

    prompt_tokens_total: int
    completion_tokens: int
    total_tokens: int


@dataclass
class LLMResponse:
    """LLM output envelope with normalized usage and optional raw payload."""

    response: str
    token_usage: TokenUsage
    model: str
    finish_reason: str | None = None
    raw: dict[str, Any] | None = None


@dataclass(frozen=True)
class QASettings:
    """Resolved answerer settings shared by the QA phase."""

    provider: str
    model: str
    base_url: str | None
    temperature: float
    max_tokens: int
    rpm: int
    timeout: float


@dataclass(frozen=True)
class JudgeSettings:
    """Resolved judge settings shared by the evaluation phase."""

    provider: str
    model: str
    base_url: str | None
    max_tokens: int
    rpm: int
    timeout: float
    reasoning_effort: str | None = None
