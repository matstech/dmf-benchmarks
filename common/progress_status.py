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

import copy
import logging
from dataclasses import dataclass, field, asdict
from rich.text import Text


@dataclass
class BenchmarkState:
    total_conversations: int = 0
    completed_conversations: int = 0
    processed: int = 0
    errors: int = 0
    current_phase: str = "init"
    current_conversation_idx: int | None = None
    current_conversation_label: str | None = None
    conversations: list[str] = field(default_factory=list)


class StatefulLogger:
    def __init__(self, logger: logging.Logger):
        self._log = logger
        self._last_snapshot: dict = {}
        self._status_message = "Phase: init | Conversation: -"

    def log_state_delta(self, state: BenchmarkState, label: str = "state update"):
        current = asdict(state)
        current_phase = current.get("current_phase", "init")
        current_conversation_label = current.get("current_conversation_label")
        current_conversation_idx = current.get("current_conversation_idx")
        total_conversations = int(current.get("total_conversations", 0) or 0)

        if current_conversation_label is not None and current_conversation_idx is not None:
            self._status_message = (
                f"Phase: {current_phase} | "
                f"Conversation: {current_conversation_label} "
                f"({current_conversation_idx + 1}/{total_conversations})"
            )
        else:
            self._status_message = f"Phase: {current_phase} | Conversation: completed"

        self._last_snapshot = copy.deepcopy(current)

    def render_status(self) -> Text:
        return Text(self._status_message)

    def set_current_conversation(
        self,
        state: BenchmarkState,
        *,
        phase: str,
        conversation_idx: int | None,
        conversation_label: str | None,
    ) -> None:
        state.current_phase = phase
        state.current_conversation_idx = conversation_idx
        state.current_conversation_label = conversation_label

    def complete_step(self, state: BenchmarkState) -> None:
        state.processed += 1

    def complete_conversation(self, state: BenchmarkState) -> None:
        if state.current_conversation_label is not None:
            state.conversations.append(state.current_conversation_label)
        state.completed_conversations += 1
