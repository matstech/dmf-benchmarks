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

"""Minimal native LongMemEval answerer prompts."""

from __future__ import annotations

import json
from typing import Any

ANSWERER_SYSTEM_PROMPT = ""

ANSWERER_USER_PROMPT_TEMPLATE = """
Use the provided native memory context to answer the LongMemEval question.
If the answer is not supported by the context, say you do not know.

Native context:
{native_context}

Question date: {question_date}
Question: {question}
Answer:
""".strip()


def build_answerer_system_prompt() -> str:
    return ANSWERER_SYSTEM_PROMPT


def build_answerer_user_prompt(
    native_context: Any,
    question: str,
    question_date: str = "",
) -> str:
    return ANSWERER_USER_PROMPT_TEMPLATE.format(
        native_context=_serialize_native_context(native_context),
        question_date=question_date or "(not specified)",
        question=question,
    )


def _serialize_native_context(native_context: Any) -> str:
    if isinstance(native_context, str):
        return native_context
    return json.dumps(native_context, ensure_ascii=False, indent=2)
