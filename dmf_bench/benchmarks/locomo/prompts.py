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

"""LoCoMo answerer prompts and answer-target normalization."""

from __future__ import annotations

import json
from typing import Any

ANSWERER_SYSTEM_PROMPT = ""

CATEGORY_2_DATE_INSTRUCTION = (
    "For time or date questions, resolve relative expressions such as yesterday, "
    "last year, this month, and next month against dates or timestamps available "
    "in the native context. Prefer an absolute date, month, or year over a "
    "relative phrase."
)

ANSWERER_USER_PROMPT_TEMPLATE = """
Use the provided native conversational memory context to answer the LoCoMo question.
Give a concise answer. Answer with exact words from the native context whenever possible.
{category_instruction}

Native context:
{native_context}

Question: {question}
Short answer:
""".strip()

NO_INFORMATION_ANSWER = "Not mentioned in the conversation"


def build_answerer_system_prompt() -> str:
    return ANSWERER_SYSTEM_PROMPT


def build_answerer_user_prompt(
    native_context: Any,
    question: str,
    *,
    category: int = 0,
) -> str:
    category_instruction = CATEGORY_2_DATE_INSTRUCTION if category == 2 else ""
    return ANSWERER_USER_PROMPT_TEMPLATE.format(
        category_instruction=category_instruction,
        native_context=_serialize_native_context(native_context),
        question=question,
    )


def _serialize_native_context(native_context: Any) -> str:
    if isinstance(native_context, str):
        return native_context
    return json.dumps(native_context, ensure_ascii=False, indent=2)


def official_ground_truth_answer(category: int, answer: str) -> str:
    """Return the official target used for a LoCoMo question category."""
    if category == 5:
        return NO_INFORMATION_ANSWER
    return answer


def normalize_category_5_prediction(prediction: str, answer: str = "") -> str:
    """Normalize the official adversarial-category multiple-choice surface."""
    normalized = prediction.strip().lower()
    if normalized in {"b", "(b)"} or normalized.startswith("(b)"):
        return NO_INFORMATION_ANSWER
    if normalized in {"a", "(a)"} or normalized.startswith("(a)"):
        return answer
    return prediction
