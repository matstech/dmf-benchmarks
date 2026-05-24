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

"""LongMemEval prompt templates for answerer and judge layers."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from longmemeval.utils import LongMemEvalStrictSession

ANSWERER_SYSTEM_PROMPT = ""


ANSWERER_USER_PROMPT_TEMPLATE = """
I will give you several history chats between you and a user.
Please answer the question based on the relevant chat history.


History Chats:

{context}

Current Date: {question_date}
Question: {question}
Answer:
""".strip()


JUDGE_SYSTEM_PROMPT = (
    "You are grading a LongMemEval answer for semantic correctness. "
    "Return JSON only with the requested format."
)


JUDGE_USER_PROMPT_TEMPLATE = """
Label the generated answer as CORRECT or WRONG.

Question type:
{question_type}

Question date:
{question_date}

## Rules

1. **SEMANTIC EQUIVALENCE FIRST**: Judge by meaning, not exact wording. Paraphrases, equivalent phrasings, and harmless extra detail are CORRECT.

2. **USE THE QUESTION TO RESOLVE THE TARGET FACT**: Read the question carefully and judge whether the model response answers that specific fact asked by the benchmark.

3. **NUMERIC AND TEMPORAL FLEXIBILITY**: Dates can vary in formatting. Relative dates that resolve to the same window are CORRECT. Small rounding or phrasing differences for durations and counts are acceptable when they preserve the same underlying answer.

4. **ABSTENTION MATCHING IS ALLOWED ONLY FOR TRUE ABSTENTIONS**: If the gold answer is genuinely an abstention or says the information is unavailable, then a semantically equivalent refusal is CORRECT. If the gold answer states a factual answer, an abstaining response is WRONG.

5. **EXTRA DETAIL IS FINE**: A longer answer is still CORRECT if it contains the gold answer's core fact and does not contradict it.

6. **SAME REFERENT**: If the generated answer identifies the same person, place, object, event, or concept as the gold answer, mark CORRECT even if it uses different wording or slightly different description.

7. **BE CONSERVATIVE WITH WRONG**: Mark WRONG only when the response misses the target fact, contradicts it, or abstains despite the gold answer being factual.

## ONLY mark WRONG if:
- The generated answer misses or contradicts the gold answer's core fact
- The generated answer abstains while the gold answer is factual and answerable
- The answer addresses a different topic

## Question
Question: {question}
Gold answer: {ground_truth_answer}
Generated answer: {generated_answer}

Return JSON with "reasoning" (one sentence) and "label" (CORRECT or WRONG). Do NOT include both labels.
""".strip()



def build_answerer_user_prompt(
    context: str,
    question: str,
    question_date: str,
) -> str:
    return ANSWERER_USER_PROMPT_TEMPLATE.format(
        context=context,
        question_date=question_date or "(not specified)",
        question=question,
    )


def build_answerer_system_prompt(_question_date: str) -> str:
    return ANSWERER_SYSTEM_PROMPT



def build_judge_user_prompt(
    question: str,
    ground_truth_answer: str,
    generated_answer: str,
    question_type: str = "",
    question_date: str = "",
) -> str:
    return JUDGE_USER_PROMPT_TEMPLATE.format(
        question_type=question_type or "(not specified)",
        question_date=question_date or "(not specified)",
        question=question,
        ground_truth_answer=ground_truth_answer,
        generated_answer=generated_answer,
    )



def format_strict_sessions_as_history_chats(
    strict_sessions: list[LongMemEvalStrictSession],
) -> str:
    """Render original dataset sessions for the strict LongMemEval direct reader."""
    if not strict_sessions:
        return "(No history chats retrieved)"

    lines: list[str] = []
    for index, session in enumerate(strict_sessions, start=1):
        session_date = str(session.get("session_date_raw", "") or "").strip() or "(date unavailable)"
        serialized_session = str(session.get("serialized_session", "") or "").strip()
        if not serialized_session:
            continue

        lines.append(f"### Session {index}:")
        lines.append(f"Session Date: {session_date}")
        lines.append("Session Content:")
        lines.append(serialized_session)

    return "\n".join(lines) if lines else "(No history chats retrieved)"


def format_search_results_as_memory_context(search_results: list[dict[str, Any]]) -> str:
    """Render retrieved memories using LongMemEval's direct reader format."""
    if not search_results:
        return "(No history chats retrieved)"

    def sort_key(item: dict[str, Any]) -> tuple[int, str, str, str]:
        created_at = str(item.get("created_at", "") or "").strip()
        session_id = str(
            (
                item.get("metadata", {}) or {}
            ).get("source_unit_id", "")
            or ""
        ).strip()
        memory = str(item.get("memory", "") or "").strip()
        if not created_at:
            return (1, "", session_id, memory)
        try:
            normalized = datetime.fromisoformat(created_at.replace("Z", "+00:00")).isoformat()
        except ValueError:
            normalized = created_at
        return (0, normalized, session_id, memory)

    ordered_results = sorted(search_results, key=sort_key)
    lines: list[str] = []
    for index, result in enumerate(ordered_results, start=1):
        memory = str(result.get("memory", "") or "").strip()
        if not memory:
            continue

        metadata = result.get("metadata", {}) or {}
        session_date_raw = str(metadata.get("session_date_raw", "") or "").strip()
        created_at = str(result.get("created_at", "") or "").strip()
        session_date = session_date_raw or created_at or "(date unavailable)"

        lines.append(f"### Session {index}:")
        lines.append(f"Session Date: {session_date}")
        lines.append("Session Content:")
        lines.append(json.dumps({"memory": memory}, ensure_ascii=False))

    return "\n".join(lines) if lines else "(No history chats retrieved)"
