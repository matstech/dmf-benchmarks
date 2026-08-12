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

"""Shared LLM-as-judge prompt definitions for benchmark runners."""

from __future__ import annotations

JUDGE_SYSTEM_PROMPT = (
    "You are evaluating conversational AI memory recall. "
    "Return JSON only with the format requested."
)

JUDGE_RETRY_INSTRUCTION = (
    "\n\nThe previous response was invalid or incomplete. Return one complete JSON "
    'object only, with exactly "reasoning" and "label"; label must be CORRECT or WRONG.'
)


def _optional_context_block(
    *,
    question_type: str = "",
    question_date: str = "",
) -> str:
    lines: list[str] = []
    normalized_question_type = question_type.strip()
    normalized_question_date = question_date.strip()

    if normalized_question_type:
        lines.extend(("Question type:", normalized_question_type, ""))
    if normalized_question_date:
        lines.extend(("Question date:", normalized_question_date, ""))

    return "\n".join(lines)


JUDGE_USER_PROMPT_TEMPLATE = """
Label the generated answer as CORRECT or WRONG.

{context_block}

## Rules

1. **PARTIAL CREDIT**: If the generated answer includes AT LEAST ONE correct item from the gold answer's list, mark CORRECT. Getting 1 out of 2, 2 out of 4, etc. is always acceptable. Only mark WRONG if NONE of the gold answer items appear.

2. **PARAPHRASES COUNT**: Same concept in different words is CORRECT. Judge semantic meaning, not exact wording.

3. **EXTRA DETAIL IS FINE**: A longer answer that includes the gold answer's key facts plus additional information is CORRECT. Never penalize for being more detailed or specific.

4. **DATE TOLERANCE**: Dates within 14 days of each other are CORRECT. Durations within 50% are CORRECT. Relative dates that point to the same time window are CORRECT.

5. **ABSTENTION MATCHING**: If the gold answer is an abstention or indicates the information is unavailable, any semantically equivalent refusal to answer is CORRECT.

6. **SEMANTIC OVERLAP**: Judge whether the generated answer addresses the same topic and captures the core idea of the gold answer. Different wording, phrasing, or level of detail should not result in WRONG if the underlying concept matches.

7. **SAME REFERENT**: If the generated answer identifies the same named entity, person, character, place, or concept as the gold answer, mark CORRECT even if it gives a different description or extra detail.

8. **FOCUS ON KNOWLEDGE, NOT WORDING**: The goal is to assess whether the system recalled the right fact. Minor differences in specificity, phrasing, or scope should not result in WRONG. Only mark WRONG when the generated answer demonstrates a genuinely different or incorrect understanding.

## ONLY mark WRONG if:
- The generated answer contains ZERO correct items from the gold answer
- The answer addresses a completely different topic

## Question
Question: {question}
Gold answer: {ground_truth_answer}
Generated answer: {generated_answer}

Return JSON with "reasoning" (one sentence) and "label" (CORRECT or WRONG). Do NOT include both labels.
""".strip()


def build_judge_user_prompt(
    question: str,
    ground_truth_answer: str,
    generated_answer: str,
    question_type: str = "",
    question_date: str = "",
) -> str:
    return JUDGE_USER_PROMPT_TEMPLATE.format(
        context_block=_optional_context_block(
            question_type=question_type,
            question_date=question_date,
        ),
        question=question,
        ground_truth_answer=ground_truth_answer,
        generated_answer=generated_answer,
    )
