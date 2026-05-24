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

"""
Prompt definitions for the benchmark pipeline.

Answer generation follows LoCoMo's official short-answer QA prompt while
remaining identical across memory backends.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

ANSWERER_SYSTEM_PROMPT = ""

NO_INFORMATION_ANSWER = "Not mentioned in the conversation"
CATEGORY_2_DATE_INSTRUCTION = (
    "Use the Session Date of the conversation to answer with an approximate date."
)

QA_PROMPT = """
Based on the above context, write an answer in the form of a short phrase for the following question.
Answer with exact words from the context whenever possible. Question: {question} Short answer:
""".strip()

QA_PROMPT_CAT_5 = """
Based on the above context, answer the following question. Question: {question} Select the correct answer: (a) {answer} (b) {no_information}.
Short answer:
""".strip()


ANSWERER_USER_PROMPT_TEMPLATE = """
{context}

{qa_prompt}
""".strip()


def build_answerer_user_prompt(
    context: str,
    question: str,
    *,
    category: int = 0,
    ground_truth_answer: str = "",
) -> str:
    normalized_question = question
    if category == 2:
        normalized_question = f"{question} {CATEGORY_2_DATE_INSTRUCTION}"

    if category == 5:
        qa_prompt = QA_PROMPT_CAT_5.format(
            question=normalized_question,
            answer=ground_truth_answer,
            no_information=NO_INFORMATION_ANSWER,
        )
    else:
        qa_prompt = QA_PROMPT.format(question=normalized_question)

    return ANSWERER_USER_PROMPT_TEMPLATE.format(
        context=context,
        qa_prompt=qa_prompt,
    )


def official_ground_truth_answer(category: int, answer: str) -> str:
    """Return the answer target used by LoCoMo's official QA protocol."""
    if category == 5:
        return NO_INFORMATION_ANSWER
    return answer


def normalize_category_5_prediction(prediction: str, answer: str = "") -> str:
    """Map option-only adversarial outputs back to the selected answer text."""
    normalized = prediction.strip().lower()
    if normalized in {"b", "(b)"} or normalized.startswith("(b)"):
        return NO_INFORMATION_ANSWER
    if normalized in {"a", "(a)"} or normalized.startswith("(a)"):
        return answer
    return prediction


def format_search_results_as_memory_context(search_results: list[dict[str, Any]]) -> str:
    if not search_results:
        return "(No relevant context found)"

    def sort_key(item: dict[str, Any]) -> tuple[int, str]:
        created_at = str(item.get("created_at", "") or "").strip()
        if not created_at:
            return (1, "")
        try:
            normalized = datetime.fromisoformat(created_at.replace("Z", "+00:00")).isoformat()
        except ValueError:
            normalized = created_at
        return (0, normalized)

    ordered_results = sorted(search_results, key=sort_key)
    blocks: list[str] = []
    for index, result in enumerate(ordered_results, start=1):
        memory = str(result.get("memory", "") or "").strip()
        if not memory:
            continue
        created_at = str(result.get("created_at", "") or "").strip()
        date_label = created_at or f"retrieved memory {index}"
        blocks.append(f"DATE: {date_label}\nCONVERSATION:\n{memory}")

    return "\n\n".join(blocks) if blocks else "(No relevant context found)"


def format_strict_dialogs_as_context(serialized_dialogs: list[str]) -> str:
    """Render the LoCoMo strict reader context from dataset-side dialogs."""
    dialogs = [dialog.strip() for dialog in serialized_dialogs if dialog and dialog.strip()]
    if not dialogs:
        return "(No relevant context found)"
    return "\n\n".join(dialogs)


JUDGE_SYSTEM_PROMPT = "You are evaluating conversational AI memory recall. Return JSON only with the format requested."


JUDGE_USER_PROMPT_TEMPLATE = """
Label the generated answer as CORRECT or WRONG.

## Rules

1. **PARTIAL CREDIT**: If the generated answer includes AT LEAST ONE correct item from the gold answer's list, mark CORRECT. Getting 1 out of 2, 2 out of 4, etc. is acceptable. Only mark WRONG if NONE of the gold answer items appear.

2. **PARAPHRASES COUNT**: Same concept in different words is CORRECT. Judge semantic meaning, not exact wording.

3. **EXTRA DETAIL IS FINE**: A longer answer that includes the gold answer's key facts plus additional information is CORRECT. Never penalize for being more detailed or specific.

4. **DATE TOLERANCE**: Dates within 14 days of each other are CORRECT. Durations within 50% are CORRECT. Relative dates that point to the same time window are CORRECT.

5. **ABSTENTION IS USUALLY WRONG**: If the gold answer states a factual answer, then responses like "I don't know", "not enough information", or other abstentions are WRONG. Treat an abstention as CORRECT only when the gold answer itself is genuinely unanswerable or abstaining.

6. **SEMANTIC OVERLAP**: Judge whether the generated answer addresses the same topic and captures the core idea of the gold answer. Different wording, phrasing, or level of detail should not result in WRONG if the underlying concept matches.

7. **SAME REFERENT**: If the generated answer identifies the same named entity, person, character, place, or concept as the gold answer, mark CORRECT even if it gives a different description or extra detail.

8. **FOCUS ON KNOWLEDGE, NOT WORDING**: The goal is to assess whether the system recalled the right fact. Minor differences in specificity, phrasing, or scope should not result in WRONG. Only mark WRONG when the generated answer demonstrates a genuinely different or incorrect understanding.

## ONLY mark WRONG if:
- The generated answer contains ZERO correct items from the gold answer
- The generated answer abstains while the gold answer is factual and answerable
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
) -> str:
    return JUDGE_USER_PROMPT_TEMPLATE.format(
        question=question,
        ground_truth_answer=ground_truth_answer,
        generated_answer=generated_answer,
    )
