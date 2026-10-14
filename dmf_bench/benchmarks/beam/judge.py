"""Strict local BEAM rubric judge; remote-judge parity is a later gate."""

from __future__ import annotations

import json
import re
from typing import Any

from dmf_bench.adapters.base import JudgeRequest
from dmf_bench.fingerprints import judge_fingerprint
from dmf_bench.metrics import BenchmarkMetrics
from dmf_bench.providers.openai_compatible import ProviderResponseError


def parse_rubric_score(text: str) -> tuple[float, str]:
    value = text.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", value, flags=re.IGNORECASE)
    if fence:
        value = fence.group(1)
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ProviderResponseError("BEAM judge response is not valid JSON.") from exc
    if not isinstance(payload, dict) or set(payload) != {"score", "reason"}:
        raise ProviderResponseError("BEAM judge requires score and reason only.")
    score = payload["score"]
    reason = payload["reason"]
    if type(score) not in (int, float) or float(score) not in (0.0, 0.5, 1.0):
        raise ProviderResponseError("BEAM judge score must be 0, 0.5, or 1.")
    if not isinstance(reason, str) or not reason.strip():
        raise ProviderResponseError("BEAM judge reason must be non-empty.")
    return float(score), reason.strip()


class BeamJudgeAdapter:
    name = "beam-rubric-judge"

    def __init__(self, *, settings: Any, transport: Any,
                 metrics: BenchmarkMetrics | None = None) -> None:
        self.settings = settings
        self.transport = transport
        self.metrics = metrics
        self.judge_fingerprint = judge_fingerprint("beam")

    def judge(self, request: JudgeRequest) -> dict[str, Any]:
        prediction = request.prediction
        rubric = prediction["ground_truth_rubric"]
        if not isinstance(rubric, list) or not rubric:
            raise ProviderResponseError("BEAM rubric is empty.")
        scores = []
        reasons = []
        usage = {"prompt_tokens_total": 0, "completion_tokens": 0, "total_tokens": 0}
        model = ""
        for criterion in rubric:
            prompt = (
                "Assess whether the response satisfies this single criterion. "
                "Use 1 for full, 0.5 for partial, or 0 for no compliance. "
                "For a negative constraint, also require a relevant answer. "
                "Accept equivalent wording and numeric forms.\n\n"
                f"Question: {prediction['question']}\n"
                f"Criterion: {criterion}\n"
                f"Response: {prediction['generated_answer']}\n\n"
                'Return JSON only: {"score": 0, "reason": "explanation"}.'
            )
            response = self.transport.generate_with_usage(
                system_prompt="You are a BEAM rubric evaluator.", user_prompt=prompt,
                temperature=self.settings.temperature, max_tokens=self.settings.max_tokens,
                reasoning_effort=self.settings.reasoning_effort,
            )
            try:
                if response.finish_reason not in (None, "stop"):
                    raise ProviderResponseError("Incomplete BEAM judge response.")
                score, reason = parse_rubric_score(response.response)
            except ProviderResponseError:
                self._record("failed", response)
                raise
            self._record("completed", response)
            scores.append(score)
            reasons.append(reason)
            model = response.model
            tokens = response.token_usage
            usage["prompt_tokens_total"] += tokens.prompt_tokens_total
            usage["completion_tokens"] += tokens.completion_tokens
            usage["total_tokens"] += tokens.total_tokens
        average = sum(scores) / len(scores)
        return {
            "judgment": "CORRECT" if average == 1 else "WRONG",
            "score": average, "reason": json.dumps(reasons, ensure_ascii=False),
            "judge_provider": self.settings.provider,
            "judge_requested_model": self.settings.requested_model,
            "judge_model": model, "judge_finish_reason": "stop",
            "judge_usage": usage, "judge_fingerprint": self.judge_fingerprint,
        }

    def _record(self, outcome: str, response: Any) -> None:
        if self.metrics is not None:
            tokens = response.token_usage
            self.metrics.record_llm_request(
                role="judge", provider=self.settings.provider, outcome=outcome,
                prompt_tokens=tokens.prompt_tokens_total,
                completion_tokens=tokens.completion_tokens,
                total_tokens=tokens.total_tokens,
            )
