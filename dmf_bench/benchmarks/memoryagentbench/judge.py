"""Official deterministic task metrics and local LongMemEval judge adapter."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from dmf_bench.adapters.base import JudgeRequest
from dmf_bench.contracts import sha256_file
from dmf_bench.fingerprints import judge_fingerprint
from dmf_bench.metrics import BenchmarkMetrics
from dmf_bench.providers.openai_compatible import ProviderResponseError

from .evaluation import deterministic_score


def longmem_prompt(prediction: dict[str, Any]) -> str:
    question_type = prediction.get("question_type")
    if prediction.get("is_abstention"):
        instruction = "Decide whether the response correctly recognizes that the question is unanswerable. Reply yes or no."
    elif question_type == "temporal-reasoning":
        instruction = "Decide whether the response gives the correct answer. Allow an off-by-one error in a duration. Reply yes or no."
    elif question_type == "knowledge-update":
        instruction = "Decide whether the response includes the updated answer, even if it also mentions older information. Reply yes or no."
    elif question_type == "single-session-preference":
        instruction = "Decide whether the response uses the relevant personal information in the rubric. Reply yes or no."
    else:
        instruction = "Decide whether the response includes the complete correct answer or an equivalent one. Reply yes or no."
    return (
        f"{instruction}\n\nQuestion: {prediction['question']}\n"
        f"Reference: {prediction['ground_truth_answer']}\n"
        f"Model response: {prediction['generated_answer']}\n\n"
        'Return JSON only: {"answer":"yes"} or {"answer":"no"}.'
    )


def parse_yes_no(value: str) -> bool:
    text = value.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        text = str(payload.get("answer", "")).strip()
    if re.fullmatch("yes", text, flags=re.IGNORECASE):
        return True
    if re.fullmatch("no", text, flags=re.IGNORECASE):
        return False
    raise ProviderResponseError("MemoryAgentBench LongMemEval judge must return yes or no.")


class MemoryAgentBenchJudgeAdapter:
    name = "memoryagentbench-task-judge"

    def __init__(self, *, settings: Any, transport: Any, config: dict[str, Any],
                 metrics: BenchmarkMetrics | None = None) -> None:
        self.settings = settings
        self.transport = transport
        self.metrics = metrics
        self.judge_fingerprint = judge_fingerprint("memoryagentbench")
        manifest_path = Path(config["dataset"]["path"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entity_path = manifest_path.parent / "entity2id.json"
        if not entity_path.is_file() or sha256_file(entity_path) != manifest["entity2id_sha256"]:
            raise ValueError("MemoryAgentBench entity mapping is absent or has the wrong SHA-256.")
        entities = json.loads(entity_path.read_text(encoding="utf-8"))
        self.movie_candidates = tuple(_movie_name(key) for key in entities)

    def judge(self, request: JudgeRequest) -> dict[str, Any]:
        prediction = request.prediction
        task = str(prediction["task"])
        if task != "longmemeval":
            result = deterministic_score(
                task, str(prediction["generated_answer"]),
                prediction["ground_truth_answer"],
                movie_candidates=self.movie_candidates if task == "recsys" else (),
            )
            score = float(result["score"])
            return self._result(score, "official-deterministic-v1", {},
                                f"{result['metric']}={score}")
        response = self.transport.generate_with_usage(
            system_prompt="Evaluate the answer using the specified MemoryAgentBench task rule.",
            user_prompt=longmem_prompt(prediction),
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
            reasoning_effort=self.settings.reasoning_effort,
        )
        try:
            if response.finish_reason not in (None, "stop"):
                raise ProviderResponseError("Incomplete LongMemEval judge response.")
            correct = parse_yes_no(response.response)
        except ProviderResponseError:
            self._record("failed", response)
            raise
        self._record("completed", response)
        usage = response.token_usage
        return self._result(float(correct), response.model, {
            "prompt_tokens_total": usage.prompt_tokens_total,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }, "LongMemEval yes/no judge")

    def _result(self, score: float, model: str, usage: dict[str, int], reason: str) -> dict[str, Any]:
        return {
            "judgment": "CORRECT" if score >= 1.0 else "WRONG",
            "score": score,
            "reason": reason,
            "judge_provider": self.settings.provider,
            "judge_requested_model": self.settings.requested_model,
            "judge_model": model,
            "judge_finish_reason": None if model == "official-deterministic-v1" else "stop",
            "judge_usage": usage,
            "judge_fingerprint": self.judge_fingerprint,
        }

    def _record(self, outcome: str, response: Any) -> None:
        if self.metrics is not None:
            usage = response.token_usage
            self.metrics.record_llm_request(
                role="judge", provider=self.settings.provider, outcome=outcome,
                prompt_tokens=usage.prompt_tokens_total,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
            )


def _movie_name(value: str) -> str:
    filename = value.split("/")[-1]
    filename = filename.replace("_", " ").replace("-", " ").replace(">", " ")
    return " ".join(re.sub(r"\([^()]*\)", "", filename).split())
