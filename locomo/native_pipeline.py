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

"""Native LoCoMo end-to-end pipeline surface.

The strict LoCoMo QA path rebuilds dataset-side dialog context. This module is
the separate native path: it passes the framework-native memory surface to the
minimal native LoCoMo prompt.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Literal
from uuid import uuid4

from rich.console import Console
from rich.logging import RichHandler
from common.dmf_native_context import build_dmf_native_context_surface
from common.mem0_local import (
    add_memory_internal_usage,
    empty_memory_internal_usage,
    normalize_memory_internal_usage,
    subtract_memory_internal_usage,
)
from common.mem0_native_context import (
    MEM0_NATIVE_SURFACE_MARKER,
    build_mem0_native_context_surface,
)
from common.native_paths import native_predicted_results_dir
from common.native_reporting import (
    build_native_secondary_rigorous_manifest,
    native_bundle_path,
    native_primary_report_path,
    native_secondary_manifest_path,
)
from common.results_io import (
    build_native_evaluation_item,
    build_native_results_bundle,
    load_mem0_question_usage,
    load_question_results,
    load_question_results_for_conversation,
    normalize_native_memory_internal_usage,
    normalize_native_resource_usage,
    normalize_pipeline_timing,
    normalize_answerer_usage,
    question_result_path,
    save_mem0_conversation_usage,
    save_mem0_question_usage,
    save_question_result,
    total_native_end_to_end_tokens,
)
from common.system_resources import (
    ProcessResourceSampler,
    detect_machine_characteristics,
    format_machine_characteristics,
)
from locomo import native_prompts

MemoryFramework = Literal["dmf", "mem0"]

BENCHMARK_NAME = "locomo"
NATIVE_BENCHMARK_RESULTS_NAME = "native/locomo"
NATIVE_PROTOCOL_MODE = "native"
NATIVE_ENTRYPOINT = "python -m locomo.native_pipeline"
log = logging.getLogger(__name__)


def _build_console() -> Console:
    return Console()


def _configure_logging(console: Console) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def native_output_dir(project_name: str) -> Path:
    """Return the isolated native output directory for one LoCoMo run."""
    return native_predicted_results_dir(
        project_name=project_name,
        benchmark_name=BENCHMARK_NAME,
    )


def build_native_run_manifest(
    *,
    project_name: str,
    framework: MemoryFramework,
) -> dict[str, Any]:
    """Return the offline manifest for the separate native runner."""
    bundle_path = native_bundle_path(
        benchmark_name=BENCHMARK_NAME,
        project_name=project_name,
    )
    return {
        "benchmark": BENCHMARK_NAME,
        "protocol_mode": NATIVE_PROTOCOL_MODE,
        "framework": framework,
        "entrypoint": NATIVE_ENTRYPOINT,
        "output_dir": str(native_output_dir(project_name)),
        "reporting": {
            "primary_quality_report": str(
                native_primary_report_path(
                    benchmark_name=BENCHMARK_NAME,
                    project_name=project_name,
                )
            ),
            "secondary_rigorous_manifest": str(
                native_secondary_manifest_path(
                    benchmark_name=BENCHMARK_NAME,
                    project_name=project_name,
                )
            ),
            "secondary_rigorous": build_native_secondary_rigorous_manifest(
                benchmark_name=BENCHMARK_NAME,
                input_path=bundle_path,
                project_name=project_name,
            ),
        },
    }


def build_dmf_embedding_engine(config: Any) -> Any:
    """Build the DMF embedding engine lazily to keep native tests offline."""
    from dmf.analysis.embedding_engine import EmbeddingEngine
    from dmf.utils.config import VectorConfig

    return EmbeddingEngine(
        VectorConfig(
            model_name=config.nlp.model_name,
            vector_dim=config.nlp.vector_dim,
            window_size=config.capacity.window_size,
        )
    )


def _bundle_memory_engine(bundle: Any) -> Any:
    if hasattr(bundle, "memory_engine"):
        return bundle.memory_engine
    memory_engine = getattr(bundle, "backend_state", {}).get("memory_engine")
    if memory_engine is None:
        raise RuntimeError("DMF native path requires a memory_engine in the bundle.")
    return memory_engine


def _bundle_mem0_backend(bundle: Any) -> Any:
    backend = getattr(bundle, "backend_state", {}).get("mem0_backend")
    if backend is None:
        raise RuntimeError("Mem0 native path requires a mem0_backend in the bundle.")
    return backend


def _bundle_mem0_user_id(bundle: Any) -> str:
    user_id = getattr(bundle, "backend_state", {}).get("user_id")
    if not isinstance(user_id, str) or not user_id:
        raise RuntimeError("Mem0 native path requires a user_id in the bundle.")
    return user_id


def build_native_context_surface_for_question(
    *,
    qa_item: dict[str, Any],
    bundle: Any,
    framework: MemoryFramework,
    retrieval_depth: int,
    dmf_config: Any | None = None,
    embedding_engine: Any | None = None,
) -> Any:
    """Build the framework-native context surface for one LoCoMo question."""
    question_text = str(qa_item.get("question", ""))

    if framework == "dmf":
        if dmf_config is None:
            raise RuntimeError("DMF native path requires dmf_config.")
        effective_embedding_engine = embedding_engine or build_dmf_embedding_engine(dmf_config)
        from dmf.memory.api import Memory
        memory = Memory.from_dmf_config(
            dmf_config,
            _bundle_memory_engine(bundle),
            effective_embedding_engine,
        )
        return build_dmf_native_context_surface(
            memory=memory,
            query_text=question_text,
            record_index=getattr(bundle, "record_index", {}),
        )

    if framework == "mem0":
        return build_mem0_native_context_surface(
            mem0_backend=_bundle_mem0_backend(bundle),
            query_text=question_text,
            user_id=_bundle_mem0_user_id(bundle),
            top_k=retrieval_depth,
        )

    raise ValueError(f"Unsupported framework: {framework}")


def build_native_answerer_inputs_for_question(
    *,
    conversation_idx: int,
    question_idx: int,
    qa_item: dict[str, Any],
    bundle: Any,
    framework: MemoryFramework,
    retrieval_depth: int,
    project_name: str | None = None,
    dmf_config: Any | None = None,
    embedding_engine: Any | None = None,
) -> dict[str, Any]:
    """Build the native context and prompt payload for one LoCoMo query."""
    surface = build_native_context_surface_for_question(
        qa_item=qa_item,
        bundle=bundle,
        framework=framework,
        retrieval_depth=retrieval_depth,
        dmf_config=dmf_config,
        embedding_engine=embedding_engine,
    )
    question = str(qa_item.get("question", ""))
    category = int(qa_item.get("category", 0) or 0)
    system_prompt = native_prompts.build_answerer_system_prompt()
    user_prompt = native_prompts.build_answerer_user_prompt(
        native_context=surface.native_context,
        question=question,
        category=category,
    )

    raw_retrieval_outputs = (
        surface.raw_search_output
        if surface.surface_marker == MEM0_NATIVE_SURFACE_MARKER
        else surface.raw_retrieval_outputs
    )
    native_surface_diagnostics: dict[str, Any] = {}
    if hasattr(surface, "context_metrics"):
        native_surface_diagnostics["context_metrics"] = dict(surface.context_metrics)
    if hasattr(surface, "recalled_section_present"):
        native_surface_diagnostics["recalled_section_present"] = bool(
            surface.recalled_section_present
        )
    if hasattr(surface, "active_section_present"):
        native_surface_diagnostics["active_section_present"] = bool(
            surface.active_section_present
        )
    if hasattr(surface, "result_count"):
        native_surface_diagnostics["result_count"] = int(surface.result_count)
    if hasattr(surface, "search_kwargs"):
        native_surface_diagnostics["search_kwargs"] = dict(surface.search_kwargs)

    question_id = f"conv{conversation_idx}_q{question_idx}"
    payload = {
        "benchmark": BENCHMARK_NAME,
        "protocol_mode": NATIVE_PROTOCOL_MODE,
        "framework": framework,
        "question_id": question_id,
        "conversation_idx": conversation_idx,
        "question_idx": question_idx,
        "question": question,
        "category": category,
        "surface_marker": surface.surface_marker,
        "native_context": surface.native_context,
        "raw_retrieval_outputs": raw_retrieval_outputs,
        "native_surface_diagnostics": native_surface_diagnostics,
        "system_prompt": system_prompt,
        "task_prompt": user_prompt,
        "retrieval_depth": retrieval_depth,
    }
    if project_name is not None:
        payload["output_dir"] = str(native_output_dir(project_name))
    return payload


def build_answerer(settings: Any) -> Any:
    """Build the answerer lazily; tests can pass a fake answerer directly."""
    from common.openai_client import OpenAIClient, resolve_provider_runtime_config

    api_key, base_url = resolve_provider_runtime_config(settings.provider)
    return OpenAIClient(
        model=settings.model,
        api_key=api_key,
        base_url=base_url,
        timeout=settings.timeout,
        rpm=settings.rpm,
    )


def _normalize_usage(token_usage: Any) -> dict[str, int]:
    return normalize_answerer_usage(
        {
            "prompt_tokens_total": getattr(token_usage, "prompt_tokens_total", 0),
            "completion_tokens": getattr(token_usage, "completion_tokens", 0),
            "total_tokens": getattr(token_usage, "total_tokens", 0),
        }
    )


def _apportion_mem0_usage(
    usage: dict[str, Any] | None,
    *,
    parts: int,
) -> list[dict[str, Any]]:
    """Split shared ingestion usage deterministically across native questions."""
    if parts <= 0:
        return []

    normalized = normalize_memory_internal_usage(usage)
    apportioned = [
        empty_memory_internal_usage(
            available=normalized["available"],
            framework=normalized["framework"],
        )
        for _ in range(parts)
    ]
    for field in ("prompt_tokens", "completion_tokens", "total_tokens", "calls"):
        base, remainder = divmod(int(normalized[field]), parts)
        for index in range(parts):
            apportioned[index][field] = base + (1 if index < remainder else 0)
    return apportioned


def _apply_mem0_usage_to_result(
    result: dict[str, Any],
    usage: dict[str, Any] | None,
) -> dict[str, Any]:
    """Overwrite one native result with restored or newly observed Mem0 usage."""
    raw_existing = result.get("memory_internal_usage")
    diagnostics = None
    if isinstance(raw_existing, dict):
        raw_diagnostics = raw_existing.get("observed_diagnostics")
        if isinstance(raw_diagnostics, dict):
            diagnostics = raw_diagnostics

    normalized_usage = normalize_native_memory_internal_usage(
        usage,
        framework="mem0",
        diagnostics=diagnostics,
    )
    result["memory_internal_usage"] = normalized_usage
    result["total_end_to_end_tokens"] = total_native_end_to_end_tokens(
        answerer_usage=result.get("answerer_usage"),
        memory_internal_usage=normalized_usage,
    )
    return result


def _restore_mem0_usage_for_results(
    *,
    results: list[dict[str, Any]],
    project_name: str,
    missing_ok: bool = False,
) -> list[dict[str, Any]]:
    """Restore Mem0 per-question accounting sidecars into native result items."""
    restored: list[dict[str, Any]] = []
    for result in results:
        if str(result.get("framework", "")) != "mem0":
            restored.append(result)
            continue

        question_id = str(result.get("question_id", ""))
        if not question_id:
            restored.append(result)
            continue

        try:
            usage = load_mem0_question_usage(
                project_name=project_name,
                question_id=question_id,
                benchmark_name=NATIVE_BENCHMARK_RESULTS_NAME,
            )
        except FileNotFoundError:
            if not missing_ok:
                raise
            log.warning(
                "Missing native Mem0 usage sidecar for question %s; token accounting "
                "will remain partial for this result.",
                question_id,
            )
            restored.append(result)
            continue

        restored.append(_apply_mem0_usage_to_result(result, usage))
    return restored


def run_native_answerer_for_question(
    *,
    conversation_idx: int,
    question_idx: int,
    qa_item: dict[str, Any],
    bundle: Any,
    framework: MemoryFramework,
    retrieval_depth: int,
    settings: Any,
    project_name: str | None = None,
    dmf_config: Any | None = None,
    embedding_engine: Any | None = None,
    answerer: Any | None = None,
    memory_internal_usage_before: dict[str, Any] | None = None,
    memory_internal_shared_usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate one answer through the separate LoCoMo native path."""
    qa_sampler = ProcessResourceSampler()
    qa_sampler.start()
    try:
        total_start = perf_counter()
        inputs = build_native_answerer_inputs_for_question(
            conversation_idx=conversation_idx,
            question_idx=question_idx,
            qa_item=qa_item,
            bundle=bundle,
            framework=framework,
            retrieval_depth=retrieval_depth,
            project_name=project_name,
            dmf_config=dmf_config,
            embedding_engine=embedding_engine,
        )
        retrieval_done = perf_counter()
        effective_answerer = answerer if answerer is not None else build_answerer(settings)
        answerer_start = perf_counter()
        response = effective_answerer.generate_with_usage(
            system_prompt=inputs["system_prompt"],
            user_prompt=inputs["task_prompt"],
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )
        answerer_done = perf_counter()
        memory_internal_usage = {"framework": framework, "available": False}
        if framework == "mem0":
            current_usage = _bundle_mem0_backend(bundle).get_usage()
            incremental_usage = subtract_memory_internal_usage(
                current_usage,
                memory_internal_usage_before,
            )
            memory_internal_usage = add_memory_internal_usage(
                memory_internal_shared_usage,
                incremental_usage,
            )
    finally:
        qa_resource_usage = qa_sampler.stop(scope="question")
    base_result = {
        **inputs,
        "ground_truth_answer": str(qa_item.get("answer", "")),
        "evidence": list(qa_item.get("evidence", [])),
        "generated_answer": response.response,
        "answerer_provider": settings.provider,
        "answerer_model": response.model,
    }
    return build_native_evaluation_item(
        base_result=base_result,
        native_context=inputs["native_context"],
        task_prompt=inputs["task_prompt"],
        surface_marker=inputs["surface_marker"],
        raw_retrieval_outputs=inputs["raw_retrieval_outputs"],
        answerer_usage=_normalize_usage(response.token_usage),
        memory_internal_usage=memory_internal_usage,
        memory_diagnostics=inputs.get("native_surface_diagnostics"),
        latency_breakdown={
            "framework_retrieval_ms": (retrieval_done - total_start) * 1000,
            "answerer_ms": (answerer_done - answerer_start) * 1000,
            "total_end_to_end_ms": (answerer_done - total_start) * 1000,
        },
        pipeline_timing={
            "qa_ms": (answerer_done - total_start) * 1000,
            "qa_scope": "question",
        },
        resource_usage={"qa": qa_resource_usage},
    )


def _native_question_result_path(*, project_name: str, question_id: str) -> Path:
    return question_result_path(
        project_name=project_name,
        question_id=question_id,
        benchmark_name=NATIVE_BENCHMARK_RESULTS_NAME,
    )


def _save_native_question_result(result: dict[str, Any], *, project_name: str) -> Path:
    return save_question_result(
        result,
        project_name=project_name,
        benchmark_name=NATIVE_BENCHMARK_RESULTS_NAME,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def save_native_bundle_and_reports(
    *,
    project_name: str,
    framework: MemoryFramework,
    evaluations: list[dict[str, Any]],
    run_metadata: dict[str, Any],
) -> Path:
    bundle_path = native_bundle_path(
        benchmark_name=BENCHMARK_NAME,
        project_name=project_name,
    )
    bundle = build_native_results_bundle(
        project_name=project_name,
        benchmark_name=BENCHMARK_NAME,
        evaluations=evaluations,
        framework=framework,
        run_metadata=run_metadata,
        results_path=bundle_path,
    )
    _write_json(bundle_path, bundle)
    _write_json(
        native_primary_report_path(
            benchmark_name=BENCHMARK_NAME,
            project_name=project_name,
        ),
        bundle["primary_quality_report"],
    )
    _write_json(
        native_secondary_manifest_path(
            benchmark_name=BENCHMARK_NAME,
            project_name=project_name,
        ),
        bundle["secondary_rigorous"],
    )
    return bundle_path


def run_secondary_reports(
    bundle_path: Path,
    *,
    project_name: str,
) -> subprocess.CompletedProcess[str]:
    manifest = build_native_secondary_rigorous_manifest(
        benchmark_name=BENCHMARK_NAME,
        input_path=bundle_path,
        project_name=project_name,
    )
    rigorous_command = [sys.executable, *manifest["commands"]["evaluate_rigorous"][1:]]
    rigorous = subprocess.run(
        rigorous_command,
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    stdout_path = Path(manifest["stdout_capture"]["evaluate_rigorous"])
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text((rigorous.stdout or "") + (rigorous.stderr or ""), encoding="utf-8")
    ablation_command = [sys.executable, *manifest["commands"]["evaluate_ablation"][1:]]
    ablation = subprocess.run(
        ablation_command,
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    if ablation.returncode != 0:
        log.warning("Native ablation evaluator exited with code %s", ablation.returncode)
    return rigorous


def parse_conversation_ids(raw: str) -> list[int] | None:
    if not raw.strip():
        return None
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_categories(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("at least one category must be specified")
    return values


def _run_mode(args: argparse.Namespace) -> str:
    if args.predict_only:
        return "predict-only"
    if args.judge_only:
        return "judge-only"
    if args.evaluate_only:
        return "evaluate-only"
    return "full"


def _load_saved_conversation_results(
    *,
    project_name: str,
    conversation_idx: int,
) -> list[dict[str, Any]]:
    results = load_question_results_for_conversation(
        project_name=project_name,
        conversation_idx=conversation_idx,
        benchmark_name=NATIVE_BENCHMARK_RESULTS_NAME,
    )
    return _restore_mem0_usage_for_results(
        results=results,
        project_name=project_name,
        missing_ok=True,
    )


def run_native_benchmark(
    args: argparse.Namespace,
    *,
    console: Console | None = None,
) -> tuple[Path, subprocess.CompletedProcess[str] | None]:
    from dotenv import load_dotenv
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    from locomo import judge, qa
    from locomo.pipeline import Pipeline

    active_console = console or _build_console()
    load_dotenv()
    conversation_indices = parse_conversation_ids(args.conversation_ids)
    categories = parse_categories(args.categories)
    machine_characteristics = detect_machine_characteristics()
    phase = Pipeline(
        project_name=args.project_name,
        framework=args.framework,
        config_path=args.config,
        conversation_indices=conversation_indices,
        categories=categories,
        resume=args.resume,
        answerer_provider=args.answerer_provider,
        answerer_model=args.answerer_model,
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
        judge_reasoning_effort=args.judge_reasoning_effort,
    )
    if phase.framework == "mem0" and phase.run_id is None:
        phase.run_id = uuid4().hex[:8]
    for line in format_machine_characteristics(machine_characteristics):
        active_console.print(line)

    qa_settings = None
    if not args.judge_only and not args.evaluate_only:
        qa_settings = qa.resolve_qa_settings(
            provider_override=args.answerer_provider,
            model_override=args.answerer_model,
        )
    judge_settings = None
    judge_client = None
    if not args.predict_only and not args.evaluate_only:
        judge_settings = judge.resolve_judge_settings(
            provider_override=args.judge_provider,
            model_override=args.judge_model,
            reasoning_effort=args.judge_reasoning_effort,
        )
        judge_client = judge.build_judge(judge_settings)

    evaluations: list[dict[str, Any]] = []
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        transient=False,
        console=active_console,
        redirect_stdout=True,
        redirect_stderr=True,
    )
    if args.evaluate_only or args.judge_only:
        steps_per_conversation = 1
    elif args.predict_only:
        steps_per_conversation = 2
    else:
        steps_per_conversation = 3
    rigorous_result: subprocess.CompletedProcess[str] | None = None
    with progress:
        global_task = progress.add_task(
            "LoCoMo native",
            total=max(1, len(phase.conversation_indices) * steps_per_conversation),
        )
        phase_task = progress.add_task(
            "Waiting...",
            total=1,
            visible=False,
        )
        if args.evaluate_only:
            progress.update(
                phase_task,
                description="Loading saved native outputs",
                total=1,
                completed=0,
                visible=True,
            )
            evaluations = load_question_results(
                project_name=args.project_name,
                benchmark_name=NATIVE_BENCHMARK_RESULTS_NAME,
            )
            allowed_conversations = set(phase.conversation_indices)
            allowed_categories = set(phase.categories)
            evaluations = [
                item
                for item in evaluations
                if int(item.get("conversation_idx", -1)) in allowed_conversations
                and int(item.get("category", 0) or 0) in allowed_categories
            ]
            evaluations = _restore_mem0_usage_for_results(
                results=evaluations,
                project_name=args.project_name,
                missing_ok=True,
            )
            progress.update(phase_task, completed=1, visible=False)
            progress.update(global_task, completed=len(phase.conversation_indices))
        else:
            for conversation_idx in phase.conversation_indices:
                conversation = phase.dataset[conversation_idx]
                question_items = phase._enumerate_conversation_questions(conversation)
                conversation_label = f"conv_{conversation_idx}"
                if args.judge_only:
                    progress.update(
                        phase_task,
                        description=f"Loading {conversation_label}",
                        total=1,
                        completed=0,
                        visible=True,
                    )
                    conversation_results = _load_saved_conversation_results(
                        project_name=args.project_name,
                        conversation_idx=conversation_idx,
                    )
                    progress.update(phase_task, completed=1)
                else:
                    expected_ids = [
                        f"conv{conversation_idx}_q{question_idx}"
                        for question_idx, _ in question_items
                    ]
                    if args.resume and all(
                        _native_question_result_path(
                            project_name=args.project_name,
                            question_id=question_id,
                        ).exists()
                        for question_id in expected_ids
                    ):
                        progress.update(
                            phase_task,
                            description=f"Resume {conversation_label}",
                            total=1,
                            completed=0,
                            visible=True,
                        )
                        conversation_results = _load_saved_conversation_results(
                            project_name=args.project_name,
                            conversation_idx=conversation_idx,
                        )
                        progress.update(phase_task, completed=1)
                        progress.advance(global_task, 2)
                    else:
                        progress.update(
                            phase_task,
                            description=f"Ingestion {conversation_label}",
                            total=max(1, phase._count_conversation_turns(conversation)),
                            completed=0,
                            visible=True,
                        )
                        ingestion_sampler = ProcessResourceSampler()
                        ingestion_sampler.start()
                        ingestion_start = perf_counter()
                        bundle = phase.process_one_locomo_conversation(
                            conversation_idx=conversation_idx,
                            conversation=conversation,
                            on_turn_completed=lambda: progress.advance(phase_task),
                        )
                        ingestion_ms = (perf_counter() - ingestion_start) * 1000
                        ingestion_resource = ingestion_sampler.stop(scope="conversation")
                        progress.advance(global_task)
                        if qa_settings is None:
                            raise RuntimeError("QA settings are not initialized.")
                        conversation_results = []
                        mem0_previous_usage = None
                        mem0_ingestion_shares: list[dict[str, Any]] = []
                        if phase.framework == "mem0":
                            mem0_previous_usage = _bundle_mem0_backend(bundle).get_usage()
                            mem0_ingestion_shares = _apportion_mem0_usage(
                                mem0_previous_usage,
                                parts=len(question_items),
                            )
                        progress.update(
                            phase_task,
                            description=f"Answerer {conversation_label}",
                            total=max(1, len(question_items)),
                            completed=0,
                            visible=True,
                        )
                        for usage_index, (question_idx, qa_item) in enumerate(question_items):
                            result = run_native_answerer_for_question(
                                conversation_idx=conversation_idx,
                                question_idx=question_idx,
                                qa_item=qa_item,
                                bundle=bundle,
                                framework=phase.framework,
                                retrieval_depth=phase.retrieval_depth,
                                settings=qa_settings,
                                project_name=args.project_name,
                                dmf_config=phase.config,
                                memory_internal_usage_before=mem0_previous_usage,
                                memory_internal_shared_usage=(
                                    mem0_ingestion_shares[usage_index]
                                    if phase.framework == "mem0"
                                    and usage_index < len(mem0_ingestion_shares)
                                    else None
                                ),
                            )
                            pipeline_timing = normalize_pipeline_timing(result.get("pipeline_timing"))
                            pipeline_timing["ingestion_ms"] = ingestion_ms
                            pipeline_timing["ingestion_scope"] = "conversation"
                            result["pipeline_timing"] = pipeline_timing
                            resource_usage = normalize_native_resource_usage(
                                result.get("resource_usage")
                            )
                            resource_usage["ingestion"] = ingestion_resource
                            result["resource_usage"] = resource_usage
                            result["category_name"] = str(qa_item.get("category", ""))
                            _save_native_question_result(result, project_name=args.project_name)
                            if phase.framework == "mem0":
                                save_mem0_question_usage(
                                    project_name=args.project_name,
                                    question_id=str(result["question_id"]),
                                    usage=result.get("memory_internal_usage"),
                                    benchmark_name=NATIVE_BENCHMARK_RESULTS_NAME,
                                )
                                mem0_previous_usage = _bundle_mem0_backend(bundle).get_usage()
                            conversation_results.append(result)
                            progress.advance(phase_task)
                        if phase.framework == "mem0":
                            save_mem0_conversation_usage(
                                project_name=args.project_name,
                                conversation_idx=conversation_idx,
                                usage=_bundle_mem0_backend(bundle).get_usage(),
                                benchmark_name=NATIVE_BENCHMARK_RESULTS_NAME,
                            )
                        progress.advance(global_task)

                if judge_client is not None and judge_settings is not None:
                    judged_results = []
                    progress.update(
                        phase_task,
                        description=f"Judge {conversation_label}",
                        total=max(1, len(conversation_results)),
                        completed=0,
                        visible=True,
                    )
                    for result in conversation_results:
                        if args.resume and result.get("judge_score") is not None:
                            judged_results.append(result)
                            progress.advance(phase_task)
                            continue
                        judged = judge.judge_one_result(
                            result=result,
                            judge_client=judge_client,
                            settings=judge_settings,
                        )
                        _save_native_question_result(judged, project_name=args.project_name)
                        judged_results.append(judged)
                        progress.advance(phase_task)
                    conversation_results = judged_results
                    progress.advance(global_task)
                evaluations.extend(conversation_results)

        if not evaluations:
            raise ValueError("No LoCoMo native evaluation items selected.")

        progress.update(
            phase_task,
            description="Saving native bundle and reports",
            total=1,
            completed=0,
            visible=True,
        )
        bundle_path = save_native_bundle_and_reports(
            project_name=args.project_name,
            framework=phase.framework,
            evaluations=evaluations,
            run_metadata={
                "dataset_path": phase.dataset_path,
                "run_mode": _run_mode(args),
                "framework": phase.framework,
                "retrieval_depth": phase.retrieval_depth,
                "conversation_ids": phase.conversation_indices,
                "categories": phase.categories,
                "answerer_provider": None if qa_settings is None else qa_settings.provider,
                "answerer_model": None if qa_settings is None else qa_settings.model,
                "judge_provider": None if judge_settings is None else judge_settings.provider,
                "judge_model": None if judge_settings is None else judge_settings.model,
                "machine_characteristics": machine_characteristics,
            },
        )
        progress.update(phase_task, completed=1)
        if not args.predict_only and not args.skip_secondary_reporting:
            progress.update(
                phase_task,
                description="Secondary rigorous reports",
                total=1,
                completed=0,
                visible=True,
            )
            rigorous_result = run_secondary_reports(
                bundle_path,
                project_name=args.project_name,
            )
            progress.update(phase_task, completed=1)
        progress.update(phase_task, visible=False)
    return bundle_path, rigorous_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the separate native LoCoMo pipeline.",
    )
    parser.add_argument("--project-name", required=True)
    parser.add_argument("--framework", choices=("dmf", "mem0"), default="dmf")
    parser.add_argument(
        "--config",
        help="Path to the framework config file (DMF TOML or Mem0 YAML).",
    )
    parser.add_argument(
        "--conversation-ids",
        default="",
        help="Comma-separated LoCoMo conversation indices to run.",
    )
    parser.add_argument(
        "--categories",
        default="1,2,3,4,5",
        help="Comma-separated LoCoMo categories to evaluate.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--judge-only", action="store_true")
    parser.add_argument(
        "--answerer-provider",
        choices=("openai", "openrouter", "ollama"),
    )
    parser.add_argument("--answerer-model")
    parser.add_argument(
        "--judge-provider",
        choices=("openai", "openrouter", "ollama"),
    )
    parser.add_argument("--judge-model")
    parser.add_argument(
        "--judge-reasoning-effort",
        choices=("low", "medium", "high"),
    )
    parser.add_argument(
        "--skip-secondary-reporting",
        action="store_true",
        help="Do not launch rigorous/ablation secondary reports after bundling.",
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Print the native runner manifest without executing provider calls.",
    )
    args = parser.parse_args()
    if sum(bool(flag) for flag in (args.predict_only, args.evaluate_only, args.judge_only)) > 1:
        parser.error("--predict-only, --evaluate-only, and --judge-only are mutually exclusive.")
    if not args.describe and not args.config:
        parser.error("--config is required unless --describe is used.")
    return args


def main() -> None:
    console = _build_console()
    _configure_logging(console)
    args = parse_args()
    manifest = build_native_run_manifest(
        project_name=args.project_name,
        framework=args.framework,
    )
    if args.describe:
        console.print(json.dumps(manifest, indent=2))
        return
    bundle_path, rigorous_result = run_native_benchmark(args, console=console)
    console.print(f"Native LoCoMo bundle: {bundle_path}")
    if rigorous_result is not None:
        if rigorous_result.stdout:
            console.print(rigorous_result.stdout.rstrip())
        if rigorous_result.returncode != 0:
            if rigorous_result.stderr:
                console.print(rigorous_result.stderr.rstrip(), style="bold red")
            raise SystemExit(rigorous_result.returncode)


if __name__ == "__main__":
    main()
