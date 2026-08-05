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

"""Native LongMemEval end-to-end pipeline surface.

This runner feeds the framework-native memory surface directly to the minimal
native prompt.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Literal
from uuid import uuid4

from rich.console import Console
from rich.logging import RichHandler
from dmf_bench.frameworks.dmf_context import (
    DMF_NATIVE_SURFACE_MARKER,
    build_dmf_native_context_surface,
)
from dmf_bench.frameworks.mem0_runtime import (
    add_memory_internal_usage,
    subtract_memory_internal_usage,
)
from dmf_bench.frameworks.mem0_context import (
    MEM0_NATIVE_SURFACE_MARKER,
    build_mem0_native_context_surface,
)
from dmf_bench.reporting.paths import native_predicted_results_dir
from dmf_bench.reporting.quality import (
    build_native_secondary_rigorous_manifest,
    native_bundle_path,
    native_primary_report_path,
    native_secondary_manifest_path,
)
from dmf_bench.reporting.results import (
    build_native_evaluation_item,
    build_native_results_bundle,
    load_mem0_question_usage,
    load_question_results,
    load_question_results_for_ids,
    normalize_native_memory_internal_usage,
    normalize_native_resource_usage,
    normalize_pipeline_timing,
    normalize_answerer_usage,
    question_result_path,
    save_mem0_question_usage,
    save_question_result,
    total_native_end_to_end_tokens,
)
from dmf_bench.reporting.resources import (
    ProcessResourceSampler,
    detect_machine_characteristics,
    format_machine_characteristics,
)
from longmemeval import native_prompts

MemoryFramework = Literal["dmf", "mem0"]

BENCHMARK_NAME = "longmemeval"
NATIVE_BENCHMARK_RESULTS_NAME = "native/longmemeval"
NATIVE_ENTRYPOINT = "python -m longmemeval.native_pipeline"
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
    """Return the isolated native output directory for one LongMemEval run."""
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
    question: dict[str, Any],
    bundle: Any,
    framework: MemoryFramework,
    retrieval_depth: int,
    dmf_config: Any | None = None,
    embedding_engine: Any | None = None,
) -> Any:
    """Build the framework-native context surface for one LongMemEval question."""
    question_text = str(question.get("question", ""))

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
    question: dict[str, Any],
    bundle: Any,
    framework: MemoryFramework,
    retrieval_depth: int,
    project_name: str | None = None,
    dmf_config: Any | None = None,
    embedding_engine: Any | None = None,
) -> dict[str, Any]:
    """Build the native context and prompt payload for one LongMemEval query."""
    surface = build_native_context_surface_for_question(
        question=question,
        bundle=bundle,
        framework=framework,
        retrieval_depth=retrieval_depth,
        dmf_config=dmf_config,
        embedding_engine=embedding_engine,
    )
    question_text = str(question.get("question", ""))
    question_date = str(question.get("question_date", "") or "")
    system_prompt = native_prompts.build_answerer_system_prompt()
    user_prompt = native_prompts.build_answerer_user_prompt(
        native_context=surface.native_context,
        question=question_text,
        question_date=question_date,
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

    payload = {
        "benchmark": BENCHMARK_NAME,
        "framework": framework,
        "question_id": str(question.get("question_id", "")),
        "question": question_text,
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
    from dmf_bench.providers.openai_compatible import OpenAIClient, resolve_provider_runtime_config

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
    question: dict[str, Any],
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
    """Generate one answer through the separate LongMemEval native path."""
    qa_sampler = ProcessResourceSampler()
    qa_sampler.start()
    try:
        total_start = perf_counter()
        inputs = build_native_answerer_inputs_for_question(
            question=question,
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
        "ground_truth_answer": str(question.get("answer", "")),
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


def _load_native_question_results(
    *,
    project_name: str,
    question_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    if question_ids is None:
        return load_question_results(
            project_name=project_name,
            benchmark_name=NATIVE_BENCHMARK_RESULTS_NAME,
        )
    return load_question_results_for_ids(
        project_name=project_name,
        question_ids=question_ids,
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


def _select_questions(args: argparse.Namespace) -> tuple[str | None, list[dict[str, Any]]]:
    from longmemeval.utils import (
        ensure_longmemeval_dataset,
        filter_questions_by_ids,
        sample_questions_stratified,
    )

    if args.evaluate_only or args.judge_only:
        saved = _load_native_question_results(project_name=args.project_name)
        if args.question_ids:
            requested = {item.strip() for item in args.question_ids.split(",") if item.strip()}
            saved = [item for item in saved if str(item.get("question_id", "")) in requested]
        return None, saved

    dataset_path, all_questions = ensure_longmemeval_dataset(args.dataset_path)
    selected_types = None
    if args.question_types:
        selected_types = [
            item.strip()
            for item in args.question_types.split(",")
            if item.strip()
        ]
    if args.question_ids:
        question_ids = [item.strip() for item in args.question_ids.split(",") if item.strip()]
        return dataset_path, filter_questions_by_ids(all_questions, question_ids)
    if args.all_questions:
        if selected_types:
            selected = [
                question
                for question in all_questions
                if question["question_type"] in set(selected_types)
            ]
            return dataset_path, selected
        return dataset_path, all_questions
    return dataset_path, sample_questions_stratified(
        all_questions,
        per_type=args.per_type,
        seed=args.seed,
        selected_types=selected_types,
    )


def _build_qa_settings(args: argparse.Namespace) -> Any:
    return SimpleNamespace(
        provider=args.answerer_provider,
        model=args.answerer_model,
        base_url=None,
        temperature=0.0,
        max_tokens=args.answerer_max_tokens,
        rpm=args.answerer_rpm,
        timeout=args.answerer_timeout,
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

    from longmemeval import judge
    from longmemeval.native_ingestion import (
        count_question_ingest_units,
        ingest_question,
        load_active_config,
    )

    active_console = console or _build_console()
    load_dotenv()
    config = load_active_config(args)
    dataset_path, questions = _select_questions(args)
    if not questions:
        raise ValueError("No LongMemEval native questions selected.")
    machine_characteristics = detect_machine_characteristics()
    for line in format_machine_characteristics(machine_characteristics):
        active_console.print(line)

    question_ids = [str(question["question_id"]) for question in questions]
    qa_settings = None if args.judge_only or args.evaluate_only else _build_qa_settings(args)
    judge_settings = None
    judge_client = None
    if not args.predict_only and not args.evaluate_only:
        judge_settings = judge.resolve_judge_settings(
            provider_override=args.judge_provider,
            model_override=args.judge_model,
            reasoning_effort=args.judge_reasoning_effort,
        )
        judge_client = judge.build_judge(judge_settings)

    results: list[dict[str, Any]] = []
    mem0_run_id = uuid4().hex[:8] if config.framework == "mem0" else None

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
    rigorous_result: subprocess.CompletedProcess[str] | None = None
    with progress:
        global_task = progress.add_task(
            "LongMemEval native",
            total=max(1, len(questions)),
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
            results = _load_native_question_results(
                project_name=args.project_name,
                question_ids=question_ids if args.question_ids else None,
            )
            if config.framework == "mem0":
                results = _restore_mem0_usage_for_results(
                    results=results,
                    project_name=args.project_name,
                    missing_ok=True,
                )
            progress.update(phase_task, completed=1, visible=False)
            progress.update(global_task, completed=len(questions))
        elif args.judge_only:
            results = _load_native_question_results(
                project_name=args.project_name,
                question_ids=question_ids if args.question_ids else None,
            )
            if config.framework == "mem0":
                results = _restore_mem0_usage_for_results(
                    results=results,
                    project_name=args.project_name,
                    missing_ok=True,
                )
            progress.update(global_task, total=max(1, len(results)), completed=0)
        else:
            for question in questions:
                qid = str(question["question_id"])
                qtype = str(question.get("question_type", ""))
                output_path = _native_question_result_path(
                    project_name=args.project_name,
                    question_id=qid,
                )
                if args.resume and output_path.exists():
                    progress.update(
                        phase_task,
                        description=f"Resume {qid} ({qtype})",
                        total=1,
                        completed=0,
                        visible=True,
                    )
                    result = json.loads(output_path.read_text(encoding="utf-8"))
                    results.append(result)
                    progress.update(phase_task, completed=1)
                    if judge_client is None:
                        progress.advance(global_task)
                    continue

                progress.update(
                    phase_task,
                    description=f"Ingestion {qid} ({qtype})",
                    total=max(1, count_question_ingest_units(question, config.framework)),
                    completed=0,
                    visible=True,
                )
                ingestion_sampler = ProcessResourceSampler()
                ingestion_sampler.start()
                ingestion_start = perf_counter()
                bundle = ingest_question(
                    question,
                    config,
                    project_name=args.project_name,
                    mem0_run_id=mem0_run_id,
                    on_turn_completed=lambda: progress.advance(phase_task),
                )
                ingestion_ms = (perf_counter() - ingestion_start) * 1000
                ingestion_resource = ingestion_sampler.stop(scope="question")
                mem0_previous_usage = None
                mem0_ingestion_usage = None
                if config.framework == "mem0":
                    mem0_previous_usage = _bundle_mem0_backend(bundle).get_usage()
                    mem0_ingestion_usage = mem0_previous_usage

                progress.update(
                    phase_task,
                    description=f"Answerer {qid} ({qtype})",
                    total=1,
                    completed=0,
                    visible=True,
                )
                if qa_settings is None:
                    raise RuntimeError("QA settings are not initialized.")
                result = run_native_answerer_for_question(
                    question=question,
                    bundle=bundle,
                    framework=config.framework,
                    retrieval_depth=config.retrieval_depth,
                    settings=qa_settings,
                    project_name=args.project_name,
                    dmf_config=config.dmf_config,
                    memory_internal_usage_before=mem0_previous_usage,
                    memory_internal_shared_usage=mem0_ingestion_usage,
                )
                pipeline_timing = normalize_pipeline_timing(result.get("pipeline_timing"))
                pipeline_timing["ingestion_ms"] = ingestion_ms
                pipeline_timing["ingestion_scope"] = "question"
                result["pipeline_timing"] = pipeline_timing
                resource_usage = normalize_native_resource_usage(result.get("resource_usage"))
                resource_usage["ingestion"] = ingestion_resource
                result["resource_usage"] = resource_usage
                result["question_type"] = qtype
                result["question_date"] = str(question.get("question_date", ""))
                _save_native_question_result(result, project_name=args.project_name)
                if config.framework == "mem0":
                    save_mem0_question_usage(
                        project_name=args.project_name,
                        question_id=str(result["question_id"]),
                        usage=result.get("memory_internal_usage"),
                        benchmark_name=NATIVE_BENCHMARK_RESULTS_NAME,
                    )
                results.append(result)
                progress.update(phase_task, completed=1)
                if judge_client is None:
                    progress.advance(global_task)

        if judge_client is not None and judge_settings is not None:
            judged_results = []
            for result in results:
                qid = str(result.get("question_id", ""))
                if args.resume and result.get("judge_score") is not None:
                    judged_results.append(result)
                    progress.advance(global_task)
                    continue
                progress.update(
                    phase_task,
                    description=f"Judge {qid}",
                    total=1,
                    completed=0,
                    visible=True,
                )
                judged = judge.judge_one_result(
                    result=result,
                    judge_client=judge_client,
                    settings=judge_settings,
                )
                _save_native_question_result(judged, project_name=args.project_name)
                judged_results.append(judged)
                progress.update(phase_task, completed=1)
                progress.advance(global_task)
            results = judged_results

        progress.update(
            phase_task,
            description="Saving native bundle and reports",
            total=1,
            completed=0,
            visible=True,
        )

        bundle_path = save_native_bundle_and_reports(
            project_name=args.project_name,
            framework=config.framework,
            evaluations=results,
            run_metadata={
                "dataset_path": dataset_path,
                "run_mode": _run_mode(args),
                "framework": args.framework,
                "retrieval_depth": config.retrieval_depth,
                "selected_question_ids": question_ids,
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


def _run_mode(args: argparse.Namespace) -> str:
    if args.predict_only:
        return "predict-only"
    if args.judge_only:
        return "judge-only"
    if args.evaluate_only:
        return "evaluate-only"
    return "full"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the separate native LongMemEval pipeline.",
    )
    parser.add_argument("--project-name", required=True)
    parser.add_argument("--framework", choices=("dmf", "mem0"), default="dmf")
    parser.add_argument(
        "--config",
        help="Framework config path: DMF TOML for dmf, Mem0 YAML for mem0.",
    )
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--all-questions", action="store_true")
    parser.add_argument("--per-type", type=int, default=5)
    parser.add_argument("--question-types", default=None)
    parser.add_argument("--question-ids", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--judge-only", action="store_true")
    parser.add_argument(
        "--answerer-provider",
        default="openai",
        choices=("openai", "openrouter", "ollama"),
    )
    parser.add_argument("--answerer-model", default="gpt-4.1-mini")
    parser.add_argument("--answerer-max-tokens", type=int, default=4096)
    parser.add_argument("--answerer-rpm", type=int, default=200)
    parser.add_argument("--answerer-timeout", type=float, default=120.0)
    parser.add_argument(
        "--judge-provider",
        default=None,
        choices=("openai", "openrouter", "ollama"),
    )
    parser.add_argument("--judge-model", default=None)
    parser.add_argument(
        "--judge-reasoning-effort",
        choices=("low", "medium", "high"),
        default=None,
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
    console.print(f"Native LongMemEval bundle: {bundle_path}")
    if rigorous_result is not None:
        if rigorous_result.stdout:
            console.print(rigorous_result.stdout.rstrip())
        if rigorous_result.returncode != 0:
            if rigorous_result.stderr:
                console.print(rigorous_result.stderr.rstrip(), style="bold red")
            raise SystemExit(rigorous_result.returncode)


if __name__ == "__main__":
    main()
