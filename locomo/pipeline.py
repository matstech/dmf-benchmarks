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
    This module relates to the ingestion part of the benchmark pipeline.
    DMF is a framework totally different with respect the others based on
    distinct phase (e.g. add, search etc).
    
    For this reason, the ingestion is handled by using directly TemporalMemory that is not
    directly exposed by DMF common interface.
    
"""

import argparse
import os
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path
from collections.abc import Callable
from typing import Any

os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")

from dmf.runtime.pipeline import InteractionPipeline
from dmf.runtime.pipeline import InteractionProvenance
from dmf.analysis.scoring_engine import ScoringEngine
from dmf.analysis.nlp_engine import NLPEngine
from dmf.memory.chroma_ltm import ChromaLTMHook
from dmf.memory.temporal_memory import TemporalMemory
from dmf.utils.config import NLPConfig, VectorConfig
from dmf.utils.config_loader import load_dmf_config, DMFConfig
from common.mem0_config import Mem0Config, load_mem0_config
from common.mem0_local import (
    LocalMem0ConversationBackend,
    add_memory_internal_usage,
    empty_memory_internal_usage,
)
from common.models import IngestedConversationBundle, MemoryFramework
from common.openai_client import resolve_provider_runtime_config
from common import progress_status, results_io
from . import judge, qa, utils
import logging
from dotenv import load_dotenv
from rich.progress import Progress, SpinnerColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.logging import RichHandler
from rich.console import Console, Group

console = Console()

# Setup logging con RichHandler (compatibile con progress bar)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
log = logging.getLogger("ingestion_phase")
logging.getLogger("chromadb.telemetry.product.posthog").disabled = True
logging.getLogger("posthog").disabled = True
logging.getLogger("mem0").setLevel(logging.WARNING)

DEFAULT_STRICT_CATEGORIES = [1, 2, 3, 4, 5]


def _namespaced_cards_collection_name(base_name: str, conversation_idx: int) -> str:
    """Return the per-conversation cards collection name for one LoCoMo run."""
    return f"{base_name}_conv_{conversation_idx}"[:63]


def _namespaced_cards_path(
    base_path: str | Path | None,
    *,
    conversation_idx: int,
    persist_directory: str | Path,
) -> Path:
    """Return the per-conversation cards audit path for one LoCoMo run."""
    path = Path(base_path) if base_path is not None else Path(persist_directory) / "ltm_cards.jsonl"
    suffix = path.suffix or ".jsonl"
    return path.with_name(f"{path.stem}_conv_{conversation_idx}{suffix}")


def validate_startup_secrets_for_mode(
    *,
    predict_only: bool,
    judge_only: bool,
    answerer_provider: str | None,
    judge_provider: str | None,
) -> None:
    """Fail fast on missing provider runtime config for the phases that will run."""
    need_answerer = not judge_only
    need_judge = not predict_only

    if need_answerer:
        resolve_provider_runtime_config(answerer_provider or "openai")

    if need_judge:
        resolve_provider_runtime_config(judge_provider or "openai")


def print_startup_configuration(
    *,
    project_name: str,
    framework: MemoryFramework,
    config_path: str,
    dataset_path: str,
    prediction_artifacts_path: Path,
    run_mode: str,
    retrieval_depth: int,
    answerer_provider: str | None,
    answerer_model: str | None,
    judge_provider: str | None,
    judge_model: str | None,
    categories: list[int],
    conversation_indices: list[int],
) -> None:
    """Print the main runtime configuration resolved for a LOCOMO run."""
    console.print("[bold]LOCOMO startup configuration[/bold]")
    console.print(f"  Run mode: [bold]{run_mode}[/bold]")
    console.print(f"  Project: [bold]{project_name}[/bold]")
    console.print(f"  Framework: [bold]{framework}[/bold]")
    console.print(f"  Framework config: {Path(config_path).resolve()}")
    console.print(f"  Dataset file: {Path(dataset_path).resolve()}")
    console.print(f"  Prediction artifacts: {prediction_artifacts_path.resolve()}")
    console.print(f"  Retrieval depth: [bold]{retrieval_depth}[/bold]")
    console.print(
        "  Answerer: "
        f"{answerer_model or '<unused>'} ({answerer_provider or '<unused>'})"
    )
    console.print(
        "  Judge: "
        f"{judge_model or '<unused>'} ({judge_provider or '<unused>'})"
    )
    console.print(
        "  Categories: " + ", ".join(str(category) for category in categories)
    )
    console.print(
        f"  Conversations selected: [bold]{len(conversation_indices)}[/bold]"
    )


class StatusProgress(Progress):
    """
        Defines the minimal set of tools for monitoring progress in benchmark execution (visually)
    """
    def __init__(self, status_provider, *args, **kwargs):
        self._status_provider = status_provider
        super().__init__(*args, **kwargs)

    def get_renderables(self):
        yield Group(self._status_provider(), self.make_tasks_table(self.tasks))


def print_storage_reset_banner(
    *,
    framework: MemoryFramework,
    resume: bool,
    dmf_config: DMFConfig | None,
    project_name: str,
) -> None:
    """Print a startup reminder about cleaning local vector stores on fresh runs."""
    if resume:
        return

    if framework == "dmf":
        chroma_path = "<unknown>"
        if dmf_config is not None:
            chroma_path = str(Path(dmf_config.ltm.chroma_path).resolve())
        predicted_path = (
            Path("results") / "locomo" / f"predicted_{project_name}"
        ).resolve()
        console.print(
            "[bold yellow]Benchmark reminder[/bold yellow]: "
            "you are starting a DMF benchmark without `--resume`. "
            f"If you want a fresh run, make sure you have cleaned the local Chroma directory `{chroma_path}` "
            f"and the project results/checkpoints directory `{predicted_path}`."
        )
        return

    if framework == "mem0":
        mem0_storage = (
            Path("results") / "locomo" / ".mem0_local" / project_name
        ).resolve()
        predicted_path = (
            Path("results") / "locomo" / f"predicted_{project_name}"
        ).resolve()
        console.print(
            "[bold yellow]Benchmark reminder[/bold yellow]: "
            "you are starting a Mem0 benchmark without `--resume`. "
            f"If you want a fresh run, make sure you have cleaned the local storage `{mem0_storage}` "
            f"and the project results/checkpoints directory `{predicted_path}`."
        )

class Pipeline:
    """
        Defines functions and tool for orchestrating the entire benchmark flow
    """
    def __init__(
        self,
        project_name: str,
        framework: MemoryFramework = "dmf",
        config_path: str | None = None,
        conversation_indices: list[int] | None = None,
        categories: list[int] | None = None,
        resume: bool = False,
        answerer_provider: str | None = None,
        answerer_model: str | None = None,
        judge_provider: str | None = None,
        judge_model: str | None = None,
        judge_reasoning_effort: str | None = None,
    ):
        if project_name is None or project_name == "":
            raise ValueError("no project_name specified")
        self.project_name = project_name
        self.framework = framework
        self.config: DMFConfig | None = None
        self.mem0_config: Mem0Config | None = None
        
        if self.framework == "dmf":
            if not config_path:
                raise ValueError("--config is required")
            # toml config utility directly imported from DMF runtime
            self.config = load_dmf_config(path=config_path)
        elif self.framework == "mem0":
            if not config_path:
                raise ValueError("--config is required")
            # mem0 yaml config load
            self.mem0_config = load_mem0_config(path=config_path)
            
        self.resume = resume
        self.answerer_provider = answerer_provider
        self.answerer_model = answerer_model
        self.judge_provider = judge_provider
        self.judge_model = judge_model
        self.judge_reasoning_effort = judge_reasoning_effort
        self.run_id: str | None = None
        self.memory_internal_usage = empty_memory_internal_usage(framework=self.framework)
        
        print_storage_reset_banner(
            framework=self.framework,
            resume=self.resume,
            dmf_config=self.config,
            project_name=self.project_name,
        )
        
        # locomo dataset get or download
        dataset_path = utils.get_locomo_dataset_path()
        self.dataset_path = str(Path(dataset_path).resolve())
        if Path(dataset_path).exists():
            print(f"Dataset already exists locally: {dataset_path}")
            self.dataset = utils.load_locomo_dataset()
        else:
            self.dataset = utils.download_locomo_dataset()
            
        self.conversation_indices = conversation_indices or list(range(len(self.dataset)))
        self.categories = categories or list(DEFAULT_STRICT_CATEGORIES)
        self.state = progress_status.BenchmarkState(total_conversations=len(self.conversation_indices))
        self.state_log = progress_status.StatefulLogger(log)

    @property
    def retrieval_depth(self) -> int:
        """Return the retrieval operating point from the active framework config."""
        if self.framework == "dmf":
            if self.config is None:
                raise RuntimeError("DMF config not loaded.")
            return int(self.config.ltm.recall_limit)
        if self.framework == "mem0":
            if self.mem0_config is None:
                raise RuntimeError("Mem0 config not loaded.")
            return self.mem0_config.top_k
        raise ValueError(f"Unsupported framework: {self.framework}")

    def _preflight_prediction_run(self) -> None:
        """Refuse fresh prediction runs on top of existing checkpoints."""
        existing = [
            conversation_idx
            for conversation_idx in self.conversation_indices
            if results_io.is_conversation_predicted(
                project_name=self.project_name,
                conversation_idx=conversation_idx,
                benchmark_name="locomo",
            )
        ]
        if not existing:
            return

        raise ValueError(
            "Fresh prediction runs require clean checkpoints for the selected "
            f"conversations. Found existing prediction artifacts for: {existing}. "
            "Use --resume to continue, or delete the project results directory and "
            "clean the framework storage before rerunning predictions."
        )

    def _enumerate_conversation_questions(
        self,
        conversation: dict[str, Any],
    ) -> list[tuple[int, dict[str, Any]]]:
        """Return the filtered QA items for one conversation with stable IDs."""
        return qa.enumerate_filtered_qa_items(
            conversation.get("qa", []),
            allowed_categories=self.categories,
        )

    def _load_saved_qa_results_for_conversation(
        self,
        *,
        conversation_idx: int,
        conversation: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Load and validate the saved QA artifacts for one conversation."""
        expected_ids = [
            f"conv{conversation_idx}_q{question_idx}"
            for question_idx, _ in self._enumerate_conversation_questions(conversation)
        ]
        saved_results = results_io.load_question_results_for_conversation(
            project_name=self.project_name,
            conversation_idx=conversation_idx,
            benchmark_name="locomo",
        )
        saved_by_id = {
            str(result.get("question_id", "")): result
            for result in saved_results
        }
        missing_ids = [
            question_id
            for question_id in expected_ids
            if question_id not in saved_by_id
        ]
        if missing_ids:
            raise ValueError(
                "Judge phase requires saved prediction outputs for every question in "
                f"conversation {conversation_idx}. Missing: {missing_ids}"
            )
        return [saved_by_id[question_id] for question_id in expected_ids]

    def _restore_mem0_usage_for_conversation(self, conversation_idx: int) -> None:
        """Restore saved Mem0 internal usage for one conversation."""
        if self.framework != "mem0":
            return

        self.memory_internal_usage = add_memory_internal_usage(
            self.memory_internal_usage,
            results_io.load_mem0_conversation_usage(
                project_name=self.project_name,
                conversation_idx=conversation_idx,
                benchmark_name="locomo",
            ),
        )

    def _persist_mem0_usage_for_conversation(
        self,
        *,
        bundle: IngestedConversationBundle,
        conversation_idx: int,
    ) -> None:
        """Persist and aggregate Mem0 internal usage after prediction."""
        if self.framework != "mem0":
            return

        backend = bundle.backend_state.get("mem0_backend")
        if not isinstance(backend, LocalMem0ConversationBackend):
            return

        conversation_usage = backend.get_usage()
        self.memory_internal_usage = add_memory_internal_usage(
            self.memory_internal_usage,
            conversation_usage,
        )
        results_io.save_mem0_conversation_usage(
            project_name=self.project_name,
            conversation_idx=conversation_idx,
            usage=conversation_usage,
            benchmark_name="locomo",
        )

    @staticmethod
    def _prepare_strict_question_results(
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Normalize one conversation batch to the canonical LoCoMo strict artifact shape."""
        return [
            results_io.prepare_locomo_strict_evaluation(result)
            for result in results
        ]

    def _build_memory_engine_for_conversation(self, conversation_idx: int) -> TemporalMemory:
        """
            A function used to clear all items to ensure a smooth flow of conversations
        """
        nlp_engine = NLPEngine(
            NLPConfig(
                spacy_model=self.config.nlp.spacy_model,
                analyze_system_prompt=False,
            )
        )

        if not self.config.ltm.enabled or self.config.ltm.storage_type != "chroma":
            return TemporalMemory.from_dmf_config(
                config=self.config,
                nlp_engine=nlp_engine,
            )
        
        # for each conversation ChromaDb has to be isoleted as well as TemporalMemory instance
        vector_config = VectorConfig(
            model_name=self.config.nlp.model_name,
            vector_dim=self.config.nlp.vector_dim,
            window_size=self.config.capacity.window_size,
        )
        ltm_hook = ChromaLTMHook(
            collection_name=f"{self.config.ltm.collection_name}_conv_{conversation_idx}",
            persist_directory=self.config.ltm.chroma_path,
            distance_threshold=self.config.ltm.distance_threshold,
            vector_config=vector_config,
            cards_enabled=self.config.ltm.cards_enabled,
            cards_path=_namespaced_cards_path(
                self.config.ltm.cards_path,
                conversation_idx=conversation_idx,
                persist_directory=self.config.ltm.chroma_path,
            ),
            cards_collection_name=_namespaced_cards_collection_name(
                self.config.ltm.cards_collection_name,
                conversation_idx,
            ),
        )
        return TemporalMemory.from_dmf_config(
            config=self.config,
            ltm_hook=ltm_hook,
            nlp_engine=nlp_engine,
        )

    @staticmethod
    def _count_conversation_turns(conversation: dict[str, Any]) -> int:
        total_turns = 0
        for session_key, session_value in conversation["conversation"].items():
            if not session_key.startswith("session_") or session_key.endswith("_date_time"):
                continue
            if isinstance(session_value, list):
                total_turns += len(session_value)
        return total_turns
    
    def process_one_locomo_conversation(
        self,
        conversation_idx: int,
        conversation: dict[str, Any],
        on_turn_completed: Callable[[], None] | None = None,
    ) -> IngestedConversationBundle:
        if self.framework == "mem0":
            return self._process_one_locomo_conversation_mem0(
                conversation_idx=conversation_idx,
                conversation=conversation,
                on_turn_completed=on_turn_completed,
            )

        pipeline = InteractionPipeline.from_dmf_config(config=self.config)
        scoring = ScoringEngine.from_dmf_config(config=self.config)
        memory_engine = self._build_memory_engine_for_conversation(
            conversation_idx=conversation_idx,
        )
        record_index: dict[str, dict[str, Any]] = {}

        session_rows = []
        for session_key in conversation["conversation"]:
            if not session_key.startswith("session_") or session_key.endswith("_date_time"):
                continue
            time_str = conversation["conversation"][f"{session_key}_date_time"]
            current_ts = utils.parse_locomo_date(date_str=time_str)
            session_rows.append((current_ts, session_key, time_str))

        session_rows.sort(key=lambda row: row[0])

        for current_ts, session_key, time_str in session_rows:
            for turn in conversation["conversation"][session_key]:
                dmf_text = utils.serialize_locomo_turn_for_dmf(turn)
                context_text = utils.render_locomo_turn_for_context(turn)
                if not dmf_text:
                    if on_turn_completed is not None:
                        on_turn_completed()
                    continue

                report, vector = pipeline.analyze_interaction_with_vector(
                    text=dmf_text,
                    is_system=False,
                    provenance=InteractionProvenance(role=turn["speaker"].lower()),
                )

                report.raw_metadata["benchmark"] = "locomo"
                report.raw_metadata["conversation_idx"] = conversation_idx
                report.raw_metadata["source_unit_type"] = "dia"
                report.raw_metadata["source_unit_id"] = turn["dia_id"]
                report.raw_metadata["session_key"] = session_key
                report.raw_metadata["session_datetime_raw"] = time_str
                report.raw_metadata["framework"] = self.framework

                scoring.calculate_score(report, text=dmf_text)

                entry = memory_engine.add_interaction(dmf_text, report, vector)
                entry.timestamp = current_ts
                record_index[entry.record_id] = {
                    "benchmark": "locomo",
                    "conversation_idx": conversation_idx,
                    "source_unit_type": "dia",
                    "source_unit_id": turn["dia_id"],
                    "source_unit_ids": [turn["dia_id"]],
                    "session_key": session_key,
                    "session_datetime_raw": time_str,
                    "speaker": turn["speaker"],
                    "text": context_text,
                    "analysis_text": dmf_text,
                    "raw_text": str(turn.get("text", "") or ""),
                    "query": str(turn.get("query", "") or ""),
                    "blip_caption": str(turn.get("blip_caption", "") or ""),
                }
                if on_turn_completed is not None:
                    on_turn_completed()

        return IngestedConversationBundle(
            conversation_idx=conversation_idx,
            framework=self.framework,
            backend_state={"memory_engine": memory_engine},
            record_index=record_index,
        )

    def _process_one_locomo_conversation_mem0(
        self,
        conversation_idx: int,
        conversation: dict[str, Any],
        on_turn_completed: Callable[[], None] | None = None,
    ) -> IngestedConversationBundle:
        if self.run_id is None:
            raise RuntimeError("Mem0 run_id was not initialized before ingestion.")

        if self.mem0_config is None:
            raise RuntimeError("Mem0 config not loaded.")
        mem0_backend = LocalMem0ConversationBackend(
            project_name=self.project_name,
            conversation_idx=conversation_idx,
            config=self.mem0_config,
        )
        conversation_data = conversation["conversation"]
        speaker_a = str(conversation_data.get("speaker_a", "") or "")
        user_id = f"locomo_{conversation_idx}_{self.run_id}"
        record_index: dict[str, dict[str, Any]] = {}

        session_rows = []
        for session_key in conversation_data:
            if not session_key.startswith("session_") or session_key.endswith("_date_time"):
                continue
            time_str = conversation_data[f"{session_key}_date_time"]
            current_ts = utils.parse_locomo_date(date_str=time_str)
            session_rows.append((current_ts, session_key, time_str))

        session_rows.sort(key=lambda row: row[0])

        for current_ts, session_key, time_str in session_rows:
            for turn in conversation_data[session_key]:
                mem0_text = utils.serialize_locomo_turn_for_mem0(turn)
                context_text = utils.render_locomo_turn_for_context(turn)
                if not mem0_text:
                    if on_turn_completed is not None:
                        on_turn_completed()
                    continue

                role = "user" if str(turn.get("speaker", "")) == speaker_a else "assistant"
                metadata = {
                    "benchmark": "locomo",
                    "conversation_idx": conversation_idx,
                    "source_unit_type": "dia",
                    "source_unit_id": turn["dia_id"],
                    "framework": self.framework,
                }
                mem0_backend.add(
                    [{"role": role, "content": mem0_text}],
                    user_id=user_id,
                    timestamp=int(current_ts),
                    metadata=metadata,
                )
                record_index[str(turn["dia_id"])] = {
                    "benchmark": "locomo",
                    "conversation_idx": conversation_idx,
                    "source_unit_type": "dia",
                    "source_unit_id": turn["dia_id"],
                    "source_unit_ids": [turn["dia_id"]],
                    "session_key": session_key,
                    "session_datetime_raw": time_str,
                    "speaker": turn["speaker"],
                    "text": context_text,
                    "ingest_text": mem0_text,
                    "raw_text": str(turn.get("text", "") or ""),
                    "query": str(turn.get("query", "") or ""),
                    "blip_caption": str(turn.get("blip_caption", "") or ""),
                }
                if on_turn_completed is not None:
                    on_turn_completed()

        return IngestedConversationBundle(
            conversation_idx=conversation_idx,
            framework=self.framework,
            backend_state={
                "mem0_backend": mem0_backend,
                "user_id": user_id,
            },
            record_index=record_index,
        )

    def run_locomo_benchmark(
        self,
        *,
        predict_only: bool = False,
        judge_only: bool = False,
        rejudge: bool = False,
    ) -> tuple[dict[int, IngestedConversationBundle], dict[int, list[dict[str, Any]]]]:
        if predict_only and judge_only:
            raise ValueError("--predict-only and --judge-only are mutually exclusive.")
        if rejudge and not judge_only:
            raise ValueError("--rejudge can only be used with --judge-only.")
        if not judge_only and not self.resume:
            self._preflight_prediction_run()
        
        processed_memories: dict[int, IngestedConversationBundle] = {}

        qa_settings = None
        if not judge_only:
            qa_settings = qa.resolve_qa_settings(
                provider_override=self.answerer_provider,
                model_override=self.answerer_model,
            )
            self.answerer_provider = qa_settings.provider
            self.answerer_model = qa_settings.model
            qa.log_qa_settings(qa_settings)

        judge_settings = None
        if not predict_only:
            judge_settings = judge.resolve_judge_settings(
                provider_override=self.judge_provider,
                model_override=self.judge_model,
                reasoning_effort=self.judge_reasoning_effort,
            )
            self.judge_provider = judge_settings.provider
            self.judge_model = judge_settings.model
            judge.log_judge_settings(judge_settings)
        if self.run_id is None:
            self.run_id = uuid.uuid4().hex[:8]

        qa_results: dict[int, list[dict[str, Any]]] = {}
        conversation_ids = [f"conv_{i}" for i in self.conversation_indices]

        progress = StatusProgress(
            self.state_log.render_status,
            SpinnerColumn(),
            "[progress.description]{task.description}",
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        )

        with progress:
            steps_per_conversation = 1 if judge_only else 2 if predict_only else 3
            global_task = progress.add_task(
                "Running LOCOMO benchmark...",
                total=len(conversation_ids) * steps_per_conversation,
            )
            phase_task = progress.add_task(
                "Current phase...",
                total=1,
            )

            for position, conversation_idx in enumerate(self.conversation_indices):
                
                conversation = self.dataset[conversation_idx]
                conversation_label = conversation_ids[position]
                turn_count = self._count_conversation_turns(conversation)
                question_items = self._enumerate_conversation_questions(conversation)
                question_count = len(question_items)
                conversation_done = results_io.is_conversation_done(
                    project_name=self.project_name,
                    conversation_idx=conversation_idx,
                    benchmark_name="locomo",
                )
                conversation_predicted = results_io.is_conversation_predicted(
                    project_name=self.project_name,
                    conversation_idx=conversation_idx,
                    benchmark_name="locomo",
                )

                if judge_only and conversation_done and not rejudge:
                    if self.framework == "mem0":
                        self._restore_mem0_usage_for_conversation(conversation_idx)
                    log.info(
                        "Skipping already judged conversation %s (--judge-only)",
                        conversation_label,
                    )
                    self.state.processed += 1
                    self.state.completed_conversations += 1
                    progress.advance(global_task, 1)
                    continue

                if predict_only and self.resume and conversation_predicted:
                    if self.framework == "mem0":
                        self._restore_mem0_usage_for_conversation(conversation_idx)
                    log.info(
                        "Skipping predicted conversation %s (--resume --predict-only)",
                        conversation_label,
                    )
                    self.state.processed += 2
                    self.state.completed_conversations += 1
                    progress.advance(global_task, 2)
                    continue

                if (
                    not predict_only
                    and not judge_only
                    and self.resume
                    and conversation_done
                ):
                    if self.framework == "mem0":
                        self._restore_mem0_usage_for_conversation(conversation_idx)
                    log.info("Skipping completed conversation %s (--resume)", conversation_label)
                    self.state.processed += 3
                    self.state.completed_conversations += 1
                    progress.advance(global_task, 3)
                    continue

                if judge_only:
                    qa_results[conversation_idx] = self._load_saved_qa_results_for_conversation(
                        conversation_idx=conversation_idx,
                        conversation=conversation,
                    )
                    if self.framework == "mem0":
                        self._restore_mem0_usage_for_conversation(conversation_idx)
                elif self.resume and conversation_predicted:
                    log.info(
                        "Resuming from saved predictions for conversation %s",
                        conversation_label,
                    )
                    qa_results[conversation_idx] = self._load_saved_qa_results_for_conversation(
                        conversation_idx=conversation_idx,
                        conversation=conversation,
                    )
                    if self.framework == "mem0":
                        self._restore_mem0_usage_for_conversation(conversation_idx)
                    self.state.processed += 2
                    progress.advance(global_task, 2)
                else:
                    self.state_log.set_current_conversation(
                        self.state,
                        phase="ingestion",
                        conversation_idx=position,
                        conversation_label=conversation_label,
                    )
                    
                    self.state_log.log_state_delta(self.state, label="ingestion")
                    
                    progress.update(
                        phase_task,
                        description=f"Ingestion {conversation_label}",
                        total=max(1, turn_count),
                        completed=0,
                    )
                    ingestion_start = time.monotonic()
                    bundle = self.process_one_locomo_conversation(
                        conversation_idx=conversation_idx,
                        conversation=conversation,
                        on_turn_completed=lambda: progress.advance(phase_task),
                    )
                    ingestion_ms = (time.monotonic() - ingestion_start) * 1000
                    processed_memories[conversation_idx] = bundle
                    self.state_log.complete_step(self.state)
                    progress.advance(global_task)

                    self.state_log.set_current_conversation(
                        self.state,
                        phase="qa",
                        conversation_idx=position,
                        conversation_label=conversation_label,
                    )
                    self.state_log.log_state_delta(self.state, label="qa")
                    progress.update(
                        phase_task,
                        description=f"QA {conversation_label}",
                        total=max(1, question_count),
                        completed=0,
                    )
                    if qa_settings is None:
                        raise RuntimeError("QA settings were not initialized for the answerer phase.")
                    qa_start = time.monotonic()
                    qa_results[conversation_idx] = qa.run_qa_for_conversation(
                        conversation=conversation,
                        bundle=bundle,
                        config=self.config,
                        settings=qa_settings,
                        retrieval_depth=self.retrieval_depth,
                        allowed_categories=self.categories,
                        on_question_completed=lambda: progress.advance(phase_task),
                    )
                    qa_ms = (time.monotonic() - qa_start) * 1000
                    qa_results[conversation_idx] = self._prepare_strict_question_results(
                        qa_results[conversation_idx]
                    )
                    for result in qa_results[conversation_idx]:
                        pipeline_timing = results_io.normalize_pipeline_timing(
                            result.get("pipeline_timing")
                        )
                        pipeline_timing["ingestion_ms"] = ingestion_ms
                        pipeline_timing["ingestion_scope"] = "conversation"
                        pipeline_timing["qa_ms"] = qa_ms
                        pipeline_timing["qa_scope"] = "conversation"
                        result["pipeline_timing"] = pipeline_timing
                    self._persist_mem0_usage_for_conversation(
                        bundle=bundle,
                        conversation_idx=conversation_idx,
                    )
                    results_io.save_question_results(
                        qa_results[conversation_idx],
                        project_name=self.project_name,
                        benchmark_name="locomo",
                    )
                    results_io.mark_conversation_predicted(
                        project_name=self.project_name,
                        conversation_idx=conversation_idx,
                        question_count=len(qa_results[conversation_idx]),
                        benchmark_name="locomo",
                    )
                    self.state_log.complete_step(self.state)
                    progress.advance(global_task)

                    if predict_only:
                        self.state_log.complete_conversation(self.state)
                        continue

                if predict_only:
                    self.state_log.complete_conversation(self.state)
                    continue

                self.state_log.set_current_conversation(
                    self.state,
                    phase="judge",
                    conversation_idx=position,
                    conversation_label=conversation_label,
                )
                self.state_log.log_state_delta(self.state, label="judge")
                progress.update(
                    phase_task,
                    description=f"Judge {conversation_label}",
                    total=max(1, question_count),
                    completed=0,
                )
                if judge_settings is None:
                    raise RuntimeError("Judge settings were not initialized for the judge phase.")
                qa_results[conversation_idx] = judge.run_judge_for_conversation(
                    qa_results[conversation_idx],
                    settings=judge_settings,
                    on_question_completed=lambda: progress.advance(phase_task),
                )
                qa_results[conversation_idx] = self._prepare_strict_question_results(
                    qa_results[conversation_idx]
                )
                results_io.save_question_results(
                    qa_results[conversation_idx],
                    project_name=self.project_name,
                    benchmark_name="locomo",
                )
                results_io.mark_conversation_done(
                    project_name=self.project_name,
                    conversation_idx=conversation_idx,
                    question_count=len(qa_results[conversation_idx]),
                    benchmark_name="locomo",
                )
                self.state_log.complete_step(self.state)
                self.state_log.complete_conversation(self.state)
                progress.advance(global_task)

        self.state_log.set_current_conversation(
            self.state,
            phase="completed",
            conversation_idx=None,
            conversation_label=None,
        )
        self.state_log.log_state_delta(self.state, label="completed")
        if self.framework != "mem0":
            self.memory_internal_usage = empty_memory_internal_usage(framework="dmf")
        return processed_memories, qa_results
        

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the LOCOMO benchmark runner with selectable memory framework.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the framework config file (DMF TOML or Mem0 YAML).",
    )
    parser.add_argument(
        "--project-name",
        required=True,
        help="Project name used for result output directories.",
    )
    parser.add_argument(
        "--framework",
        choices=("dmf", "mem0"),
        default="dmf",
        help="Memory framework backend to execute.",
    )
    parser.add_argument(
        "--conversation-ids",
        default="",
        help="Comma-separated LOCOMO conversation indices to run, e.g. '0,1,2'.",
    )
    parser.add_argument(
        "--categories",
        default="1,2,3,4,5",
        help="Comma-separated LOCOMO categories to evaluate, e.g. '1,2,3,4,5'.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip conversations already fully completed for this project.",
    )
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="Run only ingestion + QA and persist pre-judge artifacts.",
    )
    parser.add_argument(
        "--judge-only",
        action="store_true",
        help="Run only the judge on saved per-question prediction artifacts.",
    )
    parser.add_argument(
        "--rejudge",
        action="store_true",
        help="With --judge-only, re-run judge even if the conversation is already marked done.",
    )
    parser.add_argument(
        "--answerer-provider",
        choices=("openai", "openrouter", "ollama"),
        help="Provider used only by the answerer.",
    )
    parser.add_argument(
        "--answerer-model",
        help="Model used only by the answerer.",
    )
    parser.add_argument(
        "--judge-provider",
        choices=("openai", "openrouter", "ollama"),
        help="Provider used only by the judge.",
    )
    parser.add_argument(
        "--judge-model",
        help="Model used only by the judge.",
    )
    parser.add_argument(
        "--judge-reasoning-effort",
        choices=("low", "medium", "high"),
        help="Reasoning effort for the judge (only supported by o-series models).",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate framework-specific required arguments."""
    if args.predict_only and args.judge_only:
        raise ValueError("--predict-only and --judge-only cannot be used together.")
    if args.rejudge and not args.judge_only:
        raise ValueError("--rejudge requires --judge-only.")


def parse_conversation_ids(raw: str) -> list[int] | None:
    if not raw.strip():
        return None

    values: list[int] = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        values.append(int(item))
    return values


def parse_categories(raw: str) -> list[int]:
    values: list[int] = []
    seen: set[int] = set()

    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        value = int(item)
        if value in seen:
            continue
        seen.add(value)
        values.append(value)

    if not values:
        raise ValueError("at least one category must be specified")
    return values


def _run_offline_evaluator(input_path: Path) -> subprocess.CompletedProcess[str]:
    """Launch the deterministic LOCOMO evaluator for one saved bundle."""
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "locomo.evaluate_rigorous",
            "--input",
            str(input_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )


def main() -> None:
    load_dotenv()
    t0 = time.monotonic()
    try:
        args = parse_args()
        validate_args(args)
        run_mode = "judge-only" if args.judge_only else "predict-only" if args.predict_only else "full"

        qa_settings = None
        if not args.judge_only:
            qa_settings = qa.resolve_qa_settings(
                provider_override=args.answerer_provider,
                model_override=args.answerer_model,
            )
        judge_settings = None
        if not args.predict_only:
            judge_settings = judge.resolve_judge_settings(
                provider_override=args.judge_provider,
                model_override=args.judge_model,
                reasoning_effort=args.judge_reasoning_effort,
            )
        validate_startup_secrets_for_mode(
            predict_only=args.predict_only,
            judge_only=args.judge_only,
            answerer_provider=qa_settings.provider if qa_settings is not None else None,
            judge_provider=judge_settings.provider if judge_settings is not None else None,
        )
        
        phase = Pipeline(
            project_name=args.project_name,
            framework=args.framework,
            config_path=args.config,
            conversation_indices=parse_conversation_ids(args.conversation_ids),
            categories=parse_categories(args.categories),
            resume=args.resume,
            answerer_provider=args.answerer_provider,
            answerer_model=args.answerer_model,
            judge_provider=args.judge_provider,
            judge_model=args.judge_model,
            judge_reasoning_effort=args.judge_reasoning_effort,
        )
        print_startup_configuration(
            project_name=phase.project_name,
            framework=phase.framework,
            config_path=args.config,
            dataset_path=phase.dataset_path,
            prediction_artifacts_path=results_io.predicted_results_dir(
                project_name=phase.project_name,
                benchmark_name="locomo",
            ),
            run_mode=run_mode,
            retrieval_depth=phase.retrieval_depth,
            answerer_provider=qa_settings.provider if qa_settings is not None else None,
            answerer_model=qa_settings.model if qa_settings is not None else None,
            judge_provider=judge_settings.provider if judge_settings is not None else None,
            judge_model=judge_settings.model if judge_settings is not None else None,
            categories=phase.categories,
            conversation_indices=phase.conversation_indices,
        )
        processed_memories, qa_results = phase.run_locomo_benchmark(
            predict_only=args.predict_only,
            judge_only=args.judge_only,
            rejudge=args.rejudge,
        )
        final_results_path = results_io.save_final_locomo_results(
            project_name=phase.project_name,
            conversation_ids=phase.conversation_indices,
            categories=phase.categories,
            run_metadata={
                "framework": phase.framework,
                "mode": run_mode,
                "run_id": phase.run_id,
                "top_k": phase.retrieval_depth,
                "answerer_provider": phase.answerer_provider,
                "answerer_model": phase.answerer_model,
                "judge_provider": phase.judge_provider,
                "judge_model": phase.judge_model,
            },
            token_accounting={
                "memory_internal": phase.memory_internal_usage,
            },
        )
        evaluator_result = None
        if run_mode == "full":
            evaluator_result = _run_offline_evaluator(final_results_path)
    except ValueError as exc:
        log.error("Configuration error: %s", exc)
        raise SystemExit(2) from exc

    # debug_memories = {
    #     conversation_idx: bundle.memory_engine
    #     for conversation_idx, bundle in processed_memories.items()
    # }
    # snapshot = utils.temporal_memory_debug_snapshot(debug_memories, write_jsonl=True)
    # print(json.dumps(snapshot, indent=2, ensure_ascii=False)) 

    print(f"Run mode: {run_mode}")
    print(f"Ingestion conversations processed: {len(processed_memories)}")
    print(f"Ingestion conversation ids: {sorted(processed_memories.keys())}")
    print(f"Result conversations processed: {len(qa_results)}")
    print(f"Results saved to: {final_results_path}")
    if evaluator_result is not None:
        if evaluator_result.stdout:
            print(evaluator_result.stdout.rstrip())
        if evaluator_result.returncode != 0:
            if evaluator_result.stderr:
                print(evaluator_result.stderr.rstrip())
            raise SystemExit(evaluator_result.returncode)

    elapsed = time.monotonic() - t0
    minutes, seconds = divmod(elapsed, 60)
    print(f"Total elapsed time: {int(minutes)}m {seconds:.1f}s")


if __name__ == "__main__":
    main()
