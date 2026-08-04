"""Framework-aware ingestion retained by the legacy native LoCoMo runner."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")

from dmf.analysis.nlp_engine import NLPEngine
from dmf.analysis.scoring_engine import ScoringEngine
from dmf.memory.chroma_ltm import ChromaLTMHook
from dmf.memory.temporal_memory import TemporalMemory
from dmf.runtime.pipeline import InteractionPipeline, InteractionProvenance
from dmf.utils.config import NLPConfig, VectorConfig
from dmf.utils.config_loader import DMFConfig, load_dmf_config
from rich.console import Console

from common.mem0_config import Mem0Config, load_mem0_config
from common.mem0_local import LocalMem0ConversationBackend
from common.models import IngestedConversationBundle, MemoryFramework
from locomo import utils


DEFAULT_CATEGORIES = [1, 2, 3, 4, 5]
console = Console()
log = logging.getLogger(__name__)


def _namespaced_cards_collection_name(base_name: str, conversation_idx: int) -> str:
    return f"{base_name}_conv_{conversation_idx}"[:63]


def _namespaced_cards_path(
    base_path: str | Path | None,
    *,
    conversation_idx: int,
    persist_directory: str | Path,
) -> Path:
    path = (
        Path(base_path)
        if base_path is not None
        else Path(persist_directory) / "ltm_cards.jsonl"
    )
    suffix = path.suffix or ".jsonl"
    return path.with_name(f"{path.stem}_conv_{conversation_idx}{suffix}")


def _print_storage_reset_banner(
    *,
    framework: MemoryFramework,
    resume: bool,
    dmf_config: DMFConfig | None,
    project_name: str,
) -> None:
    if resume:
        return
    if framework == "dmf":
        chroma_path = (
            str(Path(dmf_config.ltm.chroma_path).resolve())
            if dmf_config is not None
            else "<unknown>"
        )
        console.print(
            "[bold yellow]Benchmark reminder[/bold yellow]: "
            "a fresh native run expects clean framework storage and checkpoints. "
            f"Chroma path: `{chroma_path}`."
        )
    elif framework == "mem0":
        storage = (Path("results") / "locomo" / ".mem0_local" / project_name).resolve()
        console.print(
            "[bold yellow]Benchmark reminder[/bold yellow]: "
            "a fresh native run expects clean framework storage and checkpoints. "
            f"Mem0 path: `{storage}`."
        )


class NativeIngestionPipeline:
    """Load one framework and ingest selected LoCoMo conversations."""

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
    ) -> None:
        if not project_name:
            raise ValueError("no project_name specified")
        if not config_path:
            raise ValueError("--config is required")

        self.project_name = project_name
        self.framework = framework
        self.config: DMFConfig | None = None
        self.mem0_config: Mem0Config | None = None
        if framework == "dmf":
            self.config = load_dmf_config(path=config_path)
        elif framework == "mem0":
            self.mem0_config = load_mem0_config(path=config_path)
        else:
            raise ValueError(f"Unsupported framework: {framework}")

        self.resume = resume
        self.answerer_provider = answerer_provider
        self.answerer_model = answerer_model
        self.judge_provider = judge_provider
        self.judge_model = judge_model
        self.judge_reasoning_effort = judge_reasoning_effort
        self.run_id: str | None = None

        _print_storage_reset_banner(
            framework=framework,
            resume=resume,
            dmf_config=self.config,
            project_name=project_name,
        )

        dataset_path = utils.get_locomo_dataset_path()
        self.dataset_path = str(Path(dataset_path).resolve())
        if Path(dataset_path).exists():
            self.dataset = utils.load_locomo_dataset()
        else:
            self.dataset = utils.download_locomo_dataset()
        self.conversation_indices = conversation_indices or list(range(len(self.dataset)))
        self.categories = categories or list(DEFAULT_CATEGORIES)

    @property
    def retrieval_depth(self) -> int:
        if self.framework == "dmf":
            if self.config is None:
                raise RuntimeError("DMF config not loaded.")
            return int(self.config.ltm.recall_limit)
        if self.mem0_config is None:
            raise RuntimeError("Mem0 config not loaded.")
        return self.mem0_config.top_k

    def _enumerate_conversation_questions(
        self,
        conversation: dict[str, Any],
    ) -> list[tuple[int, dict[str, Any]]]:
        allowed = set(self.categories)
        return [
            (question_idx, qa_item)
            for question_idx, qa_item in enumerate(conversation.get("qa", []))
            if int(qa_item.get("category", 0) or 0) in allowed
        ]

    def _build_memory_engine_for_conversation(
        self,
        conversation_idx: int,
    ) -> TemporalMemory:
        if self.config is None:
            raise RuntimeError("DMF config not loaded.")
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
        return sum(
            len(session_value)
            for session_key, session_value in conversation["conversation"].items()
            if session_key.startswith("session_")
            and not session_key.endswith("_date_time")
            and isinstance(session_value, list)
        )

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
        if self.config is None:
            raise RuntimeError("DMF config not loaded.")

        pipeline = InteractionPipeline.from_dmf_config(config=self.config)
        scoring = ScoringEngine.from_dmf_config(config=self.config)
        memory_engine = self._build_memory_engine_for_conversation(conversation_idx)
        record_index: dict[str, dict[str, Any]] = {}
        conversation_data = conversation["conversation"]
        session_rows = [
            (
                utils.parse_locomo_date(conversation_data[f"{session_key}_date_time"]),
                session_key,
                conversation_data[f"{session_key}_date_time"],
            )
            for session_key in conversation_data
            if session_key.startswith("session_")
            and not session_key.endswith("_date_time")
        ]
        session_rows.sort(key=lambda row: row[0])

        for current_ts, session_key, time_str in session_rows:
            for turn in conversation_data[session_key]:
                dmf_text = utils.serialize_locomo_turn_for_dmf(turn)
                if not dmf_text:
                    if on_turn_completed is not None:
                        on_turn_completed()
                    continue
                report, vector = pipeline.analyze_interaction_with_vector(
                    text=dmf_text,
                    is_system=False,
                    provenance=InteractionProvenance(role=turn["speaker"].lower()),
                )
                report.raw_metadata.update(
                    {
                        "benchmark": "locomo",
                        "conversation_idx": conversation_idx,
                        "source_unit_type": "dia",
                        "source_unit_id": turn["dia_id"],
                        "session_key": session_key,
                        "session_datetime_raw": time_str,
                        "framework": self.framework,
                    }
                )
                scoring.calculate_score(report, text=dmf_text)
                entry = memory_engine.add_interaction(dmf_text, report, vector)
                entry.timestamp = current_ts
                record_index[entry.record_id] = _record_metadata(
                    turn=turn,
                    conversation_idx=conversation_idx,
                    session_key=session_key,
                    time_str=time_str,
                    analysis_text=dmf_text,
                )
                if on_turn_completed is not None:
                    on_turn_completed()

        return IngestedConversationBundle(
            conversation_idx=conversation_idx,
            framework="dmf",
            backend_state={"memory_engine": memory_engine},
            record_index=record_index,
        )

    def _process_one_locomo_conversation_mem0(
        self,
        *,
        conversation_idx: int,
        conversation: dict[str, Any],
        on_turn_completed: Callable[[], None] | None,
    ) -> IngestedConversationBundle:
        if self.run_id is None:
            raise RuntimeError("Mem0 run_id was not initialized before ingestion.")
        if self.mem0_config is None:
            raise RuntimeError("Mem0 config not loaded.")
        backend = LocalMem0ConversationBackend(
            project_name=self.project_name,
            conversation_idx=conversation_idx,
            config=self.mem0_config,
        )
        conversation_data = conversation["conversation"]
        speaker_a = str(conversation_data.get("speaker_a", "") or "")
        user_id = f"locomo_{conversation_idx}_{self.run_id}"
        record_index: dict[str, dict[str, Any]] = {}
        session_rows = [
            (
                utils.parse_locomo_date(conversation_data[f"{session_key}_date_time"]),
                session_key,
                conversation_data[f"{session_key}_date_time"],
            )
            for session_key in conversation_data
            if session_key.startswith("session_")
            and not session_key.endswith("_date_time")
        ]
        session_rows.sort(key=lambda row: row[0])

        for current_ts, session_key, time_str in session_rows:
            for turn in conversation_data[session_key]:
                mem0_text = utils.serialize_locomo_turn_for_mem0(turn)
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
                backend.add(
                    [{"role": role, "content": mem0_text}],
                    user_id=user_id,
                    timestamp=int(current_ts),
                    metadata=metadata,
                )
                record_index[str(turn["dia_id"])] = _record_metadata(
                    turn=turn,
                    conversation_idx=conversation_idx,
                    session_key=session_key,
                    time_str=time_str,
                    ingest_text=mem0_text,
                )
                if on_turn_completed is not None:
                    on_turn_completed()

        return IngestedConversationBundle(
            conversation_idx=conversation_idx,
            framework="mem0",
            backend_state={"mem0_backend": backend, "user_id": user_id},
            record_index=record_index,
        )


def _record_metadata(
    *,
    turn: dict[str, Any],
    conversation_idx: int,
    session_key: str,
    time_str: str,
    analysis_text: str | None = None,
    ingest_text: str | None = None,
) -> dict[str, Any]:
    metadata = {
        "benchmark": "locomo",
        "conversation_idx": conversation_idx,
        "source_unit_type": "dia",
        "source_unit_id": turn["dia_id"],
        "source_unit_ids": [turn["dia_id"]],
        "session_key": session_key,
        "session_datetime_raw": time_str,
        "speaker": turn["speaker"],
        "text": utils.render_locomo_turn_for_context(turn),
        "raw_text": str(turn.get("text", "") or ""),
        "query": str(turn.get("query", "") or ""),
        "blip_caption": str(turn.get("blip_caption", "") or ""),
    }
    if analysis_text is not None:
        metadata["analysis_text"] = analysis_text
    if ingest_text is not None:
        metadata["ingest_text"] = ingest_text
    return metadata
