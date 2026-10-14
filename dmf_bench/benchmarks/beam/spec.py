"""Pinned BEAM release and profile identities."""

from __future__ import annotations

CODE_REVISION = "b2da22eac88bb0874c64665f13457eb99835774a"
DATASET_REVISIONS = {
    "BEAM": "3205395e897e7318c7b094ef4e6047b9b82dbb03",
    "BEAM-10M": "9b2096193fe74e2837e4713e483351e19817773c",
}
PROFILES = {
    "beam-128k-pilot-v1": "100K",
    "beam-500k-scale-v1": "500K",
    "beam-1m-primary-v1": "1M",
    "beam-10m-stress-v1": "10M",
}
ABILITIES = (
    "abstention", "contradiction_resolution", "event_ordering",
    "information_extraction", "instruction_following", "knowledge_update",
    "multi_session_reasoning", "preference_following", "summarization",
    "temporal_reasoning",
)
EVALUATOR_VERSION = "beam-rubric-evaluator-v1"
PROTOCOL_LABEL = "BEAM retrieval-controlled adaptation"
