"""Immutable offline re-evaluation bundles derived from committed runs."""

from __future__ import annotations

import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dmf_bench.artifacts import LocalArtifactStore
from dmf_bench.atomic_io import read_json, write_json_atomic
from dmf_bench.benchmarks.locomo import evaluation as locomo_evaluation
from dmf_bench.contracts import hash_canonical_json, sha256_file
from dmf_bench.state import StateError


DERIVED_ROOT_NAME = ".derived-evaluations"


def derive_locomo_evaluation(
    *,
    run_id: str,
    runs_dir: str | Path,
) -> dict[str, Any]:
    """Recompute deterministic LoCoMo reports without provider calls."""

    store = LocalArtifactStore(runs_dir)
    verification = store.verify_committed(run_id)
    run_dir = store.run_dir(run_id)
    manifest = _read_object(run_dir / "run-manifest.json")
    inputs = manifest.get("fingerprint_inputs") or {}
    if inputs.get("benchmark") != "locomo":
        raise ValueError("offline derived evaluation currently supports LoCoMo runs only")
    evaluations_path = store.committed_artifact_path(
        run_id, "evaluations/evaluations.json"
    )
    evaluations = read_json(evaluations_path)
    if not isinstance(evaluations, list):
        raise StateError("committed evaluations artifact must contain an array")

    metadata = {
        "benchmark": "locomo",
        "framework": inputs.get("framework", "unknown"),
    }
    rigorous = locomo_evaluation.rigorous_report(evaluations, metadata)
    ablation = locomo_evaluation.ablation_report(evaluations, metadata)
    version = locomo_evaluation.EVALUATOR_VERSION
    target = derived_evaluation_dir(runs_dir, run_id) / version
    source_evaluations_sha256 = sha256_file(evaluations_path)
    if target.is_dir():
        derivation = _read_object(target / "derivation.json")
        if derivation.get("source_evaluations_sha256") != source_evaluations_sha256:
            raise StateError("existing derived evaluation refers to different source evaluations")
        return {**derivation, "output_dir": str(target), "reused": True}

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{version}.", dir=target.parent))
    try:
        write_json_atomic(temporary / "rigorous_report.json", rigorous)
        write_json_atomic(temporary / "ablation_report.json", ablation)
        derivation = {
            "schema_version": 1,
            "kind": "derived-evaluation",
            "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "source_run_id": run_id,
            "source_final_manifest_sha256": verification["final_manifest_sha256"],
            "source_evaluations_path": "evaluations/evaluations.json",
            "source_evaluations_sha256": source_evaluations_sha256,
            "benchmark": "locomo",
            "framework": inputs.get("framework", "unknown"),
            "evaluator_version": version,
            "reports": {
                "rigorous_report.json": sha256_file(temporary / "rigorous_report.json"),
                "ablation_report.json": sha256_file(temporary / "ablation_report.json"),
            },
        }
        derivation["derivation_fingerprint"] = hash_canonical_json(derivation)
        write_json_atomic(temporary / "derivation.json", derivation)
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {**derivation, "output_dir": str(target), "reused": False}


def derived_evaluation_dir(runs_dir: str | Path, run_id: str) -> Path:
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be a single non-empty path component")
    return Path(runs_dir) / DERIVED_ROOT_NAME / run_id


def _read_object(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise StateError(f"expected JSON object at {path}")
    return payload
