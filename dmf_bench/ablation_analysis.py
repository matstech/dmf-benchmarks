"""Strict paired comparison of completed DMF component runs."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from pathlib import Path
from typing import Any

from dmf_bench.ablations import PROFILE_CHANGES


CONTRASTS = tuple(profile for profile in PROFILE_CHANGES if profile != "dmf-full")
ALLOWED_FINGERPRINT_DIFFERENCES = frozenset({"ablation", "framework_config"})


def _load_run(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads((path / "run-manifest.json").read_text(encoding="utf-8"))
    rows_path = path / "evaluations" / "analysis_rows.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
    marker_path = path / "final" / "COMPLETED.json"
    if not marker_path.is_file():
        raise ValueError(f"Run is not completed: {path}.")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if (marker.get("state") != "COMPLETED"
            or marker.get("run_id") != manifest.get("run_id")
            or marker.get("scientific_fingerprint") != manifest.get("scientific_fingerprint")):
        raise ValueError(f"Completion marker does not match the run: {path}.")
    inputs = manifest["fingerprint_inputs"]
    expected = inputs["expected_question_ids"]
    actual = [row.get("query_id") for row in rows]
    if actual != expected or len(set(actual)) != len(actual):
        raise ValueError(f"Missing, extra, duplicate or reordered query in {path}.")
    if any(row.get("run_id") != manifest["run_id"]
           or row.get("scientific_fingerprint") != manifest["scientific_fingerprint"]
           or row.get("benchmark") != inputs["benchmark"]
           or row.get("framework") != inputs["framework"] for row in rows):
        raise ValueError(f"Analysis rows do not match the run manifest: {path}.")
    for row in rows:
        score = row.get("primary_score")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError(f"Invalid primary score in {path}.")
        if row.get("primary_judge_id") != inputs["evaluation"]["primary_judge_id"]:
            raise ValueError(f"Primary judge mismatch in {path}.")
    return manifest, rows


def _interval(values: list[float], *, seed: int = 7, samples: int = 10000) -> list[float]:
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(samples))
    return [means[int(0.025 * samples)], means[int(0.975 * samples)]]


def _sign_flip_p(values: list[float], *, seed: int = 7) -> tuple[float, str]:
    observed = abs(sum(values))
    if len(values) <= 16:
        signs = itertools.product((-1, 1), repeat=len(values))
        totals = [abs(sum(value * sign for value, sign in zip(values, pattern)))
                  for pattern in signs]
        return sum(total >= observed - 1e-12 for total in totals) / len(totals), "exact"
    rng = random.Random(seed)
    count = 20000
    extreme = sum(
        abs(sum(value * rng.choice((-1, 1)) for value in values)) >= observed - 1e-12
        for _ in range(count)
    )
    return (extreme + 1) / (count + 1), "monte-carlo-20000"


def compare_runs(
    full_path: Path,
    variants: dict[str, Path],
    *,
    independent_control: Path | None = None,
) -> dict[str, Any]:
    """Pair every declared contrast by exact query ID and frozen controls."""
    if set(variants) != set(CONTRASTS):
        raise ValueError(f"Exactly these DMF contrasts are required: {CONTRASTS}.")
    full, full_rows = _load_run(full_path)
    full_inputs = full["fingerprint_inputs"]
    if full_inputs.get("ablation", {}).get("profile") != "dmf-full":
        raise ValueError("Control run must use dmf-full.")
    results: dict[str, Any] = {}
    for profile in CONTRASTS:
        variant, rows = _load_run(variants[profile])
        inputs = variant["fingerprint_inputs"]
        if inputs.get("ablation", {}).get("profile") != profile:
            raise ValueError(f"Run does not declare {profile}.")
        if inputs.get("framework") != "dmf" or full_inputs.get("framework") != "dmf":
            raise ValueError("Component contrasts require DMF on both sides.")
        for field in sorted(set(inputs) | set(full_inputs)):
            if field in ALLOWED_FINGERPRINT_DIFFERENCES:
                continue
            if field not in inputs or field not in full_inputs or inputs[field] != full_inputs[field]:
                raise ValueError(f"Uncontrolled cross-run difference: {field} for {profile}.")
        if inputs["ablation"]["full_config_sha256"] != full_inputs["ablation"]["full_config_sha256"]:
            raise ValueError("Component runs use different full configuration pins.")
        if inputs["ablation"].get("scientific_controls_sha256") != full_inputs["ablation"].get("scientific_controls_sha256"):
            raise ValueError("Component runs use different declared scientific controls.")
        differences = [float(row["primary_score"] - control["primary_score"])
                       for control, row in zip(full_rows, rows)]
        mean = sum(differences) / len(differences)
        variance = (sum((value - mean) ** 2 for value in differences) / (len(differences) - 1)
                    if len(differences) > 1 else 0.0)
        p_value, method = _sign_flip_p(differences)
        results[profile] = {
            "full_run_id": full["run_id"],
            "variant_run_id": variant["run_id"],
            "query_ids": full_inputs["expected_question_ids"],
            "paired_differences_variant_minus_full": differences,
            "mean_difference": mean,
            "cohen_dz": mean / math.sqrt(variance) if variance > 0 else None,
            "bootstrap_95_percentile_ci": _interval(differences),
            "sign_flip_two_sided_p": p_value,
            "sign_flip_method": method,
        }
    ranked = sorted(CONTRASTS, key=lambda profile: results[profile]["sign_flip_two_sided_p"])
    adjusted = 0.0
    for index, profile in enumerate(ranked):
        adjusted = max(adjusted, min(1.0, (len(ranked) - index) * results[profile]["sign_flip_two_sided_p"]))
        results[profile]["holm_adjusted_p"] = adjusted
    result = {
        "schema_version": 1,
        "status": "PAIRED_ANALYSIS",
        "causal_claim": False,
        "benchmark": full_inputs["benchmark"],
        "primary_contrasts": list(CONTRASTS),
        "interval_method": "paired-percentile-bootstrap-10000-seed-7",
        "multiple_test_method": "Holm step-down familywise adjustment over four prespecified contrasts",
        "contrasts": results,
    }
    if independent_control is not None:
        control, control_rows = _load_run(independent_control)
        control_inputs = control["fingerprint_inputs"]
        if control_inputs.get("framework") != "vector-rag":
            raise ValueError("Independent control must use vector-rag.")
        common_fields = (
            "scientific_profile", "benchmark", "dataset", "selection",
            "expected_question_ids", "retrieval", "context_budget",
            "answerer", "judges", "evaluation",
        )
        for field in common_fields:
            if control_inputs.get(field) != full_inputs.get(field):
                raise ValueError(f"Independent control differs in {field}.")
        differences = [float(row["primary_score"] - baseline["primary_score"])
                       for baseline, row in zip(full_rows, control_rows)]
        result["independent_control"] = {
            "profile": "vector-rag",
            "run_id": control["run_id"],
            "query_ids": full_inputs["expected_question_ids"],
            "paired_differences_control_minus_full": differences,
            "mean_difference": sum(differences) / len(differences),
            "bootstrap_95_percentile_ci": _interval(differences),
            "interpretation": "descriptive-system-control; excluded from DMF component contrasts",
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("full_run", type=Path)
    for profile in CONTRASTS:
        parser.add_argument(profile.replace("dmf-", "").replace("-", "_"), type=Path)
    parser.add_argument("--vector-rag", type=Path, help="Optional independent system control run.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    variants = {
        profile: getattr(args, profile.replace("dmf-", "").replace("-", "_"))
        for profile in CONTRASTS
    }
    result = compare_runs(args.full_run, variants, independent_control=args.vector_rag)
    content = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(content, encoding="utf-8")
    else:
        print(content, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
