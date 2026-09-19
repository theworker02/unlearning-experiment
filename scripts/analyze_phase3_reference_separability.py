from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from exact_unlearning.phase3 import (
    Phase3Outcome,
    minimum_detectable_separation,
    separability_ratio,
    write_v3_envelope,
)

EXPERIMENT_ID = "fashion-cnn-random-01pct-schema2-rocm-pilot-v1"
SOURCE = Path("results/raw") / EXPERIMENT_ID / "result.json"
DESTINATION = Path("results/processed/phase3/PILOT-FMNIST-IID-DELETE-001-v3-r2.json")


def load_test(reference: dict[str, object]) -> dict[str, np.ndarray]:
    with np.load(str(reference["path"]), allow_pickle=False) as arrays:
        return {
            "probabilities": arrays["test_probabilities"].astype(np.float64),
            "labels": arrays["test_predicted_labels"].astype(np.int64),
            "ids": arrays["test_ids"].astype(np.int64),
        }


def distances(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> dict[str, float]:
    if not np.array_equal(left["ids"], right["ids"]):
        raise ValueError("prediction artifacts do not describe identical samples")
    p = np.clip(left["probabilities"], 1e-12, 1.0)
    q = np.clip(right["probabilities"], 1e-12, 1.0)
    midpoint = (p + q) / 2
    js = np.mean(
        0.5 * np.sum(p * np.log(p / midpoint), axis=1)
        + 0.5 * np.sum(q * np.log(q / midpoint), axis=1)
    )
    return {
        "js_divergence": float(js),
        "prediction_disagreement": float(np.mean(left["labels"] != right["labels"])),
    }


def main() -> None:
    result = json.loads(SOURCE.read_text(encoding="utf-8"))
    original = load_test(result["predictions"]["original"])
    oracles = [load_test(reference) for reference in result["predictions"]["oracles"]]
    metrics = ("js_divergence", "prediction_disagreement")
    analysis = {}
    sensitive = []
    for metric in metrics:
        original_exact = np.asarray([distances(original, oracle)[metric] for oracle in oracles])
        exact_exact = np.asarray(
            [
                distances(oracles[left], oracles[right])[metric]
                for left, right in combinations(range(len(oracles)), 2)
            ]
        )
        seed_level_null = np.asarray(
            [
                np.mean(
                    [
                        distances(oracles[left], oracles[right])[metric]
                        for right in range(len(oracles))
                        if right != left
                    ]
                )
                for left in range(len(oracles))
            ]
        )
        power = minimum_detectable_separation(
            seed_level_null,
            alpha=0.05,
            target_power=0.80,
            repetitions=2000,
            seed=73192,
            candidate_ratios=np.arange(1.0, 4.01, 0.05),
        )
        ratio = separability_ratio(original_exact, exact_exact)
        threshold = power["minimum_ratio"]
        passes = threshold is not None and ratio >= threshold
        sensitive.append(passes)
        analysis[metric] = {
            "original_exact": {
                "values": original_exact.tolist(),
                "median": float(np.median(original_exact)),
            },
            "exact_exact": {
                "pair_count_not_independent_n": len(exact_exact),
                "seed_count_independent_unit": len(oracles),
                "q95": float(np.quantile(exact_exact, 0.95)),
            },
            "separability_ratio": ratio,
            "exploratory_power_simulation": power,
            "assay_sensitive": passes,
        }
    overall_sensitive = all(sensitive)
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    envelope = {
        "schema_version": 3,
        "parent": {"schema_version": 2, "path": SOURCE.as_posix(), "sha256": source_hash},
        "supersedes": {
            "path": "results/processed/phase3/PILOT-FMNIST-IID-DELETE-001-v3.json",
            "reason": "power simulation corrected to use model/seed-level summaries",
        },
        "experiment": {
            "study_phase": 3,
            "regime": "R0_IID",
            "deletion_fraction": 0.01,
            "maturity": "PILOT",
        },
        "reference_ensemble": {
            "exact_k": len(oracles),
            "convergence_status": "CONVERGED_FOR_JS_AND_DISAGREEMENT_ONLY",
        },
        "separability": {"metrics": analysis, "all_primary_metrics_sensitive": overall_sensitive},
        "functional": {
            "status": "not_evaluated_due_to_reference_gate"
            if not overall_sensitive
            else "pilot_only"
        },
        "privacy": {"status": "not_evaluated_in_v3_reanalysis"},
        "representation": {"status": "unavailable_in_schema_v2_predictions"},
        "relearning": {"status": "not_run"},
        "utility": {"status": "preserved_in_parent_schema_v2_record"},
        "statistics": {
            "unit": "model_seed",
            "alpha": 0.05,
            "target_power": 0.80,
            "confirmatory": False,
        },
        "decision": {
            "overall_class": (
                Phase3Outcome.INCONCLUSIVE.value
                if overall_sensitive
                else Phase3Outcome.ASSAY_INSENSITIVE.value
            ),
            "reason": (
                "retrospective pilot separability analysis; not a confirmatory method decision"
            ),
        },
        "provenance": {"analysis_script": Path(__file__).as_posix(), "source_sha256": source_hash},
    }
    write_v3_envelope(str(DESTINATION), envelope)
    print(DESTINATION)


if __name__ == "__main__":
    main()
