from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

SOURCE = Path("results/raw/phase3-fmnist-class-conditioned-assay-pilot-v3/result.json")
PLAN = Path("study/manifests/equivalence_margin_calibration_v2_1_plan.json")
OUTPUT = Path("results/processed/equivalence_v2_1_compute_estimate.json")
REPORT = Path("reports/equivalence_v2_1_compute_estimate.md")
LATEX = Path("paper/generated/equivalence_v2_1_compute.tex")
BOOTSTRAP_SEED = 91633
BOOTSTRAP_REPETITIONS = 10000


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def estimate() -> dict[str, object]:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    planned_conditions = {item["condition_id"] for item in plan["conditions"]}
    durations = np.asarray(
        [
            duration
            for condition in source["conditions"]
            if condition["condition_id"] in planned_conditions
            for duration in condition["exact_training_seconds"]
        ],
        dtype=float,
    )
    expected_observations = plan["compute_and_storage"]["existing_exact_models_reused"]
    if len(durations) != expected_observations or not np.isfinite(durations).all():
        raise ValueError("condition-matched source runtimes are incomplete or nonfinite")
    new_jobs = plan["compute_and_storage"]["new_exact_training_jobs"]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    projected_totals = np.sum(
        rng.choice(durations, size=(BOOTSTRAP_REPETITIONS, new_jobs), replace=True), axis=1
    )
    low, median, high = np.quantile(projected_totals, [0.025, 0.5, 0.975])
    return {
        "schema_version": 1,
        "status": "EMPIRICAL_PLANNING_ESTIMATE_NOT_A_BENCHMARK",
        "execution_authorized": False,
        "source": {
            "path": SOURCE.as_posix(),
            "sha256": _sha256(SOURCE),
            "experiment_id": source["experiment_id"],
            "condition_ids": sorted(planned_conditions),
            "runtime_environment": source["runtime"],
        },
        "plan": {"path": PLAN.as_posix(), "sha256": _sha256(PLAN)},
        "observed_exact_training_seconds": {
            "run_count": len(durations),
            "mean": float(np.mean(durations)),
            "median": float(np.median(durations)),
            "standard_deviation": float(np.std(durations, ddof=1)),
            "minimum": float(np.min(durations)),
            "maximum": float(np.max(durations)),
        },
        "projected_serial_new_training_seconds": {
            "job_count": new_jobs,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "median": float(median),
            "interval_95": [float(low), float(high)],
        },
        "not_estimated": [
            "validation_prediction_export_time",
            "data_loading_and_process_startup",
            "parallel_scheduling_efficiency",
            "peak_memory",
            "runtime_on_any_other_device_or_software_environment",
        ],
        "limitations": [
            "The estimate reuses pilot wall-clock measurements and is not a fresh benchmark.",
            "The source measurements use one WSL/ROCm software and hardware environment.",
            "Bootstrap resampling describes observed run variability, not future hardware drift.",
            "The estimate covers only the 20 planned new exact trainings, not the full workflow.",
        ],
    }


def main() -> None:
    value = estimate()
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if OUTPUT.exists() and OUTPUT.read_text(encoding="utf-8") != payload:
        raise SystemExit(f"immutable compute estimate differs: {OUTPUT}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(payload, encoding="utf-8")
    projection = value["projected_serial_new_training_seconds"]
    REPORT.write_text(
        "# Equivalence-margin v2.1 compute estimate\n\n"
        "Status: empirical planning estimate, not a benchmark or execution authorization.\n\n"
        f"The {projection['job_count']} planned new exact trainings have a serial median "
        f"estimate of {projection['median']:.2f} seconds and a 95% bootstrap interval of "
        f"{projection['interval_95'][0]:.2f}--{projection['interval_95'][1]:.2f} seconds. "
        "The estimate resamples 30 condition-matched pilot runtimes from the recorded "
        "WSL/ROCm environment.\n\n"
        "It excludes validation inference, data loading, process startup, parallel scheduling, "
        "peak memory, and transfer to another hardware or software environment.\n",
        encoding="utf-8",
    )
    LATEX.parent.mkdir(parents=True, exist_ok=True)
    LATEX.write_text(
        f"\\newcommand{{\\VTwoOneNewTrainingJobs}}{{{projection['job_count']}}}\n"
        f"\\newcommand{{\\VTwoOneTrainingMedianSeconds}}{{{projection['median']:.2f}}}\n"
        f"\\newcommand{{\\VTwoOneTrainingLowSeconds}}{{{projection['interval_95'][0]:.2f}}}\n"
        f"\\newcommand{{\\VTwoOneTrainingHighSeconds}}{{{projection['interval_95'][1]:.2f}}}\n",
        encoding="utf-8",
    )
    print(payload, end="")


if __name__ == "__main__":
    main()
