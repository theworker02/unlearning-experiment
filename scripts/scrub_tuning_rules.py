from __future__ import annotations

from statistics import mean


def select_candidate(candidate_runs: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for run in candidate_runs:
        grouped.setdefault(str(run["candidate_id"]), []).append(run)
    summaries = []
    for candidate_id, runs in grouped.items():
        drops = [float(run["validation_accuracy_drop"]) for run in runs]
        forget_losses = [float(run["forget_training_cross_entropy"]) for run in runs]
        mechanism_valid = all(bool(run["mechanism_valid"]) for run in runs)
        summaries.append(
            {
                "candidate_id": candidate_id,
                "learning_rate": float(runs[0]["hyperparameters"]["learning_rate"]),
                "alpha": float(runs[0]["hyperparameters"]["alpha"]),
                "max_steps": int(runs[0]["hyperparameters"]["max_steps"]),
                "mean_validation_accuracy_drop": mean(drops),
                "mean_forget_training_cross_entropy": mean(forget_losses),
                "mechanism_valid": mechanism_valid,
                "feasible": mechanism_valid and mean(drops) <= 0.05,
            }
        )
    feasible = [item for item in summaries if item["feasible"]]
    if not feasible:
        raise ValueError("no SCRUB candidate satisfies mechanism and validation constraints")
    selected = min(
        feasible,
        key=lambda item: (
            -item["mean_forget_training_cross_entropy"],
            item["max_steps"],
            item["learning_rate"],
            -item["alpha"],
            item["candidate_id"],
        ),
    )
    return {"selected": selected, "candidate_summaries": summaries}
