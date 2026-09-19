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
        summaries.append(
            {
                "candidate_id": candidate_id,
                "epochs": int(runs[0]["hyperparameters"]["epochs"]),
                "learning_rate": float(runs[0]["hyperparameters"]["learning_rate"]),
                "mean_validation_accuracy_drop": mean(drops),
                "mean_forget_training_cross_entropy": mean(forget_losses),
                "feasible": mean(drops) <= 0.05,
            }
        )
    feasible = [item for item in summaries if item["feasible"]]
    if not feasible:
        raise ValueError(
            "no fine-tuning candidate satisfies the frozen validation utility constraint"
        )
    selected = min(
        feasible,
        key=lambda item: (
            -item["mean_forget_training_cross_entropy"],
            item["epochs"],
            item["learning_rate"],
            item["candidate_id"],
        ),
    )
    return {"selected": selected, "candidate_summaries": summaries}
