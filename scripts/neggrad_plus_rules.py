from __future__ import annotations

from statistics import mean


def select_candidate(candidate_runs: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for run in candidate_runs:
        grouped.setdefault(str(run["candidate_id"]), []).append(run)
    summaries = []
    for candidate_id, runs in grouped.items():
        drops = [float(run["validation_accuracy_drop"]) for run in runs]
        target_losses = [float(run["target_class_validation_cross_entropy"]) for run in runs]
        summaries.append(
            {
                "candidate_id": candidate_id,
                "beta": float(runs[0]["hyperparameters"]["beta"]),
                "epochs": int(runs[0]["hyperparameters"]["epochs"]),
                "mean_validation_accuracy_drop": mean(drops),
                "mean_target_class_validation_cross_entropy": mean(target_losses),
                "feasible": mean(drops) <= 0.05,
            }
        )
    feasible = [item for item in summaries if item["feasible"]]
    if not feasible:
        raise ValueError("no NegGrad+ candidate satisfies the frozen validation utility constraint")
    selected = min(
        feasible,
        key=lambda item: (
            -item["mean_target_class_validation_cross_entropy"],
            item["epochs"],
            -item["beta"],
            item["candidate_id"],
        ),
    )
    return {"selected": selected, "candidate_summaries": summaries}
