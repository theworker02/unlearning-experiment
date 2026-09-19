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
        selected_fractions = [float(run["selected_parameter_fraction"]) for run in runs]
        summaries.append(
            {
                "candidate_id": candidate_id,
                "alpha": float(runs[0]["hyperparameters"]["alpha"]),
                "dampening_constant": float(
                    runs[0]["hyperparameters"]["dampening_constant"]
                ),
                "mean_validation_accuracy_drop": mean(drops),
                "mean_forget_training_cross_entropy": mean(forget_losses),
                "mean_selected_parameter_fraction": mean(selected_fractions),
                "feasible": mean(drops) <= 0.05,
            }
        )
    feasible = [item for item in summaries if item["feasible"]]
    if not feasible:
        raise ValueError("no SSD candidate satisfies the frozen validation utility constraint")
    selected = min(
        feasible,
        key=lambda item: (
            -item["mean_forget_training_cross_entropy"],
            item["mean_selected_parameter_fraction"],
            -item["alpha"],
            item["dampening_constant"],
            item["candidate_id"],
        ),
    )
    return {"selected": selected, "candidate_summaries": summaries}
