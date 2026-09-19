from __future__ import annotations

import math


def _finite(value: object) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def evaluate_mechanism(trace: list[dict[str, object]]) -> dict[str, bool]:
    first = trace[0]
    before_max = first["before_max"]
    after_max = first["after_max"]
    before_min = first["before_min"]
    after_min = first["after_min"]
    return {
        "initial_student_teacher_logits_are_not_identical": first["max_gradient_l2_mean"] > 0,
        "first_max_epoch_increases_forget_teacher_student_kl": (
            after_max["forget_teacher_student_kl"]
            > before_max["forget_teacher_student_kl"]
        ),
        "first_min_epoch_decreases_retain_objective_on_fixed_probe": (
            after_min["retain_objective"] < before_min["retain_objective"]
        ),
        "all_recorded_values_are_finite": _finite(trace),
    }
