from __future__ import annotations

import copy
import math
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .contracts import UnlearningContext
from .seeding import seed_everything
from .unlearning import UnlearningResult


class ScrubApplicabilityError(RuntimeError):
    """Raised when the configured model cannot produce an effective initial SCRUB max-step."""


def _has_model_derived_max_step_signal(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor
) -> bool:
    """Return whether train/eval model behavior creates a non-identical initial signal."""
    return not torch.equal(student_logits.detach(), teacher_logits.detach())


def _teacher_student_kl(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Compute d(x; w_u) from Kurmanji et al. (2023), Eq. 1--3."""
    if temperature <= 0:
        raise ValueError("SCRUB temperature must be positive")
    teacher_probabilities = torch.softmax(teacher_logits / temperature, dim=1)
    student_log_probabilities = torch.log_softmax(student_logits / temperature, dim=1)
    return nn.functional.kl_div(
        student_log_probabilities, teacher_probabilities, reduction="batchmean"
    ) * (temperature**2)


def _forget_objective(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Loss minimized during Algorithm 2 to perform gradient ascent on d(x; w_u)."""
    return -_teacher_student_kl(student_logits, teacher_logits, temperature)


def _retain_objective(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    alpha: float,
    gamma: float,
    temperature: float,
) -> torch.Tensor:
    """Algorithm 3 objective: alpha times distillation plus gamma times task loss."""
    return alpha * _teacher_student_kl(student_logits, teacher_logits, temperature) + gamma * (
        nn.functional.cross_entropy(student_logits, labels)
    )


def _gradient_l2_norm(model: nn.Module) -> float:
    squared = sum(
        float(parameter.grad.detach().square().sum())
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    return math.sqrt(squared)


@torch.inference_mode()
def _probe_mechanism(
    student: nn.Module,
    teacher: nn.Module,
    x_forget: torch.Tensor,
    y_forget: torch.Tensor,
    x_retain: torch.Tensor,
    y_retain: torch.Tensor,
    *,
    alpha: float,
    gamma: float,
    temperature: float,
) -> dict[str, float]:
    was_training = student.training
    student.eval()
    forget_student = student(x_forget)
    forget_teacher = teacher(x_forget)
    retain_student = student(x_retain)
    retain_teacher = teacher(x_retain)
    retain_kl = _teacher_student_kl(retain_student, retain_teacher, temperature)
    retain_ce = nn.functional.cross_entropy(retain_student, y_retain)
    metrics = {
        "forget_teacher_student_kl": float(
            _teacher_student_kl(forget_student, forget_teacher, temperature)
        ),
        "retain_teacher_student_kl": float(retain_kl),
        "retain_cross_entropy": float(retain_ce),
        "retain_objective": float(alpha * retain_kl + gamma * retain_ce),
        "forget_accuracy": float((forget_student.argmax(dim=1) == y_forget).float().mean()),
        "retain_accuracy": float((retain_student.argmax(dim=1) == y_retain).float().mean()),
    }
    student.train(was_training)
    return metrics


def scrub(context: UnlearningContext) -> UnlearningResult:
    """SCRUB core objective from Kurmanji et al. (NeurIPS 2023), without rewinding."""
    if len(context.x_forget) == 0:
        return UnlearningResult(copy.deepcopy(context.original), 0.0, "scrub")
    config = context.config
    if len(context.x_retain) == 0:
        raise ValueError("SCRUB requires a non-empty retain set")
    seed_everything(context.seed)
    teacher = copy.deepcopy(context.original).to(context.device).eval()
    teacher.requires_grad_(False)
    student = copy.deepcopy(context.original).to(context.device).train()
    optimizer_name = str(config.get("optimizer", "adam"))
    parameters = student.parameters()
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            parameters,
            lr=float(config["learning_rate"]),
            weight_decay=float(config.get("weight_decay", 0.0)),
        )
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            parameters,
            lr=float(config["learning_rate"]),
            momentum=float(config.get("momentum", 0.9)),
            weight_decay=float(config.get("weight_decay", 0.0)),
        )
    else:
        raise ValueError(f"unsupported SCRUB optimizer: {optimizer_name}")
    generator = torch.Generator().manual_seed(context.seed)
    forget_loader = DataLoader(
        TensorDataset(context.x_forget, context.y_forget),
        batch_size=int(config.get("forget_batch_size", config["batch_size"])),
        shuffle=True,
        generator=generator,
    )
    retain_loader = DataLoader(
        TensorDataset(context.x_retain, context.y_retain),
        batch_size=int(config.get("retain_batch_size", config["batch_size"])),
        shuffle=True,
        generator=generator,
    )
    max_steps = int(config["max_steps"])
    total_steps = int(config["steps"])
    if not 0 <= max_steps <= total_steps:
        raise ValueError("SCRUB requires 0 <= max_steps <= steps")
    alpha = float(config.get("alpha", 1.0))
    gamma = float(config.get("gamma", 1.0))
    temperature = float(config.get("temperature", 1.0))
    if alpha < 0 or gamma < 0:
        raise ValueError("SCRUB alpha and gamma must be non-negative")
    if temperature <= 0:
        raise ValueError("SCRUB temperature must be positive")
    started = time.perf_counter()
    initial_forget_signal_checked = False
    probe_count = int(config.get("mechanism_probe_size", 128))
    probe_forget_x = context.x_forget[:probe_count].to(context.device)
    probe_forget_y = context.y_forget[:probe_count].to(context.device)
    probe_retain_x = context.x_retain[:probe_count].to(context.device)
    probe_retain_y = context.y_retain[:probe_count].to(context.device)
    mechanism_trace: list[dict[str, object]] = []
    for step in range(total_steps):
        epoch_trace: dict[str, object] = {"step": step, "max_epoch_executed": step < max_steps}
        if step < max_steps:
            epoch_trace["before_max"] = _probe_mechanism(
                student,
                teacher,
                probe_forget_x,
                probe_forget_y,
                probe_retain_x,
                probe_retain_y,
                alpha=alpha,
                gamma=gamma,
                temperature=temperature,
            )
            max_gradient_norms = []
            for batch_x, _ in forget_loader:
                batch_x = batch_x.to(context.device)
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    teacher_logits = teacher(batch_x)
                student_logits = student(batch_x)
                if not initial_forget_signal_checked:
                    initial_forget_signal_checked = True
                    if not _has_model_derived_max_step_signal(student_logits, teacher_logits):
                        raise ScrubApplicabilityError(
                            "SCRUB initial max-step has no model-derived signal: student and "
                            "teacher logits are identical. Use an architecture/protocol with a "
                            "validated nonzero max-step signal; numerical optimizer noise is not "
                            "accepted as unlearning evidence."
                        )
                loss = _forget_objective(student_logits, teacher_logits, temperature)
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite SCRUB max-step objective")
                loss.backward()
                max_gradient_norms.append(_gradient_l2_norm(student))
                optimizer.step()
            epoch_trace["max_gradient_l2_mean"] = float(
                sum(max_gradient_norms) / len(max_gradient_norms)
            )
            epoch_trace["after_max"] = _probe_mechanism(
                student,
                teacher,
                probe_forget_x,
                probe_forget_y,
                probe_retain_x,
                probe_retain_y,
                alpha=alpha,
                gamma=gamma,
                temperature=temperature,
            )
        epoch_trace["before_min"] = _probe_mechanism(
            student,
            teacher,
            probe_forget_x,
            probe_forget_y,
            probe_retain_x,
            probe_retain_y,
            alpha=alpha,
            gamma=gamma,
            temperature=temperature,
        )
        min_gradient_norms = []
        for batch_x, batch_y in retain_loader:
            batch_x = batch_x.to(context.device)
            batch_y = batch_y.to(context.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                teacher_logits = teacher(batch_x)
            student_logits = student(batch_x)
            loss = _retain_objective(
                student_logits,
                teacher_logits,
                batch_y,
                alpha=alpha,
                gamma=gamma,
                temperature=temperature,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite SCRUB min-step objective")
            loss.backward()
            min_gradient_norms.append(_gradient_l2_norm(student))
            optimizer.step()
        epoch_trace["min_gradient_l2_mean"] = float(
            sum(min_gradient_norms) / len(min_gradient_norms)
        )
        epoch_trace["after_min"] = _probe_mechanism(
            student,
            teacher,
            probe_forget_x,
            probe_forget_y,
            probe_retain_x,
            probe_retain_y,
            alpha=alpha,
            gamma=gamma,
            temperature=temperature,
        )
        mechanism_trace.append(epoch_trace)
    return UnlearningResult(
        student,
        time.perf_counter() - started,
        "scrub",
        {"schema_version": 1, "mechanism_trace": mechanism_trace},
    )
