from __future__ import annotations

import copy
import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .unlearning import UnlearningResult

if TYPE_CHECKING:
    from .models import Classifier


FisherDiagonal = dict[str, torch.Tensor]


def _validate_ssd_hyperparameters(alpha: float, dampening_constant: float) -> None:
    if not torch.isfinite(torch.tensor(alpha)) or alpha <= 0:
        raise ValueError("SSD alpha must be finite and positive")
    if not torch.isfinite(torch.tensor(dampening_constant)) or dampening_constant < 0:
        raise ValueError("SSD dampening_constant must be finite and non-negative")


def estimate_diagonal_fisher(
    model: Classifier,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device | None = None,
) -> FisherDiagonal:
    """Estimate E[(grad log p(y|x, theta))^2] without batch-gradient cross terms."""
    if len(x) == 0:
        raise ValueError("diagonal Fisher estimation requires at least one sample")
    if len(x) != len(y):
        raise ValueError("features and labels must have equal lengths")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    device = device or next(model.parameters()).device
    model.to(device)
    parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    fisher = {
        name: torch.zeros_like(parameter, device=device)
        for name, parameter in parameters.items()
    }
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)
    was_training = model.training
    model.eval()
    try:
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            for sample_x, sample_y in zip(batch_x, batch_y, strict=True):
                model.zero_grad(set_to_none=True)
                log_probability = nn.functional.log_softmax(model(sample_x.unsqueeze(0)), dim=1)[
                    0, sample_y
                ]
                log_probability.backward()
                for name, parameter in parameters.items():
                    if parameter.grad is not None:
                        fisher[name].add_(parameter.grad.detach().square())
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)

    sample_count = float(len(x))
    return {name: value.div_(sample_count).detach() for name, value in fisher.items()}


def estimate_minibatch_gradient_square_fisher(
    model: Classifier,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device | None = None,
) -> FisherDiagonal:
    """Reproduce the pinned SSD code's mean of squared minibatch-mean gradients.

    This is a sensitivity comparator, not an assertion that minibatch gradient squares are the
    unique interpretation of the Fisher expectation in Foster et al. (2024).
    """
    if len(x) == 0:
        raise ValueError("minibatch Fisher estimation requires at least one sample")
    if len(x) != len(y):
        raise ValueError("features and labels must have equal lengths")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    device = device or next(model.parameters()).device
    model.to(device)
    parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    fisher = {
        name: torch.zeros_like(parameter, device=device)
        for name, parameter in parameters.items()
    }
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)
    was_training = model.training
    model.eval()
    try:
        for batch_x, batch_y in loader:
            model.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(
                model(batch_x.to(device)), batch_y.to(device), reduction="mean"
            )
            loss.backward()
            for name, parameter in parameters.items():
                if parameter.grad is not None:
                    fisher[name].add_(parameter.grad.detach().square())
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)

    batch_count = float(len(loader))
    return {name: value.div_(batch_count).detach() for name, value in fisher.items()}


def _ssd_dampening_factor(
    full_fisher: torch.Tensor,
    forget_fisher: torch.Tensor,
    *,
    alpha: float,
    dampening_constant: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the Equation 4 selection mask and multiplicative dampening factor."""
    _validate_ssd_hyperparameters(alpha, dampening_constant)
    if full_fisher.shape != forget_fisher.shape:
        raise ValueError("full and forget Fisher tensors must have identical shapes")
    if torch.any(full_fisher < 0) or torch.any(forget_fisher < 0):
        raise ValueError("Fisher diagonal entries must be non-negative")
    if not torch.all(torch.isfinite(full_fisher)) or not torch.all(torch.isfinite(forget_fisher)):
        raise ValueError("Fisher diagonal entries must be finite")

    selected = forget_fisher > alpha * full_fisher
    safe_ratio = torch.where(
        forget_fisher > 0,
        dampening_constant * full_fisher / forget_fisher,
        torch.ones_like(forget_fisher),
    )
    factor = torch.where(selected, torch.clamp(safe_ratio, max=1.0), torch.ones_like(safe_ratio))
    return selected, factor


def apply_selective_synaptic_dampening(
    original: Classifier,
    full_fisher: Mapping[str, torch.Tensor],
    forget_fisher: Mapping[str, torch.Tensor],
    *,
    alpha: float,
    dampening_constant: float,
) -> tuple[Classifier, dict[str, Any]]:
    """Apply Foster et al. (2024), Equation 4, to a copy of ``original``."""
    _validate_ssd_hyperparameters(alpha, dampening_constant)
    model = copy.deepcopy(original)
    parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    expected = set(parameters)
    if set(full_fisher) != expected or set(forget_fisher) != expected:
        raise ValueError("Fisher mappings must contain every trainable parameter exactly once")

    selected_count = 0
    parameter_count = 0
    dampening_sum = 0.0
    with torch.no_grad():
        for name, parameter in parameters.items():
            full_value = full_fisher[name].to(device=parameter.device, dtype=parameter.dtype)
            forget_value = forget_fisher[name].to(device=parameter.device, dtype=parameter.dtype)
            if full_value.shape != parameter.shape or forget_value.shape != parameter.shape:
                raise ValueError(f"Fisher shape mismatch for parameter {name}")
            selected, factor = _ssd_dampening_factor(
                full_value,
                forget_value,
                alpha=alpha,
                dampening_constant=dampening_constant,
            )
            parameter.mul_(factor)
            selected_count += int(selected.sum())
            parameter_count += selected.numel()
            dampening_sum += float(factor[selected].sum())

    diagnostics: dict[str, Any] = {
        "schema_version": 1,
        "alpha": alpha,
        "dampening_constant": dampening_constant,
        "selected_parameter_count": selected_count,
        "trainable_parameter_count": parameter_count,
        "selected_parameter_fraction": selected_count / parameter_count if parameter_count else 0.0,
        "mean_selected_dampening_factor": (
            dampening_sum / selected_count if selected_count else None
        ),
        "paper_equation": "Foster et al. (2024), Equation 4",
    }
    return model, diagnostics


def selective_synaptic_dampening(
    original: Classifier,
    x_forget: torch.Tensor,
    y_forget: torch.Tensor,
    *,
    full_fisher: Mapping[str, torch.Tensor],
    alpha: float,
    dampening_constant: float,
    batch_size: int,
    device: torch.device | None = None,
) -> UnlearningResult:
    """Clean-room SSD core; intentionally excluded from the runtime method registry."""
    if len(x_forget) == 0:
        return UnlearningResult(copy.deepcopy(original), 0.0, "ssd", {"zero_deletion": True})
    started = time.perf_counter()
    forget_fisher = estimate_diagonal_fisher(
        original,
        x_forget,
        y_forget,
        batch_size=batch_size,
        device=device,
    )
    model, diagnostics = apply_selective_synaptic_dampening(
        original,
        full_fisher,
        forget_fisher,
        alpha=alpha,
        dampening_constant=dampening_constant,
    )
    diagnostics["full_fisher_precomputed"] = True
    diagnostics["forget_sample_count"] = len(x_forget)
    return UnlearningResult(model, time.perf_counter() - started, "ssd", diagnostics)
