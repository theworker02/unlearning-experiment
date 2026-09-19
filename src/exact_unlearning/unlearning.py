from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .models import Classifier
from .seeding import seed_everything
from .training import train_model


@dataclass(frozen=True)
class UnlearningResult:
    model: Classifier
    duration_seconds: float
    method: str
    diagnostics: dict[str, Any] | None = None


def fine_tune(
    original: Classifier,
    x_retain: torch.Tensor,
    y_retain: torch.Tensor,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device | None = None,
) -> UnlearningResult:
    model = copy.deepcopy(original)
    result = train_model(
        model,
        x_retain,
        y_retain,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.0,
        device=device,
    )
    return UnlearningResult(result.model, result.duration_seconds, "fine_tune")


def naive_gradient_ascent(
    original: Classifier,
    x_retain: torch.Tensor,
    y_retain: torch.Tensor,
    x_forget: torch.Tensor,
    y_forget: torch.Tensor,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    retain_weight: float,
    forget_weight: float,
    device: torch.device | None = None,
) -> UnlearningResult:
    """Diagnostic ascent baseline; not a certified removal algorithm."""
    if len(x_forget) == 0:
        return UnlearningResult(copy.deepcopy(original), 0.0, "naive_gradient_ascent")
    model = copy.deepcopy(original)
    seed_everything(seed)
    device = device or next(model.parameters()).device
    model.to(device)
    generator = torch.Generator().manual_seed(seed)
    retain_loader = DataLoader(
        TensorDataset(x_retain, y_retain), batch_size=batch_size, shuffle=True, generator=generator
    )
    forget_loader = DataLoader(
        TensorDataset(x_forget, y_forget), batch_size=batch_size, shuffle=True, generator=generator
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    started = time.perf_counter()
    model.train()
    for _ in range(epochs):
        forget_iterator = iter(forget_loader)
        for retain_x, retain_y in retain_loader:
            try:
                forget_x, forget_y = next(forget_iterator)
            except StopIteration:
                forget_iterator = iter(forget_loader)
                forget_x, forget_y = next(forget_iterator)
            retain_x = retain_x.to(device)
            retain_y = retain_y.to(device)
            forget_x = forget_x.to(device)
            forget_y = forget_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = retain_weight * criterion(model(retain_x), retain_y)
            loss -= forget_weight * criterion(model(forget_x), forget_y)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite unlearning objective")
            loss.backward()
            optimizer.step()
    return UnlearningResult(model, time.perf_counter() - started, "naive_gradient_ascent")


def _neggrad_plus_objective(
    retain_logits: torch.Tensor,
    retain_labels: torch.Tensor,
    forget_logits: torch.Tensor,
    forget_labels: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Equation 4 of Kurmanji et al. (2023)."""
    if not 0.0 <= beta <= 1.0:
        raise ValueError("NegGrad+ beta must be in [0, 1]")
    retain_loss = nn.functional.cross_entropy(retain_logits, retain_labels)
    forget_loss = nn.functional.cross_entropy(forget_logits, forget_labels)
    return beta * retain_loss - (1.0 - beta) * forget_loss


def neggrad_plus(
    original: Classifier,
    x_retain: torch.Tensor,
    y_retain: torch.Tensor,
    x_forget: torch.Tensor,
    y_forget: torch.Tensor,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    beta: float,
    weight_decay: float,
    momentum: float = 0.9,
    device: torch.device | None = None,
) -> UnlearningResult:
    """Paper-derived NegGrad+ baseline; reproduction status is tracked separately."""
    if not 0.0 <= beta <= 1.0:
        raise ValueError("NegGrad+ beta must be in [0, 1]")
    if len(x_forget) == 0:
        return UnlearningResult(copy.deepcopy(original), 0.0, "neggrad_plus")
    if len(x_retain) == 0:
        raise ValueError("NegGrad+ requires a non-empty retain set")
    model = copy.deepcopy(original)
    seed_everything(seed)
    device = device or next(model.parameters()).device
    model.to(device)
    generator = torch.Generator().manual_seed(seed)
    retain_loader = DataLoader(
        TensorDataset(x_retain, y_retain), batch_size=batch_size, shuffle=True, generator=generator
    )
    forget_loader = DataLoader(
        TensorDataset(x_forget, y_forget), batch_size=batch_size, shuffle=True, generator=generator
    )
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    probe_count = min(128, len(x_retain), len(x_forget))

    @torch.inference_mode()
    def probe() -> dict[str, float]:
        was_training = model.training
        model.eval()
        retain_loss = nn.functional.cross_entropy(
            model(x_retain[:probe_count].to(device)), y_retain[:probe_count].to(device)
        )
        forget_loss = nn.functional.cross_entropy(
            model(x_forget[:probe_count].to(device)), y_forget[:probe_count].to(device)
        )
        model.train(was_training)
        return {
            "retain_cross_entropy": float(retain_loss),
            "forget_cross_entropy": float(forget_loss),
        }

    before = probe()
    gradient_norms: list[float] = []
    objective_values: list[float] = []
    started = time.perf_counter()
    model.train()
    for _ in range(epochs):
        forget_iterator = iter(forget_loader)
        for retain_x, retain_y in retain_loader:
            try:
                forget_x, forget_y = next(forget_iterator)
            except StopIteration:
                forget_iterator = iter(forget_loader)
                forget_x, forget_y = next(forget_iterator)
            retain_x = retain_x.to(device)
            retain_y = retain_y.to(device)
            forget_x = forget_x.to(device)
            forget_y = forget_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = _neggrad_plus_objective(
                model(retain_x), retain_y, model(forget_x), forget_y, beta
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite NegGrad+ objective")
            loss.backward()
            squared = sum(
                float(parameter.grad.detach().square().sum())
                for parameter in model.parameters()
                if parameter.grad is not None
            )
            gradient_norms.append(math.sqrt(squared))
            objective_values.append(float(loss.detach()))
            optimizer.step()
    after = probe()
    return UnlearningResult(
        model,
        time.perf_counter() - started,
        "neggrad_plus",
        {
            "schema_version": 1,
            "beta": beta,
            "before": before,
            "after": after,
            "gradient_l2_mean": float(sum(gradient_norms) / len(gradient_norms)),
            "objective_minimum": min(objective_values),
            "objective_maximum": max(objective_values),
        },
    )
