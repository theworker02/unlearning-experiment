from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .models import Classifier
from .seeding import seed_everything


@dataclass(frozen=True)
class TrainingResult:
    model: Classifier
    duration_seconds: float
    final_loss: float


def train_model(
    model: Classifier,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    deterministic: bool = True,
    device: torch.device | None = None,
    optimizer_name: str = "adamw",
    momentum: float = 0.9,
) -> TrainingResult:
    if len(x) == 0:
        raise ValueError("cannot train on an empty dataset")
    seed_everything(seed, deterministic)
    device = device or torch.device("cpu")
    model.to(device)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"unsupported optimizer: {optimizer_name}")
    criterion = nn.CrossEntropyLoss()
    started = time.perf_counter()
    final_loss = float("nan")
    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach())
    return TrainingResult(model, time.perf_counter() - started, final_loss)


@torch.inference_mode()
def per_example_loss(model: Classifier, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    model.eval()
    device = next(model.parameters()).device
    return nn.functional.cross_entropy(model(x.to(device)), y.to(device), reduction="none").cpu()
