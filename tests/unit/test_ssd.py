from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from exact_unlearning.ssd import (
    _ssd_dampening_factor,
    apply_selective_synaptic_dampening,
    estimate_diagonal_fisher,
    estimate_minibatch_gradient_square_fisher,
    selective_synaptic_dampening,
)


def test_ssd_equation_four_selection_and_dampening() -> None:
    full = torch.tensor([1.0, 2.0, 0.0, 4.0])
    forget = torch.tensor([20.0, 10.0, 3.0, 1.0])

    selected, factor = _ssd_dampening_factor(
        full, forget, alpha=5.0, dampening_constant=1.0
    )

    assert selected.tolist() == [True, False, True, False]
    assert torch.allclose(factor, torch.tensor([0.05, 1.0, 0.0, 1.0]))


def test_ssd_does_not_select_jointly_zero_fisher_entries() -> None:
    selected, factor = _ssd_dampening_factor(
        torch.zeros(2), torch.zeros(2), alpha=10.0, dampening_constant=1.0
    )
    assert not torch.any(selected)
    assert torch.equal(factor, torch.ones(2))


def test_empirical_fisher_is_per_example_nonnegative_and_restores_mode() -> None:
    model = nn.Linear(2, 2, bias=False)
    model.train()
    x = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    y = torch.tensor([0, 1])

    fisher = estimate_diagonal_fisher(model, x, y, batch_size=2)

    assert set(fisher) == {"weight"}
    assert fisher["weight"].shape == model.weight.shape
    assert torch.all(fisher["weight"] >= 0)
    assert torch.any(fisher["weight"] > 0)
    assert model.training
    assert all(parameter.grad is None for parameter in model.parameters())


def test_per_example_fisher_is_loader_batch_invariant() -> None:
    model = nn.Linear(2, 2, bias=False)
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    y = torch.tensor([0, 0])
    batch_one = estimate_diagonal_fisher(model, x, y, batch_size=1)
    batch_two = estimate_diagonal_fisher(model, x, y, batch_size=2)
    assert torch.equal(batch_one["weight"], batch_two["weight"])


def test_minibatch_gradient_square_exposes_analytic_cross_terms() -> None:
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    y = torch.tensor([0, 0])

    per_example = estimate_diagonal_fisher(model, x, y, batch_size=2)["weight"]
    minibatch_one = estimate_minibatch_gradient_square_fisher(
        model, x, y, batch_size=1
    )["weight"]
    minibatch_two = estimate_minibatch_gradient_square_fisher(
        model, x, y, batch_size=2
    )["weight"]

    assert torch.equal(per_example, torch.full((2, 2), 0.125))
    assert torch.equal(minibatch_one, per_example)
    assert torch.equal(minibatch_two, torch.full((2, 2), 0.0625))
    assert not torch.equal(minibatch_two, per_example)


def test_ssd_updates_only_copy_and_reports_selected_fraction() -> None:
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.fill_(2.0)
    original_state = copy.deepcopy(model.state_dict())
    full = {"weight": torch.ones_like(model.weight)}
    forget = {"weight": torch.full_like(model.weight, 20.0)}

    updated, diagnostics = apply_selective_synaptic_dampening(
        model, full, forget, alpha=10.0, dampening_constant=1.0
    )

    assert torch.equal(model.weight, original_state["weight"])
    assert torch.allclose(updated.weight, torch.full_like(updated.weight, 0.1))
    assert diagnostics["selected_parameter_fraction"] == 1.0
    assert diagnostics["mean_selected_dampening_factor"] == pytest.approx(0.05)


def test_ssd_zero_deletion_is_identity_without_fisher_estimation() -> None:
    model = nn.Linear(2, 2)
    outcome = selective_synaptic_dampening(
        model,
        torch.empty((0, 2)),
        torch.empty((0,), dtype=torch.long),
        full_fisher={},
        alpha=10.0,
        dampening_constant=1.0,
        batch_size=2,
    )
    assert outcome.method == "ssd"
    assert outcome.diagnostics == {"zero_deletion": True}
    assert outcome.model is not model
    for expected, actual in zip(model.parameters(), outcome.model.parameters(), strict=True):
        assert torch.equal(expected, actual)


@pytest.mark.parametrize(
    ("alpha", "dampening_constant"),
    [(0.0, 1.0), (-1.0, 1.0), (1.0, -0.1), (float("inf"), 1.0)],
)
def test_ssd_rejects_invalid_hyperparameters(alpha: float, dampening_constant: float) -> None:
    with pytest.raises(ValueError):
        _ssd_dampening_factor(
            torch.ones(1),
            torch.ones(1),
            alpha=alpha,
            dampening_constant=dampening_constant,
        )
