import torch

from exact_unlearning.contracts import UnlearningContext
from exact_unlearning.methods import run_unlearning_method
from exact_unlearning.models import build_model
from exact_unlearning.unlearning import _neggrad_plus_objective


def test_neggrad_plus_objective_matches_published_equation() -> None:
    retain_logits = torch.tensor([[2.0, -1.0], [0.5, 0.2]])
    forget_logits = torch.tensor([[-0.5, 1.0], [1.5, -0.2]])
    retain_labels = torch.tensor([0, 1])
    forget_labels = torch.tensor([1, 0])
    beta = 0.95
    expected = beta * torch.nn.functional.cross_entropy(retain_logits, retain_labels)
    expected -= (1.0 - beta) * torch.nn.functional.cross_entropy(forget_logits, forget_labels)
    assert torch.allclose(
        _neggrad_plus_objective(
            retain_logits, retain_labels, forget_logits, forget_labels, beta
        ),
        expected,
    )


def test_neggrad_plus_registry_runner_updates_copy_and_preserves_original() -> None:
    torch.manual_seed(2)
    original = build_model("mlp", 4, 8, 2)
    before = [parameter.detach().clone() for parameter in original.parameters()]
    descriptor, outcome = run_unlearning_method(
        "neggrad_plus",
        UnlearningContext(
            original=original,
            x_retain=torch.randn(16, 4),
            y_retain=torch.randint(0, 2, (16,)),
            x_forget=torch.randn(4, 4),
            y_forget=torch.randint(0, 2, (4,)),
            seed=8,
            config={
                "epochs": 1,
                "batch_size": 4,
                "learning_rate": 0.01,
                "beta": 0.95,
                "weight_decay": 0.0,
                "momentum": 0.9,
            },
            device=torch.device("cpu"),
        ),
    )
    assert descriptor.simple_baseline is False
    assert outcome.method == "neggrad_plus"
    assert outcome.diagnostics is not None
    assert outcome.diagnostics["schema_version"] == 1
    assert outcome.diagnostics["beta"] == 0.95
    assert outcome.diagnostics["gradient_l2_mean"] > 0
    assert set(outcome.diagnostics["before"]) == {
        "retain_cross_entropy",
        "forget_cross_entropy",
    }
    assert any(
        not torch.equal(left, right)
        for left, right in zip(before, outcome.model.parameters(), strict=True)
    )
    assert all(
        torch.equal(left, right) for left, right in zip(before, original.parameters(), strict=True)
    )


def test_neggrad_plus_rejects_beta_outside_unit_interval() -> None:
    logits = torch.tensor([[1.0, 0.0]])
    labels = torch.tensor([0])
    try:
        _neggrad_plus_objective(logits, labels, logits, labels, 1.01)
    except ValueError as error:
        assert "beta" in str(error)
    else:
        raise AssertionError("invalid beta must be rejected")
