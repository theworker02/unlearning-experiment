import torch

from exact_unlearning.data import load_sklearn_digits
from exact_unlearning.models import build_model
from exact_unlearning.seeding import seed_everything
from exact_unlearning.training import train_model


def _run(seed: int) -> list[torch.Tensor]:
    data = load_sklearn_digits(test_fraction=0.2, validation_fraction=0.1, split_seed=12)
    seed_everything(seed)
    model = build_model("linear", 64, 16, 10)
    result = train_model(
        model,
        data.x_train[:256],
        data.y_train[:256],
        seed=seed,
        epochs=1,
        batch_size=128,
        learning_rate=0.01,
        weight_decay=0.0,
    )
    return [parameter.detach().clone() for parameter in result.model.parameters()]


def test_cpu_training_repeats_exactly_for_same_seed() -> None:
    for left, right in zip(_run(44), _run(44), strict=True):
        assert torch.equal(left, right)
