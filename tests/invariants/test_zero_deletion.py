import numpy as np

from exact_unlearning.data import load_sklearn_digits
from exact_unlearning.deletion import select_deletion
from exact_unlearning.models import build_model


def test_zero_deletion_preserves_all_training_positions() -> None:
    data = load_sklearn_digits(test_fraction=0.2, validation_fraction=0.1, split_seed=3)
    model = build_model("linear", 64, 8, 10)
    deletion = select_deletion("random", 0.0, 1, data.train_ids, data.x_train, data.y_train, model)
    assert len(deletion.forget_ids) == 0
    assert np.array_equal(deletion.retain_positions, np.arange(len(data.train_ids)))
