import numpy as np

from exact_unlearning.data import load_sklearn_digits
from exact_unlearning.deletion import select_deletion
from exact_unlearning.models import build_model


def test_split_is_deterministic_and_disjoint() -> None:
    left = load_sklearn_digits(test_fraction=0.2, validation_fraction=0.1, split_seed=4)
    right = load_sklearn_digits(test_fraction=0.2, validation_fraction=0.1, split_seed=4)
    assert np.array_equal(left.train_ids, right.train_ids)
    assert not np.intersect1d(left.train_ids, left.test_ids).size
    assert not np.intersect1d(left.train_ids, left.validation_ids).size
    assert len(np.union1d(np.union1d(left.train_ids, left.validation_ids), left.test_ids)) == 1797


def test_deletion_is_reproducible_and_disjoint() -> None:
    data = load_sklearn_digits(test_fraction=0.2, validation_fraction=0.1, split_seed=4)
    model = build_model("linear", 64, 16, 10)
    left = select_deletion("random", 0.05, 9, data.train_ids, data.x_train, data.y_train, model)
    right = select_deletion("random", 0.05, 9, data.train_ids, data.x_train, data.y_train, model)
    assert np.array_equal(left.forget_ids, right.forget_ids)
    assert not np.intersect1d(left.forget_positions, left.retain_positions).size
    assert len(left.forget_positions) + len(left.retain_positions) == len(data.train_ids)


def test_high_loss_deletion_selects_largest_losses() -> None:
    data = load_sklearn_digits(test_fraction=0.2, validation_fraction=0.1, split_seed=4)
    model = build_model("linear", 64, 16, 10)
    deletion = select_deletion(
        "high_loss", 0.01, 9, data.train_ids, data.x_train, data.y_train, model
    )
    assert len(deletion.forget_positions) > 0
    assert not np.intersect1d(deletion.forget_positions, deletion.retain_positions).size
