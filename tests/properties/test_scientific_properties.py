import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from exact_unlearning.data import load_sklearn_digits
from exact_unlearning.deletion import select_deletion
from exact_unlearning.metrics import Predictions, model_discrepancy, predictive_metrics
from exact_unlearning.models import build_model


@given(st.integers(min_value=0, max_value=9999), st.floats(min_value=0, max_value=0.9))
@settings(max_examples=30, deadline=None)
def test_random_deletion_partitions_reconstruct_training_set(seed: int, fraction: float) -> None:
    size = 41
    ids = np.arange(size, dtype=np.int64)
    x = np.zeros((size, 2), dtype=np.float32)
    y = np.arange(size, dtype=np.int64) % 2
    import torch

    model = build_model("linear", 2, 4, 2)
    deletion = select_deletion(
        "random", fraction, seed, ids, torch.from_numpy(x), torch.from_numpy(y), model
    )
    assert np.intersect1d(deletion.forget_positions, deletion.retain_positions).size == 0
    assert np.array_equal(
        np.union1d(deletion.forget_positions, deletion.retain_positions), np.arange(size)
    )
    assert len(np.unique(deletion.forget_ids)) == len(deletion.forget_ids)


@given(st.integers(min_value=2, max_value=8), st.integers(min_value=2, max_value=8))
@settings(max_examples=20, deadline=None)
def test_probability_divergence_is_symmetric_and_bounded(rows: int, classes: int) -> None:
    rng = np.random.default_rng(rows * 100 + classes)
    left_p = rng.dirichlet(np.ones(classes), size=rows)
    right_p = rng.dirichlet(np.ones(classes), size=rows)
    features = rng.normal(size=(rows, 3))
    left = Predictions(left_p, left_p.argmax(1), features)
    right = Predictions(right_p, right_p.argmax(1), features * 1.1)
    lr = model_discrepancy(left, right)
    rl = model_discrepancy(right, left)
    assert np.isclose(lr["js_divergence"], rl["js_divergence"])
    assert 0.0 <= lr["js_divergence"] <= np.log(2.0)
    assert 0.0 <= lr["total_variation"] <= 1.0


def test_metrics_are_invariant_to_evaluation_order() -> None:
    probabilities = np.array([[0.9, 0.1], [0.3, 0.7], [0.6, 0.4], [0.2, 0.8]])
    labels = np.array([0, 1, 0, 1])
    prediction = Predictions(probabilities, probabilities.argmax(1), np.eye(4))
    permutation = np.array([2, 0, 3, 1])
    permuted = Predictions(
        probabilities[permutation], probabilities[permutation].argmax(1), np.eye(4)[permutation]
    )
    forward = predictive_metrics(prediction, labels, 5)
    reverse = predictive_metrics(permuted, labels[permutation], 5)
    assert forward.keys() == reverse.keys()
    assert all(np.isclose(forward[key], reverse[key]) for key in forward)


def test_complete_class_deletion_leaves_no_target_labels() -> None:
    data = load_sklearn_digits(test_fraction=0.2, validation_fraction=0.1, split_seed=4)
    model = build_model("linear", 64, 16, 10)
    deletion = select_deletion(
        "class", 0.1, 5, data.train_ids, data.x_train, data.y_train, model, target_class=3
    )
    assert not (data.y_train[deletion.retain_positions].numpy() == 3).any()
    assert (data.y_train[deletion.forget_positions].numpy() == 3).all()
