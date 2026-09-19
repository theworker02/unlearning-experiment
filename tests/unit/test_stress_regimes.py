import numpy as np

from exact_unlearning.stress import generate_memorization_canaries, stable_top_k


def test_hard_example_ranking_has_stable_id_tie_break() -> None:
    positions = stable_top_k(np.array([2.0, 2.0, 1.0]), np.array([20, 10, 30]), 2)
    assert positions.tolist() == [0, 1]


def test_canary_generation_is_deterministic_and_changes_labels() -> None:
    images = np.zeros((20, 8, 8), dtype=np.float32)
    labels = np.arange(20) % 10
    first = generate_memorization_canaries(images, labels, count=4, seed=9, class_count=10)
    second = generate_memorization_canaries(images, labels, count=4, seed=9, class_count=10)
    assert first.sha256 == second.sha256
    assert np.array_equal(first.source_positions, second.source_positions)
    assert np.all(first.labels != labels[first.source_positions])


def test_canary_generation_does_not_mutate_source() -> None:
    images = np.zeros((10, 8, 8), dtype=np.float32)
    labels = np.arange(10)
    generate_memorization_canaries(images, labels, count=2, seed=3, class_count=10)
    assert not images.any()
    assert np.array_equal(labels, np.arange(10))
