from __future__ import annotations

import numpy as np
import torch

from exact_unlearning.deletion import select_deletion
from exact_unlearning.models import build_model


def test_full_class_deletion_keeps_non_target_classes() -> None:
    train_ids = np.arange(8, dtype=np.int64)
    labels = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2])
    features = torch.zeros((8, 1, 2, 2), dtype=torch.float32)
    model = build_model("linear", (1, 2, 2), 4, 3)

    deletion = select_deletion(
        "class", 1.0, 17, train_ids, features, labels, model, target_class=1
    )

    assert deletion.forget_ids.tolist() == [2, 3, 4]
    assert np.intersect1d(deletion.forget_positions, deletion.retain_positions).size == 0
    assert len(deletion.retain_positions) == 5
