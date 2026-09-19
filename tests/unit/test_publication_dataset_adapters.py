from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from exact_unlearning.data import load_cifar, load_uci_dry_bean
from exact_unlearning.models import build_model


def test_cifar_adapter_fails_closed_until_a_canonical_archive_is_acquired(tmp_path) -> None:
    with pytest.raises(
        FileNotFoundError, match="acquire it through the dataset-provenance workflow"
    ):
        load_cifar(
            name="cifar10",
            root=tmp_path,
            validation_fraction=0.1,
            split_seed=1729,
            train_limit=1000,
            test_limit=200,
        )


def test_dry_bean_adapter_splits_without_scaler_leakage(tmp_path) -> None:
    raw = tmp_path / "uci_dry_bean" / "raw"
    raw.mkdir(parents=True)
    rows_per_class = 12
    frame = pd.DataFrame(
        {
            **{
                f"feature_{feature}": np.arange(rows_per_class * 7, dtype=float) + feature
                for feature in range(16)
            },
            "Class": np.repeat([f"class_{index}" for index in range(7)], rows_per_class),
        }
    )
    frame.to_csv(raw / "Dry_Bean_Dataset.csv", index=False)
    split = load_uci_dry_bean(
        root=tmp_path, test_fraction=0.2, validation_fraction=0.1, split_seed=7
    )
    assert split.x_train.shape[1:] == (16,)
    assert split.metadata.license == "CC-BY-4.0"
    assert not np.intersect1d(split.train_ids, split.validation_ids).size
    assert not np.intersect1d(split.train_ids, split.test_ids).size
    assert torch.allclose(split.x_train.mean(dim=0), torch.zeros(16), atol=1e-6)


def test_cifar_resnet18_exposes_features_and_logits() -> None:
    model = build_model("resnet18", (3, 32, 32), 16, 10)
    inputs = torch.zeros((2, 3, 32, 32))
    assert model.features(inputs).shape == (2, 128)
    assert model(inputs).shape == (2, 10)
