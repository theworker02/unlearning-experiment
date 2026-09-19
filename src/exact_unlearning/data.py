from __future__ import annotations

import gzip
import hashlib
import pickle
import struct
import tarfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from scipy.io import arff
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from .contracts import DatasetMetadata


@dataclass(frozen=True)
class DatasetSplit:
    x_train: torch.Tensor
    y_train: torch.Tensor
    train_ids: np.ndarray
    x_validation: torch.Tensor
    y_validation: torch.Tensor
    validation_ids: np.ndarray
    x_test: torch.Tensor
    y_test: torch.Tensor
    test_ids: np.ndarray
    dataset_sha256: str
    metadata: DatasetMetadata


def load_sklearn_digits(
    *, test_fraction: float, validation_fraction: float, split_seed: int
) -> DatasetSplit:
    bunch = load_digits()
    x = np.asarray(bunch.data, dtype=np.float32) / 16.0
    y = np.asarray(bunch.target, dtype=np.int64)
    ids = np.arange(len(y), dtype=np.int64)
    digest = hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest()
    train_val, test = train_test_split(
        ids, test_size=test_fraction, random_state=split_seed, stratify=y
    )
    relative_validation = validation_fraction / (1.0 - test_fraction)
    train, validation = train_test_split(
        train_val,
        test_size=relative_validation,
        random_state=split_seed + 1,
        stratify=y[train_val],
    )

    def tensor(values: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
        return torch.as_tensor(values.copy(), dtype=dtype)

    return DatasetSplit(
        tensor(x[train], torch.float32),
        tensor(y[train], torch.long),
        train,
        tensor(x[validation], torch.float32),
        tensor(y[validation], torch.long),
        validation,
        tensor(x[test], torch.float32),
        tensor(y[test], torch.long),
        test,
        digest,
        DatasetMetadata(
            name="sklearn_digits",
            version=sklearn.__version__,
            source="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html",
            license="See scikit-learn and UCI dataset records",
            sha256=digest,
        ),
    )


def load_dataset(config: dict[str, object]) -> DatasetSplit:
    """Dataset registry boundary; new adapters do not change the experiment runner."""
    name = str(config["name"])
    if name == "sklearn_digits":
        return load_sklearn_digits(
            test_fraction=float(config["test_fraction"]),
            validation_fraction=float(config["validation_fraction"]),
            split_seed=int(config["split_seed"]),
        )
    if name == "fashion_mnist":
        return load_fashion_mnist(
            root=Path(str(config["root"])),
            validation_fraction=float(config["validation_fraction"]),
            split_seed=int(config["split_seed"]),
            train_limit=int(config["train_limit"]) if config.get("train_limit") else None,
            test_limit=int(config["test_limit"]) if config.get("test_limit") else None,
        )
    if name in {"cifar10", "cifar100"}:
        return load_cifar(
            name=name,
            root=Path(str(config["root"])),
            validation_fraction=float(config["validation_fraction"]),
            split_seed=int(config["split_seed"]),
            train_limit=int(config["train_limit"]) if config.get("train_limit") else None,
            test_limit=int(config["test_limit"]) if config.get("test_limit") else None,
        )
    if name == "uci_dry_bean":
        return load_uci_dry_bean(
            root=Path(str(config["root"])),
            test_fraction=float(config["test_fraction"]),
            validation_fraction=float(config["validation_fraction"]),
            split_seed=int(config["split_seed"]),
        )
    raise ValueError(f"unsupported dataset adapter: {name}")


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, count, rows, columns = struct.unpack(">IIII", handle.read(16))
        if magic != 2051:
            raise ValueError(f"invalid IDX image magic in {path}: {magic}")
        values = np.frombuffer(handle.read(), dtype=np.uint8)
    expected = count * rows * columns
    if len(values) != expected:
        raise ValueError(f"truncated IDX image file {path}: expected {expected}, got {len(values)}")
    return values.reshape(count, 1, rows, columns)


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        if magic != 2049:
            raise ValueError(f"invalid IDX label magic in {path}: {magic}")
        values = np.frombuffer(handle.read(), dtype=np.uint8)
    if len(values) != count:
        raise ValueError(f"truncated IDX label file {path}: expected {count}, got {len(values)}")
    return values.astype(np.int64)


def _stratified_limit(
    ids: np.ndarray, labels: np.ndarray, limit: int | None, seed: int
) -> np.ndarray:
    if limit is None or limit >= len(ids):
        return ids
    selected, _ = train_test_split(ids, train_size=limit, random_state=seed, stratify=labels[ids])
    return np.sort(selected)


def load_fashion_mnist(
    *,
    root: Path,
    validation_fraction: float,
    split_seed: int,
    train_limit: int | None,
    test_limit: int | None,
) -> DatasetSplit:
    raw = root / "fashion_mnist" / "raw"
    paths = {
        "train_images": raw / "train-images-idx3-ubyte.gz",
        "train_labels": raw / "train-labels-idx1-ubyte.gz",
        "test_images": raw / "t10k-images-idx3-ubyte.gz",
        "test_labels": raw / "t10k-labels-idx1-ubyte.gz",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Fashion-MNIST files missing: {missing}; run python scripts/download_data.py"
        )
    train_x_raw = _read_idx_images(paths["train_images"])
    train_y_raw = _read_idx_labels(paths["train_labels"])
    test_x_raw = _read_idx_images(paths["test_images"])
    test_y_raw = _read_idx_labels(paths["test_labels"])
    if len(train_x_raw) != len(train_y_raw) or len(test_x_raw) != len(test_y_raw):
        raise ValueError("Fashion-MNIST image/label counts disagree")
    train_source_ids = _stratified_limit(
        np.arange(len(train_y_raw), dtype=np.int64), train_y_raw, train_limit, split_seed + 11
    )
    test_local_ids = _stratified_limit(
        np.arange(len(test_y_raw), dtype=np.int64), test_y_raw, test_limit, split_seed + 12
    )
    train_ids, validation_ids = train_test_split(
        train_source_ids,
        test_size=validation_fraction,
        random_state=split_seed,
        stratify=train_y_raw[train_source_ids],
    )
    train_ids = np.sort(train_ids)
    validation_ids = np.sort(validation_ids)
    test_ids = np.sort(test_local_ids + len(train_y_raw))
    digest = hashlib.sha256()
    for path in paths.values():
        digest.update(path.read_bytes())
    dataset_sha256 = digest.hexdigest()

    def images(values: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(values.copy(), dtype=torch.float32) / 255.0

    def labels(values: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(values.copy(), dtype=torch.long)

    return DatasetSplit(
        images(train_x_raw[train_ids]),
        labels(train_y_raw[train_ids]),
        train_ids,
        images(train_x_raw[validation_ids]),
        labels(train_y_raw[validation_ids]),
        validation_ids,
        images(test_x_raw[test_local_ids]),
        labels(test_y_raw[test_local_ids]),
        test_ids,
        dataset_sha256,
        DatasetMetadata(
            name="fashion_mnist",
            version="official IDX files, accessed 2026-08-30",
            source="https://github.com/zalandoresearch/fashion-mnist",
            license="MIT (Zalando SE, 2017)",
            sha256=dataset_sha256,
        ),
    )


def _cifar_archive_path(root: Path, name: str) -> Path:
    filenames = {
        "cifar10": "cifar-10-python.tar.gz",
        "cifar100": "cifar-100-python.tar.gz",
    }
    return root / name / "raw" / filenames[name]


def _read_cifar_member(archive: tarfile.TarFile, member_name: str) -> dict[bytes, object]:
    member = archive.getmember(member_name)
    handle = archive.extractfile(member)
    if handle is None:
        raise ValueError(f"CIFAR archive member cannot be read: {member_name}")
    return pickle.load(handle, encoding="bytes")


def _cifar_arrays(
    name: str, archive_path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read canonical CIFAR Python archives without requiring torchvision."""
    with tarfile.open(archive_path, "r:gz") as archive:
        prefix = "cifar-10-batches-py" if name == "cifar10" else "cifar-100-python"
        if name == "cifar10":
            train_batches = [
                _read_cifar_member(archive, f"{prefix}/data_batch_{index}")
                for index in range(1, 6)
            ]
            train_x = np.concatenate([np.asarray(batch[b"data"]) for batch in train_batches])
            train_y = np.concatenate(
                [np.asarray(batch[b"labels"], dtype=np.int64) for batch in train_batches]
            )
            test = _read_cifar_member(archive, f"{prefix}/test_batch")
            test_y = np.asarray(test[b"labels"], dtype=np.int64)
        else:
            train = _read_cifar_member(archive, f"{prefix}/train")
            test = _read_cifar_member(archive, f"{prefix}/test")
            train_x = np.asarray(train[b"data"])
            train_y = np.asarray(train[b"fine_labels"], dtype=np.int64)
            test_y = np.asarray(test[b"fine_labels"], dtype=np.int64)
        test_x = np.asarray(test[b"data"])
    if train_x.shape != (50000, 3072) or test_x.shape != (10000, 3072):
        raise ValueError(f"unexpected {name} image array shape")
    return (
        train_x.reshape(-1, 3, 32, 32),
        train_y,
        test_x.reshape(-1, 3, 32, 32),
        test_y,
    )


def load_cifar(
    *,
    name: str,
    root: Path,
    validation_fraction: float,
    split_seed: int,
    train_limit: int | None,
    test_limit: int | None,
) -> DatasetSplit:
    if name not in {"cifar10", "cifar100"}:
        raise ValueError(f"unsupported CIFAR variant: {name}")
    archive_path = _cifar_archive_path(root, name)
    if not archive_path.is_file():
        raise FileNotFoundError(
            f"{name} archive missing: {archive_path}; acquire it through the "
            "dataset-provenance workflow"
        )
    train_x_raw, train_y_raw, test_x_raw, test_y_raw = _cifar_arrays(name, archive_path)
    class_count = 10 if name == "cifar10" else 100
    if set(np.unique(train_y_raw)) != set(range(class_count)):
        raise ValueError(f"invalid {name} training labels")
    source_ids = _stratified_limit(
        np.arange(len(train_y_raw), dtype=np.int64), train_y_raw, train_limit, split_seed + 11
    )
    test_local_ids = _stratified_limit(
        np.arange(len(test_y_raw), dtype=np.int64), test_y_raw, test_limit, split_seed + 12
    )
    train_ids, validation_ids = train_test_split(
        source_ids,
        test_size=validation_fraction,
        random_state=split_seed,
        stratify=train_y_raw[source_ids],
    )
    train_ids = np.sort(train_ids)
    validation_ids = np.sort(validation_ids)
    test_ids = np.sort(test_local_ids + len(train_y_raw))
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    return DatasetSplit(
        torch.as_tensor(train_x_raw[train_ids].copy(), dtype=torch.float32) / 255.0,
        torch.as_tensor(train_y_raw[train_ids].copy(), dtype=torch.long),
        train_ids,
        torch.as_tensor(train_x_raw[validation_ids].copy(), dtype=torch.float32) / 255.0,
        torch.as_tensor(train_y_raw[validation_ids].copy(), dtype=torch.long),
        validation_ids,
        torch.as_tensor(test_x_raw[test_local_ids].copy(), dtype=torch.float32) / 255.0,
        torch.as_tensor(test_y_raw[test_local_ids].copy(), dtype=torch.long),
        test_ids,
        digest,
        DatasetMetadata(
            name=name,
            version="canonical Python archive",
            source="https://www.cs.toronto.edu/~kriz/cifar.html",
            license="Terms unresolved; redistribution blocked by dataset card",
            sha256=digest,
        ),
    )


def load_uci_dry_bean(
    *, root: Path, test_fraction: float, validation_fraction: float, split_seed: int
) -> DatasetSplit:
    """Load a locally acquired Dry Bean CSV and fit scaling on training rows only."""
    raw = root / "uci_dry_bean" / "raw"
    csv_path = raw / "Dry_Bean_Dataset.csv"
    arff_path = raw / "DryBeanDataset" / "Dry_Bean_Dataset.arff"
    if csv_path.is_file():
        frame = pd.read_csv(csv_path)
        path = csv_path
    elif arff_path.is_file():
        values, _ = arff.loadarff(arff_path)
        frame = pd.DataFrame(values)
        frame["Class"] = frame["Class"].str.decode("utf-8")
        path = arff_path
    else:
        raise FileNotFoundError(
            "UCI Dry Bean CSV or canonical ARFF missing; acquire it through the "
            "dataset-provenance workflow"
        )
    if "Class" not in frame.columns:
        raise ValueError("UCI Dry Bean CSV must contain the Class column")
    features = frame.drop(columns=["Class"])
    if len(features.columns) != 16 or features.isnull().any().any():
        raise ValueError("UCI Dry Bean data must contain 16 complete numeric feature columns")
    labels, classes = pd.factorize(frame["Class"], sort=True)
    if len(classes) != 7:
        raise ValueError("UCI Dry Bean data must contain seven classes")
    x = features.to_numpy(dtype=np.float32)
    y = labels.astype(np.int64)
    ids = np.arange(len(y), dtype=np.int64)
    train_val, test = train_test_split(
        ids, test_size=test_fraction, random_state=split_seed, stratify=y
    )
    relative_validation = validation_fraction / (1.0 - test_fraction)
    train, validation = train_test_split(
        train_val, test_size=relative_validation, random_state=split_seed + 1, stratify=y[train_val]
    )
    mean = x[train].mean(axis=0, keepdims=True)
    scale = x[train].std(axis=0, keepdims=True)
    if np.any(scale == 0):
        raise ValueError("UCI Dry Bean training split contains a constant feature")
    scaled = (x - mean) / scale
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return DatasetSplit(
        torch.as_tensor(scaled[train].copy(), dtype=torch.float32),
        torch.as_tensor(y[train], dtype=torch.long),
        train,
        torch.as_tensor(scaled[validation].copy(), dtype=torch.float32),
        torch.as_tensor(y[validation], dtype=torch.long),
        validation,
        torch.as_tensor(scaled[test].copy(), dtype=torch.float32),
        torch.as_tensor(y[test], dtype=torch.long),
        test,
        digest,
        DatasetMetadata(
            name="uci_dry_bean", version="UCI Dry Bean Dataset", source="https://archive.ics.uci.edu/dataset/602/dry+bean+dataset",
            license="CC-BY-4.0", sha256=digest,
        ),
    )
