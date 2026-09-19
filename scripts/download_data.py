from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

FASHION_MNIST_FILES = {
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}
BASE_URL = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
UCI_DRY_BEAN_URL = "https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip"


def md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_fashion_mnist(root: Path) -> None:
    destination = root / "fashion_mnist" / "raw"
    destination.mkdir(parents=True, exist_ok=True)
    for filename, expected in FASHION_MNIST_FILES.items():
        path = destination / filename
        if path.exists() and md5(path) == expected:
            print(f"verified {path}")
            continue
        temporary = path.with_suffix(path.suffix + ".partial")
        urllib.request.urlretrieve(BASE_URL + filename, temporary)
        actual = md5(temporary)
        if actual != expected:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"checksum mismatch for {filename}: expected {expected}, got {actual}"
            )
        temporary.replace(path)
        print(f"downloaded and verified {path}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_uci_dry_bean(root: Path) -> None:
    """Acquire the CC-BY UCI archive without redistributing it in the repository."""
    destination = root / "uci_dry_bean" / "raw"
    archive = destination / "dry+bean+dataset.zip"
    canonical_arff = destination / "DryBeanDataset" / "Dry_Bean_Dataset.arff"
    destination.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        temporary = archive.with_suffix(".zip.partial")
        urllib.request.urlretrieve(UCI_DRY_BEAN_URL, temporary)
        temporary.replace(archive)
        print(f"downloaded {archive}")
    if not canonical_arff.is_file():
        with zipfile.ZipFile(archive) as bundle:
            member = "DryBeanDataset/Dry_Bean_Dataset.arff"
            if member not in bundle.namelist():
                raise RuntimeError(f"UCI archive does not contain expected member: {member}")
            extracted = destination / member
            extracted.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, extracted.open("wb") as target:
                shutil.copyfileobj(source, target)
        print(f"extracted {canonical_arff}")
    print(f"verified source archive sha256={sha256(archive)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=["fashion_mnist", "uci_dry_bean"], default="fashion_mnist"
    )
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    arguments = parser.parse_args()
    if arguments.dataset == "fashion_mnist":
        download_fashion_mnist(arguments.root)
    else:
        download_uci_dry_bean(arguments.root)


if __name__ == "__main__":
    main()
