from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import yaml

RAW = Path("results/raw/fashion-cnn-random-01pct-schema2-rocm-pilot-v1/result.json")
PILOT = Path("research/pilots/PILOT-FMNIST-IID-DELETE-001.yaml")
ROOT = Path("experiments/P001")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)


def main() -> None:
    result = json.loads(RAW.read_text(encoding="utf-8"))
    frozen = yaml.safe_load(PILOT.read_text(encoding="utf-8"))
    write_once(ROOT / "experiment.yaml", yaml.safe_dump(frozen, sort_keys=False).encode())
    seeds = "\n".join(str(seed) for seed in result["config"]["oracle_seeds"]) + "\n"
    write_once(ROOT / "seeds/P001-exact.txt", seeds.encode())
    source_manifest = Path(result["deletion"]["manifest"]["path"])
    manifest_bytes = source_manifest.read_bytes()
    write_once(ROOT / "manifests/deletion_manifest.json", manifest_bytes)
    write_once(
        ROOT / "manifests/deletion_indices.npy",
        _npy_bytes(np.asarray(result["deletion"]["forget_positions"], dtype=np.int64)),
    )
    write_once(
        ROOT / "manifests/retained_indices.npy",
        _npy_bytes(np.asarray(result["deletion"]["retain_positions"], dtype=np.int64)),
    )
    manifest_files = sorted((ROOT / "manifests").glob("*"))
    checksums = "".join(f"{sha256(path)}  {path.name}\n" for path in manifest_files)
    write_once(ROOT / "manifests/MANIFEST.sha256", checksums.encode())
    readme = (
        "# P001\n\nImmutable archive index for the completed exploratory schema-v2 "
        "Fashion-MNIST pilot. It was not preregistered and is not publication evidence.\n"
    )
    write_once(ROOT / "README.md", readme.encode())

    figure_root = Path("figures/F002")
    figure_root.mkdir(parents=True, exist_ok=True)
    copies = {
        Path("visualizations/empirical/F002_seed_convergence.svg"): figure_root / "F002.svg",
        Path("results/processed/exact_seed_convergence_fashion_pilot.csv"): figure_root
        / "source.csv",
        Path("visualizations/metadata/F002.yaml"): figure_root / "figure.yaml",
        Path("visualizations/scripts/build_f002_seed_convergence.py"): figure_root / "generate.py",
    }
    for source, destination in copies.items():
        if destination.exists():
            raise SystemExit(f"refusing to overwrite archive artifact: {destination}")
        shutil.copyfile(source, destination)

    environment = {
        "schema_version": 1,
        "experiment": "P001",
        "source": RAW.as_posix(),
        "runtime": result["runtime"],
        "gpu_model": "AMD Radeon RX 7800 XT",
        "gpu_model_source": "verified local hardware probe; see reports/hardware_validation.md",
        "reproduction_classes": ["R-BIT", "R-NUM", "R-STAT"],
    }
    write_once(
        Path("environment/P001/environment.json"),
        (json.dumps(environment, indent=2, sort_keys=True) + "\n").encode(),
    )
    write_once(
        Path("environment/P001/python-version.txt"), f"{result['runtime']['python']}\n".encode()
    )
    write_once(
        Path("environment/P001/torch-info.json"),
        (
            json.dumps(
                {
                    "torch": result["runtime"]["torch"],
                    "rocm": result["runtime"]["rocm_version"],
                    "cuda": result["runtime"]["cuda_version"],
                    "device": result["runtime"]["device"],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode(),
    )

    tracked = [RAW, PILOT, *sorted(ROOT.rglob("*")), *sorted(figure_root.glob("*"))]
    tracked = [path for path in tracked if path.is_file()]
    provenance = Path("provenance")
    provenance.mkdir(exist_ok=True)
    artifact_lines = "".join(f"{sha256(path)}  {path.as_posix()}\n" for path in tracked)
    write_once(provenance / "artifacts.sha256", artifact_lines.encode())
    write_once(
        provenance / "manifests.sha256",
        "".join(
            f"{sha256(path)}  {path.as_posix()}\n"
            for path in sorted((ROOT / "manifests").glob("*"))
        ).encode(),
    )
    write_once(
        provenance / "figures.sha256",
        "".join(
            f"{sha256(path)}  {path.as_posix()}\n" for path in sorted(figure_root.glob("*"))
        ).encode(),
    )
    events = []
    for reference in result["checkpoints"]["oracles"]:
        events.append(
            {
                "event": "model_training_completed",
                "experiment": "P001",
                "run": f"P001-EXACT-S{int(reference['seed']):04d}",
                "seed": reference["seed"],
                "dataset_sha256": result["dataset"]["sha256_after_scaling"],
                "deletion_manifest_sha256": result["deletion"]["manifest"]["sha256"],
                "checkpoint_sha256": reference["sha256"],
                "timestamp_utc": result["completed_at_utc"],
                "status": "completed",
            }
        )
    ledger = "".join(json.dumps(event, sort_keys=True) + "\n" for event in events)
    write_once(provenance / "ledger.jsonl", ledger.encode())
    print("experiments/P001")


def _npy_bytes(values: np.ndarray) -> bytes:
    import io

    buffer = io.BytesIO()
    np.save(buffer, values, allow_pickle=False)
    return buffer.getvalue()


if __name__ == "__main__":
    main()
