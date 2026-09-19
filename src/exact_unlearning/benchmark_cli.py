from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

import yaml

from .config import load_config
from .experiment import run_experiment

REGISTRIES = {
    "datasets": ("benchmark/datasets.yaml", "datasets"),
    "methods": ("benchmark/methods.yaml", "methods"),
    "architectures": ("benchmark/architectures.yaml", "architectures"),
    "scenarios": ("benchmark/deletion_regimes.yaml", "regimes"),
    "metrics": ("benchmark/metrics.yaml", "metrics"),
}


def _list_registry(name: str) -> None:
    path, key = REGISTRIES[name]
    entries = yaml.safe_load(Path(path).read_text(encoding="utf-8"))[key]
    for entry in entries:
        print(f"{entry['id']}\t{entry.get('status', 'defined')}\t{entry.get('role', '')}")


def _validate() -> None:
    manifest = json.loads(Path("benchmark/benchmark_manifest.json").read_text(encoding="utf-8"))
    errors = []
    for item in manifest["files"]:
        path = Path(item["path"])
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if actual != item["sha256"]:
            errors.append(item["path"])
    if errors:
        raise SystemExit(f"benchmark validation failed for: {errors}")
    print(f"Benchmark {manifest['benchmark_version']} manifest verified")


def _status() -> None:
    database = Path("results/processed/experiments.sqlite")
    if not database.is_file():
        raise SystemExit("experiment database missing; run make corpus")
    connection = sqlite3.connect(database)
    try:
        total = connection.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
        eligible = connection.execute(
            "SELECT COUNT(*) FROM experiments WHERE publication_eligibility = 1"
        ).fetchone()[0]
        failures = connection.execute(
            "SELECT COUNT(*) FROM experiments WHERE validation_status LIKE 'FAIL%'"
        ).fetchone()[0]
    finally:
        connection.close()
    print(
        f"experiment rows: {total}\nfailed gate rows: {failures}\npublication eligible: {eligible}"
    )


def _evidence(claim_id: str | None) -> None:
    claims = yaml.safe_load(Path("research/claims.yaml").read_text(encoding="utf-8"))["claims"]
    if claim_id is None:
        if not claims:
            print("No registered claims. Insufficient publication evidence.")
            return
        for claim in claims:
            print(f"{claim['claim_id']}\t{claim.get('verification_status')}\t{claim['text']}")
        return
    matches = [claim for claim in claims if claim.get("claim_id") == claim_id]
    if not matches:
        raise SystemExit(f"unknown claim: {claim_id}")
    print(yaml.safe_dump(matches[0], sort_keys=False))


def main() -> None:
    parser = argparse.ArgumentParser(prog="unlearn-bench")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in REGISTRIES:
        command = subparsers.add_parser(name)
        command.add_argument("action", choices=["list"])
    subparsers.add_parser("validate")
    subparsers.add_parser("status")
    evidence = subparsers.add_parser("evidence")
    evidence.add_argument("claim_id", nargs="?")
    run = subparsers.add_parser("run")
    run.add_argument("--config", required=True)
    arguments = parser.parse_args()
    if arguments.command in REGISTRIES:
        _list_registry(arguments.command)
    elif arguments.command == "validate":
        _validate()
    elif arguments.command == "status":
        _status()
    elif arguments.command == "evidence":
        _evidence(arguments.claim_id)
    else:
        print(run_experiment(load_config(arguments.config)))


if __name__ == "__main__":
    main()
