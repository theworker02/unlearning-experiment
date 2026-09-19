from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

SCHEMA_VERSION = 1


def main() -> None:
    source = Path("research/experiment_ledger.csv")
    if not source.is_file():
        raise SystemExit("build the experiment ledger first")
    output = Path("results/processed/experiments.sqlite")
    if output.exists():
        output.unlink()
    connection = sqlite3.connect(output)
    try:
        connection.executescript(
            """
            CREATE TABLE schema_metadata (schema_version INTEGER NOT NULL);
            INSERT INTO schema_metadata VALUES (1);
            CREATE TABLE experiments (
              experiment_id TEXT NOT NULL, phase TEXT, maturity TEXT, status TEXT,
              dataset TEXT, architecture TEXT, method TEXT, deletion_regime TEXT,
              deletion_fraction REAL, seed INTEGER, commit_hash TEXT, start_time TEXT,
              finish_time TEXT, runtime_seconds REAL, validation_status TEXT,
              publication_eligibility INTEGER NOT NULL, failure_reason TEXT,
              superseding_run TEXT, raw_artifact TEXT NOT NULL,
              PRIMARY KEY (experiment_id, method, seed)
            );
            """
        )
        with source.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                connection.execute(
                    "INSERT INTO experiments VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        row["experiment_id"],
                        row["phase"],
                        row["maturity"],
                        row["status"],
                        row["dataset"],
                        row["architecture"],
                        row["method"],
                        row["deletion_regime"],
                        row["deletion_fraction"] or None,
                        row["seed"] or None,
                        row["commit"],
                        row["start_time"],
                        row["finish_time"],
                        row["runtime_seconds"] or None,
                        row["validation_status"],
                        int(row["publication_eligibility"] == "True"),
                        row["failure_reason"],
                        row["superseding_run"],
                        row["raw_artifact"],
                    ),
                )
        connection.commit()
        count = connection.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
    finally:
        connection.close()
    print(f"{output}: schema {SCHEMA_VERSION}, {count} rows")


if __name__ == "__main__":
    main()
