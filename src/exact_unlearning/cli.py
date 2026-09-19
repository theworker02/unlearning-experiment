from __future__ import annotations

import argparse
import datetime as dt
import platform
import traceback
import uuid

from .config import load_config
from .experiment import run_experiment
from .registry import write_failure_record


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an immutable unlearning experiment")
    parser.add_argument("--config", required=True, help="Path to a YAML experiment configuration")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    try:
        output = run_experiment(config)
    except Exception as error:
        failure = {
            "schema_version": 1,
            "failure_id": str(uuid.uuid4()),
            "experiment_id": config.experiment_id,
            "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
            "failure_category": type(error).__name__,
            "exception": str(error),
            "traceback": traceback.format_exc(),
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
            "configuration": config.raw,
            "configuration_sha256": config.sha256,
            "invalidates_batch": None,
            "remediation": None,
            "statistical_observation": False,
        }
        path = write_failure_record(config.raw["output_root"], config.experiment_id, failure)
        print(f"experiment failed; immutable failure record: {path}")
        raise
    print(output)


if __name__ == "__main__":
    main()
