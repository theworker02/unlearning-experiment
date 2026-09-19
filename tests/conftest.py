from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path


def pytest_configure(config) -> None:
    """Isolate pytest from stale Windows and OneDrive temporary-directory ACLs."""
    if config.option.basetemp is None:
        config.option.basetemp = Path(tempfile.gettempdir()) / (
            f"exact-unlearning-pytest-{os.getpid()}"
        )


def pytest_sessionfinish(session, exitstatus) -> None:
    basetemp = session.config.option.basetemp
    if basetemp is not None:
        shutil.rmtree(Path(basetemp), ignore_errors=True)
