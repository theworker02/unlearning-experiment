from pathlib import Path

import pytest

from exact_unlearning.config import ConfigError, load_config


def test_checked_in_tiny_config_is_valid() -> None:
    config = load_config(Path("configs/experiments/digits_linear_tiny.yaml"))
    assert config.experiment_id == "digits-linear-random-02pct-tiny-v1"
    assert len(config.sha256) == 64


def test_duplicate_oracle_seeds_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        """experiment_id: bad
maturity: VALIDATION
dataset: {}
model: {}
training: {}
original_seed: 1
oracle_seeds: [2, 2]
deletion: {fraction: 0.1}
unlearning: {}
evaluation: {}
output_root: out
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="distinct"):
        load_config(path)
