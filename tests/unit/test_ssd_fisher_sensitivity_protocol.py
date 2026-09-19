from pathlib import Path

import pytest
import yaml

from scripts.validate_ssd_fisher_sensitivity_protocol import validate


def test_ssd_fisher_sensitivity_protocol_is_bounded_and_execution_disabled() -> None:
    result = validate()
    assert result["seed_count"] == 3
    assert result["estimator_conditions_per_seed"] == 6
    assert result["planned_full_fisher_estimates"] == 18
    assert result["planned_ssd_updates"] == 18


def test_ssd_fisher_sensitivity_protocol_rejects_test_access(tmp_path: Path) -> None:
    protocol = yaml.safe_load(
        Path("study/preregistration/ssd_fisher_estimator_sensitivity_v1.yaml").read_text()
    )
    protocol["data_access"]["test_split"] = "permitted"
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    with pytest.raises(SystemExit, match="test split must be prohibited"):
        validate(path)
