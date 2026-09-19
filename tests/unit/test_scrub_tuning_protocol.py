from pathlib import Path

import pytest
import yaml

from scripts.validate_scrub_tuning_protocol import validate


def test_scrub_tuning_protocol_is_bounded_and_execution_disabled() -> None:
    result = validate()
    assert result["candidate_count"] == 8
    assert result["candidate_runs"] == 24
    assert result["planned_total_model_updates"] == 24


def test_scrub_tuning_protocol_rejects_test_access(tmp_path: Path) -> None:
    protocol = yaml.safe_load(Path("study/preregistration/scrub_tuning_v1.yaml").read_text())
    protocol["validation_only_tuning"]["selection_data"]["test_split"] = "permitted"
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    with pytest.raises(SystemExit, match="test split must be prohibited"):
        validate(path)
