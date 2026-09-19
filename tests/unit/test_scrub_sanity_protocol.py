from pathlib import Path

import yaml

from scripts.validate_scrub_sanity_protocol import validate


def test_scrub_sanity_protocol_is_fixed_setting_and_explicitly_authorized() -> None:
    validate()
    protocol = yaml.safe_load(
        Path("study/preregistration/scrub_sanity_reproduction_v1.yaml").read_text()
    )
    assert protocol["scope"]["architecture"] == "small_cnn_bn"
    assert protocol["compute_control"]["planned_unlearning_runs"] == 3
    assert protocol["status"] == "LOCKED"
    assert protocol["execution_authorized"] is True
