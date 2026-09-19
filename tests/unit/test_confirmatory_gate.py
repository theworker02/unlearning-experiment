from __future__ import annotations

import pytest

from exact_unlearning.quality import QualityGateError, validate_launch_preconditions


def test_confirmatory_launch_fails_closed_without_frozen_protocol() -> None:
    with pytest.raises(QualityGateError, match="confirmatory preconditions failed"):
        validate_launch_preconditions({"maturity": "CONFIRMATORY", "quality": {}})


def test_publication_is_not_a_runnable_maturity() -> None:
    with pytest.raises(QualityGateError, match="promotion states"):
        validate_launch_preconditions({"maturity": "PUBLICATION"})
