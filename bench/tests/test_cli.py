from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_bench.cli import _arms
from llm_bench.config import Config


def test_arm_filter_requires_enabled_arms():
    config = Config(
        raw={"arms": ["direct_blind"], "context_experiment": {"arms": ["r0_imp"]}},
        path=Path("config_reviewer.yaml"),
    )
    assert _arms(config, SimpleNamespace(arms=None)) == ["direct_blind", "r0_imp"]
    assert _arms(config, SimpleNamespace(arms="r0_imp, direct_blind, r0_imp")) == ["r0_imp", "direct_blind"]
    with pytest.raises(SystemExit, match="not enabled.*r0_ctx"):
        _arms(config, SimpleNamespace(arms="r0_ctx"))
