import re

import pandas as pd
import pytest

from llm_bench.config import load_config
from llm_bench.data import list_bundles, load_calendar, load_oag_monthly, load_observed
from llm_bench.prompts import build_prompt


@pytest.fixture(scope="module")
def ctx():
    cfg = load_config()
    bundles = list_bundles(cfg, smoke=False) or list_bundles(cfg, smoke=True)
    if not bundles:
        pytest.skip("no Stan export yet")
    return cfg, load_observed(cfg), bundles[0], load_oag_monthly(cfg), load_calendar(cfg)


def test_blind_prompt_leaks_nothing(ctx):
    cfg, obs, b, oag, cal = ctx
    p = build_prompt("direct_blind", cfg, obs, b, oag, cal)
    text = p.user
    assert not re.search(r"\b20\d\d\b", text), "a year leaked"
    for word in ["Hong Kong", "influenza", "COVID", "RSV", "HFMD", "H3", "H1"]:
        assert word.lower() not in text.lower()
    assert sorted(p.keys) == ["A", "B", "C", "D", "E", "F"]


def test_blind_scale_round_trips(ctx):
    cfg, obs, b, oag, cal = ctx
    p = build_prompt("direct_blind", cfg, obs, b, oag, cal)
    last_shown = pd.Series(p.mock["last"], index=cfg.strains)
    last_true = obs.loc[:b.origin].iloc[-1]
    assert ((last_shown * p.scale - last_true).abs() < 1e-6 + 1e-9 * last_true.abs()).all()


def test_semi_prompt_hides_year_and_city(ctx):
    cfg, obs, b, oag, cal = ctx
    p = build_prompt("r0_semi", cfg, obs, b, oag, cal)
    # observed counts can look like years (H1 was 2022 on 2025-05-10), so look for date forms
    assert not re.search(r"\b20\d\d-\d\d", p.user), "ISO date leaked"
    assert not re.search(r"[A-Z][a-z]+,? 20\d\d\b", p.user), "month-year leaked"
    assert str(b.origin.year) + "-" not in p.user
    assert "Hong Kong" not in p.user
    assert "multipliers" in p.user


def test_named_prompt_has_targets(ctx):
    cfg, obs, b, oag, cal = ctx
    p = build_prompt("direct_named", cfg, obs, b, oag, cal)
    target = (b.origin + pd.Timedelta(weeks=4)).strftime("%Y-%m-%d")
    assert target in p.user and "Hong Kong" in p.user
