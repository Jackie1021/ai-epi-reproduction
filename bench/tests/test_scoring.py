import numpy as np

from llm_bench.scoring import interval_wis, wis

LEVELS = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975]
ALPHAS = [0.05, 0.1, 0.2, 0.5]


def test_wis_matches_interval_definition():
    rng = np.random.default_rng(0)
    for _ in range(200):
        q = np.sort(rng.gamma(2.0, 50.0, size=len(LEVELS)))
        y = float(rng.gamma(2.0, 60.0))
        ref = interval_wis(y, dict(zip([round(v, 4) for v in LEVELS], q)), ALPHAS)
        got = wis(np.array([y]), q[None, :], np.array(LEVELS))[0]
        assert np.isclose(got, ref, rtol=1e-12), (got, ref)


def test_wis_zero_for_point_mass_on_truth():
    q = np.full((1, len(LEVELS)), 42.0)
    assert wis(np.array([42.0]), q, np.array(LEVELS))[0] == 0.0


def test_wis_grows_with_error():
    q = np.linspace(90, 110, len(LEVELS))[None, :]
    near = wis(np.array([101.0]), q, np.array(LEVELS))[0]
    far = wis(np.array([300.0]), q, np.array(LEVELS))[0]
    assert far > near > 0
