from pathlib import Path

import numpy as np
import pandas as pd
from llm_bench.config import load_config
from llm_bench.data import OriginBundle, list_bundles
from llm_bench.simulator import MechanisticForecaster, sample_multiplier_paths, simulate_weeks


def _random_state(D=5, N=6, seed=1):
    rng = np.random.default_rng(seed)
    y = rng.dirichlet([50, 1, 1, 20], size=(D, N))
    C = np.zeros((D, N, N))
    C[:, 1, 2] = C[:, 2, 1] = 0.3
    return rng, y, C


def test_compartments_stay_normalised():
    rng, y, C = _random_state()
    D, N = y.shape[:2]
    R0 = rng.uniform(0.8, 2.5, size=(D, N, 28))
    sig, gam = np.full(N, 1 / 3), np.full(N, 1 / 5)
    om = np.full((D, N), 1 / 365)
    inc, S_w, Rt_w, y_end = simulate_weeks(y, R0, sig, gam, om, C)
    assert inc.shape == (D, 4, N)
    assert np.allclose(y_end.sum(-1), 1.0)
    assert np.all(y_end > 0) and np.all(inc >= 0)


def test_higher_r0_means_more_cases():
    rng, y, C = _random_state()
    D, N = y.shape[:2]
    sig, gam, om = np.full(N, 1 / 3), np.full(N, 1 / 5), np.full((D, N), 1 / 365)
    low, *_ = simulate_weeks(y, np.full((D, N, 28), 1.0), sig, gam, om, C)
    high, *_ = simulate_weeks(y, np.full((D, N, 28), 2.0), sig, gam, om, C)
    assert np.all(high[:, -1] > low[:, -1])


def test_multiplier_paths_hit_requested_quantiles():
    q = np.array([[[0.8, 1.0, 1.3]]])
    paths = sample_multiplier_paths(q, 200000, np.random.default_rng(0))
    got = np.quantile(paths[:, 0, 0], [0.1, 0.5, 0.9])
    assert np.allclose(got, [0.8, 1.0, 1.3], rtol=0.01)


def test_matches_stan_on_exported_bundle():
    cfg = load_config()
    bundles = list_bundles(cfg, smoke=False) or list_bundles(cfg, smoke=True)
    b = bundles[0] if bundles else OriginBundle(Path(__file__).parent / "fixtures/stan_smoke/2025-06-07")
    f = MechanisticForecaster(b, cfg.strains)
    mu = f.expected_cases(np.ones((f.n_draws, len(cfg.strains), cfg.horizons)))
    st = b.stan_mu_fc[b.stan_mu_fc["fc_mode"] == 1].pivot_table(
        index=["draw", "horizon"], columns="strain", values="mu")
    py = pd.DataFrame(mu.reshape(-1, len(cfg.strains)), columns=cfg.strains,
                      index=pd.MultiIndex.from_product([f.draw_ids, range(1, cfg.horizons + 1)]))
    st = st.loc[py.index, cfg.strains]
    assert np.allclose(py.to_numpy(), st.to_numpy(), rtol=1e-9)
