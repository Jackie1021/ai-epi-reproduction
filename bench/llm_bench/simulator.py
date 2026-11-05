"""机制模型前推：ai_epi.stan 里 seirs_cross_step / simulate_weeks 的 numpy 逐行移植

所有数组第一维是后验样本 D；状态 y 形状 (D, N, 4)，最后一维依次是 S, E, I, R 的人群比例。
和 Stan 的一致性由 `llm_bench validate-sim` 和 tests/test_simulator.py 检查。
"""
from __future__ import annotations

import numpy as np

from .data import OriginBundle

Z90 = 1.2815515655446004  # standard-normal 90th percentile


def seirs_cross_step(y, beta, sig, gam, om, C):
    """One day. beta, om: (D, N); sig, gam: (N,); C: (D, N, N) with C[d, j, k] = j -> k."""
    S, E, I, R = y[..., 0], y[..., 1], y[..., 2], y[..., 3]
    new_inf = np.minimum(beta * I * S, 0.5 * S)
    cross_out = S * np.einsum("djk,dj->dk", C, new_inf)  # S_k * sum_j C[j,k] * new_inf_j
    out = new_inf + cross_out
    over = out > 0.9 * S                                  # outflow capped at 90% of S_k
    scale = np.ones_like(out)
    scale[over] = 0.9 * S[over] / out[over]
    ni, co = new_inf * scale, cross_out * scale
    y_new = np.stack([S - ni - co + om * R,
                      E + ni - sig * E,
                      I + sig * E - gam * I,
                      R + gam * I - om * R + co], axis=-1)
    y_new = np.maximum(y_new, 1e-10)
    return y_new / y_new.sum(axis=-1, keepdims=True)


def simulate_weeks(y0, R0_daily, sig, gam, om, C):
    """R0_daily: (D, N, 7*W). Returns weekly incidence fraction, mean S, mean Rt (each (D, W, N))
    and the final state."""
    D, N, T = R0_daily.shape
    W = T // 7
    inc = np.zeros((D, W, N))
    S_w = np.zeros((D, W, N))
    Rt_w = np.zeros((D, W, N))
    y = y0.copy()
    for w in range(W):
        for d in range(7):
            t = 7 * w + d
            S = y[..., 0]
            inc[:, w] += sig * y[..., 1]
            S_w[:, w] += S / 7
            Rt_w[:, w] += R0_daily[:, :, t] * S / 7
            y = seirs_cross_step(y, R0_daily[:, :, t] * gam, sig, gam, om, C)
    return inc, S_w, Rt_w, y


class MechanisticForecaster:
    """Propagates every exported posterior draw of one origin under a given R0 path."""

    def __init__(self, bundle: OriginBundle, strains: list[str], max_draws: int | None = None):
        n = len(bundle.draws)
        keep = np.arange(n) if not max_draws or max_draws >= n else \
            np.unique(np.round(np.linspace(0, n - 1, max_draws)).astype(int))
        self.strains = strains
        self.draw_ids = bundle.draws["draw"].to_numpy()[keep]
        self.y0 = bundle.state0(strains)[keep]
        self.R0_end = bundle.per_strain("R0_end", strains)[keep]
        self.om = bundle.per_strain("omega", strains)[keep]
        self.obs_scale = bundle.per_strain("obs_scale", strains)[keep]
        self.phi = bundle.per_strain("phi", strains)[keep]
        self.C = bundle.cross(strains)[keep]
        self.sig = bundle.constant("sigma_E", strains)
        self.gam = bundle.constant("gamma_I", strains)
        self.R0_min = float(bundle.meta["constants"]["R0_min"])
        self.R0_max = float(bundle.meta["constants"]["R0_max"])

    @property
    def n_draws(self) -> int:
        return len(self.draw_ids)

    def expected_cases(self, mult: np.ndarray) -> np.ndarray:
        """mult: (D, N, W) weekly multipliers on R0_end (1 = Stan's constant-R0 mode).
        Returns expected reported cases (D, W, N), exactly as mu_fc in Stan."""
        R0_daily = np.clip(self.R0_end[:, :, None] * np.repeat(mult, 7, axis=2),
                           self.R0_min, self.R0_max)
        inc, _, _, _ = simulate_weeks(self.y0, R0_daily, self.sig, self.gam, self.om, self.C)
        return np.maximum(0.1, inc * self.obs_scale[:, None, :])

    def predictive(self, mu: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Negative-binomial draws via gamma-Poisson, as safe_neg_binomial_2_rng in Stan."""
        phi = self.phi[:, None, :]
        lam = rng.gamma(shape=phi, scale=mu / phi)
        return rng.poisson(np.minimum(lam, 1e9)).astype(float)


def sample_multiplier_paths(q: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """q: (N, W, 3) = p10/p50/p90 multipliers per strain and week -> paths (n, N, W).

    Split-normal on the log scale (each side's width set by p10 / p90); one z per path and
    strain shared across weeks, so a path that runs hot in week 1 stays hot."""
    lq = np.log(q)
    mid = lq[..., 1]
    sd_lo = np.maximum((lq[..., 1] - lq[..., 0]) / Z90, 1e-6)
    sd_hi = np.maximum((lq[..., 2] - lq[..., 1]) / Z90, 1e-6)
    z = rng.standard_normal((n, q.shape[0], 1))
    return np.exp(mid + np.where(z < 0, z * sd_lo, z * sd_hi))
