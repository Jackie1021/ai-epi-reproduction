"""非 LLM 的简单基线：持平预测（FluSight baseline 的做法）"""
from __future__ import annotations

import numpy as np


def persistence_quantiles(history: np.ndarray, horizons: int, levels: list[float],
                          rng: np.random.Generator, n: int = 20000) -> np.ndarray:
    """Median = last value; spread = sum of h resampled past week-to-week changes
    (symmetrised). Returns (horizons, len(levels)), floored at 0."""
    y = history[np.isfinite(history)]
    diffs = np.diff(y)
    diffs = np.concatenate([diffs, -diffs]) if diffs.size else np.zeros(1)
    out = np.empty((horizons, len(levels)))
    for h in range(1, horizons + 1):
        paths = y[-1] + rng.choice(diffs, size=(n, h)).sum(axis=1)
        out[h - 1] = np.maximum(np.quantile(paths, levels), 0.0)
    return out
