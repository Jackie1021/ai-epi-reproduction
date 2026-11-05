"""评分：WIS（加权区间分数）、中位数绝对误差、区间覆盖率、相对 WIS

WIS 按 Bracher et al. (2021) 定义，等价于 2 x 各分位数 pinball loss 的平均。
log 尺度（Bosse et al. 2023）：对观测和分位数同时取 log1p 后再算，适合跨病原体比较。
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

KEYS = ["origin", "target_date", "horizon", "strain", "model", "arm"]


def wis(y: np.ndarray, q: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """y: (n,), q: (n, Q) sorted by level -> (n,)"""
    below = (y[:, None] <= q).astype(float)
    pinball = (below - levels[None, :]) * (q - y[:, None])   # (1{y<=q} - tau)(q - y) >= 0
    return 2.0 * pinball.mean(axis=1)


def interval_wis(y: float, q: dict[float, float], alphas: list[float]) -> float:
    """Reference implementation from the interval-score definition (used in tests)."""
    total = 0.5 * abs(y - q[0.5])
    for a in alphas:
        lo, hi = q[round(a / 2, 4)], q[round(1 - a / 2, 4)]
        interval = (hi - lo) + 2 / a * max(lo - y, 0) + 2 / a * max(y - hi, 0)
        total += a / 2 * interval
    return total / (len(alphas) + 0.5)


def score_forecasts(fc: pd.DataFrame, obs_wide: pd.DataFrame, levels: list[float]) -> pd.DataFrame:
    levels_r = [round(q, 4) for q in levels]
    f = fc.assign(quantile=fc["quantile"].round(4))
    f = f[f["quantile"].isin(levels_r)]
    wide = f.pivot_table(index=KEYS, columns="quantile", values="value", aggfunc="first")
    complete = wide.notna().all(axis=1)
    if (~complete).any():
        print(f"warning: {(~complete).sum()} forecasts miss some quantile levels and are skipped")
    wide = wide[complete][levels_r].reset_index()

    obs = obs_wide.stack().rename("y").reset_index()
    obs.columns = ["target_date", "strain", "y"]
    df = wide.merge(obs, on=["target_date", "strain"], how="inner")
    q = np.sort(df[levels_r].to_numpy(float), axis=1)
    y = df["y"].to_numpy(float)
    lv = np.array(levels_r)
    df["wis"] = wis(y, q, lv)
    df["wis_log"] = wis(np.log1p(y), np.log1p(np.maximum(q, 0)), lv)
    df["ae_median"] = np.abs(y - q[:, levels_r.index(0.5)])
    for width in (50, 80, 90, 95):
        lo, hi = round((1 - width / 100) / 2, 4), round(1 - (1 - width / 100) / 2, 4)
        if lo in levels_r and hi in levels_r:
            df[f"cov{width}"] = ((y >= q[:, levels_r.index(lo)]) & (y <= q[:, levels_r.index(hi)])).astype(float)
    return df.drop(columns=levels_r)


def relative_wis(scores: pd.DataFrame, reference: str, col: str = "wis") -> pd.DataFrame:
    """Per (model, arm, strain, horizon): sum WIS / sum reference WIS over the same targets.
    Pooled row per (model, arm, horizon) with strain = 'all': geometric mean over strains."""
    ref = scores[scores["model"] == reference][["origin", "strain", "horizon", col]]
    ref = ref.rename(columns={col: "ref"})
    m = scores.merge(ref, on=["origin", "strain", "horizon"], how="inner")
    g = m.groupby(["model", "arm", "strain", "horizon"])
    per = (g[col].sum() / g["ref"].sum()).rename("rel_wis").to_frame()
    per["n"] = g.size()
    per = per.reset_index()
    pooled = (per.assign(log=np.log(per["rel_wis"].clip(lower=1e-12)))
              .groupby(["model", "arm", "horizon"])
              .agg(log=("log", "mean"), n=("n", "sum")).reset_index())
    pooled["rel_wis"] = np.exp(pooled.pop("log"))
    pooled["strain"] = "all"
    return pd.concat([per, pooled[per.columns]], ignore_index=True)


def coverage_table(scores: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in scores.columns if c.startswith("cov")]
    return scores.groupby(["model", "arm"])[cols].mean().reset_index()


def paired_gain(scores: pd.DataFrame, base_arm: str, new_arm: str, col: str = "wis",
                n_boot: int = 4000, seed: int = 20260925, strains: list | None = None) -> pd.DataFrame:
    """Same model, two designs, matched targets (origin, strain, horizon).

    ratio = WIS(new_arm) / WIS(base_arm): per pathogen the summed WIS ratio, combined across
    pathogens by the geometric mean (as in relative_wis); < 1 means new_arm is better.
    95% interval: bootstrap over forecast origins (whole origins resampled, all pathogens and
    horizons of an origin kept together). p_better = share of bootstrap ratios below 1.
    One row per (model, horizon) plus horizon = 'all'."""
    keys = ["model", "origin", "strain", "horizon"]
    if strains is not None:                               # e.g. influenza only
        scores = scores[scores["strain"].isin(strains)]
    a = scores.loc[scores["arm"] == base_arm, keys + [col]]
    b = scores.loc[scores["arm"] == new_arm, keys + [col]]
    m = a.merge(b, on=keys, suffixes=("_a", "_b"))
    rng = np.random.default_rng(seed)
    rows = []
    for model, g in m.groupby("model"):
        origins = np.sort(g["origin"].unique())
        strains = np.sort(g["strain"].unique())
        idx = rng.integers(0, len(origins), size=(n_boot, len(origins)))
        for h in sorted(g["horizon"].unique()) + ["all"]:
            gh = g if h == "all" else g[g["horizon"] == h]
            tab = gh.pivot_table(index="origin", columns="strain", values=[f"{col}_a", f"{col}_b"],
                                 aggfunc="sum").reindex(origins).fillna(0.0)
            A, B = tab[f"{col}_a"][strains].to_numpy(), tab[f"{col}_b"][strains].to_numpy()

            def gm_ratio(sa, sb):
                r = np.clip(sb, 1e-12, None) / np.clip(sa, 1e-12, None)
                return np.exp(np.log(r).mean(axis=-1))

            est = float(gm_ratio(A.sum(0), B.sum(0)))
            boot = gm_ratio(A[idx].sum(1), B[idx].sum(1))
            lo, hi = np.percentile(boot, [2.5, 97.5])
            rows.append((model, h, len(origins), est, lo, hi, float((boot < 1).mean())))
    return pd.DataFrame(rows, columns=["model", "horizon", "n_origins", "ratio", "lo95", "hi95",
                                       "p_better"])


# ---------------------------------------------------------------- external-information experiment
# 2x2 factorial (calendar & weather x importation pressure) + placebo; protocol:
# bench/protocol_context_experiment.md. Each effect is a contrast of log summed WIS:
# per model and pathogen, L[arm] = log(sum of WIS over the matched targets); effect =
# mean over pathogens of sum_arm weight * L[arm]; reported as a ratio exp(effect) (< 1 = better).
FLU = ["B", "H3", "H1"]
CONTEXT_EFFECTS = {   # name -> (pathogens, weights on log WIS); the first four are pre-specified
    "calendar_weather": ("all", {"r0_ctx": .5, "r0_semi": -.5, "r0_imp": .5, "r0_imponly": -.5}),
    "importation": ("influenza", {"r0_imponly": .5, "r0_semi": -.5, "r0_imp": .5, "r0_ctx": -.5}),
    "real_vs_placebo": ("influenza", {"r0_imp": 1, "r0_placebo": -1}),
    "interaction": ("influenza", {"r0_imp": 1, "r0_ctx": -1, "r0_imponly": -1, "r0_semi": 1}),
    "importation_alone": ("influenza", {"r0_imponly": 1, "r0_semi": -1}),
    "importation_given_context": ("influenza", {"r0_imp": 1, "r0_ctx": -1}),
    "placebo_vs_none": ("influenza", {"r0_placebo": 1, "r0_ctx": -1}),
    "calendar_weather_alone": ("all", {"r0_ctx": 1, "r0_semi": -1}),
}
STRATIFIED = ("importation", "importation_given_context", "real_vs_placebo")


def import_signal(imp: pd.DataFrame, origins, delay_days: int = 14, high: float = 1.0,
                  rise: float = 1.3) -> pd.DataFrame:
    """Importation signal per (origin, influenza subtype), from what the prompt shows: the latest
    4 weeks published `delay_days` before the origin. 'high or rising' = latest relative pressure
    > `high` or latest / oldest of the 4 weeks > `rise`; otherwise 'low and flat'."""
    rows = []
    for o in pd.to_datetime(pd.Series(origins)).unique():
        for k in FLU:
            d = imp[(imp["subtype"] == k) & (imp["week_end"] <= o - pd.Timedelta(days=delay_days))]
            d = d.sort_values("week_end").tail(4)
            if len(d) < 4:
                continue
            p, t = float(d["pressure_rel"].iloc[-1]), float(d["pressure_rel"].iloc[-1] / d["pressure_rel"].iloc[0])
            rows.append((o, k, p, t, "high_or_rising" if (p > high or t > rise) else "low_and_flat"))
    return pd.DataFrame(rows, columns=["origin", "strain", "pressure_rel", "trend", "stratum"])


def context_effects(scores: pd.DataFrame, strains: list[str], signal: pd.DataFrame | None = None,
                    col: str = "wis", n_boot: int = 4000, seed: int = 20260925) -> pd.DataFrame:
    """Effects of CONTEXT_EFFECTS per model and pooled over models, overall, by horizon, by pathogen
    and (STRATIFIED effects) by importation signal.

    Analysis set: per model, the targets (origin, pathogen, horizon) scored for all designs.
    Per-model 95% interval: bootstrap over forecast origins. Pooled = equal-weight mean of the
    per-model log effects; its interval resamples origins (the same draw for every model, since the
    models forecast the same targets) and then models with replacement, so it reflects both the
    sampling of origins and the heterogeneity between models (a random-effects-type interval that
    does not assume independent models)."""
    arms = sorted({a for _, w in CONTEXT_EFFECTS.values() for a in w})
    keys = ["model", "origin", "strain", "horizon"]
    s = scores[scores["arm"].isin(arms)]
    wide = s.pivot_table(index=keys, columns="arm", values=col, aggfunc="first")
    if wide.empty or not set(arms) <= set(wide.columns):
        return pd.DataFrame()
    wide = wide[arms].dropna().reset_index()                     # complete cases
    models = sorted(wide["model"].unique())
    origins = np.sort(wide["origin"].unique())
    o_idx = {o: i for i, o in enumerate(origins)}
    rng = np.random.default_rng(seed)
    boot_o = rng.integers(0, len(origins), size=(n_boot, len(origins)))
    counts = np.zeros((n_boot, len(origins)))                   # times each origin is drawn
    np.add.at(counts, (np.arange(n_boot)[:, None], boot_o), 1.0)
    boot_m = rng.integers(0, len(models), size=(n_boot, len(models)))
    strata = None
    if signal is not None and len(signal):
        strata = signal.assign(origin=pd.to_datetime(signal["origin"])).set_index(["origin", "strain"])["stratum"]

    def cube(d, ss):
        """-> (M, A, O, K) summed WIS and (M, O, K) presence"""
        X = np.zeros((len(models), len(arms), len(origins), len(ss)))
        P = np.zeros((len(models), len(origins), len(ss)), bool)
        g = d.groupby(["model", "origin", "strain"])[arms].sum()
        for (m, o, k), v in g.iterrows():
            mi, oi, ki = models.index(m), o_idx[o], ss.index(k)
            X[mi, :, oi, ki] = v.to_numpy(float)
            P[mi, oi, ki] = True
        return X, P

    def effect(X, P, w, cnt):
        """X (M, A, O, K); cnt (B, O) -> point (M,), bootstrap per-model log effect (B, M)"""
        with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            def log_eff(S, n):                                    # S (..., A, K), n (..., K)
                L = np.log(np.where(S > 0, S, np.nan))
                e = np.tensordot(w, L, axes=([0], [-2]))           # (..., K)
                e = np.where(n > 0, e, np.nan)
                return np.nanmean(e, axis=-1)
            point = log_eff(X.sum(2), P.sum(1))                    # (M,)
            Xb = np.einsum("maok,bo->bmak", X, cnt)
            nb = np.einsum("mok,bo->bmk", P.astype(float), cnt)
            boot = log_eff(Xb, nb)                                 # (B, M)
        return point, boot

    rows = []

    def record(name, paths, stratum, h, d, ss, w):
        X, P = cube(d, ss)
        point, boot = effect(X, P, w, counts)
        n_or = P.any(2).sum(1)
        for mi, m in enumerate(models):
            if n_or[mi] == 0 or not np.isfinite(point[mi]):
                continue
            b = boot[:, mi][np.isfinite(boot[:, mi])]
            lo, hi = np.percentile(b, [2.5, 97.5]) if len(b) else (np.nan, np.nan)
            rows.append((name, paths, stratum, m, h, int(n_or[mi]), int(P[mi].sum()),
                         float(np.exp(point[mi])), float(np.exp(lo)), float(np.exp(hi)),
                         float((b < 0).mean()) if len(b) else np.nan))
        ok = np.isfinite(point) & (n_or > 0)
        if ok.sum() >= 2:
            pb = np.take_along_axis(boot, boot_m, axis=1)          # resample models
            okb = np.take_along_axis(np.broadcast_to(ok, boot.shape), boot_m, axis=1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                pooled_b = np.nanmean(np.where(okb, pb, np.nan), axis=1)
            pooled_b = pooled_b[np.isfinite(pooled_b)]
            lo, hi = np.percentile(pooled_b, [2.5, 97.5])
            rows.append((name, paths, stratum, "pooled", h, int(n_or[ok].min()), int(P[ok].sum()),
                         float(np.exp(point[ok].mean())), float(np.exp(lo)), float(np.exp(hi)),
                         float((pooled_b < 0).mean())))

    for name, (paths, weights) in CONTEXT_EFFECTS.items():
        ss = [k for k in (FLU if paths == "influenza" else strains) if k in set(wide["strain"])]
        w = np.array([weights.get(a, 0.0) for a in arms])
        base = wide[wide["strain"].isin(ss)]
        for h in ["all"] + sorted(base["horizon"].unique()):
            record(name, paths, "all", h, base if h == "all" else base[base["horizon"] == h], ss, w)
        for k in ss:                                              # by pathogen, all horizons
            record(name, k, "all", "all", base[base["strain"] == k], [k], w)
        if name in STRATIFIED and strata is not None:
            lab = strata.reindex(pd.MultiIndex.from_frame(base[["origin", "strain"]].assign(
                origin=pd.to_datetime(base["origin"])))).to_numpy()
            for st in ("high_or_rising", "low_and_flat"):
                record(name, paths, st, "all", base[lab == st], ss, w)
    return pd.DataFrame(rows, columns=["effect", "pathogens", "stratum", "model", "horizon", "n_origins",
                                       "n_cells", "ratio", "lo95", "hi95", "p_better"])


# ---------------------------------------------------------------- when does the LLM help? (amendment 7)
def phase_at_origin(bundles, strains, grow: float = 1.05, decline: float = 0.95) -> pd.DataFrame:
    """Epidemic phase per (origin, pathogen) known at the origin: the fitted model's Rt in the last
    week before the origin (posterior median). growing: Rt >= grow; declining: Rt <= decline."""
    rows = []
    for b in bundles:
        lat = b.latent
        for s in strains:
            r = lat[(lat["variable"] == "Rt") & (lat["strain"] == s)].sort_values("date")
            if r.empty:
                continue
            rt = float(r["med"].iloc[-1])
            ph = "growing" if rt >= grow else ("declining" if rt <= decline else "stable")
            rows.append((b.origin, s, rt, ph))
    return pd.DataFrame(rows, columns=["origin", "strain", "Rt", "phase"])


def stratified_gain(d: pd.DataFrame, by: str, new: str, base: str, n_boot: int = 4000,
                    seed: int = 20260927, strata: str | None = None) -> pd.DataFrame:
    """Paired WIS ratio new / base within each level of `by`, per model and pooled.

    d: one row per (model, origin, strain, horizon) with columns `by`, `new`, `base`.
    Per model and level: summed WIS per pathogen, log ratio averaged over pathogens (as elsewhere);
    95% interval: bootstrap over origins. Pooled: equal-weight mean of the per-model log ratios,
    two-level bootstrap (origins shared across models, then models resampled).
    strata: column of `d` (constant within an origin, e.g. the study period); origins are then
    resampled within each stratum (protocol amendment 8, pooled 2015-2026 analysis)."""
    rng = np.random.default_rng(seed)
    rows = []
    for level, sub in d.groupby(by):
        models = sorted(sub["model"].unique())
        origins = np.sort(sub["origin"].unique())
        strains = sorted(sub["strain"].unique())
        oi = {o: i for i, o in enumerate(origins)}
        si = {s: i for i, s in enumerate(strains)}
        A = np.zeros((len(models), len(origins), len(strains)))
        B = np.zeros_like(A)
        P = np.zeros_like(A, dtype=bool)
        g = sub.groupby(["model", "origin", "strain"])[[new, base]].sum()
        for (m, o, s), v in g.iterrows():
            mi = models.index(m)
            A[mi, oi[o], si[s]], B[mi, oi[o], si[s]], P[mi, oi[o], si[s]] = v[new], v[base], True
        if strata is None:
            idx = rng.integers(0, len(origins), size=(n_boot, len(origins)))
        else:
            lab = sub.drop_duplicates("origin").set_index("origin")[strata].reindex(origins).to_numpy()
            idx = np.concatenate([pos[rng.integers(0, len(pos), size=(n_boot, len(pos)))]
                                  for pos in (np.where(lab == v)[0] for v in pd.unique(lab))], axis=1)
        cnt = np.zeros((n_boot, len(origins)))
        np.add.at(cnt, (np.arange(n_boot)[:, None], idx), 1.0)
        with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            def lr(a, b, n):
                x = np.log(np.where(a > 0, a, np.nan)) - np.log(np.where(b > 0, b, np.nan))
                return np.nanmean(np.where(n > 0, x, np.nan), axis=-1)
            point = lr(A.sum(1), B.sum(1), P.sum(1))                                  # (M,)
            boot = lr(np.einsum("mok,bo->bmk", A, cnt), np.einsum("mok,bo->bmk", B, cnt),
                      np.einsum("mok,bo->bmk", P.astype(float), cnt))                 # (B, M)
            n_or = P.any(2).sum(1)
            for mi, m in enumerate(models):
                bb = boot[:, mi][np.isfinite(boot[:, mi])]
                if not np.isfinite(point[mi]) or not len(bb):
                    continue
                lo, hi = np.percentile(bb, [2.5, 97.5])
                rows.append((level, m, int(n_or[mi]), int(P[mi].sum()), float(np.exp(point[mi])),
                             float(np.exp(lo)), float(np.exp(hi)), float((bb < 0).mean())))
            ok = np.isfinite(point)
            if ok.sum() >= 2:
                mres = rng.integers(0, len(models), size=(n_boot, len(models)))
                pb = np.take_along_axis(boot, mres, axis=1)
                okb = np.take_along_axis(np.broadcast_to(ok, boot.shape), mres, axis=1)
                pooled = np.nanmean(np.where(okb, pb, np.nan), axis=1)
                pooled = pooled[np.isfinite(pooled)]
                lo, hi = np.percentile(pooled, [2.5, 97.5])
                rows.append((level, "pooled", int(n_or[ok].min()), int(P[ok].sum()),
                             float(np.exp(point[ok].mean())), float(np.exp(lo)), float(np.exp(hi)),
                             float((pooled < 0).mean())))
    out = pd.DataFrame(rows, columns=["level", "model", "n_origins", "n_cells", "ratio", "lo95", "hi95",
                                      "p_better"])
    out.insert(0, "by", by)
    return out


def direction_accuracy(fc: pd.DataFrame, obs_wide: pd.DataFrame, band: float = 1.1) -> pd.DataFrame:
    """Per forecast target: does the forecast median move in the same direction as the observation from
    the last observed week (up: > x band, down: < x 1/band, flat otherwise)?"""
    med = fc[fc["quantile"].round(4) == 0.5][["origin", "target_date", "horizon", "strain", "model", "arm", "value"]]
    o = obs_wide.stack().rename("y").reset_index()
    o.columns = ["date", "strain", "y"]
    d = med.merge(o.rename(columns={"date": "origin", "y": "y0"}), on=["origin", "strain"]) \
           .merge(o.rename(columns={"date": "target_date", "y": "yt"}), on=["target_date", "strain"])
    def cls(r):
        return np.select([r > band, r < 1 / band], [1, -1], 0)
    d["obs_dir"] = cls((d["yt"] + 1) / (d["y0"] + 1))
    d["fc_dir"] = cls((d["value"] + 1) / (d["y0"] + 1))
    d["correct"] = (d["obs_dir"] == d["fc_dir"]).astype(float)
    return d
