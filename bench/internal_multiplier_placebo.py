"""INTERNAL CHECK, not part of the protocol (post hoc; decided 2 October 2026 before any computation).

Does the LLM's judgement in design (iii) carry information, beyond the extra spread its multipliers add?
No new LLM request: the saved design-(iii) answers are re-propagated with altered multipliers.

  P1 "no judgement":  each answer's p50 set to 1 for every pathogen and week; p10 and p90 divided by p50,
                      so the LLM's own log-scale widths are kept (same split-normal shape).
  P2 "wrong time":    the same model's multipliers taken from another origin (one random derangement of
                      that model's origins, seed 20261002), same pathogen and week.
Monte Carlo seeds are those of the original forecast (runner.build_forecasts), so P1 differs from the LLM
forecast only through the median.

Comparisons (paired, same targets; pooled over models as in the main analysis, two-level bootstrap):
  primary  LLM / P1, natural scale, all phases
  also     log scale; by phase at the origin; by horizon; LLM / P2; P1 / constant R0; 90% coverage
Models: at least 20 (main) or 52 (2015-2019) scored origins for design (iii).

  cd bench && .venv/bin/python internal_multiplier_placebo.py     -> runs/<run>/internal/
"""
from __future__ import annotations

import json
import zlib

import numpy as np
import pandas as pd

from llm_bench.config import load_config
from llm_bench.data import list_bundles, load_observed
from llm_bench.runner import Runner, run_dir
from llm_bench.scoring import score_forecasts, stratified_gain
from llm_bench.simulator import MechanisticForecaster, sample_multiplier_paths

K = ["origin", "target_date", "horizon", "strain"]
ARM = "r0_semi"


def forecasts(cfg, run: str, models: list[str]) -> pd.DataFrame:
    strains, levels, H, seed = cfg.strains, cfg.quantiles, cfg.horizons, int(cfg.get("seed", 1))
    runner = Runner(cfg, run, [], list(cfg["arms"]))
    ans = {(r["origin"], r["model"]): np.array(r["multipliers"]) for r in runner._all_saved()
           if r["arm"] == ARM and r["sample"] == 0 and r["model"] in models and "multipliers" in r}
    # P2: one derangement of each model's origins
    rng0 = np.random.default_rng(20261002)
    swap = {}
    for m in models:
        os_ = sorted(o for (o, mm) in ans if mm == m)
        perm = list(rng0.permutation(os_))
        for a, b in zip(perm, perm[1:] + perm[:1]):
            swap[(a, m)] = (b, m)
    rows = []
    for b in list_bundles(cfg):
        okey = f"{b.origin:%Y-%m-%d}"
        sim = None
        for m in models:
            if (okey, m) not in ans:
                continue
            sim = sim or MechanisticForecaster(b, strains, cfg.get("sim_draws"))
            q = ans[(okey, m)]
            variants = {"llm": q, "P1": q / q[..., 1:2], "P2": ans[swap[(okey, m)]]}
            for name, qm in variants.items():
                rng = np.random.default_rng([seed, zlib.crc32("|".join(map(str, (okey, ARM, m, "main"))).encode())])
                paths = sample_multiplier_paths(qm, sim.n_draws, rng)
                pred = sim.predictive(sim.expected_cases(paths), rng)
                qq = np.quantile(pred, levels, axis=0).transpose(2, 1, 0)        # (N, H, Q)
                for i, s in enumerate(strains):
                    for h in range(H):
                        for j, lev in enumerate(levels):
                            rows.append((b.origin, b.origin + pd.Timedelta(weeks=h + 1), h + 1, s, m, name, lev,
                                         float(qq[i, h, j])))
    return pd.DataFrame(rows, columns=["origin", "target_date", "horizon", "strain", "model", "arm", "quantile", "value"])


def analyse(run: str, cfg_file: str | None, min_origins: int, *, replay: bool = False) -> pd.DataFrame:
    cfg = load_config(cfg_file)
    d = run_dir(run)
    sc_main = pd.read_parquet(d / "scores.parquet")
    n = sc_main[sc_main["arm"] == ARM].groupby("model")["origin"].nunique()
    models = sorted(n[n >= min_origins].index)
    out = d / "internal"
    out.mkdir(exist_ok=True)
    predictions = out / "forecasts_multiplier_placebo.parquet"
    if replay:
        fc = pd.read_parquet(predictions)
    else:
        fc = forecasts(cfg, run, models)
        fc.to_parquet(predictions, index=False)
    sc = score_forecasts(fc, load_observed(cfg), cfg.quantiles)
    sc.to_parquet(out / "scores_multiplier_placebo.parquet", index=False)
    # reproduction check against the stored design-(iii) scores
    chk = sc[sc["arm"] == "llm"][K + ["model", "wis"]].merge(
        sc_main[sc_main["arm"] == ARM][K + ["model", "wis"]], on=K + ["model"], suffixes=("", "_stored"))
    print(f"{run}: {len(models)} models; reproduction max |WIS diff| = {np.abs(chk.wis - chk.wis_stored).max():.3g} "
          f"over {len(chk)} targets")
    ph = pd.read_csv(d / "table_phase_at_origin.csv", parse_dates=["origin"])[["origin", "strain", "phase"]]
    ref = sc_main[sc_main["model"] == "stan_const"][K + ["wis", "wis_log"]]
    wide = {a: sc[sc["arm"] == a][K + ["model", "wis", "wis_log", "cov90"]] for a in ("llm", "P1", "P2")}
    res = []
    for new, base, label in (("llm", "P1", "LLM / P1 (no judgement)"), ("llm", "P2", "LLM / P2 (wrong time)"),
                             ("P1", "ref", "P1 / constant R0"), ("P2", "ref", "P2 / constant R0"),
                             ("llm", "ref", "LLM / constant R0")):
        a = wide[new]
        b = ref if base == "ref" else wide[base]
        on = K if base == "ref" else K + ["model"]
        x = a.merge(b, on=on, suffixes=("", "_b")).merge(ph, on=["origin", "strain"])
        x["all"] = "all"
        for col, scale in (("wis", "natural"), ("wis_log", "log")):
            for by in ("all", "phase", "horizon"):
                g = stratified_gain(x, by, col, col + "_b")
                g = g[g["model"] == "pooled"]
                for r in g.itertuples():
                    res.append((run, label, scale, by, str(r.level), r.n_origins, r.ratio, r.lo95, r.hi95, r.p_better))
    per = []                                     # per model, LLM / P2 (reported as the post hoc check)
    x = wide["llm"].merge(wide["P2"], on=K + ["model"], suffixes=("", "_b"))
    x["all"] = "all"
    for col, scale in (("wis", "natural"), ("wis_log", "log")):
        g = stratified_gain(x, "all", col, col + "_b")
        per.append(g[g["model"] != "pooled"].assign(run=run, scale=scale))
    pd.concat(per, ignore_index=True).to_csv(out / "table_multiplier_placebo_P2_per_model.csv", index=False)
    cov = {a: wide[a].groupby("model")["cov90"].mean().mean() for a in wide}
    tab = pd.DataFrame(res, columns=["run", "comparison", "scale", "by", "level", "n_origins", "ratio", "lo95", "hi95",
                                     "p_better"])
    tab.to_csv(out / "table_multiplier_placebo.csv", index=False)
    (out / "multiplier_placebo_coverage.json").write_text(json.dumps({k: round(v, 4) for k, v in cov.items()}))
    v = tab[tab["by"].isin(["all", "phase"])]
    print(v[["comparison", "scale", "level", "ratio", "lo95", "hi95", "p_better"]].round(3).to_string(index=False))
    print("90% coverage:", {k: round(v_, 3) for k, v_ in cov.items()})
    return tab


if __name__ == "__main__":
    analyse("main", None, 20)
    analyse("hist", "config_hist.yaml", 52)


def figure(path) -> None:
    """Per-model forest plot of LLM / P2 (natural and log scale) in both periods, in the style of fig_a8"""
    import matplotlib.pyplot as plt
    from llm_bench.plots import (_LABELS, GRID, INK, INK2, MUTED, SURFACE, WIDTH, _colors, _label, _log_axis,
                                 _panel_title, _style)
    _style()
    cfg = load_config()
    order = [m.id for m in cfg.models() if m.enabled]
    colors = _colors(order, {m.id: m.family for m in cfg.models() if m.family})
    _LABELS.clear()
    _LABELS.update({m.id: m.label for m in cfg.models() if m.label})
    per, pooled = {}, {}
    for run in ("main", "hist"):
        d = run_dir(run) / "internal"
        per[run] = pd.read_csv(d / "table_multiplier_placebo_P2_per_model.csv")
        t = pd.read_csv(d / "table_multiplier_placebo.csv")
        pooled[run] = t[(t["comparison"] == "LLM / P2 (wrong time)") & (t["by"] == "all")]
    models = [m for m in order if m in set(per["main"]["model"]) | set(per["hist"]["model"])]
    rows = models + ["pooled"]
    fig = plt.figure(figsize=(WIDTH, 3.3))
    panels = [("main", "natural", "A", "2024–2026, natural scale"), ("main", "log", "B", "2024–2026, log scale"),
              ("hist", "natural", "C", "2015–2019, natural scale"), ("hist", "log", "D", "2015–2019, log scale")]
    x0, w, step, yb, hb = 0.2, 0.17, 0.2, 0.13, 0.72
    vals = [v for run in per for v in list(per[run]["lo95"]) + list(per[run]["hi95"])]
    axes = []
    for j, (run, scale, letter, title) in enumerate(panels):
        ax = fig.add_axes([x0 + j * step, yb, w, hb], sharex=axes[0] if axes else None)
        axes.append(ax)
        p = per[run][per[run]["scale"] == scale].set_index("model")
        pr = pooled[run][pooled[run]["scale"] == scale]
        for i, m in enumerate(rows):
            if m == "pooled":
                if len(pr):
                    r = pr.iloc[0]
                    ax.plot([r.lo95, r.hi95], [i, i], color=INK, lw=1.5, solid_capstyle="butt", zorder=2)
                    ax.plot(r.ratio, i, ls="", marker="D", ms=4.3, color=INK, mec=SURFACE, mew=0.6, zorder=3)
                continue
            if m not in p.index:
                continue
            r = p.loc[m]
            c, mk = colors[m][:2]
            ax.plot([r.lo95, r.hi95], [i, i], color=c, lw=1.3, solid_capstyle="butt", zorder=2)
            ax.plot(r.ratio, i, ls="", marker=mk, ms=4.3, color=c, mec=SURFACE, mew=0.6, zorder=3)
        ax.axvline(1, color=INK, lw=0.7, zorder=1)
        ax.axhline(len(rows) - 1.5, color=GRID, lw=0.8, zorder=0)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("← own answer better   WIS ratio", fontsize=6, color=INK2, labelpad=2)
        _panel_title(ax, letter, title, dy=6)
        if j:
            ax.tick_params(labelleft=False)
    _log_axis(axes[0], vals, axis="x")
    n_models = len(per["main"][per["main"]["scale"] == "natural"])
    axes[0].set_yticks(range(len(rows)), [f"Pooled" if m == "pooled" else _label(m) for m in rows],
                       fontsize=6.8, color=INK)
    axes[0].get_yticklabels()[-1].set_fontweight("bold")
    fig.savefig(f"{path}.png", dpi=400, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(f"{path}.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
