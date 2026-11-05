"""Protocol amendment 8: historical replication period 2015-2019 and the pooled 2015-2026 analysis.

    bash llm.sh a8          (after `bash llm.sh score` and `bash llm.sh score --config config_hist.yaml`)

Reads the scores and phase tables of both runs (runs/main, runs/hist) and writes
runs/hist/table_a8.csv and runs/hist/figures/fig_a8_replication.png:
  A8-1  (primary) design (iii) vs constant R0, growing phase, 2015-2019, pooled over models
  A8-2  the same over all phases and in each phase, 2015-2019
  A8-3  2015-2026: both periods together, five pathogens (no COVID-19); origins resampled within period
  A8-4  designs (i), (ii), (iv) vs constant R0 in 2015-2019
  contamination check: direct named (ii) / direct blinded (i), by period
Model eligibility: at least `phase_min_origins` scored origins for the design in each run
(main 20, historical 52).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .runner import run_dir
from .scoring import stratified_gain

REF = "stan_const"
KEY = ["origin", "target_date", "horizon", "strain"]
PERIOD = {"main": "2024-2026", "hist": "2015-2019"}


def _load(run: str) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    d = run_dir(run)
    if not (d / "scores.parquet").exists() or not (d / "table_phase_at_origin.csv").exists():
        return None
    sc = pd.read_parquet(d / "scores.parquet")
    ph = pd.read_csv(d / "table_phase_at_origin.csv", parse_dates=["origin"])
    return sc, ph


def _paired(sc: pd.DataFrame, ph: pd.DataFrame, new_arm: str, base: str | tuple[str, str],
            min_origins: int, period: str) -> pd.DataFrame:
    """rows (model, origin, strain, horizon) with new and base WIS; base = constant R0 (a model) or
    another design of the same model (arm, e.g. ('arm', 'direct_blind'))"""
    a = sc[sc["arm"] == new_arm]
    n = a.groupby("model")["origin"].nunique()
    a = a[a["model"].isin(n[n >= min_origins].index)][KEY + ["model", "wis", "wis_log"]]
    if isinstance(base, tuple):
        b = sc[sc["arm"] == base[1]][KEY + ["model", "wis", "wis_log"]]
        d = a.merge(b, on=KEY + ["model"], suffixes=("", "_base"))
    else:
        b = sc[sc["model"] == base][KEY + ["wis", "wis_log"]]
        d = a.merge(b, on=KEY, suffixes=("", "_base"))
    d = d.merge(ph[["origin", "strain", "phase"]], on=["origin", "strain"])
    d["period"] = period
    d["all"] = "all"
    return d


def a8_tables(main_cfg=None, hist_cfg=None) -> pd.DataFrame | None:
    main_cfg = main_cfg or load_config()
    hist_cfg = hist_cfg or load_config("config_hist.yaml")
    runs = {"main": _load(main_cfg.get("run_name", "main")), "hist": _load(hist_cfg.get("run_name", "hist"))}
    if runs["hist"] is None:
        print("no scored historical run yet (bash llm.sh score --config config_hist.yaml)")
        return None
    thr = {"main": int(main_cfg.get("phase_min_origins", 20)), "hist": int(hist_cfg.get("phase_min_origins", 52))}
    common = [s for s in hist_cfg.strains if s in main_cfg.strains]
    rows = []

    def add(analysis, d, by, scale, strata=None, **kw):
        if d.empty or d["model"].nunique() == 0:
            return
        col = "wis" if scale == "natural" else "wis_log"
        g = stratified_gain(d, by, col, col + "_base", strata=strata)
        rows.append(g.assign(analysis=analysis, scale=scale, **kw))

    sc_h, ph_h = runs["hist"]
    for scale in ("natural", "log"):
        # A8-1 / A8-2: design (iii) vs constant R0 in 2015-2019
        d = _paired(sc_h, ph_h, "r0_semi", REF, thr["hist"], PERIOD["hist"])
        add("A8-1/A8-2", d, "phase", scale, arm="r0_semi", period=PERIOD["hist"])
        add("A8-1/A8-2", d, "all", scale, arm="r0_semi", period=PERIOD["hist"])
        add("A8-2 by horizon", d, "horizon", scale, arm="r0_semi", period=PERIOD["hist"])
        add("A8-2 by pathogen", d, "strain", scale, arm="r0_semi", period=PERIOD["hist"])
        # A8-4: the other designs in 2015-2019
        for arm in ("direct_blind", "direct_named", "r0_ctx"):
            d4 = _paired(sc_h, ph_h, arm, REF, thr["hist"], PERIOD["hist"])
            add("A8-4", d4, "all", scale, arm=arm, period=PERIOD["hist"])
            add("A8-4", d4, "phase", scale, arm=arm, period=PERIOD["hist"])
        # contamination check: (ii) / (i), each period
        for run, (sc, ph) in ((r, v) for r, v in runs.items() if v is not None):
            dc = _paired(sc, ph, "direct_named", ("arm", "direct_blind"), thr[run], PERIOD[run])
            add("contamination (ii)/(i)", dc, "all", scale, arm="direct_named/direct_blind", period=PERIOD[run])
        if runs["main"] is not None:
            sc_m, ph_m = runs["main"]
            # the main period on its own (amendment 7 test, six pathogens), for comparison in the figure
            dm = _paired(sc_m, ph_m, "r0_semi", REF, thr["main"], PERIOD["main"])
            add("amendment 7 (main period)", dm, "phase", scale, arm="r0_semi", period=PERIOD["main"])
            add("amendment 7 (main period)", dm, "all", scale, arm="r0_semi", period=PERIOD["main"])
            # A8-3: 2015-2026, five common pathogens, models eligible in both periods
            dh = d[d["strain"].isin(common)]
            dm5 = dm[dm["strain"].isin(common)]
            both = sorted(set(dh["model"]) & set(dm5["model"]))
            dc = pd.concat([dh[dh["model"].isin(both)], dm5[dm5["model"].isin(both)]], ignore_index=True)
            add("A8-3", dc, "phase", scale, strata="period", arm="r0_semi", period="2015-2026")
            add("A8-3", dc, "all", scale, strata="period", arm="r0_semi", period="2015-2026")
    if not rows:
        return None
    tab = pd.concat(rows, ignore_index=True)
    tab["level"] = tab["level"].astype(str)
    cols = ["analysis", "arm", "period", "scale", "by", "level", "model", "n_origins", "n_cells", "ratio",
            "lo95", "hi95", "p_better"]
    tab = tab[cols]
    out = run_dir(hist_cfg.get("run_name", "hist"))
    tab.to_csv(out / "table_a8.csv", index=False)
    v = tab[(tab["model"] == "pooled") & (tab["scale"] == "natural") & (tab["by"].isin(["all", "phase"]))
            & (tab["arm"] == "r0_semi")]
    print("design (iii) / constant R0, pooled over models (natural scale):")
    print(v[["analysis", "period", "level", "n_origins", "ratio", "lo95", "hi95", "p_better"]].round(3).to_string(index=False))
    p = tab[(tab["analysis"] == "A8-1/A8-2") & (tab["level"] == "growing") & (tab["model"] == "pooled")
            & (tab["scale"] == "natural")]
    if len(p):
        r = p.iloc[0]
        verdict = "SUPPORTED" if r.hi95 < 1 else "NOT SUPPORTED"
        print(f"\nA8-1 (primary): growing phase 2015-2019, pooled ratio {r.ratio:.3f} "
              f"[{r.lo95:.3f}, {r.hi95:.3f}] -> {verdict}")
    return tab


def fig_a8(tab: pd.DataFrame, colors: dict, out: Path) -> Path | None:
    """A-C: 2015-2019 by phase, each model + pooled; D: pooled by phase in each period and 2015-2026;
    E: every design vs constant R0 in 2015-2019 (pooled); F: contamination check by period"""
    import matplotlib.pyplot as plt
    from .plots import GRID, INK, INK2, MUTED, PHASES, SURFACE, WIDTH, _label, _log_axis, _panel_title, _save, _style
    _style()
    t = tab[tab["scale"] == "natural"]
    ph = t[(t["analysis"] == "A8-1/A8-2") & (t["by"] == "phase")]
    models = [m for m in colors if m in set(ph["model"])]
    if not models:
        return None
    rows = models + (["pooled"] if "pooled" in set(ph["model"]) else [])
    fig = plt.figure(figsize=(WIDTH, 4.6))
    yb, hb, x0, w, step = 0.57, 0.31, 0.165, 0.24, 0.275
    vals = list(ph["lo95"]) + list(ph["hi95"])
    fax = []
    for j, (key, title, sub) in enumerate(PHASES):
        ax = fig.add_axes([x0 + j * step, yb, w, hb], sharex=fax[0] if fax else None)
        fax.append(ax)
        d = ph[ph["level"] == key].set_index("model")
        for i, m in enumerate(rows):
            if m not in d.index:
                continue
            r = d.loc[m]
            c, mk = (INK, "D") if m == "pooled" else colors[m][:2]
            ax.plot([r.lo95, r.hi95], [i, i], color=c, lw=1.5 if m == "pooled" else 1.3, solid_capstyle="butt", zorder=2)
            ax.plot(r.ratio, i, ls="", marker=mk, ms=4.3, color=c, mec=SURFACE, mew=0.6, zorder=3)
        ax.axvline(1, color=INK, lw=0.7, zorder=1)
        if "pooled" in rows:
            ax.axhline(len(rows) - 1.5, color=GRID, lw=0.8, zorder=0)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("← LLM better   WIS ratio   worse →", fontsize=6, color=INK2, labelpad=2)
        _panel_title(ax, "ABC"[j], f"{title}, 2015–2019", dy=13)
        ax.annotate(sub, xy=(0, 1), xycoords="axes fraction", xytext=(11, 4), textcoords="offset points",
                    va="baseline", fontsize=6, color=MUTED)
        if j:
            ax.tick_params(labelleft=False)
    _log_axis(fax[0], vals, axis="x")
    fax[0].set_yticks(range(len(rows)), [f"Pooled ({len(models)} models)" if m == "pooled" else _label(m) for m in rows],
                      fontsize=6.8, color=INK)
    if "pooled" in rows:
        fax[0].get_yticklabels()[-1].set_fontweight("bold")

    def forest(ax, items, letter, title, sub, groups=0):
        """items: list of (label, row or None, emphasised); groups: rows per group (separator lines)"""
        ys = range(len(items))
        if groups:
            for g in range(groups, len(items), groups):
                ax.axhline(g - 0.5, color=GRID, lw=0.8, zorder=0)
        for y, (lab, r, em) in zip(ys, items):
            if r is None:
                continue
            ax.plot([r.lo95, r.hi95], [y, y], color=INK if em else INK2, lw=1.5 if em else 1.2, solid_capstyle="butt")
            ax.plot(r.ratio, y, ls="", marker="D", ms=4 if em else 3.5, color=INK if em else INK2, mec=SURFACE, mew=0.6)
        ax.axvline(1, color=INK, lw=0.7, zorder=0)
        ax.set_ylim(len(items) - 0.5, -0.5)
        ax.set_yticks(list(ys), [lab for lab, _, _ in items], fontsize=6.2, color=INK)
        ax.grid(axis="y", visible=False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        vv = [x for _, r, _ in items if r is not None for x in (r.lo95, r.hi95)]
        if vv:
            _log_axis(ax, vv, axis="x")
        ax.set_xlabel("WIS ratio (pooled over models)", fontsize=6, color=INK2, labelpad=2)
        _panel_title(ax, letter, title, dy=13)
        ax.annotate(sub, xy=(0, 1), xycoords="axes fraction", xytext=(11, 4), textcoords="offset points",
                    va="baseline", fontsize=6, color=MUTED)

    def pick(analysis, level, by="phase", arm="r0_semi", period=None):
        q = t[(t["analysis"] == analysis) & (t["by"] == by) & (t["level"] == level) & (t["model"] == "pooled")
              & (t["arm"] == arm)]
        if period:
            q = q[q["period"] == period]
        return q.iloc[0] if len(q) else None

    items = []
    for key, title, _ in PHASES:
        items += [(f"{title}: 2024–2026", pick("amendment 7 (main period)", key), False),
                  (f"{title}: 2015–2019", pick("A8-1/A8-2", key), key == "growing"),
                  (f"{title}: 2015–2026", pick("A8-3", key), False)]
    forest(fig.add_axes([0.165, 0.08, 0.2, 0.34]), items, "D", "LLM R0 + model vs constant R0",
           "design (iii), by period and phase", groups=3)
    arms = [("direct_blind", "(i) Direct, blinded"), ("direct_named", "(ii) Direct, named"),
            ("r0_semi", "(iii) R0 + model"), ("r0_ctx", "(iv) + calendar, weather")]
    items = [(lab, pick("A8-4" if a != "r0_semi" else "A8-1/A8-2", "all", by="all", arm=a), a == "r0_semi")
             for a, lab in arms]
    forest(fig.add_axes([0.545, 0.08, 0.16, 0.34]), items, "E", "Every design, 2015–2019",
           "vs constant R0, all phases")
    items = [(f"{p}", pick("contamination (ii)/(i)", "all", by="all", arm="direct_named/direct_blind", period=p), False)
             for p in ("2024–2026".replace("–", "-"), "2015-2019")]
    items = [(lab.replace("-", "–"), r, e) for lab, r, e in items]
    forest(fig.add_axes([0.815, 0.08, 0.165, 0.34]), items, "F", "Recall check",
           "direct named / blinded")
    _save(fig, out, "fig_a8_replication")
    return out / "fig_a8_replication.png"


def cmd_a8(cfg, args) -> None:
    """cfg: the historical config (--config, default config_hist.yaml); the main study uses config.yaml"""
    from .plots import _LABELS, _colors
    hist_cfg = cfg if cfg.get("run_name", "main") != "main" else load_config("config_hist.yaml")
    main_cfg = load_config()
    tab = a8_tables(main_cfg, hist_cfg)
    if tab is None:
        return
    order = [m.id for m in main_cfg.models() if m.enabled] + ["mock"]
    families = {m.id: m.family for m in main_cfg.models() if m.family}
    _LABELS.clear()
    _LABELS.update({m.id: m.label for m in main_cfg.models() if m.label})
    out = run_dir(hist_cfg.get("run_name", "hist")) / "figures"
    p = fig_a8(tab, _colors(order, families), out)
    if p:
        print("wrote", p)
