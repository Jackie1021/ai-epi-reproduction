"""Fig. 4: the LLM judgements carry real information (A-D, answers swapped between origins) and follow transmission
mechanisms (E-G, counterfactual prompts that change one driver; main-study settings).
The frontier extension (GPT-6.1 Sol, GPT-6 Astra; runs/frontier*/) is shown next to the main-study models when its
tables exist; pooled estimates stay those of the main-study models.

    .venv/bin/python fig4_combined.py <path without extension>
"""
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import cf_analysis as c
from llm_bench.config import load_config
from llm_bench.plots import (_EXTENSION, _LABELS, GRID, INK, INK2, MUTED, SURFACE, WIDTH, _colors, _label, _log_axis,
                             _panel_title, _row_rules, _rows, _style)
from llm_bench.runner import run_dir

ACC = "#D55E00"
CF_ORDER = ["h3_import", "variant", "school"]
CF_TITLE = {"h3_import": "High A(H3N2) importation", "variant": "New SARS-CoV-2 variant", "school": "School holidays"}
PCT = lambda x: 100 * (np.exp(x) - 1)
FRONTIER = {"main": "frontier", "hist": "frontier_hist"}      # extension runs of the swap test (A-D)
FRONTIER_CF = "frontier_cf"                                    # and of the mechanism test (E-G)


def mechanism_data(run: str = "cf", exclude: frozenset = frozenset({"gem38f", "ling3s"})):
    c.RUN, c.EXCLUDE = run_dir(run), set(exclude)
    t, d = c.analyse(), c.load()
    per = {}
    for name, (a0, a1, target, sign) in c.PAIRS.items():
        e = c.effects(d, a0, a1)
        m = e.groupby(["model", "strain"])["lr"].mean().unstack()
        w = e.pivot_table(index=["model", "origin"], columns="strain", values="lr")
        spec = (w[target].mean(axis=1) - w[[s for s in w if s not in target]].mean(axis=1)).groupby("model").mean()
        per[name] = (m, spec)
    return t, per


def figure(path):
    _style()
    ext = (run_dir(FRONTIER["main"]) / "internal" / "table_multiplier_placebo_P2_per_model.csv").exists()
    cfg = load_config(str(c.HERE / "config_main_frontier.yaml") if ext else None)
    _EXTENSION.clear()
    _EXTENSION.update(cfg.get("extension_models") or () if ext else ())
    order = [m.id for m in cfg.models() if m.enabled]
    colors = _colors(order, {m.id: m.family for m in cfg.models() if m.family})
    _LABELS.clear()
    _LABELS.update({m.id: m.label for m in cfg.models() if m.label})
    H = 6.3
    fig = plt.figure(figsize=(WIDTH, H))
    # ---- A-D: answers swapped between origins (as the former Fig. 4)
    per, pooled = {}, {}
    for run in ("main", "hist"):
        dd = run_dir(run) / "internal"
        per[run] = pd.read_csv(dd / "table_multiplier_placebo_P2_per_model.csv")
        fx = run_dir(FRONTIER[run]) / "internal" / "table_multiplier_placebo_P2_per_model.csv"
        if ext and fx.exists():
            per[run] = pd.concat([per[run], pd.read_csv(fx)], ignore_index=True)
        tt = pd.read_csv(dd / "table_multiplier_placebo.csv")
        pooled[run] = tt[(tt["comparison"] == "LLM / P2 (wrong time)") & (tt["by"] == "all")]
    models = [m for m in order if m in set(per["main"]["model"]) | set(per["hist"]["model"])]
    rows, _ = _rows(models, True)
    panels = [("main", "natural", "A", "2024–2026, natural scale"), ("main", "log", "B", "2024–2026, log scale"),
              ("hist", "natural", "C", "2015–2019, natural scale"), ("hist", "log", "D", "2015–2019, log scale")]
    x0, w, step = 0.2, 0.17, 0.2
    hb = 2.376 / H
    yb = 1 - 0.3 / H - hb
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
            col, mk = colors[m][:2]
            ax.plot([r.lo95, r.hi95], [i, i], color=col, lw=1.3, solid_capstyle="butt", zorder=2)
            ax.plot(r.ratio, i, ls="", marker=mk, ms=4.3, color=col, mec=SURFACE, mew=0.6, zorder=3)
        ax.axvline(1, color=INK, lw=0.7, zorder=1)
        _row_rules(ax, rows)
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
    axes[0].set_yticks(range(len(rows)), ["Pooled" if m == "pooled" else _label(m) for m in rows], fontsize=6.8,
                       color=INK)
    axes[0].get_yticklabels()[rows.index("pooled")].set_fontweight("bold")
    # ---- E-G: counterfactual prompts
    t, mper = mechanism_data()
    tf = mechanism_data(FRONTIER_CF, frozenset())[0] if ext and (run_dir(FRONTIER_CF) / "table_mechanism.csv").exists() else None
    dy = 0.17 if tf is not None else 0.0              # main-study models above, frontier models below
    strains = c.STRAINS + ["spec"]
    ypos = {s: i for i, s in enumerate(c.STRAINS)} | {"spec": len(c.STRAINS) + 0.55}
    wb, stepb, hb2, yb2 = 0.235, 0.27, 2.05 / H, 0.62 / H
    first = None
    for k, name in enumerate(CF_ORDER):
        ax = fig.add_axes([x0 + k * stepb, yb2, wb, hb2], sharey=first)
        first = first or ax
        target = c.PAIRS[name][2]
        m, spec = mper[name]
        for s in strains:
            y = ypos[s]
            col = ACC if (s == "spec" or s in target) else MUTED
            lab = "target minus other" if s == "spec" else c.LABEL[s]
            r = t[(t["pair"] == name) & (t["pathogen"] == lab)].iloc[0]
            v = PCT(spec.values) if s == "spec" else PCT(m[s].dropna().values)
            jit = np.linspace(-0.12, 0.12, len(v)) if len(v) > 1 else [0]
            ax.plot(v, y - dy + np.asarray(jit) * (1 - dy * 2.5), "o", ms=2.4, color=col, alpha=0.35, mew=0, zorder=2)
            ax.plot([r.pct_lo, r.pct_hi], [y - dy, y - dy], color=col, lw=1.4, solid_capstyle="butt", zorder=3)
            ax.plot(r.pct, y - dy, "o", ms=4.6, color=col, mec=SURFACE, mew=0.6, zorder=4)
            if tf is not None:                        # frontier GPT models, pooled over the two
                f = tf[(tf["pair"] == name) & (tf["pathogen"] == lab)].iloc[0]
                ax.plot([f.pct_lo, f.pct_hi], [y + dy, y + dy], color=col, lw=1.0, solid_capstyle="butt", zorder=3)
                ax.plot(f.pct, y + dy, "D", ms=3.6, mfc=SURFACE, mec=col, mew=0.9, zorder=4)
        ax.axvline(0, color=INK, lw=0.7, zorder=1)
        ax.axhline(len(c.STRAINS) - 0.2, color=GRID, lw=0.8, zorder=0)
        ax.set_ylim(ypos["spec"] + 0.6, -0.6)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Change in R$_0$ multiplier (%)", fontsize=6.5, labelpad=2)
        _panel_title(ax, "EFG"[k], CF_TITLE[name], dy=6)
        if k:
            ax.tick_params(labelleft=False)
        else:
            ax.set_yticks([ypos[s] for s in strains], [c.LABEL[s] for s in c.STRAINS] + ["Expected − others"],
                          fontsize=6.8, color=INK)
            ax.get_yticklabels()[-1].set_fontweight("bold")
    handles = [Line2D([], [], color=INK2, marker="o", ms=4.6, mec=SURFACE, mew=0.6, lw=1.4),
               Line2D([], [], color=INK2, marker="o", ms=2.4, alpha=0.4, mew=0, lw=0),
               Line2D([], [], color=ACC, marker="s", ms=5, lw=0), Line2D([], [], color=MUTED, marker="s", ms=5, lw=0)]
    names = ["Pooled over models (95% CI)", "Single models", "Pathogens expected to respond", "Other pathogens"]
    if tf is not None:
        handles.insert(1, Line2D([], [], color=INK2, marker="D", ms=3.6, mfc=SURFACE, mec=INK2, mew=0.9, lw=1.0))
        names = ["Main-study models, pooled (95% CI)", "Frontier GPT models, pooled", "Single main-study models",
                 "Pathogens expected to respond", "Other pathogens"]
    fig.legend(handles, names, loc="lower center", ncol=len(names), bbox_to_anchor=(0.58, 0.0),
               handlelength=1.8, columnspacing=1.4 if tf is not None else 1.8)
    fig.savefig(f"{path}.png", dpi=400, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(f"{path}.pdf", bbox_inches="tight", pad_inches=0.04)
    print(f"{path}.png")


if __name__ == "__main__":
    figure(sys.argv[1] if len(sys.argv) > 1 else str(run_dir("cf") / "figures" / "fig4_combined"))
