"""Extended Data Fig. 3: retrospective evaluation 2015-2019, same comparisons as the main study (grounded vs direct LLM
forecasts), for the models scored at 52 or more of the 65 origins in each design. GPT-6.1 Sol (frontier extension,
runs/frontier_hist/) is shown below the pooled estimate, which stays that of the main-study models.

    .venv/bin/python edfig3_hist.py <path without extension>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from llm_bench.config import load_config
from llm_bench.plots import (_EXTENSION, _LABELS, GRID, INK, INK2, MUTED, SURFACE, WIDTH, _colors, _label, _log_axis,
                             _panel_title, _row_rules, _rows, _style)
from llm_bench.runner import run_dir
from llm_bench.scoring import stratified_gain

DESIGNS = ["direct_blind", "direct_named", "r0_semi", "r0_ctx"]
KEY = ["origin", "target_date", "horizon", "strain", "model"]


def data(min_origins=52, run="hist"):
    sc = pd.read_parquet(run_dir(run) / "scores.parquet")
    n = sc[sc["arm"].isin(DESIGNS)].groupby(["model", "arm"])["origin"].nunique().unstack()
    el = set(n[(n[DESIGNS] >= min_origins).all(axis=1)].index)
    def gain(new, base):
        d = sc[sc["arm"] == new][KEY + ["wis_log"]].merge(sc[sc["arm"] == base][KEY + ["wis_log"]], on=KEY,
                                                           suffixes=("", "_base"))
        d = d[d["model"].isin(el)].copy()
        d["all"] = "all"
        return stratified_gain(d, "all", "wis_log", "wis_log_base").set_index("model")
    cov = sc[sc["model"].isin(el) & sc["arm"].isin(["direct_blind", "r0_semi"])].groupby(["model", "arm"])["cov90"]\
        .mean().unstack()
    return el, {"A": gain("r0_semi", "direct_blind"), "B": gain("r0_semi", "direct_named"),
                "D": gain("direct_named", "direct_blind")}, cov


def figure(path):
    _style()
    ext = (run_dir("frontier_hist") / "scores.parquet").exists()
    cfg = load_config(str(Path(__file__).resolve().parent / "config_main_frontier.yaml") if ext else None)
    _EXTENSION.clear()
    _EXTENSION.update(cfg.get("extension_models") or () if ext else ())
    order = [m.id for m in cfg.models() if m.enabled]
    colors = _colors(order, {m.id: m.family for m in cfg.models() if m.family})
    _LABELS.clear()
    _LABELS.update({m.id: m.label for m in cfg.models() if m.label})
    el, g, cov = data()
    if ext:                                          # frontier models: bootstrapped on their own, not pooled
        el2, g2, cov2 = data(run="frontier_hist")
        el2 = {m for m in el2 if m in _EXTENSION}
        for k in g:
            g[k] = pd.concat([g[k], g2[k].loc[sorted(el2)]])
        cov_all = pd.concat([cov, cov2.loc[sorted(el2)]])
        el = el | el2
    else:
        cov_all = cov
    models = [m for m in order if m in el]
    rows, n_pooled = _rows(models, True)
    fig = plt.figure(figsize=(WIDTH, 3.3))
    x0, w, step, yb, hb = 0.2, 0.17, 0.2, 0.2, 0.67
    panels = [("A", "Grounded / blinded direct"), ("B", "Grounded / named direct"), ("C", "90% interval coverage"),
              ("D", "Named / blinded direct")]
    axes = []
    for j, (letter, title) in enumerate(panels):
        ax = fig.add_axes([x0 + j * step, yb, w, hb], sharey=axes[0] if axes else None)
        axes.append(ax)
        for i, m in enumerate(rows):
            col, mk = (INK, "D") if m == "pooled" else colors[m][:2]
            if letter == "C":
                a, b = (cov["direct_blind"].mean(), cov["r0_semi"].mean()) if m == "pooled" else cov_all.loc[m]
                ax.plot([100 * a, 100 * b], [i, i], color=GRID if m != "pooled" else MUTED, lw=1.2, zorder=1)
                ax.plot(100 * a, i, ls="", marker=mk, ms=4.3, mfc=SURFACE, mec=col, mew=0.9, zorder=3)
                ax.plot(100 * b, i, ls="", marker=mk, ms=4.3, color=col, mec=SURFACE, mew=0.6, zorder=3)
                continue
            r = g[letter].loc[m]
            ax.plot([r.lo95, r.hi95], [i, i], color=col, lw=1.5 if m == "pooled" else 1.3, solid_capstyle="butt",
                    zorder=2)
            ax.plot(r.ratio, i, ls="", marker=mk, ms=4.3, color=col, mec=SURFACE, mew=0.6, zorder=3)
        if letter == "C":
            ax.axvline(90, color=INK, lw=0.7, ls=(0, (3, 2)), zorder=0)
            ax.set_xlim(0, 100)
            ax.set_xticks([0, 25, 50, 75, 90])
            ax.set_xlabel("Observations inside (%)", fontsize=6, color=INK2, labelpad=2)
        else:
            ax.axvline(1, color=INK, lw=0.7, zorder=1)
            vals = list(g[letter]["lo95"]) + list(g[letter]["hi95"])
            _log_axis(ax, vals, axis="x")
            better = "named" if letter == "D" else "grounded"
            ax.set_xlabel(f"← {better} better   WIS ratio", fontsize=6, color=INK2, labelpad=2)
        _row_rules(ax, rows)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        _panel_title(ax, letter, title, dy=6)
        if j:
            ax.tick_params(labelleft=False)
    axes[0].set_yticks(range(len(rows)), [f"Pooled ({n_pooled} models)" if m == "pooled" else _label(m)
                                          for m in rows], fontsize=6.8, color=INK)
    axes[0].get_yticklabels()[rows.index("pooled")].set_fontweight("bold")
    handles = [Line2D([], [], ls="", marker="o", ms=4.3, mfc=SURFACE, mec=INK2, mew=0.9),
               Line2D([], [], ls="", marker="o", ms=4.3, color=INK2, mec=SURFACE, mew=0.6)]
    fig.legend(handles, ["Direct, blinded", "Grounded"], loc="lower center", ncol=2, bbox_to_anchor=(0.69, 0.02),
               handlelength=1.0, columnspacing=1.6)
    fig.savefig(f"{path}.png", dpi=400, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(f"{path}.pdf", bbox_inches="tight", pad_inches=0.04)
    print(f"{path}.png")
    for k in "ABD":
        r = g[k].loc["pooled"]; print(k, f"{r.ratio:.2f} [{r.lo95:.2f}–{r.hi95:.2f}]")


if __name__ == "__main__":
    figure(sys.argv[1] if len(sys.argv) > 1 else str(run_dir("hist") / "figures" / "edfig3_draft"))
