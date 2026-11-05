"""Draft figure of the mechanism test (Amendment 15): pooled effect per pathogen and target-minus-other specificity,
main settings (filled) with individual models (faint dots) and reasoning on (open).

    .venv/bin/python cf_figure.py      -> runs/cf/figures/fig_mechanism_draft.png/.pdf
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import cf_analysis as c
from llm_bench.plots import INK, INK2, MUTED, GRID, _panel_title, _style

ACC = "#D55E00"
ORDER = ["h3_import", "variant", "school"]
TITLE = {"h3_import": "A(H3N2) importation: high vs low", "variant": "SARS-CoV-2 variant news vs none",
         "school": "School holidays vs term"}
PCT = lambda x: 100 * (np.exp(x) - 1)


def run(run, exclude):
    c.RUN, c.EXCLUDE = c.HERE / "runs" / run, set(exclude)
    t = c.analyse()
    d = c.load()
    per = {}
    for name, (a0, a1, target, sign) in c.PAIRS.items():
        e = c.effects(d, a0, a1)
        if e.empty:
            continue
        m = e.groupby(["model", "strain"])["lr"].mean().unstack()
        w = e.pivot_table(index=["model", "origin"], columns="strain", values="lr")
        spec = (w[target].mean(axis=1) - w[[s for s in w if s not in target]].mean(axis=1)).groupby("model").mean()
        per[name] = (m, spec)
    return t, per


main_t, main_per = run("cf", ["gem38f", "ling3s"])
rea_t, _ = run("cf_reason", ["kimik3r", "gptoss20r"])

_style()
plt.rcParams["axes.grid"] = False
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), sharey=True)
rows = c.STRAINS + ["spec"]
ypos = {s: i for i, s in enumerate(c.STRAINS)} | {"spec": len(c.STRAINS) + 0.55}
for k, (ax, name) in enumerate(zip(axes, ORDER)):
    target = c.PAIRS[name][2]
    m, spec = main_per[name]
    for s in rows:
        y = ypos[s]
        is_t = s == "spec" or s in target
        col = ACC if is_t else MUTED
        lab = "target minus other" if s == "spec" else c.LABEL[s]
        r = main_t[(main_t["pair"] == name) & (main_t["pathogen"] == lab)].iloc[0]
        q = rea_t[(rea_t["pair"] == name) & (rea_t["pathogen"] == lab)].iloc[0]
        vals = PCT(spec.values) if s == "spec" else PCT(m[s].dropna().values)
        jit = np.linspace(-0.07, 0.07, len(vals)) if len(vals) > 1 else [0]
        ax.plot(vals, y - 0.13 + np.asarray(jit), "o", ms=2.4, color=col, alpha=0.35, mew=0, zorder=2)
        ax.plot([r.pct_lo, r.pct_hi], [y - 0.13] * 2, color=col, lw=1.3, zorder=3)
        ax.plot(r.pct, y - 0.13, "o", ms=4.6, color=col, mec="white", mew=0.6, zorder=4)
        ax.plot([q.pct_lo, q.pct_hi], [y + 0.2] * 2, color=col, lw=0.8, zorder=3)
        ax.plot(q.pct, y + 0.2, "o", ms=3.9, mfc="white", mec=col, mew=0.9, zorder=4)
    ax.axvline(0, color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.axhline(len(c.STRAINS) - 0.2, color=GRID, lw=0.6, zorder=0)
    ax.set_yticks([ypos[s] for s in rows])
    ax.set_yticklabels([c.LABEL[s] for s in c.STRAINS] + ["Expected − others"])
    for tl in ax.get_yticklabels():
        tl.set_color(INK)
    if k:
        ax.tick_params(axis="y", labelleft=False)
    ax.set_ylim(ypos["spec"] + 0.6, -0.6)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Change in R$_0$ multiplier (%)")
    _panel_title(ax, "ABC"[k], TITLE[name])
handles = [Line2D([], [], color=INK2, marker="o", ms=4.6, mec="white", mew=0.6, lw=1.3),
           Line2D([], [], color=INK2, marker="o", ms=3.9, mfc="white", mec=INK2, mew=0.9, lw=0.8),
           Line2D([], [], color=INK2, marker="o", ms=2.4, alpha=0.4, mew=0, lw=0),
           Line2D([], [], color=ACC, marker="s", ms=5, lw=0), Line2D([], [], color=MUTED, marker="s", ms=5, lw=0)]
fig.legend(handles, ["Main settings, pooled (95% CI)", "Reasoning on, 4 models", "Single models", "Expected to respond",
                     "Other pathogens"], loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0), handlelength=1.8,
           columnspacing=1.6)
fig.tight_layout(rect=(0, 0.07, 1, 0.97), w_pad=2.2)
out = c.HERE / "runs" / "cf" / "figures" / "fig_mechanism_draft.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=300)
fig.savefig(out.with_suffix(".pdf"))
print(out)
