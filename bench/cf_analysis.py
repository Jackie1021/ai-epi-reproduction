"""Amendment 15: mechanism test with counterfactual prompts (runs/cf/).

For each pair of prompts that differ in one input, the effect is the log ratio of the median R0 multipliers
(second / first prompt) for each model, origin, pathogen and week, averaged over weeks. Pooled: equal-weight mean
over models, 95% interval by a two-level bootstrap (origins with the same draw for every model, then models).

    .venv/bin/python cf_analysis.py            -> runs/cf/table_mechanism.csv, runs/cf/figures/fig_mechanism.png
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from llm_bench.config import load_config
from llm_bench.runner import runs_root

HERE = Path(__file__).resolve().parent
RUN = runs_root() / __import__("os").environ.get("CF_RUN", "cf")
CFG = load_config(str(HERE / "config_cf.yaml"))
STRAINS = list(CFG.strains)
PAIRS = {   # name: (first prompt, second prompt, target pathogens, expected sign of the target effect)
    "school": ("cf_term", "cf_holiday", ["B", "H3", "H1", "RSV", "HFMD"], -1),
    "h3_import": ("cf_h3_low", "cf_h3_high", ["H3"], +1),
    "variant": ("cf_news_none", "cf_news_variant", ["COVID"], +1),
}
TITLE = {"school": "School holidays vs school term", "h3_import": "High vs low A(H3N2) importation",
         "variant": "New SARS-CoV-2 variant vs no news"}
LABEL = {"B": "B", "H3": "A(H3N2)", "H1": "A(H1N1)", "COVID": "SARS-CoV-2", "RSV": "RSV", "HFMD": "HFMD"}
N_BOOT, SEED = 4000, 20261013


EXCLUDE = {m for m in __import__("os").environ.get("CF_EXCLUDE", "").split(",") if m}


def load() -> pd.DataFrame:
    snapshot = RUN / "mechanism_medians.parquet"
    if snapshot.is_file():
        data = pd.read_parquet(snapshot)
        return data[~data["model"].isin(EXCLUDE)]
    rows = []
    for p in RUN.glob("responses/*/*/*_s0.json"):
        r = json.loads(p.read_text(encoding="utf-8"))
        if r.get("status") != "ok" or not r.get("multipliers") or r.get("model") in EXCLUDE:
            continue
        for k, s in enumerate(STRAINS):
            for w, q in enumerate(r["multipliers"][k], 1):
                rows.append((r["origin"], r["arm"], r["model"], s, w, float(q[1])))
    return pd.DataFrame(rows, columns=["origin", "arm", "model", "strain", "week", "p50"])


def effects(d: pd.DataFrame, a0: str, a1: str) -> pd.DataFrame:
    x = d[d["arm"].isin([a0, a1])].pivot_table(index=["model", "origin", "strain", "week"], columns="arm",
                                                values="p50")
    if a0 not in x or a1 not in x:
        return pd.DataFrame(columns=["model", "origin", "strain", "lr"])
    x = x.dropna()
    x["lr"] = np.log(x[a1]) - np.log(x[a0])
    return x.groupby(["model", "origin", "strain"])["lr"].mean().reset_index()


def pooled(e: pd.DataFrame, value: str = "lr") -> tuple[float, float, float, int, int]:
    """equal-weight mean over models of each model's mean over origins; two-level bootstrap"""
    rng = np.random.default_rng(SEED)
    models = sorted(e["model"].unique())
    origins = sorted(e["origin"].unique())
    M = np.full((len(models), len(origins)), np.nan)
    for r in e.itertuples():
        M[models.index(r.model), origins.index(r.origin)] = getattr(r, value)
    point = np.nanmean(np.nanmean(M, axis=1))
    oi = rng.integers(0, len(origins), size=(N_BOOT, len(origins)))
    mi = rng.integers(0, len(models), size=(N_BOOT, len(models)))
    with np.errstate(invalid="ignore"), np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        per_model = np.nanmean(M[:, oi], axis=2).T                       # (B, models)
        boot = np.nanmean(np.take_along_axis(per_model, mi, axis=1), axis=1)
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return float(point), float(lo), float(hi), len(models), len(origins)


def analyse() -> pd.DataFrame:
    d = load()
    out = []
    for name, (a0, a1, target, sign) in PAIRS.items():
        e = effects(d, a0, a1)
        if e.empty:
            continue
        for s in STRAINS:
            x = e[e["strain"] == s]
            if len(x):
                p, lo, hi, nm, no = pooled(x)
                out.append((name, LABEL[s], "target" if s in target else "other", p, lo, hi, nm, no, np.nan))
        # specificity: target minus the mean of the other pathogens, within model and origin
        w = e.pivot_table(index=["model", "origin"], columns="strain", values="lr")
        spec = (w[[s for s in target if s in w]].mean(axis=1) - w[[s for s in w if s not in target]].mean(axis=1))
        spec = spec.dropna().rename("lr").reset_index()
        tgt = w[[s for s in target if s in w]].mean(axis=1).dropna()
        share = float((np.sign(tgt) == sign).mean())
        p, lo, hi, nm, no = pooled(spec)
        out.append((name, "target minus other", "specificity", p, lo, hi, nm, no, share))
    t = pd.DataFrame(out, columns=["pair", "pathogen", "role", "log_ratio", "lo95", "hi95", "n_models", "n_origins",
                                   "share_expected_direction"])
    for c in ("log_ratio", "lo95", "hi95"):
        t[c.replace("log_ratio", "pct").replace("lo95", "pct_lo").replace("hi95", "pct_hi")] = 100 * (np.exp(t[c]) - 1)
    return t


def figure(t: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from llm_bench.plots import INK, MUTED, _panel_title, _style
    _style()
    pairs = [p for p in PAIRS if p in set(t["pair"])]
    fig, axes = plt.subplots(1, len(pairs), figsize=(7.2, 2.4), sharey=False)
    axes = np.atleast_1d(axes)
    for k, (ax, name) in enumerate(zip(axes, pairs)):
        x = t[(t["pair"] == name) & (t["role"] != "specificity")].reset_index(drop=True)
        for i, r in x.iterrows():
            c = "#c0392b" if r["role"] == "target" else MUTED
            ax.plot([r["pct_lo"], r["pct_hi"]], [i, i], color=c, lw=1.4)
            ax.plot(r["pct"], i, "o", color=c, ms=4.5, mec="white", mew=0.6)
        ax.axvline(0, color=INK, lw=0.7, ls=(0, (3, 2)))
        ax.set_yticks(range(len(x)), x["pathogen"])
        ax.invert_yaxis()
        ax.set_xlabel("Change in R0 multiplier (%)")
        _panel_title(ax, "ABC"[k], TITLE[name])
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=400)
    fig.savefig(path.with_suffix(".pdf"))


if __name__ == "__main__":
    t = analyse()
    RUN.mkdir(parents=True, exist_ok=True)
    t.to_csv(RUN / "table_mechanism.csv", index=False)
    print(t[["pair", "pathogen", "role", "pct", "pct_lo", "pct_hi", "n_models", "n_origins",
             "share_expected_direction"]].round(1).to_string(index=False))
    if len(t):
        figure(t, RUN / "figures" / "fig_mechanism.png")
        print("figure:", RUN / "figures" / "fig_mechanism.png")
