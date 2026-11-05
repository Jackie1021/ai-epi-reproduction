"""Protocol amendments 12 and 14: sensitivity analyses with the longest-chain Stan fits.

    .venv/bin/python -m llm_bench.sensitivity     (after `bash llm.sh score` for both runs;
                                                    manuscript/build_manuscript.py runs it too)

Scenarios (each compared with the main analysis on the same answers):
  main study   S1  without 2026-03-14 (pre-specified, amendment 6)
               S2  2026-03-14 replaced by its amendment-14 fit
  2015-2019    S1  without the five origins of amendment 12
               S2  the five origins replaced by their longest-chain fit (amendment 14: 2016-11-05,
                   2016-12-31, 2017-02-25; amendment 12: 2016-01-02, 2016-12-03)
In S2 the mechanistic forecast is recomputed from the new posterior, the LLM-guided forecasts propagate
the multipliers already given by each LLM through the new draws (same seeds), and the epidemic phase at
the origin is taken from the new fit. No new LLM request is sent. Direct forecasts do not use the fit and
are unchanged. Model eligibility is fixed by the main analysis.

Writes runs/<run>/sensitivity/{forecasts,scores}_S2.parquet, runs/main/table_convergence_fits.csv (R-hat of
every fit of these origins) and runs/<run>/table_convergence_sensitivity.csv.
"""
from __future__ import annotations

import json
import os

import pandas as pd

from .config import load_config
from .data import OriginBundle, load_observed
from .runner import BENCH_DIR, Runner, load_forecasts, run_dir
from .scoring import phase_at_origin, score_forecasts, stratified_gain

K = ["origin", "target_date", "horizon", "strain"]
OUT = BENCH_DIR.parent / "output"
REPLACE = {
    "main": {"2026-03-14": OUT / "origins_longer" / "2026-03-14"},
    "hist": {"2016-01-02": OUT / "origins_hist_long" / "2016-01-02",
             "2016-11-05": OUT / "origins_hist_longer" / "2016-11-05",
             "2016-12-03": OUT / "origins_hist_long" / "2016-12-03",
             "2016-12-31": OUT / "origins_hist_longer" / "2016-12-31",
             "2017-02-25": OUT / "origins_hist_longer" / "2017-02-25"},
}
# every fit of these origins, in the order they were made
FITS = [("original", "archive/refit_originals", "main"), ("refit", "archive/refit_attempts", "main"),
        ("main analysis", "origins", "main"), ("amendment 14", "origins_longer", "main"),
        ("original", "archive/hist_refit_originals", "hist"), ("main analysis (amendment 8 refit)", "origins_hist", "hist"),
        ("amendment 12", "origins_hist_long", "hist"), ("amendment 14", "origins_hist_longer", "hist")]


def fit_table() -> pd.DataFrame:
    rows = []
    for label, sub, study in FITS:
        for o in REPLACE[study]:
            for p in sorted((OUT / sub).glob(f"{o}*/meta.json")):
                m = json.loads(p.read_text())
                d, s = m.get("diagnostics", {}), m.get("sampling", {})
                n_samp = (s.get("iter") or 0) - (s.get("warmup") or 0)
                worst = max((b.get("divergent", 0) for b in d.get("blocks", [])), default=d.get("n_divergent", 0))
                rows.append((study, o, label, p.parent.relative_to(OUT).as_posix(), s.get("warmup"), n_samp,
                             s.get("adapt_delta"), s.get("max_treedepth"), d.get("max_rhat"), d.get("min_n_eff"),
                             d.get("n_divergent"), worst / max(1, s.get("chains", 4) * n_samp)))
    # divergent_share: worst fitting block, divergent transitions / post-warm-up draws of that block
    return pd.DataFrame(rows, columns=["study", "origin", "fit", "dir", "warmup", "sampling", "adapt_delta",
                                       "max_treedepth", "max_rhat", "min_ess", "n_divergent", "divergent_share"])


def s2_scores(cfg, run: str, sc: pd.DataFrame, ph: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """scores and phases with the replaced origins rebuilt from the longest-chain fits"""
    rep = {pd.Timestamp(o): p for o, p in REPLACE[run].items()}
    out = run_dir(run) / "sensitivity"
    out.mkdir(exist_ok=True)
    replay = os.environ.get("AI_EPI_OFFLINE") == "1"
    if replay:
        fc = pd.read_parquet(out / "forecasts_S2.parquet")
        changes = pd.read_csv(out / "phase_changes_S2.csv", parse_dates=["origin"])
        ph_new = ph[ph["origin"].isin(rep)].copy()
        if not changes.empty:
            ph_new = ph_new.merge(
                changes[["origin", "strain", "Rt_S2", "phase_S2"]], on=["origin", "strain"], how="left"
            )
            for column in ("Rt", "phase"):
                ph_new[column] = ph_new.pop(f"{column}_S2").combine_first(ph_new[column])
    else:
        bundles = [OriginBundle(p) for p in rep.values() if (p / "DONE").exists()]
        runner = Runner(cfg, run, [], list(cfg["arms"]))
        saved = [r for r in runner._all_saved() if pd.Timestamp(r["origin"]) in rep]
        fc = runner.build_forecasts(bundles, saved)["main"]
        fc.to_parquet(out / "forecasts_S2.parquet", index=False)
        ph_new = phase_at_origin(bundles, cfg.strains)
    new = score_forecasts(fc, load_observed(cfg), cfg.quantiles)
    # keep exactly the (origin, model, arm) cells the main analysis scored
    cells = sc[sc["origin"].isin(rep)][["origin", "model", "arm"]].drop_duplicates()
    new = new.merge(cells, on=["origin", "model", "arm"])
    sc2 = pd.concat([sc[~sc["origin"].isin(rep)], new], ignore_index=True)
    sc2.to_parquet(out / "scores_S2.parquet", index=False)
    ph2 = pd.concat([ph[~ph["origin"].isin(rep)], ph_new], ignore_index=True)
    if not replay:
        chg = ph[ph["origin"].isin(rep)].merge(ph_new, on=["origin", "strain"], suffixes=("", "_S2"))
        chg[chg["phase"] != chg["phase_S2"]].to_csv(out / "phase_changes_S2.csv", index=False)
    return sc2, ph2


def comparisons(sc: pd.DataFrame, ph: pd.DataFrame, models: dict) -> list[tuple]:
    """(analysis, scale, level, row) for the headline comparisons"""
    base = sc[sc["model"] == "stan_const"][K + ["wis", "wis_log"]]
    r0 = sc[(sc["arm"] == "r0_semi") & sc["model"].isin(models["r0_semi"])][K + ["model", "wis", "wis_log"]]
    d = r0.merge(base, on=K, suffixes=("", "_base")).merge(ph[["origin", "strain", "phase"]], on=["origin", "strain"])
    d["all"] = "all"
    out = []
    for col, scale in (("wis", "natural"), ("wis_log", "log")):
        for by in ("all", "phase"):
            g = stratified_gain(d, by, col, col + "_base")
            for r in g[g["model"] == "pooled"].itertuples():
                out.append(("(iii) vs mechanistic", scale, str(r.level), r))
    mods = sorted(set(models["r0_semi"]) & set(models["direct_blind"]))
    a = sc[(sc["arm"] == "r0_semi") & sc["model"].isin(mods)][K + ["model", "wis", "wis_log"]]
    b = sc[(sc["arm"] == "direct_blind") & sc["model"].isin(mods)][K + ["model", "wis", "wis_log"]]
    d = a.merge(b, on=K + ["model"], suffixes=("", "_base"))
    d["all"] = "all"
    for col, scale in (("wis", "natural"), ("wis_log", "log")):
        g = stratified_gain(d, "all", col, col + "_base")
        out.append(("(iii) vs direct (i)", scale, "all", g[g["model"] == "pooled"].iloc[0]))
    return out


def run_sensitivity(run: str, cfg_file: str | None, min_origins: int) -> pd.DataFrame | None:
    cfg = load_config(cfg_file)
    d = run_dir(run)
    if not (d / "scores.parquet").exists() or not (d / "table_phase_at_origin.csv").exists():
        print(f"{run}: not scored yet")
        return None
    sc = pd.read_parquet(d / "scores.parquet")
    ph = pd.read_csv(d / "table_phase_at_origin.csv", parse_dates=["origin"])
    n = sc.groupby(["model", "arm"])["origin"].nunique()
    models = {a: sorted(m for (m, a2), v in n.items() if a2 == a and v >= min_origins)
              for a in ("r0_semi", "direct_blind")}
    if not models["r0_semi"]:
        print(f"{run}: no model has {min_origins} scored origins yet")
        return None
    rep = [pd.Timestamp(o) for o in REPLACE[run]]
    sc2, ph2 = s2_scores(cfg, run, sc, ph)
    scen = {"main analysis": (sc, ph),
            "S1 without these origins": (sc[~sc["origin"].isin(rep)], ph[~ph["origin"].isin(rep)]),
            "S2 longest-chain fits": (sc2, ph2)}
    rows = []
    for name, (s, p) in scen.items():
        for analysis, scale, level, r in comparisons(s, p, models):
            rows.append((run, name, analysis, scale, level, int(r.n_origins), r.ratio, r.lo95, r.hi95, r.p_better))
        for arm, mods in (("r0_semi", models["r0_semi"]), ("stan_const", ["stan_const"])):
            x = s[(s["model"].isin(mods)) & ((s["arm"] == arm) | (s["model"] == arm))]
            cov = x.groupby("model")["cov90"].mean().mean()
            rows.append((run, name, f"coverage 90% {arm}", "natural", "all", x["origin"].nunique(), cov, None, None, None))
    tab = pd.DataFrame(rows, columns=["run", "scenario", "analysis", "scale", "level", "n_origins", "ratio",
                                      "lo95", "hi95", "p_better"])
    tab.attrs["models"] = models
    tab.to_csv(d / "table_convergence_sensitivity.csv", index=False)
    v = tab[tab["level"].isin(["all", "growing"])].pivot_table(index=["analysis", "scale", "level"], columns="scenario",
                                                                values="ratio", aggfunc="first")
    print(f"\n{run}: models (iii) {models['r0_semi']}")
    print(v.round(3).to_string())
    return tab


def cmd_sens(cfg=None, args=None):
    ft = fit_table()
    ft.to_csv(run_dir("main") / "table_convergence_fits.csv", index=False)
    print(ft[["study", "origin", "fit", "warmup", "sampling", "adapt_delta", "max_rhat", "n_divergent"]].to_string(index=False))
    main_cfg, hist_cfg = load_config(), load_config("config_hist.yaml")
    run_sensitivity("main", None, int(main_cfg.get("phase_min_origins", 20)))
    run_sensitivity("hist", "config_hist.yaml", int(hist_cfg.get("phase_min_origins", 52)))


if __name__ == "__main__":
    cmd_sens()
