"""Protocol amendment 10 (exploratory): how the LLMs use last-year R0, and the reasoning-mode test.

A10-1  For every answer of designs (iii)-(v) whose prompt showed last-year R0, pathogen k and week h:
       y = log(p50 multiplier), x = log(R0 in week h one year earlier / current R0), both as printed in
       the prompt. Slope beta of y = a + b x per model (least squares); 95% interval from a bootstrap over
       origins (4,000 resamples, seed 20260929); pooled = mean of per-model slopes, two-level bootstrap.
       beta = 1: follows last year's pattern; beta = 0: ignores it. Also by epidemic phase and pathogen.
A10-2  Reasoning versions (runs/reason) against the same models' main-study answers, paired on the same
       targets: WIS ratio reasoning / non-reasoning (natural and log), relative WIS against constant R0,
       coverage, and beta.

  python -m llm_bench.amendment10            writes runs/main/table_a10_lastyear_use.csv and, when the
                                             reasoning run has been scored, runs/reason/table_a10_reasoning.csv
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .runner import runs_root

MAIN, REASON = runs_root() / "main", runs_root() / "reason"
SEED, NBOOT = 20260929, 4000
K = ["origin", "target_date", "horizon", "strain"]
PAIRS = {"gptoss20r": "gptoss20", "kimik3r": "kimik3"}          # reasoning version -> main-study version


def prompt_r0(run: Path = MAIN) -> pd.DataFrame:
    """Current R0 and last-year R0 per origin, pathogen and week, exactly as printed in the R0 prompts."""
    rows = []
    for f in sorted(run.glob("prompts/*/r0_semi.txt")):
        origin = pd.Timestamp(f.parent.name)
        for block in re.split(r"\n### ", f.read_text(encoding="utf-8"))[1:]:
            strain = block.split(" ", 1)[0]
            cur = re.search(r"Current R0 \(mean of the last 7 days\): ([0-9.]+)", block)
            ly = re.search(r"one year earlier \(median\): (.*)", block)
            if not cur or not ly or "not available" in ly.group(1):
                continue
            for h, v in re.findall(r"week (\d+): ([0-9.]+)", ly.group(1)):
                rows.append((origin, strain, int(h), float(cur.group(1)), float(v)))
    return pd.DataFrame(rows, columns=["origin", "strain", "week", "r0_now", "r0_last_year"])


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean()
    return float((x * (y - y.mean())).sum() / (x * x).sum()) if (x * x).sum() > 0 else np.nan


def lastyear_use(mult: pd.DataFrame, pr: pd.DataFrame, arms, by: str | None = None) -> pd.DataFrame:
    """beta per model (and pooled) of log p50 on log(last-year R0 / current R0)."""
    d = mult[mult["arm"].isin(arms)].merge(pr, on=["origin", "strain", "week"])
    d = d[(d["p50"] > 0) & (d["r0_now"] > 0) & (d["r0_last_year"] > 0)].copy()
    d["x"], d["y"] = np.log(d["r0_last_year"] / d["r0_now"]), np.log(d["p50"])
    d["level"] = d[by].astype(str) if by else "all"
    rng = np.random.default_rng(SEED)
    out = []
    for level, sub in d.groupby("level"):
        models = sorted(sub["model"].unique())
        origins = np.sort(sub["origin"].unique())
        oi = {o: i for i, o in enumerate(origins)}
        idx = rng.integers(0, len(origins), size=(NBOOT, len(origins)))
        W = np.stack([np.bincount(r, minlength=len(origins)) for r in idx])        # resample weights per origin
        per = {}
        for m in models:
            g = sub[sub["model"] == m].assign(xx=lambda z: z.x * z.x, xy=lambda z: z.x * z.y, n=1)
            s = g.groupby("origin")[["n", "x", "y", "xx", "xy"]].sum()
            T = np.zeros((len(origins), 5))                                        # sufficient statistics
            T[[oi[o] for o in s.index]] = s.to_numpy()
            n, sx, sy, sxx, sxy = (W @ T).T
            with np.errstate(invalid="ignore", divide="ignore"):
                b = (sxy - sx * sy / n) / (sxx - sx * sx / n)
            per[m] = np.where(n >= 3, b, np.nan)
            out.append((level, m, g["origin"].nunique(), len(g), _slope(g["x"].to_numpy(), g["y"].to_numpy()),
                        *np.nanpercentile(per[m], [2.5, 97.5])))
        if len(models) > 1:
            pick = rng.integers(0, len(models), size=(NBOOT, len(models)))
            boot = np.nanmean(np.stack([per[m] for m in models], axis=1)[np.arange(NBOOT)[:, None], pick], axis=1)
            point = np.nanmean([r[4] for r in out if r[0] == level and r[1] in models])
            out.append((level, "pooled", sub["origin"].nunique(), len(sub), point, *np.nanpercentile(boot, [2.5, 97.5])))
    return pd.DataFrame(out, columns=["level", "model", "n_origins", "n_points", "beta", "lo95", "hi95"]).assign(by=by or "all")


def reasoning_comparison() -> pd.DataFrame:
    """A10-2: paired WIS ratios (reasoning / non-reasoning), relative WIS against constant R0, coverage."""
    from .scoring import stratified_gain
    sr, sm = pd.read_parquet(REASON / "scores.parquet"), pd.read_parquet(MAIN / "scores.parquet")
    base = sm[sm["model"] == "stan_const"][K + ["wis", "wis_log"]]
    rows = []
    for new, old in PAIRS.items():
        for arm in ("r0_semi", "direct_blind"):
            a = sr[(sr["model"] == new) & (sr["arm"] == arm)]
            b = sm[(sm["model"] == old) & (sm["arm"] == arm)]
            if a.empty or b.empty:
                continue
            for col, scale in (("wis", "natural"), ("wis_log", "log")):
                d = a[K + [col]].merge(b[K + [col]], on=K, suffixes=("", "_old")).merge(
                    base[K + [col]].rename(columns={col: col + "_ref"}), on=K)
                d["model"], d["all"] = new, "all"
                vs_old = stratified_gain(d, "all", col, col + "_old", seed=SEED).query("model != 'pooled'").iloc[0]
                new_ref = stratified_gain(d, "all", col, col + "_ref", seed=SEED).query("model != 'pooled'").iloc[0]
                old_ref = stratified_gain(d.assign(model=old), "all", col + "_old", col + "_ref", seed=SEED) \
                    .query("model != 'pooled'").iloc[0]
                rows.append((new, arm, scale, int(d["origin"].nunique()),
                             vs_old.ratio, vs_old.lo95, vs_old.hi95,
                             new_ref.ratio, old_ref.ratio,
                             a["cov90"].mean(), b[b.set_index(K).index.isin(a.set_index(K).index)]["cov90"].mean()))
    return pd.DataFrame(rows, columns=["model", "arm", "scale", "n_origins", "reasoning_vs_off", "lo95", "hi95",
                                       "relwis_reasoning", "relwis_off", "cov90_reasoning", "cov90_off"])


def main():
    pr = prompt_r0(MAIN)
    mult = pd.read_parquet(MAIN / "r0_multipliers.parquet")
    ph = pd.read_csv(MAIN / "table_phase_at_origin.csv", parse_dates=["origin"])[["origin", "strain", "phase"]]
    arms = ["r0_semi", "r0_ctx", "r0_imp"]
    m = mult.merge(ph, on=["origin", "strain"], how="left")
    tab = pd.concat([lastyear_use(m, pr, arms), lastyear_use(m, pr, arms, "phase"),
                     lastyear_use(m, pr, arms, "strain")], ignore_index=True)
    tab.to_csv(MAIN / "table_a10_lastyear_use.csv", index=False)
    print("A10-1 beta (log p50 on log last-year/current R0), designs iii-v, main study")
    print(tab[tab["by"] == "all"].round(3).to_string(index=False))
    print(tab[(tab["model"] == "pooled") & (tab["by"] != "all")].round(3).to_string(index=False))
    if (REASON / "scores.parquet").exists() and (REASON / "r0_multipliers.parquet").exists():
        rc = reasoning_comparison()
        mr = pd.read_parquet(REASON / "r0_multipliers.parquet")
        pr_r = prompt_r0(REASON)
        beta_r = lastyear_use(mr, pr_r, ["r0_semi"])
        beta_o = lastyear_use(mult[mult["model"].isin(PAIRS.values())], pr, ["r0_semi"])
        rc.to_csv(REASON / "table_a10_reasoning.csv", index=False)
        pd.concat([beta_r, beta_o]).to_csv(REASON / "table_a10_reasoning_beta.csv", index=False)
        print("\nA10-2 reasoning versus off (paired, same targets)")
        print(rc.round(3).to_string(index=False))
        print(pd.concat([beta_r, beta_o]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------- amendment 11 (pre-specified extension)
EXT = runs_root() / "reason_ext"
PAIRS_EXT = {"nemo3ur": "nemo3u", "ling3sr": "ling3s", "qwenr": "qwen", "glm52r": "glm52", "dsv41fr": "dsv41f",
             "gem25fr": "gem25f", "gem38fr": "gem38f"}


def amendment11(min_origins: int = 20) -> pd.DataFrame:
    """R1: paired reasoning / main study (design iii); R2: reasoning vs constant R0, growing phase; R3: all phases.
    Natural scale is primary; log scale also reported. First the new models, then new + A10-2 pilot."""
    from .scoring import stratified_gain
    sm = pd.read_parquet(MAIN / "scores.parquet")
    base = sm[sm["model"] == "stan_const"][K + ["wis", "wis_log"]]
    ph = pd.read_csv(MAIN / "table_phase_at_origin.csv", parse_dates=["origin"])[["origin", "strain", "phase"]]
    parts = []
    for src, pairs, tag in ((EXT, PAIRS_EXT, "new"), (REASON, PAIRS, "pilot")):
        if not (src / "scores.parquet").exists():
            continue
        s = pd.read_parquet(src / "scores.parquet")
        s = s[(s["arm"] == "r0_semi") & s["model"].isin(pairs)]
        n = s.groupby("model")["origin"].nunique()
        for m in n[n >= min_origins].index:
            parts.append(s[s["model"] == m].assign(set=tag, old=pairs[m]))
    if not parts:
        return pd.DataFrame()
    r = pd.concat(parts, ignore_index=True)
    main_iii = sm[sm["arm"] == "r0_semi"][K + ["model", "wis", "wis_log"]].rename(columns={"model": "old"})
    rows = []
    for sets, label in ((["new"], "new models"), (["new", "pilot"], "new + pilot")):
        rs = r[r["set"].isin(sets)]
        for col, scale in (("wis", "natural"), ("wis_log", "log")):
            # R1: paired with the same model's main-study answer on the same targets
            d1 = rs[K + ["model", "old", col]].merge(main_iii[K + ["old", col]], on=K + ["old"], suffixes=("", "_old"))
            d1["all"] = "all"
            g1 = stratified_gain(d1, "all", col, col + "_old", seed=SEED)
            # R2 / R3: against the mechanistic model with constant R0
            d2 = rs[K + ["model", col]].merge(base[K + [col]], on=K, suffixes=("", "_ref")).merge(ph, on=["origin", "strain"], how="left")
            d2["all"] = "all"
            g2 = stratified_gain(d2, "phase", col, col + "_ref", seed=SEED)
            g3 = stratified_gain(d2, "all", col, col + "_ref", seed=SEED)
            for g, hyp in ((g1, "R1 reasoning vs off"), (g2[g2["level"].astype(str) == "growing"], "R2 vs constant R0, growing"),
                           (g3, "R3 vs constant R0, all phases")):
                for x in g.itertuples():
                    rows.append((label, hyp, scale, x.model, x.n_origins, x.ratio, x.lo95, x.hi95,
                                 "supported" if (x.model == "pooled" and x.hi95 < 1) else ("not supported" if x.model == "pooled" else "")))
    out = pd.DataFrame(rows, columns=["set", "hypothesis", "scale", "model", "n_origins", "ratio", "lo95", "hi95", "decision"])
    out.to_csv(EXT / "table_a11_hypotheses.csv", index=False)
    return out


def replay_reasoning_beta() -> None:
    """Repeat the last-year R0 analysis from saved numerical prompt inputs."""
    main_r0 = pd.read_parquet(MAIN / "lastyear_r0_inputs.parquet")
    reason_r0 = pd.read_parquet(REASON / "lastyear_r0_inputs.parquet")
    main = pd.read_parquet(MAIN / "r0_multipliers.parquet")
    reason = pd.read_parquet(REASON / "r0_multipliers.parquet")
    phase = pd.read_csv(MAIN / "table_phase_at_origin.csv", parse_dates=["origin"])
    arms = ["r0_semi", "r0_ctx", "r0_imp"]
    phased = main.merge(phase[["origin", "strain", "phase"]], on=["origin", "strain"], how="left")
    pd.concat([lastyear_use(phased, main_r0, arms, by) for by in (None, "phase", "strain")],
              ignore_index=True).to_csv(MAIN / "table_a10_lastyear_use.csv", index=False)
    pd.concat([
        lastyear_use(reason, reason_r0, ["r0_semi"]),
        lastyear_use(main[main["model"].isin(PAIRS.values())], main_r0, ["r0_semi"]),
    ]).to_csv(REASON / "table_a10_reasoning_beta.csv", index=False)
