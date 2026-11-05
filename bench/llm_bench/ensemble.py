"""LLM 预测与季节性机制基线（去年同期 R0）的组合

组合方式：逐分位数加权平均（quantile averaging / Vincentization，预测汇总平台常用的做法）。
  equal    LLM 和去年同期 R0 各占 0.5
  learned  每个起点只用它之前已经观测到的目标（target_date <= 起点）选权重（0-1，步长 0.1），
           目标函数与评分一致：各病原体 log(WIS 之和) 的平均；没有历史时用 0.5
对照：去年同期 R0 单独用、LLM 单独用、两个机制基线（固定 R0 + 去年同期 R0）等权组合
（检验收益是否只来自"组合"这个动作本身）。
llm_ens：起点齐全的几个 LLM 先取平均，再与季节基线组合。
只用主结果的第 1 次调用（sample 0）。事后分析，结果按探索性处理。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .scoring import paired_gain, score_forecasts

KEYS = ["origin", "target_date", "horizon", "strain"]
GRID = np.round(np.linspace(0, 1, 11), 1)          # weight on the LLM
SEASONAL, CONSTANT = "stan_lastyear", "stan_const"


def _wide(fc: pd.DataFrame, model: str, arm: str) -> pd.DataFrame:
    d = fc[(fc["model"] == model) & (fc["arm"] == arm)]
    return d.pivot_table(index=KEYS, columns="quantile", values="value").sort_index()


def _long(w: pd.DataFrame, model: str, arm: str) -> pd.DataFrame:
    out = w.stack().rename("value").reset_index()
    out["model"], out["arm"] = model, arm
    return out


def _log_score(sc: pd.DataFrame, col: str) -> float:
    """mean over pathogens of log(sum of WIS): the quantity that relative WIS compares"""
    s = sc.groupby("strain")[col].sum()
    return float(np.log(s.clip(lower=1e-12)).mean()) if len(s) else np.nan


def build(fc: pd.DataFrame, obs: pd.DataFrame, levels: list[float], llm_models: list[str],
          arms: list[str], ens_members: list[str] | None = None, min_origins: int = 5):
    """-> (scores of every method, chosen weights per origin)"""
    ly, const = _wide(fc, SEASONAL, "baseline"), _wide(fc, CONSTANT, "baseline")
    parts = []
    for arm in arms:
        sources = {m: _wide(fc, m, arm) for m in llm_models}
        sources = {m: w for m, w in sources.items()
                   if len(w) and w.index.get_level_values("origin").nunique() >= min_origins}
        members = [m for m in (ens_members or []) if m in sources]
        if len(members) >= 2:                          # multi-LLM mean, targets that every member has
            common = sources[members[0]].index
            for m in members[1:]:
                common = common.intersection(sources[m].index)
            sources["llm_ens"] = sum(sources[m].loc[common] for m in members) / len(members)
        for m, w in sources.items():
            idx = w.index.intersection(ly.index)
            a, b = w.loc[idx], ly.loc[idx]
            for g in GRID:                             # g = weight on the LLM
                parts.append(_long(g * a + (1 - g) * b, m, f"{arm}|{g:.1f}"))
            parts.append(_long(0.5 * const.loc[idx] + 0.5 * b, m, f"{arm}|mech_pair"))
            parts.append(_long(const.loc[idx], m, f"{arm}|const"))
    sc = score_forecasts(pd.concat(parts, ignore_index=True), obs, levels)
    sc[["design", "method"]] = sc["arm"].str.split("|", expand=True)

    # online weights: for origin t use only targets observed by t
    rows, chosen = [], []
    for (m, design), g in sc[sc["method"].isin([f"{x:.1f}" for x in GRID])].groupby(["model", "design"]):
        for t in sorted(g["origin"].unique()):
            past = g[(g["origin"] < t) & (g["target_date"] <= t)]
            if past.empty:
                best = 0.5
            else:
                obj = {float(k): _log_score(d, "wis") for k, d in past.groupby("method")}
                lo = min(obj.values())
                best = min((k for k, v in obj.items() if v <= lo + 1e-12), key=lambda k: abs(k - 0.5))
            chosen.append((m, design, t, best))
            rows.append(g[(g["origin"] == t) & (g["method"] == f"{best:.1f}")].assign(method="learned"))
    sc = pd.concat([sc] + rows, ignore_index=True)
    names = {"1.0": "llm", "0.0": "seasonal", "0.5": "equal", "mech_pair": "mech_pair", "const": "const",
             "learned": "learned"}
    sc = sc[sc["method"].isin(names)].assign(method=lambda d: d["method"].map(names))
    weights = pd.DataFrame(chosen, columns=["model", "design", "origin", "llm_weight"])
    return sc, weights


def summarise(sc: pd.DataFrame, n_boot: int = 4000) -> pd.DataFrame:
    """relative WIS vs constant R0 (natural and log scale) and ratio vs the seasonal baseline"""
    out = []
    for (m, design), g in sc.groupby(["model", "design"]):
        ref = g[g["method"] == "const"]
        row = {"model": m, "design": design, "n_origins": g["origin"].nunique()}
        for method in ("llm", "seasonal", "mech_pair", "equal", "learned"):
            x = g[g["method"] == method]
            for col, tag in (("wis", ""), ("wis_log", "_log")):
                row[f"{method}{tag}"] = float(np.exp(_log_score(x, col) - _log_score(ref, col)))
        base = g.assign(model=m)
        for method in ("equal", "learned", "mech_pair"):
            pg = paired_gain(base.assign(arm=base["method"]), "seasonal", method, n_boot=n_boot)
            r = pg[pg["horizon"].astype(str) == "all"].iloc[0]
            row.update({f"{method}_vs_seasonal": r["ratio"], f"{method}_lo95": r["lo95"],
                        f"{method}_hi95": r["hi95"], f"{method}_p_better": r["p_better"]})
        out.append(row)
    return pd.DataFrame(out)
