"""各实验组的 prompt

direct_blind  LLM 直接给病例分位数；日期、地点、病原体名全部隐去，每条序列重缩放到最大值 1000
direct_named  LLM 直接给病例分位数；真实日期、香港、病原体名
r0_semi       LLM 给未来 4 周的 R0 乘数，机制模型负责前推；保留病原体名和季节，隐去年份和城市
r0_named      同上，信息全开
r0_ctx        同 r0_semi，另加背景信息：校历/公众假期、天气（与 r0_semi 只差这两块，二者之差 = 背景信息的价值）
r0_imp        同 r0_ctx，再加流感输入压力（航空 + 陆路/海路入境 x 来源地阳性率，见 Data/context/importation；与 r0_ctx 之差 = 输入压力的价值）
r0_imponly    同 r0_semi，只加流感输入压力（不给校历、天气）；和上面三组构成 2x2 析因设计
r0_placebo    同 r0_imp，但输入压力换成起点前 13-39 周（按起点日期固定种子抽取）的数据：格式、长度、
              数值范围都一样，只是和当前时间对不上；r0_imp 与它之差 = 真实输入信号的价值
              （外部信息实验的设计和分析计划见 bench/protocol_context_experiment.md）

每个 R0 组给哪些背景信息由 ARM_CONTEXT 决定；r0_semi 固定只有航空旅客（和已收回的回答保持同一个 prompt）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config
from .data import OriginBundle

ARMS = {
    "direct_blind": ("direct", False),
    "direct_named": ("direct", True),
    "r0_semi": ("r0", False),
    "r0_named": ("r0", True),
    "r0_ctx": ("r0", False),
    "r0_imp": ("r0", False),
    "r0_imponly": ("r0", False),
    "r0_placebo": ("r0", False),
    # mechanism test (counterfactual prompts, runs/cf/): same epidemic state, one input changed
    "cf_term": ("r0", False), "cf_holiday": ("r0", False),
    "cf_h3_low": ("r0", False), "cf_h3_high": ("r0", False),
    "cf_news_none": ("r0", False), "cf_news_variant": ("r0", False),
}
ARM_CONTEXT = {                       # R0 组附带的背景信息块
    "r0_semi": ("oag",),
    "r0_named": ("oag", "calendar"),
    "r0_ctx": ("oag", "calendar", "weather"),
    "r0_imp": ("oag", "calendar", "weather", "import"),
    "r0_imponly": ("oag", "import"),
    "r0_placebo": ("oag", "calendar", "weather", "import_placebo"),
    "cf_term": ("oag", "cal_term", "weather"),
    "cf_holiday": ("oag", "cal_holiday", "weather"),
    "cf_h3_low": ("oag", "calendar", "weather", "import_h3_low"),
    "cf_h3_high": ("oag", "calendar", "weather", "import_h3_high"),
    "cf_news_none": ("oag", "news_none"),
    "cf_news_variant": ("oag", "news_variant"),
}
# counterfactual inputs of the mechanism test
CF_H3 = {"import_h3_low": ([1.0, 1.0, 1.0, 1.0], 0.20),        # A(H3N2) importation only; fixed levels
         "import_h3_high": ([8.0, 11.0, 15.0, 20.0], 2.50)}  # weighted positivity (%), relative pressure
CF_NEWS = {
    "news_none": "Recent public-health news: no new variants or notable changes have been reported for any of the "
                 "pathogens.",
    "news_variant": "Recent public-health news: a new SARS-CoV-2 variant with marked immune escape is spreading in "
                    "neighbouring regions and accounts for a rapidly growing share of local sequences; no notable "
                    "changes have been reported for the other pathogens.",
}


def _cf_calendar(kind: str, H: int) -> str:
    end = 7 * H
    if kind == "cal_term":
        rows = [f"- day -13 to day +{end} relative to the origin: school term (primary and secondary schools in "
                "session; no school or public holidays)"]
    else:
        rows = ["- day -13 to day +0 relative to the origin: school term (primary and secondary schools in session)",
                f"- day +1 to day +{end} relative to the origin: school holidays (primary and secondary schools closed)"]
    return "Calendar events near the forecast window:\n" + "\n".join(rows)


def _cf_import(imp: pd.DataFrame | None, kind: str, origin: pd.Timestamp) -> pd.DataFrame | None:
    """the A(H3N2) rows shown in the prompt (latest 4 weeks available at the origin) set to fixed levels"""
    if imp is None:
        return None
    x = imp.copy()
    rows = x[(x["subtype"] == "H3") & (x["week_end"] <= origin - pd.Timedelta(days=IMPORT_DELAY_DAYS))]
    rows = rows.sort_values("week_end").tail(4)
    vals, rel = CF_H3[kind]
    x.loc[rows.index, "weighted_positivity"] = vals[-len(rows):]
    x.loc[rows.index[-1], "pressure_rel"] = rel
    return x
IMPORT_DELAY_DAYS = 14                # FluNet 周报按报告延迟 2 周处理：只给起点前已能拿到的周
# 安慰剂：整块输入压力信息（各型别、覆盖率、主要来源地、台湾）取自起点前 s 周，s 在 13-39 周之间，
# 由起点日期和固定种子决定（与参加实验的起点集合无关）；只往过去移，不会带进起点之后的信息；
# 避开 48-56 周，免得正好对上去年同季
PLACEBO_SHIFT_WEEKS = (13, 39)
PLACEBO_SEED = 20260925


def placebo_shift_weeks(origin: pd.Timestamp) -> int:
    lo, hi = PLACEBO_SHIFT_WEEKS
    rng = np.random.default_rng([PLACEBO_SEED, int(origin.strftime("%Y%m%d"))])
    return int(rng.integers(lo, hi + 1))

SYSTEM_PROMPT = (
    "You are an expert infectious-disease epidemiologist and forecaster. Your probabilistic "
    "forecasts are well calibrated: each central interval should contain the truth at its "
    "nominal rate, neither more nor less often. Reply with exactly one JSON object and "
    "nothing else."
)

R0_MULT_RANGE = (0.3, 3.0)
COUNT_WORD = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}


@dataclass
class PromptSpec:
    arm: str
    kind: str                         # "direct" | "r0"
    system: str
    user: str
    keys: list[str]                   # answer keys, in config strain order
    scale: np.ndarray | None = None   # direct_blind: answer x scale = original units
    mock: dict = field(default_factory=dict)


def _num(v: float, decimals: int = 0) -> str:
    if v is None or not np.isfinite(v):
        return "NA"
    return f"{v:.{decimals}f}" if decimals else f"{int(round(v))}"


def _period(ts: pd.Timestamp) -> str:
    part = "early" if ts.day <= 10 else ("mid" if ts.day <= 20 else "late")
    return f"{part} {ts.strftime('%B')}"


def build_direct(cfg: Config, obs: pd.DataFrame, origin: pd.Timestamp, named: bool) -> PromptSpec:
    strains, H, levels = cfg.strains, cfg.horizons, cfg.quantiles
    hist = obs.loc[:origin].tail(int(cfg["history_weeks_direct"]))
    qtxt = ", ".join(f"{q:g}" for q in levels)
    arm = "direct_named" if named else "direct_blind"

    if named:
        keys = list(strains)
        scale = None
        desc = "\n".join(f"- {s}: {cfg['series_descriptions'].get(s, s)}" for s in strains)
        rows = ["week_ending," + ",".join(strains)] + [
            d.strftime("%Y-%m-%d") + "," + ",".join(_num(v) for v in r)
            for d, r in zip(hist.index, hist.to_numpy(float))]
        targets = [(origin + pd.Timedelta(weeks=h)).strftime("%Y-%m-%d") for h in range(1, H + 1)]
        pop = cfg["setting"]["population_millions"]
        user = (
            f"Setting: {cfg['setting']['named']} (population about {pop} million, subtropical climate).\n"
            f"Data: weekly surveillance indicators for {len(strains)} pathogens; each week ends on a Saturday.\n"
            f"{desc}\n{cfg['series_note']}\n\n"
            f"The last observed week ends on {origin:%Y-%m-%d} (the forecast origin).\n\n"
            f"History ({len(hist)} weeks, oldest first):\n```csv\n" + "\n".join(rows) + "\n```\n\n"
            f"Task: forecast each indicator for the weeks ending {', '.join(targets[:-1])} and "
            f"{targets[-1]} (horizons 1-{H}). For each pathogen and horizon give the quantiles at "
            f"levels [{qtxt}], in that order (non-decreasing, in the indicator's own units).\n\n"
            "Return exactly this JSON on a single line, whole numbers only (no decimals, no spaces):\n"
            + _direct_shape(keys, H, len(levels)))
        last = hist.iloc[-1].to_numpy(float)
    else:
        rng = np.random.default_rng(int(origin.strftime("%Y%m%d")))
        perm = rng.permutation(len(strains))             # letter i shows strain perm[i]
        letters = [chr(ord("A") + i) for i in range(len(strains))]
        shown = [strains[p] for p in perm]
        maxes = hist[shown].max().to_numpy(float)
        factor = np.where(maxes > 0, 1000.0 / maxes, 1.0)
        table = hist[shown].to_numpy(float) * factor
        t_idx = np.arange(-(len(hist) - 1), 1)
        rows = ["t," + ",".join(letters)] + [
            f"{t}," + ",".join(_num(v, 1) for v in r) for t, r in zip(t_idx, table)]
        pos = [int(np.where(perm == i)[0][0]) for i in range(len(strains))]
        keys = [letters[p] for p in pos]
        scale = np.array([1.0 / factor[p] for p in pos])
        user = (
            f"You will see {len(strains)} anonymised weekly time series from infectious-disease "
            f"surveillance, labelled {letters[0]}-{letters[-1]}. Each series was rescaled so that its "
            "maximum over the history shown equals 1000; values are non-negative. Time is in weeks "
            "relative to the last observed week (t = 0). Everything else - the pathogens, the place "
            "and the calendar dates - is deliberately withheld.\n\n"
            f"History ({len(hist)} weeks, oldest first):\n```csv\n" + "\n".join(rows) + "\n```\n\n"
            f"Task: forecast each series at t = {', '.join(str(h) for h in range(1, H + 1))}, in the "
            f"same rescaled units. For each series and horizon give the quantiles at levels "
            f"[{qtxt}], in that order (non-decreasing).\n\n"
            "Return exactly this JSON on a single line, whole numbers only (no decimals, no spaces):\n"
            + _direct_shape(letters, H, len(levels)))
        last = table[-1][pos]
    return PromptSpec(arm=arm, kind="direct", system=SYSTEM_PROMPT, user=user, keys=keys,
                      scale=scale, mock={"last": last.tolist(), "levels": levels, "horizons": H})


def _direct_shape(keys: list[str], H: int, n_q: int) -> str:
    # 紧凑格式（每个序列一个按周排列的嵌套列表），一次回答约 700 token，适合有输出限额的模型
    inner = "[" + ", ".join(f"[{n_q} values for week {h}]" for h in range(1, H + 1)) + "]"
    return '{"forecasts": {' + ", ".join(f'"{k}": {inner}' for k in keys) + "}}"


def build_r0(cfg: Config, obs: pd.DataFrame, bundle: OriginBundle, oag: pd.DataFrame | None,
             calendar: pd.DataFrame | None, named: bool, weather: pd.DataFrame | None = None,
             context_blocks: tuple = ("oag", "calendar"), arm: str | None = None,
             importation: pd.DataFrame | None = None) -> PromptSpec:
    strains, H = cfg.strains, cfg.horizons
    origin = bundle.origin
    nw = int(cfg["history_weeks_r0"])
    lat = bundle.latent
    recent = obs.loc[:origin].index[-nw:]
    targets = [origin + pd.Timedelta(weeks=h) for h in range(1, H + 1)]
    pop = cfg["setting"]["population_millions"]
    place = cfg["setting"]["named"] if named else cfg["setting"]["semi"]
    lo, hi = R0_MULT_RANGE

    if named:
        timing = (f"Forecast origin: the week ending {origin:%Y-%m-%d} is the last observed week. "
                  f"Forecast weeks 1-{H} end on " + ", ".join(f"{d:%Y-%m-%d}" for d in targets) + ".")
    else:
        timing = (f"Forecast origin: ISO week {origin.isocalendar().week} ({_period(origin)}) is the "
                  "last observed week; the year is withheld. Forecast weeks 1-"
                  f"{H} are ISO weeks " + ", ".join(str(d.isocalendar().week) for d in targets) + ".")

    def series(var: str, s: str) -> pd.DataFrame:
        return lat[(lat["variable"] == var) & (lat["strain"] == s)].set_index("date")

    blocks = []
    r0_end = bundle.per_strain("R0_end", strains)
    for i, s in enumerate(strains):
        R0, Rt, S = series("R0", s), series("Rt", s), series("S", s)
        lines = ["week | observed | R0 median [95% CrI] | Rt median [95% CrI] | immunity 1-S (median)"]
        for j, d in enumerate(recent):
            label = f"{d:%Y-%m-%d}" if named else f"w{j - (len(recent) - 1)}"
            r0 = R0.loc[d] if d in R0.index else None
            rt = Rt.loc[d] if d in Rt.index else None
            sv = S.loc[d] if d in S.index else None
            lines.append(
                f"{label} | {_num(obs.at[d, s])} | "
                + (f"{r0['med']:.2f} [{r0['lo95']:.2f}, {r0['hi95']:.2f}]" if r0 is not None else "NA")
                + " | "
                + (f"{rt['med']:.2f} [{rt['lo95']:.2f}, {rt['hi95']:.2f}]" if rt is not None else "NA")
                + " | " + (f"{1 - sv['med']:.2f}" if sv is not None else "NA"))
        cur = r0_end[:, i]
        lines.append(f"Current R0 (mean of the last 7 days): {np.median(cur):.2f} "
                     f"[{np.quantile(cur, 0.025):.2f}, {np.quantile(cur, 0.975):.2f}]")
        ly = [origin - pd.Timedelta(days=364) + pd.Timedelta(weeks=h) for h in range(1, H + 1)]
        if all(d in R0.index for d in ly):
            lines.append("R0 in the same weeks one year earlier (median): "
                         + ", ".join(f"week {h}: {R0.loc[d, 'med']:.2f}" for h, d in enumerate(ly, 1)))
        else:
            lines.append("R0 in the same weeks one year earlier: not available (under one year of data).")
        name = cfg["series_descriptions"].get(s, s)
        blocks.append(f"### {s} ({name})\n" + "\n".join(lines))

    cf_cal = next((k for k in ("cal_term", "cal_holiday") if k in context_blocks), None)
    cf_imp = next((k for k in CF_H3 if k in context_blocks), None)
    cf_news = next((k for k in CF_NEWS if k in context_blocks), None)
    context = [b for b in ((_oag_block(oag, origin, named) if "oag" in context_blocks else ""),
                           (_calendar_block(calendar, origin, H, named) if "calendar" in context_blocks else ""),
                           (_cf_calendar(cf_cal, H) if cf_cal else ""),
                           (_weather_block(weather, origin, H, named) if "weather" in context_blocks else ""),
                           (_import_block(importation, origin) if "import" in context_blocks else ""),
                           (_import_block(importation, origin - pd.Timedelta(weeks=placebo_shift_weeks(origin)))
                            if "import_placebo" in context_blocks else ""),
                           (_import_block(_cf_import(importation, cf_imp, origin), origin) if cf_imp else ""),
                           (CF_NEWS[cf_news] if cf_news else ""))
               if b]
    shape_inner = "[" + ", ".join(f"[p10, p50, p90 for week {h}]" for h in range(1, H + 1)) + "]"
    shape = '{"multipliers": {' + ", ".join(f'"{s}": {shape_inner}' for s in strains) + "}}"
    user = (
        f"Setting: {place} (population about {pop} million).\n\n"
        "Background: we fitted a Bayesian multi-pathogen SEIRS transmission model "
        f"({COUNT_WORD.get(len(strains), len(strains))} pathogens, "
        "with cross-immunity between influenza subtypes) to weekly surveillance data up to the "
        "forecast origin. In this model the coming weeks are fully determined by (a) the current "
        "state - who is susceptible, exposed and infectious - which the model estimates, and (b) "
        "the future basic reproduction number R0(t) of each pathogen, which the model cannot "
        "foresee. R0(t) absorbs everything outside the model: school terms and holidays, weather "
        "and season, contact behaviour, importations from abroad, and new variants or antigenic "
        "drift. Susceptible depletion and waning immunity are handled by the model (Rt = R0 x S), "
        "so do NOT adjust R0 for immunity build-up.\n\n"
        "Your task: for each pathogen, forecast how R0 will change over the next "
        f"{H} weeks relative to its current value (multiplier 1.0 = unchanged).\n\n"
        f"{timing}\n\n"
        f"{cfg['series_note']}\n"
        "Recent history per pathogen (model estimates are posterior medians with 95% credible "
        "intervals):\n\n" + "\n\n".join(blocks) + "\n\n"
        + ("\n\n".join(context) + "\n\n" if context else "")
        + "Return exactly this JSON on a single line (at most 2 decimals, no spaces):\n" + shape + "\n"
        f"Multipliers scale the current R0; p10 <= p50 <= p90, each between {lo} and {hi}. "
        "Use wide intervals when the drivers are uncertain.")
    arm = arm or ("r0_named" if named else "r0_semi")
    return PromptSpec(arm=arm, kind="r0", system=SYSTEM_PROMPT, user=user, keys=list(strains),
                      mock={"horizons": H})


def _oag_block(oag: pd.DataFrame | None, origin: pd.Timestamp, named: bool) -> str:
    if oag is None:
        return ""
    month0 = origin.to_period("M").to_timestamp()
    total = oag.groupby("month_start")["inbound"].sum()
    prev = total[total.index < month0].tail(6)       # complete months before the origin month
    if prev.empty:
        return ""
    lines = []
    for m, v in prev.items():
        label = f"{m:%Y-%m}" if named else f"{m:%B}"
        yoy = total.get(m - pd.DateOffset(years=1))
        change = f" ({(v / yoy - 1) * 100:+.0f}% vs same month a year earlier)" if yoy else ""
        lines.append(f"- {label}: {v / 1e6:.2f} million{change}")
    text = "Inbound air passengers per month (all origins), last complete months:\n" + "\n".join(lines)
    if named:
        last3 = oag[oag["month_start"].isin(prev.index[-3:])]
        share = last3.groupby("partner")["inbound"].sum()
        share = (share / share.sum()).sort_values(ascending=False).head(5)
        text += "\nLargest origins over the last 3 months: " + ", ".join(
            f"{p} {v * 100:.0f}%" for p, v in share.items())
    return text


def _calendar_block(cal: pd.DataFrame | None, origin: pd.Timestamp, H: int, named: bool) -> str:
    if cal is None:
        return ""
    lo, hi = origin - pd.Timedelta(days=13), origin + pd.Timedelta(weeks=H)
    ev = cal[(cal["end"] >= lo) & (cal["start"] <= hi)]
    if ev.empty:
        return "Calendar: no listed school terms or public holidays between two weeks before the origin and the end of the forecast window."
    rows = []
    for _, e in ev.iterrows():
        if named:
            rows.append(f"- {e['start']:%Y-%m-%d} to {e['end']:%Y-%m-%d}: {e['event']}")
        else:
            label = e["event_semi"] if isinstance(e.get("event_semi"), str) and e["event_semi"] else e["event"]
            rows.append(f"- day {(e['start'] - origin).days:+d} to day {(e['end'] - origin).days:+d} "
                        f"relative to the origin: {label}")
    return "Calendar events near the forecast window:\n" + "\n".join(rows)


def _weather_block(w: pd.DataFrame | None, origin: pd.Timestamp, H: int, named: bool) -> str:
    """最近 4 周的实测天气（与同周 30 年常年值 1991-2020 的差）+ 预测期 4 周的常年值（只用常年值，不用实测）；
    半盲组不写年份，所以说"30-year normal"而不写 1991-2020"""
    if w is None:
        return ""
    w = w.set_index("week_end")
    past = [origin - pd.Timedelta(weeks=k) for k in range(3, -1, -1)]
    fut = [origin + pd.Timedelta(weeks=h) for h in range(1, H + 1)]
    if any(d not in w.index for d in past + fut):
        return ""
    lines = []
    for k, d in zip(range(-3, 1), past):
        r = w.loc[d]
        label = f"week ending {d:%Y-%m-%d}" if named else f"w{k}"
        lines.append(f"- {label}: {r.temp:.1f} C ({r.temp - r.temp_norm:+.1f}), relative humidity "
                     f"{r.rh:.0f}% ({r.rh - r.rh_norm:+.0f}), absolute humidity {r.ah:.1f} g/m3 "
                     f"({r.ah - r.ah_norm:+.1f}), rainfall {r.rain:.0f} mm (normal {r.rain_norm:.0f})")
    normals = "; ".join(
        f"{f'week ending {d:%Y-%m-%d}' if named else f'w+{h}'}: {w.loc[d, 'temp_norm']:.1f} C, "
        f"absolute humidity {w.loc[d, 'ah_norm']:.1f} g/m3" for h, d in enumerate(fut, 1))
    return ("Weather, weekly means at the main observatory (in brackets: difference from the "
            "30-year normal for the same week):\n" + "\n".join(lines)
            + "\nNormal conditions for the forecast weeks (long-term averages, not a forecast): "
            + normals + ".")


def _import_block(imp: pd.DataFrame | None, origin: pd.Timestamp) -> str:
    """流感输入压力（航空 + 陆路/海路）：最近 4 个在起点时已发布的周（报告延迟 2 周）；
    台湾没有检测总数和分型，不并入指数，单独给出相对活动水平；半盲组只写地区不写国家。
    安慰剂组传入的是往前移过的起点，文字完全一样"""
    if imp is None:
        return ""
    avail = imp[imp["week_end"] <= origin - pd.Timedelta(days=IMPORT_DELAY_DAYS)]
    lines, cover, tw = [], None, {}
    for k, name in (("H1", "A(H1N1)"), ("H3", "A(H3N2)"), ("B", "B")):
        d = avail[avail["subtype"] == k].sort_values("week_end").tail(4)
        if len(d) < 4:
            continue
        last = d.iloc[-1]
        cover = last["coverage"]
        tw["B" if k == "B" else "A"] = last["taiwan_rel"]
        tw["share"] = last["taiwan_air_share"]
        lines.append(f"- influenza {name}: " + ", ".join(f"{v:.1f}%" for v in d["weighted_positivity"])
                     + f" (oldest to latest); relative pressure {last['pressure_rel']:.2f} "
                     f"(1.00 = average of an earlier reference year); main origins: {last['top_regions']}")
    if not lines:
        return ""
    text = ("Influenza importation pressure: all arrivals by air, land and sea, weighted by the share "
            "of respiratory specimens testing positive for each subtype at the travellers' places of "
            "origin (WHO FluNet for air origins, provincial surveillance for the neighbouring mainland; "
            "weekly values, the latest 4 weeks already reported two weeks before the forecast origin; "
            f"origins covering {cover:.0%} of arrivals; influenza only, other pathogens not covered):\n"
            + "\n".join(lines))
    if all(np.isfinite(tw.get(k, np.nan)) for k in ("A", "B")):
        text += (f"\nA nearby origin not included above (about {tw['share']:.0%} of air arrivals; no testing "
                 f"denominators) reports laboratory-confirmed influenza A at {tw['A']:.2f} and influenza B "
                 f"at {tw['B']:.2f} times their average of the same reference year.")
    return text


def build_prompt(arm: str, cfg: Config, obs: pd.DataFrame, bundle: OriginBundle,
                 oag: pd.DataFrame | None, calendar: pd.DataFrame | None,
                 weather: pd.DataFrame | None = None,
                 importation: pd.DataFrame | None = None) -> PromptSpec:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm}; choose from {sorted(ARMS)}")
    kind, named = ARMS[arm]
    if kind == "direct":
        return build_direct(cfg, obs, bundle.origin, named)
    return build_r0(cfg, obs, bundle, oag, calendar, named, weather=weather,
                    context_blocks=ARM_CONTEXT.get(arm, ("oag", "calendar")), arm=arm,
                    importation=importation)
