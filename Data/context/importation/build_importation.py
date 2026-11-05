"""Influenza importation pressure into Hong Kong by air, land and sea (LLM context, arm r0_imp).

Method: the standard importation-risk formulation, expected imported infections = number of
travellers from each source x infection prevalence at the source (Wu JT, Leung K, Leung GM.
Lancet 2020;395:689-97), used as a relative index of travel flows weighted by each source's
epidemic activity (Gilbert M, Pullano G, Pinotti F, et al. Lancet 2020;395:871-7). Air passenger
flux is the dominant predictor of the global spread of seasonal influenza (Lemey P, Rambaut A,
Bedford T, et al. PLoS Pathog 2014;10:e1003932). Subtype-specific activity = proportion of
specimens positive for the subtype (as in Goldstein E, Cobey S, Takahashi S, Miller JC, Lipsitch M.
PLoS Med 2011;8:e1001051).

Sources (all official, downloaded unchanged into raw/):
  raw/flunet_VIW_FNT_2023on.csv           WHO FluNet, weekly specimens tested and positive by
                                          subtype for every reporting country (public API)
  raw/cnic_weekly_reports/*.pdf           Chinese National Influenza Center weekly reports,
                                          Table 1: southern provinces, specimens tested and
                                          positive by subtype (and the southern ILI%)
  raw/immd_daily_passenger_traffic.csv    HK Immigration Department, daily arrivals by control point
  raw/taiwan_lars_influenza_2023w27_2026w20.html
                                          Taiwan CDC NIDSS, weekly laboratory-confirmed influenza A
                                          and B positives (LARS, 29 hospitals); no denominators
  ../../OAG_HongKong_2011-2026Mar.xlsx      OAG monthly inbound air passengers by origin country

Construction, for influenza subtype k (A(H1N1)pdm09, A(H3N2), B) and ISO week w:
  arrivals: air arrivals = ImmD airport arrivals of week w, split across origin countries by
    their OAG share of inbound air passengers in the previous month (OAG counts transfer
    passengers who do not enter Hong Kong; its totals are ~1.9x the immigration arrivals, so OAG
    is used only for the origin mix); land and sea arrivals = ImmD arrivals at the mainland
    land and ferry control points (incl. the Hong Kong-Zhuhai-Macao Bridge);
    Macau Ferry Terminal arrivals are not covered (no Macau data); cruise arrivals are excluded
  positivity p_jk: air origins from FluNet (unsubtyped A shared between H1 and H3 in proportion
    to the subtyped A; if a country does not report specimens processed, its median over the
    period is used, flag denom_imputed); land/sea arrivals from the southern-provinces positivity
    of the CNIC reports; 3-week trailing mean (past weeks only)
  weighted_positivity_k = sum_j arrivals_j p_jk / sum_j arrivals_j over covered arrivals (%)
  pressure_k            = sum_j arrivals_j p_jk (positive-equivalent arrivals per week)
  pressure_rel_k        = pressure_k / its mean from 2023-07-03 to 2024-06-30
  coverage              = covered arrivals / all arrivals (air + land + sea)
Taiwan reports no denominators or subtypes, so it is not merged into the index; its influenza A
(for the H1 and H3 rows) or B (for the B row) positives relative to their mean over the same
reference year are given separately (taiwan_rel, 3-week trailing mean; taiwan_air_share = its
share of inbound air passengers in OAG).

Checks printed on every build: every CNIC table parsed; rows where south + north != total, or
where the southern subtypes do not add up, are listed (typos in the published tables);
CNIC national positivity vs WHO FluNet China (a different, larger specimen stream, so only the
positivity is compared); OAG vs immigration airport arrivals; week coverage.

Outputs: importation_weekly.csv (one row per week and subtype; used in the prompt)
         importation_by_source.csv (arrivals, positivity and contribution per source and week)
Usage: python build_importation.py   (needs pypdf for the CNIC reports)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pypdf

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
OAG = HERE.parents[1] / "OAG_HongKong_2011-2026Mar.xlsx"      # OAG 2011-01 to 2026-03
START, END = "2023-07-03", "2026-04-27"               # ISO weeks 2023-W27 .. 2026-W18 (air origin mix: OAG of the previous month, to 2026-03)
BASELINE = ("2023-07-03", "2024-06-30")
SUB = ("H1", "H3", "B")
NAME = {"Korea Republic of": "Republic of Korea", "USA": "United States of America",
        "Netherlands": "Netherlands (Kingdom of the)", "Turkiye": "Türkiye",
        "Ireland Republic of": "Ireland",
        "United Kingdom": "United Kingdom of Great Britain and Northern Ireland"}
ITZ = {"FLU_EST_ASIA": "East Asia", "FLU_SE_ASIA": "South-East Asia", "FLU_STH_ASIA": "South Asia",
       "FLU_WST_ASIA": "West Asia", "FLU_NRT_AMR": "North America", "FLU_OCE_MEL_POL": "Oceania",
       "FLU_NTH_EUR": "Europe", "FLU_SW_EUR": "Europe", "FLU_EST_EUR": "Europe",
       "FLU_CNT_ASIA": "Central Asia", "FLU_CNT_AMC": "Central America and Caribbean",
       "FLU_TRP_SAMR": "South America", "FLU_TEMP_SAMR": "South America", "FLU_NRT_AFR": "Africa",
       "FLU_WST_AFR": "Africa", "FLU_MID_AFR": "Africa", "FLU_EST_AFR": "Africa", "FLU_STH_AFR": "Africa"}
LAND_SEA = ["Lo Wu", "Lok Ma Chau", "Lok Ma Chau Spur Line", "Shenzhen Bay", "Man Kam To", "Sha Tau Kok",
            "Heung Yuen Wai", "Hung Hom", "Express Rail Link West Kowloon",
            "Hong Kong-Zhuhai-Macao Bridge", "China Ferry Terminal", "Tuen Mun Ferry Terminal",
            "Harbour Control"]
UNCOVERED = ["Macau Ferry Terminal", "Macao Ferry Terminal"]
LAND_LABEL = "Neighbouring mainland (land and sea)"


def iso_monday(d: pd.Series) -> pd.Series:
    return d - pd.to_timedelta(d.dt.weekday, unit="D")


def trailing(s: pd.Series, by: pd.Series | None = None) -> pd.Series:
    f = (lambda x: x.rolling(3, min_periods=2).mean())
    return s.groupby(by).transform(f) if by is not None else f(s)


def split_a(a, h1, h3):
    """Share unsubtyped A between H1 and H3 in proportion to the subtyped A (same row)."""
    sub = h1 + h3
    extra = (a - sub).clip(lower=0)
    w1 = (h1 / sub).where(sub > 0)
    return h1 + extra * w1.fillna(0), h3 + extra * (1 - w1).fillna(0), sub


# ------------------------------------------------------------------ sources
def flunet() -> pd.DataFrame:
    f = pd.read_csv(RAW / "flunet_VIW_FNT_2023on.csv", low_memory=False)
    for c in ["SPEC_PROCESSED_NB", "INF_ALL", "INF_A", "INF_B", "AH1N12009", "AH1", "AH3", "INF_NEGATIVE"]:
        f[c] = pd.to_numeric(f[c], errors="coerce")
    uk = f["COUNTRY_AREA_TERRITORY"].str.contains("United Kingdom", na=False)
    f.loc[uk, "COUNTRY_AREA_TERRITORY"] = NAME["United Kingdom"]
    f["week"] = pd.to_datetime(f["ISO_WEEKSTARTDATE"])
    f["proc"] = f["SPEC_PROCESSED_NB"].where(f["SPEC_PROCESSED_NB"] > 0, f["INF_ALL"].fillna(0) + f["INF_NEGATIVE"])
    keys = ["COUNTRY_AREA_TERRITORY", "week"]
    cols = ["INF_A", "INF_B", "AH1N12009", "AH1", "AH3"]
    g = (f[f["proc"] > 0].groupby(keys)[cols + ["proc"]].sum(min_count=1)
         .combine_first(f.groupby(keys)[cols].sum(min_count=1)).reset_index())
    g["denom_imputed"] = g["proc"].isna().astype(int)
    g["proc"] = g["proc"].fillna(g.groupby("COUNTRY_AREA_TERRITORY")["proc"].transform("median"))
    itz = (f.dropna(subset=["ITZ"]).drop_duplicates("COUNTRY_AREA_TERRITORY")
           .set_index("COUNTRY_AREA_TERRITORY")["ITZ"])
    g["region"] = g["COUNTRY_AREA_TERRITORY"].map(itz).map(ITZ).fillna("Other")
    g = g.sort_values(keys)
    h1, h3, sub = split_a(g["INF_A"].fillna(g["AH1N12009"].fillna(0) + g["AH1"].fillna(0) + g["AH3"].fillna(0)),
                          g["AH1N12009"].fillna(0) + g["AH1"].fillna(0), g["AH3"].fillna(0))
    # weeks with influenza A but no subtyping: use the country's subtype share of the previous 4 weeks
    share = (h1 / (h1 + h3)).where(sub > 0)
    share = share.groupby(g["COUNTRY_AREA_TERRITORY"]).transform(
        lambda s: s.fillna(s.shift(1).rolling(4, min_periods=1).mean()))
    a = g["INF_A"].fillna(0)
    g["pos_H1"] = np.where(sub > 0, h1, a * share)
    g["pos_H3"] = np.where(sub > 0, h3, a * (1 - share))
    g["pos_B"] = g["INF_B"]
    for k in SUB:
        g[f"p_{k}"] = trailing((g[f"pos_{k}"] / g["proc"]).clip(0, 1), g["COUNTRY_AREA_TERRITORY"])
    return g.rename(columns={"COUNTRY_AREA_TERRITORY": "country"})


ROWS = [("tested", r"检测数"), ("positive", r"阳性数\s*[（(]\s*%\s*[)）]"), ("A", r"A\s*型"),
        ("H1", r"A\s*[（(]\s*H1N1\s*[)）]\s*pdm\s*09"), ("H3", r"A\s*[（(]\s*H3N2\s*[)）]"),
        ("A_unsub", r"A\s*[（(]\s*unsubtyped\s*[)）]?"), ("B", r"B\s*型")]
NUM = r"(\d[\d,]*)(?:\s*[（(][^)）\s]*[)）]+)?"   # a count, optionally followed by "(xx.x%)"
WARN: list[str] = []                                  # inconsistencies found in the published tables


def cnic_table(pdf: Path) -> dict:
    t = "\n".join(p.extract_text() or "" for p in pypdf.PdfReader(pdf).pages)
    i = t.find("流感样病例监测实验室检测结果")        # the table caption (the text also mentions "表1")
    if i < 0:
        raise ValueError(f"{pdf.name}: Table 1 not found")
    tab = t[i:i + 2500]
    out = {}
    for key, pat in ROWS:
        m = re.search(pat + r"\s+" + NUM + r"\s+" + NUM + r"\s+" + NUM, tab)
        if not m and key == "A_unsub":                    # derive it and note it
            out["south_A_unsub"] = out["south_A"] - out["south_H1"] - out["south_H3"]
            out["total_A_unsub"] = out["total_A"] - out["total_H1"] - out["total_H3"]
            WARN.append(f"{pdf.stem}: A(unsubtyped) row not found, derived as A - H1 - H3")
            continue
        if not m:                                         # garbled percentages: keep the southern count
            m1 = re.search(pat + r"\s+(\d[\d,]*)", tab)
            if not m1:
                raise ValueError(f"{pdf.name}: row {key} not found")
            out[f"south_{key}"], out[f"total_{key}"] = int(m1.group(1).replace(",", "")), np.nan
            WARN.append(f"{pdf.stem} {key}: row garbled in the report, southern count {out[f'south_{key}']} kept")
            continue
        s, n, tot = (int(x.replace(",", "")) for x in m.groups())
        if s + n != tot:                                  # typos in the published tables: note, keep
            WARN.append(f"{pdf.stem} {key}: south {s} + north {n} != total {tot}")
        out[f"south_{key}"], out[f"total_{key}"] = s, tot
    if out["south_H1"] + out["south_H3"] + out["south_A_unsub"] != out["south_A"] or \
            out["south_A"] + out["south_B"] != out["south_positive"]:
        WARN.append(f"{pdf.stem}: southern subtypes do not add up to the type totals")
    m = re.search(r"(\d{4})\s*年\s*第\s*(\d+)\s*周\s*[（(]\s*(\d{4})\s*年\s*(\d+)\s*月\s*(\d+)\s*日"
                  r"\s*[－\-—~～至]+\s*(\d{4})\s*年\s*(\d+)\s*月\s*(\d+)\s*日", t)
    start = pd.Timestamp(int(m.group(3)), int(m.group(4)), int(m.group(5)))
    end = pd.Timestamp(int(m.group(6)), int(m.group(7)), int(m.group(8)))
    out["week"] = end - pd.Timedelta(days=6)            # week = Monday; the end date is used (see check)
    out["label"] = f"{m.group(1)}-W{int(m.group(2)):02d}"
    if start != out["week"]:                            # e.g. 2025-W52 prints 12月12日 for 12月22日
        WARN.append(f"{pdf.stem}: printed week start {start.date()} does not match the week end "
                    f"{end.date()}; week start taken as {out['week'].date()}")
    m = re.search(r"南方省份哨点医院报告的\s*ILI\s*%\s*为\s*([\d.]+)\s*%", t)
    out["south_ili_pct"] = float(m.group(1)) if m else np.nan
    if out["label"] != pdf.stem:
        raise ValueError(f"{pdf.name}: report is for {out['label']}")
    return out


def cnic() -> pd.DataFrame:
    rows = [cnic_table(f) for f in sorted((RAW / "cnic_weekly_reports").glob("*.pdf"))]
    c = pd.DataFrame(rows).sort_values("week").reset_index(drop=True)
    h1, h3, _ = split_a(c["south_A"], c["south_H1"], c["south_H3"])
    for k, pos in (("H1", h1), ("H3", h3), ("B", c["south_B"])):
        c[f"p_{k}"] = trailing(pos / c["south_tested"])
    return c


def taiwan() -> pd.DataFrame:
    r = (RAW / "taiwan_lars_influenza_2023w27_2026w20.html").read_text(encoding="utf-8")
    cats = json.loads(re.search(r'"xAxis_categories":(\[[^\]]*\])', r).group(1))
    ser = {n: json.loads(re.search(r'\{"name":"' + n + r'","type":"\w+","yAxis":\d+,"data":(\[[^\]]*\])', r).group(1))
           for n in ("A", "B")}
    # Taiwan CDC weeks run Sunday-Saturday (MMWR rule: week 1 contains 4 January)
    starts = []
    for yw in cats:
        y, w = int(yw[:4]), int(yw[4:])
        jan4 = pd.Timestamp(y, 1, 4)
        starts.append(jan4 - pd.Timedelta(days=(jan4.weekday() + 1) % 7) + pd.Timedelta(weeks=w - 1))
    t = pd.DataFrame({"week": [s + pd.Timedelta(days=1) for s in starts], "tw_A": ser["A"], "tw_B": ser["B"]})
    base = t[(t["week"] >= BASELINE[0]) & (t["week"] <= BASELINE[1])]
    for k in ("A", "B"):
        t[f"tw_rel_{k}"] = trailing(t[f"tw_{k}"]) / base[f"tw_{k}"].mean()
    return t


def arrivals() -> tuple[pd.DataFrame, pd.DataFrame]:
    im = pd.read_csv(RAW / "immd_daily_passenger_traffic.csv", encoding="utf-8-sig")
    im = im.loc[:, ~im.columns.str.startswith("Unnamed")]
    im["Date"] = pd.to_datetime(im["Date"], format="%d-%m-%Y")
    im = im[im["Arrival / Departure"] == "Arrival"]
    im["week"] = iso_monday(im["Date"])
    cp = im.groupby(["week", "Control Point"])["Total"].sum().unstack(fill_value=0)
    wk = pd.DataFrame({"air": cp["Airport"],
                       "land_sea": cp[[c for c in LAND_SEA if c in cp]].sum(axis=1),
                       "uncovered_sea": cp[[c for c in UNCOVERED if c in cp]].sum(axis=1),
                       "all": cp.drop(columns=[c for c in cp if "Cruise" in c]).sum(axis=1)})
    days = im.groupby("week")["Date"].nunique()
    wk = wk[days == 7]
    m = pd.read_excel(OAG, sheet_name="HK_Monthly_RoundTrip")
    m.columns = ["ts", "year", "month", "partner", "outbound", "inbound", "roundtrip"]
    m["month"] = pd.to_datetime(dict(year=m["year"], month=m["month"], day=1))
    top = m[m["year"] >= 2023].groupby("partner")["inbound"].sum().nlargest(40).index
    tot = m.groupby("month")["inbound"].sum()
    m = m[m["partner"].isin(top)].copy()
    m["share"] = m["inbound"] / m["month"].map(tot)
    m["country"] = m["partner"].map(lambda p: NAME.get(p, p))
    return wk, m[["month", "partner", "country", "share", "inbound"]]


# ------------------------------------------------------------------ build
def main() -> None:
    fl, cn, tw = flunet(), cnic(), taiwan()
    wk, oag = arrivals()
    weeks = pd.date_range(START, END, freq="W-MON")
    wk = wk.reindex(weeks)
    if wk["air"].isna().any():
        raise ValueError("immigration data missing for some weeks")

    # air arrivals by origin: immigration airport arrivals x OAG origin share (previous month)
    wkdf = pd.DataFrame({"week": weeks, "pax_month": (weeks.to_period("M") - 1).to_timestamp()})
    air = wkdf.merge(oag, left_on="pax_month", right_on="month")
    air["arrivals"] = air["share"] * air["week"].map(wk["air"])
    air = air.merge(fl[["country", "week", "region", "denom_imputed"] + [f"p_{k}" for k in SUB]],
                    on=["country", "week"], how="left")
    air["source"] = air["partner"]
    land = pd.DataFrame({"week": weeks, "source": LAND_LABEL, "region": LAND_LABEL,
                         "arrivals": wk["land_sea"].to_numpy(), "denom_imputed": 0})
    land = land.merge(cn[["week"] + [f"p_{k}" for k in SUB]], on="week", how="left")
    src = pd.concat([air[land.columns], land], ignore_index=True)

    rows = []
    for w in weeks:
        s = src[src["week"] == w]
        for k in SUB:
            ok = s[s[f"p_{k}"].notna()]
            contrib = ok["arrivals"] * ok[f"p_{k}"]
            reg = contrib.groupby(ok["region"]).sum().sort_values(ascending=False)
            reg = reg / reg.sum() if reg.sum() > 0 else reg
            rows.append(dict(week_start=w, subtype=k,
                             weighted_positivity=100 * contrib.sum() / ok["arrivals"].sum(),
                             pressure=contrib.sum(), coverage=ok["arrivals"].sum() / wk.loc[w, "all"],
                             land_share=(contrib[ok["region"] == LAND_LABEL].sum() / contrib.sum()
                                         if contrib.sum() > 0 else np.nan),
                             top_regions="; ".join(f"{r} {v * 100:.0f}%" for r, v in reg.head(3).items() if v > 0)))
    out = pd.DataFrame(rows)
    base = out[(out["week_start"] >= BASELINE[0]) & (out["week_start"] <= BASELINE[1])]
    out["pressure_rel"] = out["pressure"] / out["subtype"].map(base.groupby("subtype")["pressure"].mean())
    tw_share = air[air["partner"] == "Chinese Taipei"].set_index("week")["share"]
    out["taiwan_air_share"] = out["week_start"].map(tw_share)
    twk = tw.set_index("week")
    out["taiwan_rel"] = [twk["tw_rel_B" if k == "B" else "tw_rel_A"].get(w, np.nan)
                         for w, k in zip(out["week_start"], out["subtype"])]
    out["week_end"] = out["week_start"] + pd.Timedelta(days=6)
    out = out[["week_start", "week_end", "subtype", "weighted_positivity", "pressure", "pressure_rel",
               "coverage", "land_share", "top_regions", "taiwan_rel", "taiwan_air_share"]].sort_values(
        ["subtype", "week_start"])
    out.round(4).to_csv(HERE / "importation_weekly.csv", index=False)
    src.sort_values(["week", "arrivals"], ascending=[True, False]).round(5).to_csv(
        HERE / "importation_by_source.csv", index=False)

    # ---------------------------------------------------------------- checks
    print(f"CNIC reports parsed: {len(cn)} weeks ({cn['label'].iloc[0]} to {cn['label'].iloc[-1]}); "
          f"inconsistencies in the published tables: {len(WARN)}")
    for w in WARN:
        print("   ", w)
    # CNIC Table 1 counts ILI specimens only; FluNet China is a larger stream (about 2.4x the
    # specimens), so the counts differ by design: compare positivity instead
    chn = fl[fl["country"] == "China"].set_index("week")
    j = cn.set_index("week")[["total_tested", "total_positive"]].join(chn[["proc", "INF_A", "INF_B"]], how="inner")
    p_cn = j["total_positive"] / j["total_tested"]
    p_fl = (j["INF_A"].fillna(0) + j["INF_B"].fillna(0)) / j["proc"]
    print(f"CNIC national vs WHO FluNet China positivity ({len(j)} weeks): correlation "
          f"{p_cn.corr(p_fl):.3f}, median CNIC/FluNet ratio {(p_cn / p_fl).median():.2f} "
          f"(different specimen streams: CNIC Table 1 = ILI specimens)")
    mon = wk["air"].groupby(wk.index.to_period("M")).sum()
    oagm = oag.groupby(oag["month"].dt.to_period("M"))["inbound"].sum()
    r = (oagm / mon).dropna()
    print(f"OAG inbound (top 40 origins) / immigration airport arrivals: median {r.median():.2f} "
          f"(range {r.min():.2f}-{r.max():.2f}); OAG used only for the origin mix")
    print(f"Taiwan LARS weeks: {len(tw)}; weeks {weeks[0]:%Y-%m-%d} to {weeks[-1]:%Y-%m-%d}: {len(weeks)}; "
          f"median coverage {out['coverage'].median():.0%}; median land/sea share of pressure "
          f"{out['land_share'].median():.0%}")


if __name__ == "__main__":
    main()
