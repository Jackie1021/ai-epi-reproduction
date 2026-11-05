"""Build the weekly model input for the six pathogens from the Centre for Health Protection (CHP) data.

    python build_hk_weekly.py          (run inside AI_Epi/Data; prints its checks)

Output: HK_weekly_6pathogens.csv, same columns as HK_ILI_COVID_Sep.csv
(date, COVID, RSV, HFMD, H1, H3, B, HFMD_ori), one row per week from 2014-01-04; `date` is the
week-ending Saturday (CHP "To" date). Values are rounded to integers as before. COVID is empty
before 2023 (no series of the same construction; 2020-2022 are not evaluated). RSV is empty in the
weeks of 2020-2021 in which the laboratory reported children (<18 years) only.

Formulas (POP = 7,524,000, the population used in the Stan model):
  H1, H3, B  PMP ILI consultation rate / 1000 x share of respiratory specimens positive for the
             subtype x POP. Source: flux_data.xlsx (Flu Express). The share is computed as
             positives / specimens, specimens = all influenza positives / all-influenza positive
             share; it equals the published subtype positive% up to its rounding (checked below)
             and keeps precision in weeks with few positives.
  COVID      PMP consultation rate for COVID-19 / 1000 x SARS-CoV-2 positive detection rate x POP.
             Source: covidx_data.xlsx. This is the formula of the COVID column of
             HK_ILI_COVID_Sep.csv (reproduced exactly, checked below). The weeks before the PMP
             COVID rate starts (to 2023-02-04) are taken from HK_ILI_COVID_Sep.csv, whose dates are
             one week early, with the scale factor estimated on the overlap (1.000).
  RSV        PMP ILI consultation rate / 1000 x RSV positives / specimens tested x POP.
             Sources: PMP rate from flux_data.xlsx; RSV positives and specimens from CHP
             "Detection of pathogens from respiratory specimens - other respiratory viruses"
             (chp_raw/resp_other_viruses_<year>.html).
  HFMD       hospital admission episodes of HFMD without complication / 0.013 (the scaling of
             HFMD_2020-2025.xlsx, Proxy = count x 76.923); HFMD_ori = the count itself. Source: CHP
             "Weekly number of hospital admission episodes of HFMD reported by Hospital Authority"
             (chp_raw/hfmd_admissions_<year>.html). The Stan model treats it as a hospitalisation
             series (population x child ratio, hospitalisation-rate prior).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE / "chp_raw"
POP = 7_524_000
HFMD_FACTOR = 1 / 0.013
START = pd.Timestamp("2014-01-04")          # first week of the CHP Flu Express data (2014-W01)
OUT = HERE / "HK_weekly_6pathogens.csv"


def _body(path: Path) -> pd.DataFrame:
    d = pd.read_excel(path, header=None, skiprows=3)
    return d[pd.to_numeric(d[0], errors="coerce").notna()].reset_index(drop=True)


def flu() -> pd.DataFrame:
    d = _body(HERE / "flux_data.xlsx")
    out = pd.DataFrame({"date": pd.to_datetime(d[3]), "pmp_ili": pd.to_numeric(d[5], errors="coerce")})
    n = {k: pd.to_numeric(d[c], errors="coerce") for k, c in (("H1", 6), ("H3", 7), ("B", 8), ("all", 9))}
    p = {k: pd.to_numeric(d[c], errors="coerce") for k, c in (("H1", 10), ("H3", 11), ("B", 12), ("all", 13))}
    specimens = n["all"] / p["all"]
    for k in ("H1", "H3", "B"):
        share = np.where(n["all"] > 0, n[k] / specimens, 0.0)
        out[f"share_{k}"] = share
        out[f"pub_{k}"] = p[k]
    return out


def covid() -> pd.DataFrame:
    d = _body(HERE / "covidx_data.xlsx")
    return pd.DataFrame({"date": pd.to_datetime(d[3]),
                         "cov_positive_rate": pd.to_numeric(d[5], errors="coerce"),
                         "cov_pmp_rate": pd.to_numeric(d[12], errors="coerce")})


def _week_end(label: str, year: int, week: int) -> pd.Timestamp:
    """'29/12/2024 - 04/01' or '05/01 - 11/01' -> end date in `year` (next year if week 52/53 ends in Jan)"""
    end = label.split("-")[-1].strip()
    dd, mm = (int(x) for x in end.split("/")[:2])
    y = year + 1 if (week >= 50 and mm == 1) else year
    return pd.Timestamp(y, mm, dd)


def rsv() -> pd.DataFrame:
    rows = []
    for f in sorted(RAW.glob("resp_other_viruses_*.html")):
        year = int(re.search(r"(\d{4})", f.name).group(1))
        t = pd.read_html(f)[0]
        t.columns = [" | ".join(dict.fromkeys(c)) for c in t.columns]
        for _, r in t.iterrows():
            week = int(r["Week number"])
            if "Age" in t.columns and str(r["Age"]).strip() != "All ages":   # 2020-2021: children only
                rows.append((_week_end(str(r["Date"]), year, week), np.nan, np.nan, np.nan))
                continue
            rows.append((_week_end(str(r["Date"]), year, week), float(r["No. of specimen tested"]),
                         float(r["RSV | No."]), float(r["RSV | %"])))
    return pd.DataFrame(rows, columns=["date", "specimens", "rsv_n", "rsv_pct"])


def hfmd() -> pd.DataFrame:
    rows = []
    for f in sorted(RAW.glob("hfmd_admissions_*.html")):
        year = int(re.search(r"(\d{4})", f.name).group(1))
        t = pd.read_html(f)[1]
        for _, r in t.iterrows():
            label, without, with_ = r.iloc[0], r.iloc[1], r.iloc[2]
            if pd.isna(without):
                continue
            dd, mm = (int(x) for x in str(label).split("/")[:2])
            rows.append((pd.Timestamp(year, mm, dd), float(without), float(with_)))
    return pd.DataFrame(rows, columns=["date", "hfmd_without", "hfmd_with"])


def old_csv() -> pd.DataFrame:
    o = pd.read_csv(HERE / "HK_ILI_COVID_Sep.csv")
    o["date"] = pd.to_datetime(o["date"], format="%Y/%m/%d") + pd.Timedelta(days=7)   # its dates are one week early
    for c in ["COVID", "RSV", "HFMD", "H1", "H3", "B", "HFMD_ori"]:
        o[c] = pd.to_numeric(o[c].astype(str).str.strip(), errors="coerce")
    return o


def build() -> pd.DataFrame:
    f, c, r, h, o = flu(), covid(), rsv(), hfmd(), old_csv()
    end = min(f["date"].max(), c["date"].max(), r["date"].max(), h["date"].max())
    weeks = pd.DataFrame({"date": pd.date_range(START, end, freq="7D")})
    m = weeks.merge(f, on="date", how="left").merge(c, on="date", how="left") \
             .merge(r, on="date", how="left").merge(h, on="date", how="left")
    out = pd.DataFrame({"date": m["date"]})
    for k in ("H1", "H3", "B"):
        out[k] = m["pmp_ili"] / 1000 * m[f"share_{k}"] * POP
    out["RSV"] = m["pmp_ili"] / 1000 * m["rsv_n"] / m["specimens"] * POP
    out["COVID"] = m["cov_pmp_rate"] / 1000 * m["cov_positive_rate"] * POP
    # COVID weeks before the PMP COVID rate: HK_ILI_COVID_Sep.csv, rescaled on the overlap
    ov = out.merge(o[["date", "COVID"]], on="date", suffixes=("", "_old")).dropna()
    ov = ov[ov["COVID"] > 0]
    scale = float(np.median(ov["COVID"] / ov["COVID_old"]))
    fill = out["COVID"].isna() & out["date"].isin(o["date"])
    out.loc[fill, "COVID"] = out.loc[fill, "date"].map(o.set_index("date")["COVID"]) * scale
    out["HFMD_ori"] = m["hfmd_without"]
    out["HFMD"] = m["hfmd_without"] * HFMD_FACTOR

    # ---------------------------------------------------------------- checks
    print(f"weeks: {len(out)} ({out['date'].min().date()} to {out['date'].max().date()}), consecutive: "
          f"{(out['date'].diff().dropna() == pd.Timedelta(days=7)).all()}")
    print("missing values:", out.drop(columns="date").isna().sum().to_dict(),
          "| COVID from", out.loc[out["COVID"].notna(), "date"].min().date())
    for k in ("H1", "H3", "B"):
        dev = (m[f"share_{k}"] - m[f"pub_{k}"]).abs().max()
        print(f"{k}: share from counts vs published positive% - max difference {dev:.5f} (published rounding 0.00005)")
    rd = (m["rsv_n"] / m["specimens"] * 100 - m["rsv_pct"]).abs()
    print(f"RSV: positives / specimens vs published % - above rounding (0.005) in {int((rd > 0.0051).sum())} "
          f"week(s): {', '.join(str(d.date()) for d in m.loc[rd > 0.0051, 'date'])} (counts used)")
    print(f"COVID: {int(fill.sum())} early weeks from HK_ILI_COVID_Sep.csv "
          f"({', '.join(str(d.date()) for d in out.loc[fill, 'date'])}), scale factor {scale:.4f} "
          f"from {len(ov)} overlapping weeks")
    comp = out.merge(o, on="date", suffixes=("", "_old"))
    print(f"\ncomparison with HK_ILI_COVID_Sep.csv (its dates + 7 days), {len(comp)} common weeks:")
    for k in ("H1", "H3", "B", "COVID", "RSV", "HFMD", "HFMD_ori"):
        a, b = comp[k], comp[f"{k}_old"]
        ok = a.notna() & b.notna()
        exact = ((a - b).abs() <= 1)[ok].mean()
        ratio = np.median((a / b)[ok & (b > 0)])
        print(f"  {k:8s} correlation {np.corrcoef(a[ok], b[ok])[0, 1]:.4f}  median new/old {ratio:.3f}  "
              f"identical (within 1) {exact:.0%}")
    return out


if __name__ == "__main__":
    out = build()
    w = out.copy()
    w["date"] = w["date"].map(lambda d: f"{d.year}/{d.month}/{d.day}")      # same format as before
    for k in ("COVID", "RSV", "HFMD", "H1", "H3", "B", "HFMD_ori"):
        w[k] = w[k].round().astype("Int64")
    w[["date", "COVID", "RSV", "HFMD", "H1", "H3", "B", "HFMD_ori"]].to_csv(OUT, index=False)
    print(f"\nwrote {OUT} ({len(w)} weeks)")
