"""Build the weekly weather table used as LLM context (arm r0_ctx).

Source: Hong Kong Observatory "Daily Extract" for the HKO headquarters station
  https://www.hko.gov.hk/cis/dailyExtract/dailyExtract_YYYYMM.xml   (JSON, available from 2015-01)
  columns per day: mean pressure, max / mean / min temperature, mean dew point,
  mean relative humidity, mean cloud amount, total rainfall; each monthly file also has a
  "Normal" row with HKO's official monthly normals. Files up to 2021 carry the 1981-2010 normals
  (annual rainfall 2398.5 mm), files from 2022 the 1991-2020 normals (2431.2 mm); only the latter
  are used here (checked below).

Output (next to this script):
  hk_weather_daily.csv   daily observations + daily normals
  hk_weather_weekly.csv  weeks ending Saturday (same as the surveillance data): means of
                         temperature, relative humidity and absolute humidity, total rainfall,
                         and the matching normals
Absolute humidity (g/m3) = 216.7 * e / (T + 273.15), e = 6.112 * exp(17.67 Td / (Td + 243.5)) hPa.
The raw monthly files are kept in raw/hko_daily_extract/.

A few monthly files stop 1-3 days before the month end. Those days (from 2019-09 on) are filled
from the HKO open data API and flagged (imputed = 1): daily mean temperature from CLMTEMP
(raw/hko_CLMTEMP_daily_mean.csv), rainfall and max/min relative humidity from the daily
"Weather and Radiation Level Report" (RYES, raw/hko_ryes/); mean relative humidity = (max + min) / 2
and the dew point is derived from temperature and relative humidity.

Usage: python build_weather.py [--download]
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw" / "hko_daily_extract"
URL = "https://www.hko.gov.hk/cis/dailyExtract/dailyExtract_{ym}.xml"
FIRST, LAST = "2015-01", "2026-09"
COLS = ["pressure", "temp_max", "temp", "temp_min", "dewpoint", "rh", "cloud", "rain"]


def download() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for p in pd.period_range(FIRST, LAST, freq="M"):
        f = RAW / f"{p.strftime('%Y%m')}.json"
        if f.exists() and f.read_bytes()[:1] == b"{":
            continue
        req = urllib.request.Request(URL.format(ym=p.strftime("%Y%m")), headers={"User-Agent": "Mozilla/5.0"})
        f.write_bytes(urllib.request.urlopen(req, timeout=30).read())
        time.sleep(0.5)


RYES = "https://data.weather.gov.hk/weatherAPI/opendata/opendata.php?dataType=RYES&date={d}&lang=en&station=HKO"


def fill_missing(d: pd.DataFrame) -> pd.DataFrame:
    full = pd.date_range(d.index.min(), d.index.max())
    miss = [t for t in full.difference(d.index) if t >= pd.Timestamp("2019-09-10")]
    d = d.reindex(full)
    d["imputed"] = 0
    if not miss:
        return d
    clm = pd.read_csv(HERE / "raw" / "hko_CLMTEMP_daily_mean.csv", skiprows=2, header=0,
                      names=["y", "m", "day", "v", "flag"], on_bad_lines="skip")
    clm = clm[pd.to_numeric(clm["y"], errors="coerce").notna()]
    clm.index = pd.to_datetime(clm["y"].astype(int).astype(str) + "-" + clm["m"].astype(int).astype(str)
                               + "-" + clm["day"].astype(int).astype(str), errors="coerce")   # 表里有 2 月 30 日之类的空行
    clm = clm[clm.index.notna()]
    (HERE / "raw" / "hko_ryes").mkdir(parents=True, exist_ok=True)
    for t in miss:
        f = HERE / "raw" / "hko_ryes" / f"{t:%Y%m%d}.json"
        if not f.exists():
            f.write_bytes(urllib.request.urlopen(RYES.format(d=f"{t:%Y%m%d}"), timeout=30).read())
            time.sleep(0.5)
        r = json.loads(f.read_text())
        rh = (float(r["HKOReadingsMaxRH"]) + float(r["HKOReadingsMinRH"])) / 2
        temp = float(clm.loc[t, "v"])
        es = 6.112 * np.exp(17.67 * temp / (temp + 243.5))
        g = np.log(rh / 100 * es / 6.112)
        d.loc[t, ["temp", "temp_max", "temp_min", "rh", "rain", "dewpoint", "imputed"]] = [
            temp, float(r["HKOReadingsMaxTemp"]), float(r["HKOReadingsMinTemp"]), rh,
            value(r["HKOReadingsRainfall"]), 243.5 * g / (17.67 - g), 1]
    return d


def value(x: str) -> float:
    x = x.strip()
    if x == "Trace":
        return 0.05
    try:
        return float(x)
    except ValueError:
        return np.nan


def abs_humidity(t, td):
    e = 6.112 * np.exp(17.67 * td / (td + 243.5))
    return 216.7 * e / (t + 273.15)


def main() -> None:
    if "--download" in sys.argv:
        download()
    days, normals = [], {}
    for f in sorted(RAW.glob("*.json")):
        year, month = int(f.stem[:4]), int(f.stem[4:])
        for block in json.loads(f.read_text())["stn"]["data"]:
            for row in block["dayData"]:
                if row[0].isdigit():
                    days.append([pd.Timestamp(year, month, int(row[0]))] + [value(v) for v in row[1:9]])
                elif row[0] == "Normal" and year >= 2022:          # 1991-2020 normals
                    normals[month] = [value(v) for v in row[1:9]]
    d = pd.DataFrame(days, columns=["date"] + COLS).set_index("date").sort_index()
    d = fill_missing(d)
    d.index.name = "date"
    d["ah"] = abs_humidity(d["temp"], d["dewpoint"])

    # daily normals: monthly normals placed mid-month, linearly interpolated (periodic)
    nm = pd.DataFrame.from_dict(normals, orient="index", columns=COLS).sort_index()
    assert len(nm) == 12 and abs(nm["rain"].sum() - 2431.2) < 1, "expected HKO 1991-2020 normals"
    nm["rain"] = nm["rain"] / pd.Series({m: pd.Period(f"2001-{m:02d}").days_in_month for m in nm.index})
    doy = d.index.dayofyear.to_numpy()
    mid = np.array([pd.Timestamp(2001, m, 15).dayofyear for m in nm.index], float)
    xp = np.concatenate([mid - 365, mid, mid + 365])
    for col in ["temp", "dewpoint", "rh", "rain"]:
        fp = np.tile(nm[col].to_numpy(float), 3)
        d[f"{col}_norm"] = np.interp(doy, xp, fp)
    d["ah_norm"] = abs_humidity(d["temp_norm"], d["dewpoint_norm"])
    d.round(3).reset_index().to_csv(HERE / "hk_weather_daily.csv", index=False)

    # weeks ending Saturday
    d["week_end"] = d.index + pd.to_timedelta((5 - d.index.weekday) % 7, unit="D")
    w = d.groupby("week_end").agg(
        n_days=("temp", "count"), temp=("temp", "mean"), rh=("rh", "mean"), ah=("ah", "mean"),
        rain=("rain", "sum"), temp_norm=("temp_norm", "mean"), rh_norm=("rh_norm", "mean"),
        ah_norm=("ah_norm", "mean"), rain_norm=("rain_norm", "sum"))
    w = w[w["n_days"] == 7]
    n_imp = int(d["imputed"].sum())
    w.round(3).reset_index().to_csv(HERE / "hk_weather_weekly.csv", index=False)
    print(f"daily {d.index.min():%Y-%m-%d} to {d.index.max():%Y-%m-%d} ({len(d)} days); "
          f"weekly {len(w)} complete weeks; normals for {len(nm)} months; {n_imp} days filled")


if __name__ == "__main__":
    main()
