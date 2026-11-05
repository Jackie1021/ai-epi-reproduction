"""数据读取：ILI 观测、OAG 航空客流、可选校历、每个预测起点的 Stan 导出包"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config


def load_observed(cfg: Config) -> pd.DataFrame:
    """Wide table: index = week-ending date, columns = strains (config order)."""
    df = pd.read_csv(cfg.data_path("ili_csv"))
    df["date"] = pd.to_datetime(df["date"], format="%Y/%m/%d")
    if cfg.get("observed_from"):                 # the study period decides how far back the data go
        df = df[df["date"] >= pd.Timestamp(str(cfg.get("observed_from")))]
    for s in cfg.strains:
        df[s] = pd.to_numeric(df[s].astype(str).str.strip(), errors="coerce")
    return df.set_index("date").sort_index()[cfg.strains]


def load_oag_monthly(cfg: Config) -> pd.DataFrame | None:
    """Monthly inbound/outbound air passengers by partner country (OAG est. pax)."""
    path = cfg.data_path("oag_xlsx")
    if path is None or not path.exists():
        return None
    m = pd.read_excel(path, sheet_name="HK_Monthly_RoundTrip")
    m.columns = ["ts", "year", "month", "partner", "outbound", "inbound", "roundtrip"]
    m["month_start"] = pd.to_datetime(dict(year=m["year"], month=m["month"], day=1))
    return m


def load_calendar(cfg: Config) -> pd.DataFrame | None:
    """Optional CSV with columns start,end,event (school terms, holidays, ...)."""
    path = cfg.data_path("calendar_csv")
    if path is None or not path.exists():
        return None
    cal = pd.read_csv(path, parse_dates=["start", "end"])
    return cal.sort_values("start").reset_index(drop=True)


def load_weather(cfg: Config) -> pd.DataFrame | None:
    """Optional weekly weather table (week_end, temp, rh, ah, rain and *_norm columns)."""
    path = cfg.data_path("weather_csv")
    if path is None or not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["week_end"])


def load_importation(cfg: Config) -> pd.DataFrame | None:
    """Optional weekly influenza importation pressure (build_importation.py)."""
    path = cfg.data_path("importation_csv")
    if path is None or not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["week_start", "week_end"])


@dataclass
class OriginBundle:
    """One forecast origin exported by export_origins.R."""

    dir: Path

    @cached_property
    def meta(self) -> dict:
        return json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))

    @property
    def origin(self) -> pd.Timestamp:
        return pd.Timestamp(self.meta["origin"])

    @property
    def smoke(self) -> bool:
        return bool(self.meta.get("smoke", False))

    @cached_property
    def draws(self) -> pd.DataFrame:
        return pd.read_parquet(self.dir / "draws.parquet")

    @cached_property
    def latent(self) -> pd.DataFrame:
        d = pd.read_parquet(self.dir / "latent_weekly.parquet")
        d["date"] = pd.to_datetime(d["date"])
        return d

    @cached_property
    def stan_fc(self) -> pd.DataFrame:
        d = pd.read_parquet(self.dir / "stan_fc_quantiles.parquet")
        d["origin"] = pd.to_datetime(d["origin"])
        d["target_date"] = pd.to_datetime(d["target_date"])
        return d

    @cached_property
    def stan_mu_fc(self) -> pd.DataFrame:
        return pd.read_parquet(self.dir / "stan_mu_fc.parquet")

    # ---- arrays for the simulator; strain order is always the caller's ----
    def state0(self, strains: list[str]) -> np.ndarray:
        cols = [f"{c}_{s}" for s in strains for c in "SEIR"]
        return self.draws[cols].to_numpy(float).reshape(len(self.draws), len(strains), 4)

    def per_strain(self, prefix: str, strains: list[str]) -> np.ndarray:
        return self.draws[[f"{prefix}_{s}" for s in strains]].to_numpy(float)

    def cross(self, strains: list[str]) -> np.ndarray:
        """(D, N, N); [d, j, k] = protection against k after infection with j."""
        cols = [f"C_{a}_{b}" for a in strains for b in strains]
        return self.draws[cols].to_numpy(float).reshape(len(self.draws), len(strains), len(strains))

    def constant(self, name: str, strains: list[str]) -> np.ndarray:
        order = self.meta["strain_names"]
        vals = self.meta["constants"][name]
        return np.array([vals[order.index(s)] for s in strains], dtype=float)


def list_bundles(cfg: Config, smoke: bool = False) -> list[OriginBundle]:
    root = cfg.origins_dir(smoke)
    if not root.exists():
        return []
    # REFIT_PENDING: fit failed the convergence criteria and waits for its pre-specified refit (amendment 8)
    dirs = sorted(p for p in root.iterdir() if (p / "DONE").exists() and (p / "meta.json").exists()
                  and not (p / "REFIT_PENDING").exists())
    return [OriginBundle(p) for p in dirs]


def select_bundles(cfg: Config, origins: str | None, smoke: bool = False,
                   max_origins: int | None = None) -> list[OriginBundle]:
    """origins: None/'all' | comma-separated dates | 'first' | 'last'."""
    bundles = list_bundles(cfg, smoke)
    if origins and origins not in ("all",):
        if origins == "first":
            bundles = bundles[:1]
        elif origins == "last":
            bundles = bundles[-1:]
        else:
            wanted = {pd.Timestamp(x.strip()) for x in origins.split(",")}
            bundles = [b for b in bundles if b.origin in wanted]
            missing = wanted - {b.origin for b in bundles}
            if missing:
                raise SystemExit(f"没有这些起点的 Stan 导出（先跑 export_origins.R）: "
                                 f"{sorted(d.date().isoformat() for d in missing)}")
    if max_origins:
        bundles = bundles[:max_origins]
    return bundles
