# Influenza importation pressure into Hong Kong

Context for the LLM design "R0 + mechanistic model + context + importation" (arm `r0_imp`).
Built by `build_importation.py` from official sources only; the script prints its data checks.

## Files
| file | content |
|---|---|
| `importation_weekly.csv` | one row per ISO week (2023-W27 to 2026-W18) and influenza subtype; used in the prompts |
| `importation_by_source.csv` | arrivals, subtype positivity and contribution per source and week (for checking) |
| `raw/` | the downloaded source files, unchanged |

## Method
Expected imported infections = number of travellers from each source x infection prevalence at
the source (Wu, Leung & Leung, Lancet 2020;395:689-97). With surveillance data this gives a
relative index: travel flows weighted by each source's epidemic activity, as in flight-weighted
importation-risk indices (Gilbert et al., Lancet 2020;395:871-7). Air passenger flux is the
dominant predictor of the global spread of seasonal influenza (Lemey et al., PLoS Pathog
2014;10:e1003932). Subtype-specific activity is the proportion of specimens positive for the
subtype (as in Goldstein et al., PLoS Med 2011;8:e1001051).

For influenza subtype k (A(H1N1)pdm09, A(H3N2), B) and week w:

- **Arrivals.** All arrivals from the Immigration Department (daily, by control point). Air
  arrivals are split across origin countries by each country's share of OAG inbound air passengers
  in the previous month. OAG also counts transfer passengers who never enter Hong Kong (its totals
  are about 1.9 times the immigration arrivals), so it is used only for the origin mix. Land and
  sea arrivals at the mainland control points (Lo Wu, Lok Ma Chau, Shenzhen Bay, the Express Rail
  Link, Heung Yuen Wai, the Hong Kong-Zhuhai-Macao Bridge, the China Ferry Terminal and others)
  are one source, "neighbouring mainland". Land and sea arrivals are about 83% of all arrivals.
- **Positivity at the source.** Air origins: WHO FluNet, specimens positive for k / specimens
  processed; unsubtyped influenza A is shared between H1 and H3 in proportion to the subtyped A.
  Neighbouring mainland: southern-provinces laboratory results in Table 1 of the Chinese National
  Influenza Center weekly reports. 3-week trailing mean (past weeks only).
- **Indices.** `weighted_positivity` = arrivals-weighted positivity over covered arrivals (%);
  `pressure` = sum of arrivals x positivity (positive-equivalent arrivals per week);
  `pressure_rel` = pressure / its mean from July 2023 to June 2024; `coverage` = share of all
  arrivals whose origin has data; `land_share` = share of the pressure from the neighbouring
  mainland; `top_regions` = main contributors (WHO influenza transmission zones).
- **Taiwan** reports only positive counts (no specimens tested, no subtypes), so it is not merged
  into the index. `taiwan_rel` = its laboratory-confirmed influenza A (rows H1 and H3) or B
  (row B) positives relative to their mean over the same reference year; `taiwan_air_share` = its
  share of OAG inbound air passengers.

In the prompt: for each subtype the latest 4 weeks already published two weeks before the forecast
origin, the relative pressure and the main origin regions, then Taiwan's relative activity as a
separate sentence; countries are not named (semi-blinded design).

## Sources (in `raw/`)
| file | source |
|---|---|
| `flunet_VIW_FNT_2023on.csv` | WHO FluNet public API (xmart-api-public.who.int/FLUMART/VIW_FNT), ISO year 2023 on |
| `cnic_weekly_reports/*.pdf` | Chinese National Influenza Center, 中国流感监测周报, 2023-W27 onwards (ivdc.chinacdc.cn/cnic/zyzx/lgzb/; page of each report in `cnic_weekly_report_urls.json`; large, can be deleted after the build and downloaded again) |
| `immd_daily_passenger_traffic.csv` | Hong Kong Immigration Department, Statistics on Daily Passenger Traffic (open data) |
| `taiwan_lars_influenza_2023w27_2026w20.html` | Taiwan CDC NIDSS, Trend of Influenza virus (LARS), weekly A / B positives (identical to the earlier download for the 123 overlapping weeks) |
| `../../OAG_HongKong_2022-2026Mar.xlsx` | OAG monthly inbound air passengers by origin (provided; 2022-01 to 2026-03, see its README sheet) |

## Checks (printed by the build; results of the current build)
- CNIC reports: all weekly Table 1s from 2023-W27 parsed (148 weeks to 2026-W18 used). 17 small inconsistencies
  in the published tables are listed by the build (e.g. 2024-W41: the H1N1 total is printed as 220
  instead of 287 + 38 = 325; 2024-W38: 256 A(H1N1)pdm09 in the table vs 265 in the text); they
  change the southern positivity by at most about 0.1 percentage point and are kept as published.
  2025-W52 prints its week as 12月12日-12月28日; the week start is taken from the end date (22 December).
  2025-W47 has garbled percentages in two rows; the southern counts are kept (continuous with W46 and W48).
- CNIC national positivity vs WHO FluNet China over the overlapping weeks: correlation 0.993, median
  ratio 1.04. The counts differ by design (CNIC Table 1 covers ILI specimens only; FluNet China is a
  larger stream, about 2.4 times as many specimens).
- OAG inbound passengers / immigration airport arrivals: median 1.91 (range 1.51-2.24), i.e. OAG
  includes transfer passengers; OAG is therefore used only for the origin mix.
- Coverage: median 96% of all arrivals (air, land and sea) have source data; the neighbouring
  mainland (land and sea) contributes a median 68% of the pressure.

## Limitations
- Macau (Macau Ferry Terminal, about 2% of arrivals) has no data and is not covered; the
  Hong Kong-Zhuhai-Macao Bridge is counted with the neighbouring mainland although part of its
  traffic is from Macau.
- Japan does not report specimens processed in 59% of weeks; its median over the period is used
  (`denom_imputed` in `importation_by_source.csv`).
- Air arrivals are split by the OAG origin mix, which includes transfer passengers; origins with
  many transfers are over-weighted.
- WHO FluNet data are the current release and may contain revisions made after the forecast dates.
- OAG estimates end in March 2026; the series therefore ends in 2026-W18 (air origin mix of the previous month).
- RSV, SARS-CoV-2 and enteroviruses are not included.
