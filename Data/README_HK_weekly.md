# Weekly model input for the six pathogens (`HK_weekly_6pathogens.csv`)

Built by `build_hk_weekly.py` (run it in this folder; it prints its checks). Same columns and date
format as `HK_ILI_COVID_Sep.csv`: `date, COVID, RSV, HFMD, H1, H3, B, HFMD_ori`. One row per week
from 2014-01-04 to 2026-09-19 (664 weeks); `date` is the week-ending Saturday (the CHP "To" date). The
latest weeks are provisional in the CHP sources and may be revised.

**Which years each analysis uses.** The main study (24 origins, 2024-07-06 to 2026-04-11) reads only
the weeks from 2023-01-07, as frozen in its protocol (`observed_from` in `bench/config.yaml`,
`data_from` in `ai_epi_core.R`). The weeks 2014-2022 were added for the historical study
(protocol amendment 8: origins 2015-2019; 2020-2022 are not evaluated because of the COVID-19
interventions). Rebuilding with the longer range leaves every week from 2023-01-07 unchanged
(checked against the values stored in the main study's prompts: 248 weeks x 6 series identical).

**Gaps by design.**
- COVID is empty before 2023-01-07 (no series of the same construction).
- RSV is empty in the 126 weeks of 2020-2022 in which the laboratory table reported children
  (<18 years) only (2020-02-22 to 2022-07-16); the rows labelled "All ages" are used.

| column | formula (POP = 7,524,000) | source |
|---|---|---|
| H1, H3, B | PMP ILI consultation rate / 1000 x share of respiratory specimens positive for the subtype x POP | `flux_data.xlsx` (CHP Flu Express) |
| COVID | PMP consultation rate for COVID-19 / 1000 x SARS-CoV-2 positive detection rate x POP; weeks to 2023-02-04 from `HK_ILI_COVID_Sep.csv` (scale factor 1.000 on 142 overlapping weeks) | `covidx_data.xlsx` (CHP COVID-19 & Flu Express) |
| RSV | PMP ILI consultation rate / 1000 x RSV positives / specimens tested x POP | PMP rate: `flux_data.xlsx`; RSV: CHP "Detection of pathogens from respiratory specimens", other respiratory viruses (`chp_raw/resp_other_viruses_<year>.html`) |
| HFMD | hospital admission episodes of HFMD without complication / 0.013 (as in `HFMD_2020-2025.xlsx`) | CHP "Weekly number of hospital admission episodes of HFMD reported by Hospital Authority" (`chp_raw/hfmd_admissions_<year>.html`) |
| HFMD_ori | the admission count itself | as above |

One RSV week disagrees with its own published percentage: 2018-07-28, 197 positives of 4,801
specimens (4.10%) against 4.83% published; the count fits the neighbouring weeks (207, 222, 213) and
is used. The CHP series were also checked against `Sentinel_ILI_2015-2025.xlsx`: the PMP ILI rate is
identical in 569 of 571 weeks (2015-2025) and the RSV counts in all 184 weeks it covers (2020-2025).

The influenza share is computed from counts (positives / specimens, specimens = all influenza
positives / all-influenza positive share); it equals the published positive% up to rounding and
keeps precision in weeks with few positives. HFMD is a hospitalisation series; the Stan model
handles it with its own population (population x child ratio) and hospitalisation-rate prior.

## Differences from `HK_ILI_COVID_Sep.csv`
- **Dates:** the previous file is dated one week early (its date + 7 days = the CHP week end).
- **Influenza:** the previous file used the CHP laboratory table "Detection of pathogens from respiratory
  specimens" (Public Health Laboratory Services Branch, positives / specimens tested); this file uses
  Flu Express, which reports more positives. Correlation 0.98-0.998; median new/old 1.07 (H1),
  1.13 (H3), 1.34 (B).
- **COVID, RSV, HFMD:** same construction, and the values agree:
  - COVID: 99% of weeks identical;
  - RSV: within 4.1% (revisions);
  - HFMD: 1 revised week.
- **Coverage:** the series now runs to 2026-09-19 (previously to 2025-11-08).

`HK_weekly_6pathogens_check.png` overlays the new and the previous series.
