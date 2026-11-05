# Context data for the LLM designs with added information

| design (arm) | information added to the LLM R0 prompt |
|---|---|
| R0 + mechanistic model (r0_semi) | none beyond the model estimates and monthly air passengers |
| + context (r0_ctx) | school calendar and public holidays, weekly weather |
| + context + importation (r0_imp) | the above plus influenza importation pressure (air, land and sea) |

Each design adds one block, so comparing neighbouring designs measures the value of that block.

## hk_calendar.csv (in results/: hk_school_calendar_and_public_holidays.csv)
School holidays and Hong Kong general (public) holidays:
- January 2015 to January 2020, for the historical study. `add_calendar_2015_2020.py` holds these
  rows and their sources.
- June 2024 to August 2026, for the main study.

For 2015-2020 the sources are:
- the EDB example school calendars 2015/16 to 2019/20 (the 2015/16 and 2016/17 calendars in their
  Chinese version);
- the EDB general-holiday lists 2014/15 to 2019/20.

The 2014/15 example calendar is not archived, so January-August 2015 has general holidays only. The
first school day of each year is the first weekday of September.

| column | meaning |
|---|---|
| start, end | first and last day (inclusive) |
| event | full name (used in prompts that name the place) |
| event_semi | name without place-identifying words (used in the semi-blinded prompts) |
| type | school_holiday, school_term (first school day after the summer vacation), general_holiday |
| source | document the dates come from |

Sources: Education Bureau (EDB) "Example school calendar" for the 2023/24, 2024/25 and 2025/26
school years (primary and secondary schools; summer, Christmas/New Year, Lunar New Year and Easter
holidays) and EDB "General Holidays" lists, retrieved from the EDB website archive
(web.archive.org; the 2025/26 example calendar and general-holiday list are kept in `raw/edb/`).
Individual schools set their own calendars within EDB guidelines, so dates can
differ by a few days between schools; kindergarten calendars are not included.

In the prompt, the events from 13 days before the forecast origin to the end of the 4-week
forecast window are listed with days relative to the origin (no dates).

## hk_weather_weekly.csv / hk_weather_daily.csv
Hong Kong Observatory headquarters station, 2015-01-01 to 2026-09-24, built by
`build_weather.py` from the HKO Daily Extract (raw files in `raw/`).

| column | meaning |
|---|---|
| week_end | Saturday ending the week (same weeks as the surveillance data) |
| temp | mean daily mean temperature, deg C |
| rh | mean relative humidity, % |
| ah | mean absolute humidity, g/m3, from temperature and dew point |
| rain | total rainfall, mm |
| *_norm | HKO official 1991-2020 normal for the same days (monthly normals interpolated to days) |
| n_days | days in the week (only complete weeks are kept) |

Nine days missing from the monthly extract files (2019-12-31, 2022-01-31, 2024-06-28 to 30,
2024-08-30 to 31, 2024-09-30, 2026-07-31) were filled from the HKO open data API and are flagged `imputed = 1`
in the daily file (mean temperature from the daily-mean series; rainfall and relative humidity
from the daily weather report, mean humidity = (max + min) / 2).

In the prompt: the 4 weeks up to the forecast origin (observed, with the difference from normal)
and the normals for the 4 forecast weeks. Observed weather after the origin is never shown.

## Influenza importation pressure
Built in `importation/` (script, sources, outputs and methods in `importation/README.md`): all
arrivals by air, land and sea (Immigration Department), weighted by the proportion of specimens
positive for each influenza subtype at the travellers' origins (WHO FluNet for air origins, the
Chinese National Influenza Center weekly reports for the neighbouring mainland; Taiwan CDC given
separately because it reports no denominators).
