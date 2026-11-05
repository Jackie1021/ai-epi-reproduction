"""School holidays and general holidays for the historical study (protocol amendment 8), added to
hk_calendar.csv. Dates transcribed from the EDB documents in raw/edb/ (web.archive.org copies):
  Example school calendars 2015/16, 2016/17 (Chinese version), 2017/18, 2018/19, 2019/20
  General holiday lists 2014/15 - 2019/20 (2017/18 Chinese version)
The 2014/15 example school calendar is not archived, so January-August 2015 has general holidays
only (no school Lunar New Year, Easter or summer holidays). The first school day of each year is the
first weekday of September (the EDB school year starts on 1 September).

    python add_calendar_2015_2020.py        (idempotent: replaces the rows before 2024-06-01)
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CAL = HERE / "hk_calendar.csv"
GH = "general_holiday"
SH = "school_holiday"
ST = "school_term"

def gh(start, end, event, semi, year):
    return (start, end, f"General holiday{'s' if start != end else ''}: {event}", semi, GH,
            f"EDB General Holidays {year}")

def school(kind, start, end, year, lang=""):
    names = {
        "xmas": ("School Christmas and New Year holidays", "School Christmas and New Year holidays (schools closed)"),
        "lny": ("School Lunar New Year holidays", "School Lunar New Year holidays (schools closed)"),
        "easter": ("School Easter holidays", "School Easter holidays (schools closed)"),
        "summer": ("School summer vacation (primary and secondary schools)",
                   "School summer vacation (primary and secondary schools closed)"),
    }
    e, s = names[kind]
    return (start, end, e, s, SH, f"EDB example school calendar {year}{lang}")

def reopen(day, year):
    return (day, day, f"Schools reopen: first day of the {year} school year",
            "Schools reopen after the summer vacation", ST, f"EDB school year {year} (first weekday of September)")

ROWS = [
    # ---- 2014/15: general holidays only (example school calendar not archived)
    gh("2015-02-19", "2015-02-21", "the first to third day of Lunar New Year", "Public holidays (Lunar New Year)", "2014/15"),
    gh("2015-04-03", "2015-04-07", "Good Friday, the day following Good Friday, the day following Ching Ming Festival and the day following Easter Monday",
       "Public holidays (Easter and Ching Ming Festival)", "2014/15"),
    gh("2015-05-01", "2015-05-01", "Labour Day", "Public holiday (Labour Day)", "2014/15"),
    gh("2015-05-25", "2015-05-25", "the Birthday of the Buddha", "Public holiday (Buddha's Birthday)", "2014/15"),
    gh("2015-06-20", "2015-06-20", "Tuen Ng Festival", "Public holiday (Dragon Boat Festival)", "2014/15"),
    gh("2015-07-01", "2015-07-01", "HKSAR Establishment Day", "Public holiday", "2014/15"),
    # ---- 2015/16
    reopen("2015-09-01", "2015/16"),
    gh("2015-09-03", "2015-09-03", "the 70th anniversary day of the victory of the Chinese people's war of resistance against Japanese aggression (one-off)",
       "Public holiday (one-off anniversary)", "2015/16"),
    gh("2015-09-28", "2015-09-28", "the day following the Chinese Mid-Autumn Festival", "Public holiday (day after Mid-Autumn Festival)", "2015/16"),
    gh("2015-10-01", "2015-10-01", "National Day", "Public holiday (National Day)", "2015/16"),
    gh("2015-10-21", "2015-10-21", "Chung Yeung Festival", "Public holiday (Chung Yeung Festival)", "2015/16"),
    school("xmas", "2015-12-21", "2016-01-03", "2015/16", " (Chinese version)"),
    gh("2015-12-25", "2015-12-26", "Christmas Day and the first weekday after Christmas", "Public holidays (Christmas)", "2015/16"),
    gh("2016-01-01", "2016-01-01", "the first day of January", "Public holiday (New Year's Day)", "2015/16"),
    school("lny", "2016-02-05", "2016-02-14", "2015/16", " (Chinese version)"),
    gh("2016-02-08", "2016-02-10", "the first to third day of Lunar New Year", "Public holidays (Lunar New Year)", "2015/16"),
    gh("2016-03-25", "2016-03-28", "Good Friday to Easter Monday", "Public holidays (Easter)", "2015/16"),
    school("easter", "2016-03-25", "2016-04-04", "2015/16", " (Chinese version)"),
    gh("2016-04-04", "2016-04-04", "Ching Ming Festival", "Public holiday (Ching Ming Festival)", "2015/16"),
    gh("2016-05-02", "2016-05-02", "the day following Labour Day", "Public holiday (Labour Day)", "2015/16"),
    gh("2016-05-14", "2016-05-14", "the Birthday of the Buddha", "Public holiday (Buddha's Birthday)", "2015/16"),
    gh("2016-06-09", "2016-06-09", "Tuen Ng Festival", "Public holiday (Dragon Boat Festival)", "2015/16"),
    gh("2016-07-01", "2016-07-01", "HKSAR Establishment Day", "Public holiday", "2015/16"),
    school("summer", "2016-07-14", "2016-08-31", "2015/16", " (Chinese version)"),
    # ---- 2016/17
    reopen("2016-09-01", "2016/17"),
    gh("2016-09-16", "2016-09-16", "the day following the Chinese Mid-Autumn Festival", "Public holiday (day after Mid-Autumn Festival)", "2016/17"),
    gh("2016-10-01", "2016-10-01", "National Day", "Public holiday (National Day)", "2016/17"),
    gh("2016-10-10", "2016-10-10", "the day following Chung Yeung Festival", "Public holiday (Chung Yeung Festival)", "2016/17"),
    school("xmas", "2016-12-21", "2017-01-02", "2016/17", " (Chinese version)"),
    gh("2016-12-26", "2016-12-27", "the first and second weekdays after Christmas Day", "Public holidays (Christmas)", "2016/17"),
    gh("2017-01-02", "2017-01-02", "the day following the first day of January", "Public holiday (New Year's Day)", "2016/17"),
    school("lny", "2017-01-27", "2017-02-04", "2016/17", " (Chinese version)"),
    gh("2017-01-28", "2017-01-31", "Lunar New Year's Day and the third and fourth day of Lunar New Year", "Public holidays (Lunar New Year)", "2016/17"),
    gh("2017-04-04", "2017-04-04", "Ching Ming Festival", "Public holiday (Ching Ming Festival)", "2016/17"),
    gh("2017-04-14", "2017-04-17", "Good Friday to Easter Monday", "Public holidays (Easter)", "2016/17"),
    school("easter", "2017-04-14", "2017-04-24", "2016/17", " (Chinese version)"),
    gh("2017-05-01", "2017-05-01", "Labour Day", "Public holiday (Labour Day)", "2016/17"),
    gh("2017-05-03", "2017-05-03", "the Birthday of the Buddha", "Public holiday (Buddha's Birthday)", "2016/17"),
    gh("2017-05-30", "2017-05-30", "Tuen Ng Festival", "Public holiday (Dragon Boat Festival)", "2016/17"),
    gh("2017-07-01", "2017-07-01", "HKSAR Establishment Day", "Public holiday", "2016/17"),
    school("summer", "2017-07-14", "2017-08-31", "2016/17", " (Chinese version)"),
    # ---- 2017/18
    reopen("2017-09-01", "2017/18"),
    gh("2017-10-02", "2017-10-02", "the day following National Day", "Public holiday (National Day)", "2017/18"),
    gh("2017-10-05", "2017-10-05", "the day following the Chinese Mid-Autumn Festival", "Public holiday (day after Mid-Autumn Festival)", "2017/18"),
    gh("2017-10-28", "2017-10-28", "Chung Yeung Festival", "Public holiday (Chung Yeung Festival)", "2017/18"),
    school("xmas", "2017-12-20", "2018-01-01", "2017/18"),
    gh("2017-12-25", "2017-12-26", "Christmas Day and the first weekday after Christmas", "Public holidays (Christmas)", "2017/18"),
    gh("2018-01-01", "2018-01-01", "the first day of January", "Public holiday (New Year's Day)", "2017/18"),
    school("lny", "2018-02-14", "2018-02-22", "2017/18"),
    gh("2018-02-16", "2018-02-19", "Lunar New Year's Day, the second and the fourth day of Lunar New Year", "Public holidays (Lunar New Year)", "2017/18"),
    gh("2018-03-30", "2018-04-02", "Good Friday to Easter Monday", "Public holidays (Easter)", "2017/18"),
    school("easter", "2018-03-30", "2018-04-09", "2017/18"),
    gh("2018-04-05", "2018-04-05", "Ching Ming Festival", "Public holiday (Ching Ming Festival)", "2017/18"),
    gh("2018-05-01", "2018-05-01", "Labour Day", "Public holiday (Labour Day)", "2017/18"),
    gh("2018-05-22", "2018-05-22", "the Birthday of the Buddha", "Public holiday (Buddha's Birthday)", "2017/18"),
    gh("2018-06-18", "2018-06-18", "Tuen Ng Festival", "Public holiday (Dragon Boat Festival)", "2017/18"),
    gh("2018-07-02", "2018-07-02", "the day following HKSAR Establishment Day", "Public holiday", "2017/18"),
    school("summer", "2018-07-13", "2018-08-31", "2017/18"),
    # ---- 2018/19
    reopen("2018-09-03", "2018/19"),
    gh("2018-09-25", "2018-09-25", "the day following the Chinese Mid-Autumn Festival", "Public holiday (day after Mid-Autumn Festival)", "2018/19"),
    gh("2018-10-01", "2018-10-01", "National Day", "Public holiday (National Day)", "2018/19"),
    gh("2018-10-17", "2018-10-17", "Chung Yeung Festival", "Public holiday (Chung Yeung Festival)", "2018/19"),
    school("xmas", "2018-12-21", "2019-01-01", "2018/19"),
    gh("2018-12-25", "2018-12-26", "Christmas Day and the first weekday after Christmas", "Public holidays (Christmas)", "2018/19"),
    gh("2019-01-01", "2019-01-01", "the first day of January", "Public holiday (New Year's Day)", "2018/19"),
    school("lny", "2019-02-01", "2019-02-11", "2018/19"),
    gh("2019-02-05", "2019-02-07", "the first to third day of Lunar New Year", "Public holidays (Lunar New Year)", "2018/19"),
    gh("2019-04-05", "2019-04-05", "Ching Ming Festival", "Public holiday (Ching Ming Festival)", "2018/19"),
    gh("2019-04-19", "2019-04-22", "Good Friday to Easter Monday", "Public holidays (Easter)", "2018/19"),
    school("easter", "2019-04-19", "2019-04-29", "2018/19"),
    gh("2019-05-01", "2019-05-01", "Labour Day", "Public holiday (Labour Day)", "2018/19"),
    gh("2019-05-13", "2019-05-13", "the day following the Birthday of the Buddha", "Public holiday (Buddha's Birthday)", "2018/19"),
    gh("2019-06-07", "2019-06-07", "Tuen Ng Festival", "Public holiday (Dragon Boat Festival)", "2018/19"),
    gh("2019-07-01", "2019-07-01", "HKSAR Establishment Day", "Public holiday", "2018/19"),
    school("summer", "2019-07-15", "2019-08-31", "2018/19"),
    # ---- 2019/20 (to the last forecast target, 2020-01-25)
    reopen("2019-09-02", "2019/20"),
    gh("2019-09-14", "2019-09-14", "the day following the Chinese Mid-Autumn Festival", "Public holiday (day after Mid-Autumn Festival)", "2019/20"),
    gh("2019-10-01", "2019-10-01", "National Day", "Public holiday (National Day)", "2019/20"),
    gh("2019-10-07", "2019-10-07", "Chung Yeung Festival", "Public holiday (Chung Yeung Festival)", "2019/20"),
    school("xmas", "2019-12-20", "2020-01-01", "2019/20"),
    gh("2019-12-25", "2019-12-26", "Christmas Day and the first weekday after Christmas", "Public holidays (Christmas)", "2019/20"),
    gh("2020-01-01", "2020-01-01", "the first day of January", "Public holiday (New Year's Day)", "2019/20"),
    school("lny", "2020-01-22", "2020-02-01", "2019/20"),
    gh("2020-01-25", "2020-01-28", "Lunar New Year's Day, the third and the fourth day of Lunar New Year", "Public holidays (Lunar New Year)", "2019/20"),
]

if __name__ == "__main__":
    cal = pd.read_csv(CAL)
    keep = cal[pd.to_datetime(cal["start"]) >= "2024-06-01"]
    new = pd.DataFrame(ROWS, columns=cal.columns)
    for c in ("start", "end"):
        assert (pd.to_datetime(new[c]).dt.strftime("%Y-%m-%d") == new[c]).all()
    assert (pd.to_datetime(new["end"]) >= pd.to_datetime(new["start"])).all()
    out = pd.concat([new, keep]).sort_values(["start", "type"], kind="stable")
    out.to_csv(CAL, index=False)
    print(f"{len(new)} rows for 2015-2020 + {len(keep)} existing rows -> {CAL}")
