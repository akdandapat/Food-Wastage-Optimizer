from __future__ import annotations

from datetime import date

import pandas as pd


def season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "summer"
    if month in (6, 7, 8, 9):
        return "monsoon"
    return "autumn"


def default_menu_for_date(value: pd.Timestamp | date) -> str:
    timestamp = pd.Timestamp(value)
    weekday_defaults = {
        0: "protein_rich",
        1: "regular",
        2: "regional_special",
        3: "comfort_food",
        4: "regular",
        5: "festive",
        6: "light_weekend",
    }
    return weekday_defaults[timestamp.dayofweek]


def _holiday_dates_for_year(year: int) -> set[pd.Timestamp]:
    fixed_dates = {
        pd.Timestamp(year=year, month=1, day=1),
        pd.Timestamp(year=year, month=1, day=26),
        pd.Timestamp(year=year, month=3, day=8),
        pd.Timestamp(year=year, month=4, day=14),
        pd.Timestamp(year=year, month=5, day=1),
        pd.Timestamp(year=year, month=8, day=15),
        pd.Timestamp(year=year, month=10, day=2),
        pd.Timestamp(year=year, month=12, day=25),
    }
    durga_puja = set(pd.date_range(f"{year}-10-09", f"{year}-10-14", freq="D"))
    kali_puja = set(pd.date_range(f"{year}-11-01", f"{year}-11-03", freq="D"))
    summer_break = set(pd.date_range(f"{year}-05-20", f"{year}-07-05", freq="D"))
    winter_break = set(pd.date_range(f"{year}-12-20", f"{year + 1}-01-05", freq="D"))
    return fixed_dates | durga_puja | kali_puja | summer_break | winter_break


def is_holiday_date(value: pd.Timestamp | date) -> bool:
    timestamp = pd.Timestamp(value).normalize()
    holidays = _holiday_dates_for_year(timestamp.year) | _holiday_dates_for_year(
        timestamp.year - 1
    )
    return timestamp in holidays


def is_exam_period(value: pd.Timestamp | date) -> bool:
    timestamp = pd.Timestamp(value).normalize()
    year = timestamp.year
    exam_ranges = [
        (pd.Timestamp(year=year, month=4, day=10), pd.Timestamp(year=year, month=5, day=5)),
        (pd.Timestamp(year=year, month=11, day=10), pd.Timestamp(year=year, month=11, day=30)),
    ]
    return any(start <= timestamp <= end for start, end in exam_ranges)


def event_name_for_date(value: pd.Timestamp | date) -> str | None:
    timestamp = pd.Timestamp(value).normalize()
    year = timestamp.year
    event_windows = {
        "orientation": pd.date_range(f"{year}-07-12", f"{year}-07-18", freq="D"),
        "tech_fest": pd.date_range(f"{year}-02-08", f"{year}-02-11", freq="D"),
        "sports_meet": pd.date_range(f"{year}-09-02", f"{year}-09-05", freq="D"),
        "cultural_fest": pd.date_range(f"{year}-12-02", f"{year}-12-04", freq="D"),
    }
    for event_name, event_dates in event_windows.items():
        if timestamp in set(event_dates):
            return event_name
    return None


def is_event_date(value: pd.Timestamp | date) -> bool:
    return event_name_for_date(value) is not None


def annotate_calendar(frame: pd.DataFrame) -> pd.DataFrame:
    annotated = frame.copy()
    annotated["date"] = pd.to_datetime(annotated["date"])
    annotated["season"] = annotated["date"].dt.month.map(season_from_month)

    inferred_holiday = annotated["date"].map(lambda value: int(is_holiday_date(value)))
    inferred_exam = annotated["date"].map(lambda value: int(is_exam_period(value)))
    inferred_event = annotated["date"].map(lambda value: int(is_event_date(value)))
    inferred_event_name = annotated["date"].map(event_name_for_date)

    if "is_holiday" in annotated.columns:
        annotated["is_holiday"] = (
            pd.to_numeric(annotated["is_holiday"], errors="coerce")
            .fillna(inferred_holiday)
            .astype(int)
        )
    else:
        annotated["is_holiday"] = inferred_holiday.astype(int)

    if "is_exam_week" in annotated.columns:
        annotated["is_exam_week"] = (
            pd.to_numeric(annotated["is_exam_week"], errors="coerce")
            .fillna(inferred_exam)
            .astype(int)
        )
    else:
        annotated["is_exam_week"] = inferred_exam.astype(int)

    if "is_event_day" in annotated.columns:
        annotated["is_event_day"] = (
            pd.to_numeric(annotated["is_event_day"], errors="coerce")
            .fillna(inferred_event)
            .astype(int)
        )
    else:
        annotated["is_event_day"] = inferred_event.astype(int)

    if "event_name" in annotated.columns:
        annotated["event_name"] = annotated["event_name"].fillna(inferred_event_name)
    else:
        annotated["event_name"] = inferred_event_name

    return annotated
