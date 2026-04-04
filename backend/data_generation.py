from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from backend.calendar_utils import annotate_calendar, default_menu_for_date
from backend.config import ForecastConfig


def _attendance_multiplier(default_band: str) -> float:
    return {"high": 0.93, "medium": 0.88, "low": 0.82}.get(default_band, 0.86)


def _menu_sampler(timestamp: pd.Timestamp, rng: np.random.Generator) -> str:
    default_menu = default_menu_for_date(timestamp)
    weekday = timestamp.dayofweek
    if weekday >= 5:
        return rng.choice(["light_weekend", "festive", default_menu], p=[0.45, 0.30, 0.25])
    if weekday == 0:
        return rng.choice(["protein_rich", "regular", default_menu], p=[0.35, 0.40, 0.25])
    if weekday == 2:
        return rng.choice(["regional_special", "comfort_food", default_menu], p=[0.34, 0.18, 0.48])
    return rng.choice([default_menu, "regular", "comfort_food"], p=[0.45, 0.38, 0.17])


def generate_synthetic_operations(
    kitchens: pd.DataFrame,
    config: ForecastConfig | None = None,
    weather_override: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Generate multi-kitchen historical operations data for Kolkata hostels.

    When ``weather_override`` is provided (columns: date, temperature, rainfall) it
    replaces the analytic climate curve so fallback data still uses measured Kolkata
    weather from ingestion.
    """

    config = config or ForecastConfig()
    rng = np.random.default_rng(config.random_state)

    end_date = pd.Timestamp.today().normalize() - timedelta(days=1)
    start_date = end_date - timedelta(days=config.synthetic_period_days - 1)
    calendar = pd.DataFrame({"date": pd.date_range(start_date, end_date, freq="D")})
    calendar = annotate_calendar(calendar)
    calendar["day_of_week"] = calendar["date"].dt.day_name()
    calendar["month"] = calendar["date"].dt.month

    if weather_override is not None and not weather_override.empty:
        wo = weather_override.copy()
        wo["date"] = pd.to_datetime(wo["date"]).dt.normalize()
        calendar = calendar.merge(wo[["date", "temperature", "rainfall"]], on="date", how="left")
        day_of_year = calendar["date"].dt.dayofyear.to_numpy()
        monsoon_signal = np.clip(np.sin(2 * np.pi * (day_of_year - 160) / 365.25), 0, None)
        summer_signal = np.clip(np.sin(2 * np.pi * (day_of_year - 85) / 365.25), 0, None)
        fill_temp = (
            28.0
            + 7.8 * summer_signal
            - 3.2 * monsoon_signal
            + rng.normal(0, 1.6, len(calendar))
        )
        rain_probability = 0.10 + 0.60 * monsoon_signal
        rain_intensity = rng.gamma(2.5 + 3.0 * monsoon_signal, 5.2, len(calendar))
        fill_rain = np.where(rng.random(len(calendar)) < rain_probability, rain_intensity, 0.0)
        calendar["temperature"] = calendar["temperature"].fillna(fill_temp).round(1)
        calendar["rainfall"] = calendar["rainfall"].fillna(fill_rain).round(1)
    else:
        day_of_year = calendar["date"].dt.dayofyear.to_numpy()
        monsoon_signal = np.clip(np.sin(2 * np.pi * (day_of_year - 160) / 365.25), 0, None)
        summer_signal = np.clip(np.sin(2 * np.pi * (day_of_year - 85) / 365.25), 0, None)

        city_temperature = (
            28.0
            + 7.8 * summer_signal
            - 3.2 * monsoon_signal
            + rng.normal(0, 1.6, len(calendar))
        )
        rain_probability = 0.10 + 0.60 * monsoon_signal
        rain_intensity = rng.gamma(2.5 + 3.0 * monsoon_signal, 5.2, len(calendar))
        city_rainfall = np.where(rng.random(len(calendar)) < rain_probability, rain_intensity, 0.0)

        calendar["temperature"] = city_temperature.round(1)
        calendar["rainfall"] = city_rainfall.round(1)

    menu_effect = {
        "regular": 0.00,
        "protein_rich": 0.035,
        "regional_special": 0.028,
        "comfort_food": 0.018,
        "festive": 0.08,
        "light_weekend": -0.045,
    }
    weekday_effect = {0: 1.03, 1: 1.01, 2: 1.00, 3: 0.99, 4: 0.97, 5: 0.90, 6: 0.86}
    zone_effect = {
        "south": 0.00,
        "central": 0.02,
        "west": 0.015,
        "east": -0.01,
        "northwest": 0.025,
    }

    records: list[dict] = []
    for kitchen in kitchens.to_dict("records"):
        kitchen_rng = np.random.default_rng(
            config.random_state + sum(ord(value) for value in kitchen["kitchen_id"])
        )
        capacity = float(kitchen["capacity"])
        base_occupancy = _attendance_multiplier(kitchen["default_attendance_band"])
        kitchen_bias = zone_effect.get(kitchen["campus_zone"], 0.0) + kitchen_rng.normal(0, 0.012)

        for row in calendar.itertuples(index=False):
            menu_type = _menu_sampler(pd.Timestamp(row.date), kitchen_rng)
            attendance_variation = (
                kitchen_rng.normal(0, 0.028)
                - 0.15 * row.is_holiday
                - 0.05 * row.is_exam_week
                + 0.07 * row.is_event_day
                - max(row.rainfall - 28.0, 0.0) * 0.0012
                + menu_effect[menu_type] * 0.45
            )
            attendance_variation = float(np.clip(attendance_variation, -0.30, 0.18))

            occupancy = base_occupancy * weekday_effect[pd.Timestamp(row.date).dayofweek]
            occupancy *= 1 + kitchen_bias + attendance_variation

            weather_penalty = (
                max(abs(row.temperature - 31.0) - 3.0, 0.0) * 12.0
                + max(row.rainfall - 35.0, 0.0) * 2.5
            )
            demand = (
                capacity
                * occupancy
                * (1 + menu_effect[menu_type])
                * (1 - 0.07 * row.is_exam_week)
                * (1 - 0.18 * row.is_holiday)
                * (1 + 0.10 * row.is_event_day)
                - weather_penalty
                + kitchen_rng.normal(0, 42.0)
            )
            actual_demand = int(np.clip(round(demand), capacity * 0.42, capacity * 1.08))

            prep_buffer = (
                0.06
                + 0.015 * row.is_event_day
                + 0.010 * (menu_type == "festive")
                + kitchen_rng.normal(0, 0.01)
            )
            prepared_quantity = int(max(round(actual_demand * (1 + prep_buffer)), actual_demand))
            waste_quantity = float(max(prepared_quantity - actual_demand + kitchen_rng.normal(0, 8.0), 0.0))
            shortage_quantity = float(max(actual_demand - prepared_quantity, 0.0))

            records.append(
                {
                    "kitchen_id": kitchen["kitchen_id"],
                    "date": row.date,
                    "actual_demand": actual_demand,
                    "prepared_quantity": prepared_quantity,
                    "waste_quantity": round(waste_quantity, 2),
                    "shortage_quantity": round(shortage_quantity, 2),
                    "attendance_variation": round(attendance_variation, 4),
                    "menu_type": menu_type,
                    "is_holiday": int(row.is_holiday),
                    "is_exam_week": int(row.is_exam_week),
                    "is_event_day": int(row.is_event_day),
                    "event_name": row.event_name,
                    "temperature": round(float(row.temperature + kitchen_rng.normal(0, 0.4)), 1),
                    "rainfall": round(float(max(row.rainfall + kitchen_rng.normal(0, 1.2), 0.0)), 1),
                    "predicted_demand": None,
                    "selected_model": None,
                    "data_source": "synthetic",
                    "meal_session": "daily_aggregate",
                    "is_augmented": 0,
                }
            )

    observations = pd.DataFrame(records)

    # Sparse missingness keeps the feature and imputation paths realistic.
    for column, fraction in (("temperature", 0.01), ("rainfall", 0.015), ("attendance_variation", 0.01)):
        missing_count = max(1, int(len(observations) * fraction))
        missing_indices = rng.choice(observations.index.to_numpy(), size=missing_count, replace=False)
        observations.loc[missing_indices, column] = np.nan

    return observations.sort_values(["date", "kitchen_id"]).reset_index(drop=True)
