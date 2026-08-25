"""Synthetic operations data generator for Kolkata hostel kitchens.

Produces realistic multi-kitchen demand, preparation, waste, and shortage
records embedding Kolkata-specific climate curves, academic calendar effects,
day-of-week patterns, menu-type mixes, and zone-level capacity biases.
All randomness is seeded via ``ForecastConfig.random_state``.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.random import Generator

from backend.calendar_utils import annotate_calendar, default_menu_for_date
from backend.config import ForecastConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level look-up tables
# ---------------------------------------------------------------------------

#: Menu-type demand multiplier.  Positive = higher attendance draw.
MENU_DEMAND_EFFECT: Dict[str, float] = {
    "regular": 0.00,
    "protein_rich": 0.035,
    "regional_special": 0.028,
    "comfort_food": 0.018,
    "festive": 0.08,
    "light_weekend": -0.045,
}

#: ISO weekday → demand scale factor.
#: Mon–Sat are flat at 1.0 — at n≈45–190 per meal the observed day-to-day
#: spread does not exceed the Poisson noise floor.  This omission is
#: deliberate.  Sunday effect is handled via SUNDAY_DAILY_MULTIPLIER.
WEEKDAY_DEMAND_SCALE: Dict[int, float] = {
    0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0,
}

#: Sunday net daily-aggregate multiplier (~−18%).
#: The underlying driver is that Sunday night is vegetarian and attendance
#: collapses by ~45%, while Sunday morning ("grand meal") rises ~10–15%.
#: The daily aggregate of those two effects is approximately −18%.
SUNDAY_DAILY_MULTIPLIER: float = 0.82

#: Sunday night veg multiplier — for use once meal_session is split
#: into breakfast/dinner (Task 8).  Kept here as a named constant so the
#: planted value is visible and recoverable.
SUNDAY_VEG_NIGHT_MULTIPLIER: float = 0.55

#: Campus zone → incremental demand bias.
#: With all three hostels on the same campus, zone bias is zero.
ZONE_DEMAND_BIAS: Dict[str, float] = {
    "campus": 0.0,
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _attendance_multiplier(default_band: str) -> float:
    """Map an attendance band to a base-occupancy fraction.

    Args:
        default_band: One of ``"high"``, ``"medium"``, ``"low"``.

    Returns:
        Float in [0.82, 0.93] — fraction of capacity typically occupied.
        Higher bands indicate hostels with stricter meal-attendance
        policies.
    """
    band_map: Dict[str, float] = {"high": 0.93, "medium": 0.88, "low": 0.82}
    return band_map.get(default_band, 0.86)


def _menu_sampler(timestamp: pd.Timestamp, rng: Generator) -> str:
    """Stochastically select a menu type conditioned on day of week.

    Sampling probabilities reflect institutional kitchen patterns:
    weekends favour lighter meals, Mondays skew protein-rich,
    Wednesdays lean into regional specials ("special day").

    Args:
        timestamp: Date for which to sample a menu.
        rng: Seeded NumPy ``Generator`` for reproducibility.

    Returns:
        Menu-type string (key in ``MENU_DEMAND_EFFECT``).
    """
    default_menu: str = default_menu_for_date(timestamp)
    weekday: int = timestamp.dayofweek

    if weekday >= 5:
        return str(rng.choice(
            ["light_weekend", "festive", default_menu], p=[0.45, 0.30, 0.25]))
    if weekday == 0:
        return str(rng.choice(
            ["protein_rich", "regular", default_menu], p=[0.35, 0.40, 0.25]))
    if weekday == 2:
        return str(rng.choice(
            ["regional_special", "comfort_food", default_menu], p=[0.34, 0.18, 0.48]))
    return str(rng.choice(
        [default_menu, "regular", "comfort_food"], p=[0.45, 0.38, 0.17]))


# ---------------------------------------------------------------------------
# Kolkata climate helpers
# ---------------------------------------------------------------------------


def _kolkata_climate_signals(
    day_of_year: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute seasonal monsoon and summer sinusoidal signals.

    Kolkata has a summer peak (~day 85-145) and monsoon onset
    (~day 160-270).  The sinusoidal approximation captures these
    envelopes without a full meteorological dataset.

    Args:
        day_of_year: 1-D integer array of ordinal day-of-year values.

    Returns:
        ``(monsoon_signal, summer_signal)`` — non-negative float arrays.
    """
    monsoon: np.ndarray = np.clip(
        np.sin(2 * np.pi * (day_of_year - 160) / 365.25), 0, None)
    summer: np.ndarray = np.clip(
        np.sin(2 * np.pi * (day_of_year - 85) / 365.25), 0, None)
    return monsoon, summer


def _synthesize_temperature(
    monsoon: np.ndarray, summer: np.ndarray,
    rng: Generator, n: int,
) -> np.ndarray:
    """Generate synthetic daily-mean temperature for Kolkata (°C).

    Baseline 28 °C + summer boost (+7.8) - monsoon depression (-3.2)
    with Gaussian jitter (σ=1.6).

    Args:
        monsoon: Non-negative monsoon envelope.
        summer: Non-negative summer envelope.
        rng: Seeded Generator.
        n: Number of days.

    Returns:
        1-D float array of temperatures.
    """
    return 28.0 + 7.8 * summer - 3.2 * monsoon + rng.normal(0, 1.6, n)


def _synthesize_rainfall(
    monsoon: np.ndarray, rng: Generator, n: int,
) -> np.ndarray:
    """Generate synthetic daily rainfall for Kolkata (mm).

    Two-stage model: Bernoulli occurrence (10-70 % by monsoon) then
    Gamma intensity.  Produces realistic right-skewed tropical rainfall.

    Args:
        monsoon: Non-negative monsoon envelope.
        rng: Seeded Generator.
        n: Number of days.

    Returns:
        1-D float array (zero for dry days).
    """
    prob: np.ndarray = 0.10 + 0.60 * monsoon
    intensity: np.ndarray = rng.gamma(2.5 + 3.0 * monsoon, 5.2, n)
    return np.where(rng.random(n) < prob, intensity, 0.0)


# ---------------------------------------------------------------------------
# Calendar construction
# ---------------------------------------------------------------------------


def _build_calendar(
    config: ForecastConfig,
    rng: Generator,
    weather_override: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Build date-indexed calendar with weather and academic annotations.

    Real weather from *weather_override* is used first; gaps are filled
    with the analytic Kolkata climate curve.

    Args:
        config: Forecast configuration.
        rng: Seeded Generator.
        weather_override: Optional real Kolkata weather observations.

    Returns:
        DataFrame with one row per day containing date, weather, and
        academic calendar columns.
    """
    end_date: pd.Timestamp = pd.Timestamp.today().normalize() - timedelta(days=1)
    start_date: pd.Timestamp = end_date - timedelta(days=config.synthetic_period_days - 1)

    calendar: pd.DataFrame = pd.DataFrame(
        {"date": pd.date_range(start_date, end_date, freq="D")})
    calendar = annotate_calendar(calendar)
    calendar["day_of_week"] = calendar["date"].dt.day_name()
    calendar["month"] = calendar["date"].dt.month

    doy: np.ndarray = calendar["date"].dt.dayofyear.to_numpy()
    n: int = len(calendar)
    monsoon, summer = _kolkata_climate_signals(doy)

    if weather_override is not None and not weather_override.empty:
        wo: pd.DataFrame = weather_override.copy()
        wo["date"] = pd.to_datetime(wo["date"]).dt.normalize()
        calendar = calendar.merge(wo[["date", "temperature", "rainfall"]], on="date", how="left")
        calendar["temperature"] = calendar["temperature"].fillna(
            pd.Series(_synthesize_temperature(monsoon, summer, rng, n))).round(1)
        calendar["rainfall"] = calendar["rainfall"].fillna(
            pd.Series(_synthesize_rainfall(monsoon, rng, n))).round(1)
    else:
        calendar["temperature"] = _synthesize_temperature(monsoon, summer, rng, n).round(1)
        calendar["rainfall"] = _synthesize_rainfall(monsoon, rng, n).round(1)

    return calendar


# ---------------------------------------------------------------------------
# Core entry-point
# ---------------------------------------------------------------------------


def generate_synthetic_operations(
    kitchens: pd.DataFrame,
    config: Optional[ForecastConfig] = None,
    weather_override: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Generate multi-kitchen historical operations data.

    Produces a date × kitchen panel embedding day-of-week cyclicality,
    menu-driven demand shift, weather sensitivity, academic calendar
    effects, zone biases, preparation buffer, waste, and sparse
    missingness for robust model training.

    Args:
        kitchens: DataFrame with columns ``kitchen_id``, ``capacity``,
            ``default_attendance_band``, ``campus_zone``.
        config: Forecast hyper-parameters (defaults to ``ForecastConfig()``).
        weather_override: Optional real weather to overlay.

    Returns:
        DataFrame sorted by ``(date, kitchen_id)`` with demand, waste,
        weather, calendar flags, and lineage metadata.

    Raises:
        KeyError: If *kitchens* is missing required columns.
    """
    config = config or ForecastConfig()
    rng: Generator = np.random.default_rng(config.random_state)
    calendar: pd.DataFrame = _build_calendar(config, rng, weather_override)
    n_days: int = len(calendar)

    # Pre-extract vectorised calendar arrays.
    cal_dates: np.ndarray = calendar["date"].to_numpy()
    cal_weekdays: np.ndarray = calendar["date"].dt.dayofweek.to_numpy()
    cal_is_holiday: np.ndarray = calendar["is_holiday"].to_numpy(dtype=float)
    cal_is_exam: np.ndarray = calendar["is_exam_week"].to_numpy(dtype=float)
    cal_is_event: np.ndarray = calendar["is_event_day"].to_numpy(dtype=float)
    cal_event_name: np.ndarray = calendar["event_name"].to_numpy()
    cal_temperature: np.ndarray = calendar["temperature"].to_numpy(dtype=float)
    cal_rainfall: np.ndarray = calendar["rainfall"].to_numpy(dtype=float)
    weekday_scale: np.ndarray = np.array([WEEKDAY_DEMAND_SCALE[w] for w in cal_weekdays])

    # Sunday flag (ISO weekday 6).
    is_sunday: np.ndarray = (cal_weekdays == 6).astype(float)

    # Shared campus shock: one draw per day, applied identically to all
    # three hostels.  Three hostels 300 m apart are hit by the same
    # festival, the same rainstorm, the same exam week.
    shared_campus_shock: np.ndarray = rng.normal(0, 0.02, n_days)

    panels: List[pd.DataFrame] = []

    for kitchen in kitchens.to_dict("records"):
        kid: str = kitchen["kitchen_id"]
        cap: float = float(kitchen["capacity"])
        base_occ: float = _attendance_multiplier(kitchen["default_attendance_band"])
        k_rng: Generator = np.random.default_rng(
            config.random_state + sum(ord(c) for c in kid))
        k_bias: float = ZONE_DEMAND_BIAS.get(kitchen["campus_zone"], 0.0) + k_rng.normal(0, 0.012)

        # Menu sampling (stochastic per-day).
        menus: List[str] = [_menu_sampler(pd.Timestamp(d), k_rng) for d in cal_dates]
        m_eff: np.ndarray = np.array([MENU_DEMAND_EFFECT[m] for m in menus])

        # --- Attendance variation (Task 3b fix) ---
        # att_var is *only* idiosyncratic day-to-day noise plus the shared
        # campus shock.  Calendar/menu effects are applied once in the
        # demand expression below — not here.
        att_var: np.ndarray = np.clip(
            k_rng.normal(0, 0.028, n_days) + shared_campus_shock,
            -0.30, 0.18)

        # Occupancy (vectorised).
        occ: np.ndarray = base_occ * weekday_scale * (1.0 + k_bias + att_var)

        # Weather penalty — multiplicative on occupancy (Task 3c fix).
        # Temperature: each °C above 34 or below 28 reduces attendance by 0.8%.
        # Rainfall: each mm above 35 reduces attendance by 0.15%.
        weather_factor: np.ndarray = (
            1.0
            - np.clip(np.abs(cal_temperature - 31.0) - 3.0, 0.0, None) * 0.008
            - np.clip(cal_rainfall - 35.0, 0.0, None) * 0.0015
        )

        # Structural demand before count noise (effects applied ONCE here).
        structural_demand: np.ndarray = (
            cap * occ * weather_factor
            * (1.0 + m_eff)
            * (1.0 - 0.07 * cal_is_exam) * (1.0 - 0.18 * cal_is_holiday)
            * (1.0 + 0.10 * cal_is_event)
            * (1.0 + (SUNDAY_DAILY_MULTIPLIER - 1.0) * is_sunday)
        )

        # --- Terminal noise: Negative Binomial (Task 3c fix) ---
        # At μ≈50 the normal approximation puts real probability mass below
        # zero and requires constant clipping; count data needs a count
        # distribution.  We use NegBin with dispersion index ~1.3
        # (mild overdispersion — what mess attendance actually looks like).
        # Var = μ + μ²/r  ⟹  dispersion index = 1 + μ/r = 1.3  ⟹  r = μ/0.3
        mu_structural = np.clip(structural_demand, 1.0, None)
        r_dispersion = mu_structural / 0.3  # gives dispersion index ~1.3
        # NegBin parameterisation: n=r, p=r/(r+μ)
        p_nb = r_dispersion / (r_dispersion + mu_structural)
        demand: np.ndarray = k_rng.negative_binomial(
            np.clip(r_dispersion, 1, None).astype(int), np.clip(p_nb, 1e-6, 1.0 - 1e-6)
        ).astype(float)

        act_dem: np.ndarray = np.clip(np.round(demand), cap * 0.20, cap * 1.25).astype(int)

        # Preparation buffer & waste (vectorised) — Task 3a fix.
        # prep_buf straddles zero so ~10–15% of days end in genuine shortfall.
        festive_flag: np.ndarray = np.array([1.0 if m == "festive" else 0.0 for m in menus])
        prep_buf: np.ndarray = 0.06 + 0.015 * cal_is_event + 0.010 * festive_flag + k_rng.normal(0, 0.05, n_days)
        # Floor at 0 — NOT at act_dem — so preparation can genuinely fall short.
        prep_qty: np.ndarray = np.maximum(np.round(act_dem * (1.0 + prep_buf)), 0).astype(int)
        # Waste noise proportional to demand (Task 3c fix).
        waste: np.ndarray = np.maximum(prep_qty - act_dem + k_rng.normal(0, 0.04 * act_dem, n_days), 0.0)
        short: np.ndarray = np.maximum(act_dem - prep_qty, 0.0).astype(float)

        # Per-kitchen micro-climate jitter.
        loc_temp: np.ndarray = np.round(cal_temperature + k_rng.normal(0, 0.4, n_days), 1)
        loc_rain: np.ndarray = np.round(np.maximum(cal_rainfall + k_rng.normal(0, 1.2, n_days), 0.0), 1)

        panels.append(pd.DataFrame({
            "kitchen_id": kid, "date": cal_dates,
            "actual_demand": act_dem, "prepared_quantity": prep_qty,
            "waste_quantity": np.round(waste, 2),
            "shortage_quantity": np.round(short, 2),
            "attendance_variation": np.round(att_var, 4),
            "menu_type": menus,
            "is_holiday": cal_is_holiday.astype(int),
            "is_exam_week": cal_is_exam.astype(int),
            "is_event_day": cal_is_event.astype(int),
            "event_name": cal_event_name,
            "temperature": loc_temp, "rainfall": loc_rain,
            "predicted_demand": np.nan, "selected_model": np.nan,
            "data_source": "synthetic", "meal_session": "daily_aggregate",
            "is_augmented": 0,
        }))

    observations: pd.DataFrame = pd.concat(panels, ignore_index=True)

    # --- Post-generation shortage assertions (Task 3a) ---
    total_rows = len(observations)
    shortage_rows = int((observations["shortage_quantity"] > 0).sum())
    shortage_rate = shortage_rows / total_rows
    print(f"[data_generation] shortage_quantity.sum() = {observations['shortage_quantity'].sum():.1f}")
    print(f"[data_generation] shortage rate = {shortage_rate:.1%} ({shortage_rows}/{total_rows} rows)")
    if observations["shortage_quantity"].sum() == 0:
        logger.warning(
            "shortage_quantity is identically zero — the newsvendor has no shortage signal."
        )
    if not (0.05 <= shortage_rate <= 0.20):
        logger.warning(
            f"Shortage rate {shortage_rate:.1%} is outside the target 5–20% band."
        )

    # --- MAR missingness (Task 3g) ---
    # attendance_variation logs go missing on holidays (4x higher probability)
    # because the mess runs on skeleton staff.
    # Weather readings go missing during heavy rain (3x higher probability).
    _corrupt_mar(observations, "attendance_variation", base_rate=0.01,
                 condition_col="is_holiday", condition_val=1, boost=4.0, rng=rng)
    _corrupt_mar(observations, "temperature", base_rate=0.01,
                 condition_col="rainfall", condition_threshold=30.0, boost=3.0, rng=rng)
    _corrupt_mar(observations, "rainfall", base_rate=0.015,
                 condition_col="rainfall", condition_threshold=30.0, boost=3.0, rng=rng)

    return observations.sort_values(["date", "kitchen_id"]).reset_index(drop=True)


def _corrupt_mar(
    df: pd.DataFrame,
    target_col: str,
    base_rate: float,
    rng: Generator,
    condition_col: str | None = None,
    condition_val: float | None = None,
    condition_threshold: float | None = None,
    boost: float = 1.0,
) -> None:
    """Introduce Missing-At-Random (MAR) corruption to a column.

    Missingness probability is ``base_rate`` for rows not matching the
    condition, and ``base_rate * boost`` for rows that do match.
    """
    n = len(df)
    probs = np.full(n, base_rate)
    if condition_col is not None:
        if condition_val is not None:
            mask = df[condition_col].to_numpy() == condition_val
        elif condition_threshold is not None:
            # Use the *pre-corruption* values for the condition check
            vals = pd.to_numeric(df[condition_col], errors="coerce").to_numpy()
            mask = vals > condition_threshold
        else:
            mask = np.zeros(n, dtype=bool)
        probs[mask] = base_rate * boost
    # Normalise so overall rate stays close to base_rate
    draws = rng.random(n) < probs
    df.loc[draws, target_col] = np.nan
