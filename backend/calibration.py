"""
Prediction interval calibration for tabular and TFT outputs.

Computes multiplicative scales on a holdout set so empirical coverage tracks
the nominal level (e.g. 90%). When coverage is too low, widen; when it is too
high with excessive width, tighten slightly.

The scales are persisted inside the production model bundle and applied at
inference time in ``KitchenForecastSystem``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def empirical_coverage(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Fraction of observations inside [lower, upper]."""
    mask = (actual >= lower) & (actual <= upper)
    return float(np.mean(mask)) if len(actual) else 0.0


def mean_interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean(upper - lower)) if len(lower) else 0.0


def calibrate_symmetric_scale(
    actual: np.ndarray,
    predicted: np.ndarray,
    base_sigma: float,
    z_score: float,
    nominal_coverage: float,
    min_scale: float = 0.85,
    max_scale: float = 2.5,
) -> dict[str, float]:
    """
    Find ``sigma_scale`` such that intervals ``pred ± z * scale * sigma`` meet coverage.

    Uses monotone bisection on the scale factor; ``base_sigma`` is the per-row
    or global residual dispersion already used by the forecaster.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    sigma = max(float(base_sigma), 1e-6)
    z = float(z_score)

    def coverage_for_scale(scale: float) -> float:
        lo = predicted - z * sigma * scale
        hi = predicted + z * sigma * scale
        return empirical_coverage(actual, lo, hi)

    target = float(nominal_coverage)
    lo_s, hi_s = min_scale, max_scale
    cov_lo = coverage_for_scale(lo_s)
    cov_hi = coverage_for_scale(hi_s)
    if cov_lo >= target:
        chosen = lo_s
        note = "coverage_met_at_min_scale"
    elif cov_hi <= target:
        chosen = max_scale
        note = "coverage_still_below_target_at_max_scale"
    else:
        for _ in range(40):
            mid = (lo_s + hi_s) / 2.0
            if coverage_for_scale(mid) >= target:
                hi_s = mid
            else:
                lo_s = mid
        chosen = hi_s
        note = "bisection_converged"

    width_before = 2 * z * sigma
    width_after = 2 * z * sigma * chosen
    return {
        "sigma_scale": float(chosen),
        "empirical_coverage_before": float(coverage_for_scale(1.0)),
        "empirical_coverage_after": float(coverage_for_scale(chosen)),
        "mean_width_before": float(width_before),
        "mean_width_after": float(width_after),
        "nominal_coverage": target,
        "calibration_note": note,
    }


def calibrate_tft_quantile_spread(
    actual: np.ndarray,
    lower: np.ndarray,
    median: np.ndarray,
    upper: np.ndarray,
    nominal_coverage: float,
    min_scale: float = 0.85,
    max_scale: float = 2.5,
) -> dict[str, float]:
    """
    Widen or tighten TFT quantile outputs symmetrically around the median forecast.
    """
    actual = np.asarray(actual, dtype=float)
    lower = np.asarray(lower, dtype=float)
    median = np.asarray(median, dtype=float)
    upper = np.asarray(upper, dtype=float)

    def coverage_for_scale(scale: float) -> float:
        lo = median - (median - lower) * scale
        hi = median + (upper - median) * scale
        return empirical_coverage(actual, lo, hi)

    target = float(nominal_coverage)
    lo_s, hi_s = min_scale, max_scale
    if coverage_for_scale(1.0) >= target and mean_interval_width(lower, upper) > 0:
        # Optionally tighten if intervals are overly conservative
        chosen = 1.0
        for trial in np.linspace(0.85, 1.0, 8):
            if coverage_for_scale(trial) >= target - 0.02:
                chosen = trial
        note = "tightened_or_unity"
    elif coverage_for_scale(max_scale) < target:
        chosen = max_scale
        note = "max_spread_reached"
    else:
        lo_s, hi_s = 1.0, max_scale
        for _ in range(40):
            mid = (lo_s + hi_s) / 2.0
            if coverage_for_scale(mid) >= target:
                hi_s = mid
            else:
                lo_s = mid
        chosen = hi_s
        note = "bisection_widen"

    return {
        "tft_quantile_scale": float(chosen),
        "empirical_coverage_before": float(coverage_for_scale(1.0)),
        "empirical_coverage_after": float(coverage_for_scale(chosen)),
        "nominal_coverage": target,
        "tft_calibration_note": note,
    }


def build_interval_calibration_payload(
    evaluation_frame: pd.DataFrame,
    z_interval: float,
    residual_std: float,
    nominal_coverage: float,
    model_name: str,
) -> dict[str, Any]:
    """
    Package calibration metrics for persistence alongside the model bundle.
    """
    if evaluation_frame.empty or "actual_demand" not in evaluation_frame.columns:
        return {
            "tabular_sigma_scale": 1.0,
            "tft_quantile_scale": 1.0,
            "details": {"reason": "empty_evaluation"},
        }

    actual = evaluation_frame["actual_demand"].to_numpy()
    pred = evaluation_frame["predicted_demand"].to_numpy()
    sigma = float(
        evaluation_frame["sigma"].mean()
        if "sigma" in evaluation_frame.columns
        else residual_std
    )

    tab = calibrate_symmetric_scale(actual, pred, sigma, z_interval, nominal_coverage)

    payload: dict[str, Any] = {
        "tabular_sigma_scale": tab["sigma_scale"],
        "tft_quantile_scale": 1.0,
        "details": {
            "tabular": tab,
            "model_calibrated_for": model_name,
        },
    }

    if model_name == "tft" and {"lower_bound", "upper_bound"}.issubset(evaluation_frame.columns):
        tft = calibrate_tft_quantile_spread(
            actual,
            evaluation_frame["lower_bound"].to_numpy(),
            pred,
            evaluation_frame["upper_bound"].to_numpy(),
            nominal_coverage,
        )
        payload["tft_quantile_scale"] = tft["tft_quantile_scale"]
        payload["details"]["tft"] = tft

    return payload
