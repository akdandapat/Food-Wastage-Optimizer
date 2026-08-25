#!/usr/bin/env python
"""Recovery test: verify that the simulator's key structural assumptions
are recoverable from a single generated panel.

Checks:
  1. Shortage rate is in [5%, 20%].
  2. Sunday multiplier on demand is recoverable (~0.82 ± 0.10).
  3. NegBin dispersion index is ~1.3 ± 0.3.
  4. Total shortage_quantity > 0.

Run: python scripts/recovery_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from backend.config import DEFAULT_KITCHENS, ForecastConfig
from backend.data_generation import generate_synthetic_operations
from backend.database import initialize_database, SQLiteRepository


def main() -> None:
    config = ForecastConfig()
    initialize_database()
    repo = SQLiteRepository()
    kitchens = repo.list_kitchens()

    panel = generate_synthetic_operations(kitchens, config)
    n_rows = len(panel)
    results: list[tuple[str, bool, str]] = []

    # --- Check 1: Shortage rate ---
    shortage_rows = int((panel["shortage_quantity"] > 0).sum())
    shortage_rate = shortage_rows / n_rows
    ok = 0.05 <= shortage_rate <= 0.20
    results.append((
        "Shortage rate in [5%, 20%]",
        ok,
        f"{shortage_rate:.1%} ({shortage_rows}/{n_rows})",
    ))

    # --- Check 2: Total shortage > 0 ---
    total_shortage = float(panel["shortage_quantity"].sum())
    ok = total_shortage > 0
    results.append((
        "Total shortage > 0",
        ok,
        f"{total_shortage:.1f}",
    ))

    # --- Check 3: Sunday multiplier recovery ---
    panel["weekday"] = pd.to_datetime(panel["date"]).dt.weekday  # 6 = Sunday
    # For each kitchen, compute mean demand on Sunday vs non-Sunday weekdays
    sunday_ratios: list[float] = []
    for _kid, group in panel.groupby("kitchen_id"):
        sun = group[group["weekday"] == 6]["actual_demand"].mean()
        non_sun = group[group["weekday"] != 6]["actual_demand"].mean()
        if non_sun > 0:
            sunday_ratios.append(sun / non_sun)
    avg_sunday_ratio = float(np.mean(sunday_ratios))
    ok = 0.72 <= avg_sunday_ratio <= 0.92  # target is ~0.82
    results.append((
        "Sunday multiplier ~0.82 (±0.10)",
        ok,
        f"{avg_sunday_ratio:.3f}",
    ))

    # --- Check 4: Dispersion index recovery ---
    # Dispersion index = Var(demand) / Mean(demand)
    # For NegBin with dispersion_index=1.3, this should be around 1.3 on baseline days
    disp_indices: list[float] = []
    for _kid, group in panel.groupby("kitchen_id"):
        clean = group[
            (group["menu_type"] == "regular")
            & (group["is_holiday"] == 0)
            & (group["is_exam_week"] == 0)
            & (group["is_event_day"] == 0)
        ]
        demands = clean["actual_demand"].to_numpy(dtype=float)
        if len(demands) > 30 and np.mean(demands) > 0:
            weekdays = pd.to_datetime(clean["date"]).dt.weekday.to_numpy()
            weekday_means = np.array([
                demands[weekdays == w].mean() if (weekdays == w).sum() > 0 else np.mean(demands)
                for w in range(7)
            ])
            residuals = demands - weekday_means[weekdays]
            var_r = float(np.var(residuals))
            mean_d = float(np.mean(demands))
            disp_indices.append(var_r / mean_d)
    avg_disp = float(np.mean(disp_indices)) if disp_indices else 0.0
    ok = 1.0 <= avg_disp <= 1.6
    results.append((
        "Dispersion index ~1.3 (±0.3)",
        ok,
        f"{avg_disp:.2f}",
    ))

    # --- Print results ---
    print(f"\n{'='*60}")
    print("Recovery Test Results")
    print(f"{'='*60}")
    all_pass = True
    for name, passed, detail in results:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {name}: {detail}")
    print(f"{'='*60}")
    if all_pass:
        print("All checks passed.")
    else:
        print("Some checks FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
