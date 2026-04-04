from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler

from backend.calendar_utils import annotate_calendar, default_menu_for_date
from backend.config import ForecastConfig, ensure_directories
from backend.forecasting import KitchenForecastSystem


_SYSTEM: KitchenForecastSystem | None = None


def get_system(config: ForecastConfig | None = None) -> KitchenForecastSystem:
    global _SYSTEM
    if _SYSTEM is None:
        _SYSTEM = KitchenForecastSystem(config=config or ForecastConfig())
    return _SYSTEM


def bootstrap_system(config: ForecastConfig | None = None) -> dict:
    ensure_directories()
    system = get_system(config)
    system.ensure_seed_data()
    metrics = system.get_metrics_payload()
    if not metrics.get("trained_at"):
        return system.train_all_models()
    return metrics


def run_retraining_job(config: ForecastConfig | None = None) -> dict:
    return get_system(config).train_all_models()


def ingest_feedback(
    feedback_payload: dict,
    config: ForecastConfig | None = None,
    retrain_on_feedback: bool = False,
) -> dict:
    system = get_system(config)
    result = system.log_feedback(feedback_payload)
    if retrain_on_feedback:
        retrain = system.train_all_models()
        result["retraining"] = retrain
    return result


def ingest_uploaded_dataset(
    uploaded_frame: pd.DataFrame,
    config: ForecastConfig | None = None,
    retrain: bool = True,
) -> dict:
    system = get_system(config)
    prepared = _prepare_uploaded_frame(uploaded_frame, system)
    system.repository.upsert_observations(prepared)
    training_result = system.train_all_models() if retrain else {}
    return {
        "rows_ingested": int(len(prepared)),
        "training_result": training_result,
    }


def get_dashboard_payload(config: ForecastConfig | None = None) -> dict:
    system = get_system(config)
    return {
        "metrics": system.get_metrics_payload(),
        "history": system.get_dashboard_history(),
    }


def start_scheduler(config: ForecastConfig | None = None) -> BackgroundScheduler:
    config = config or ForecastConfig()
    scheduler = BackgroundScheduler(timezone=ZoneInfo("Asia/Kolkata"))
    scheduler.add_job(
        run_retraining_job,
        trigger="interval",
        hours=config.retrain_interval_hours,
        id="nightly-kitchen-retraining",
        replace_existing=True,
    )
    scheduler.start()
    return scheduler


def _prepare_uploaded_frame(
    frame: pd.DataFrame,
    system: KitchenForecastSystem,
) -> pd.DataFrame:
    working = frame.copy()
    working.columns = [column.strip() for column in working.columns]
    required = {"kitchen_id", "date", "actual_demand"}
    missing = required - set(working.columns)
    if missing:
        raise ValueError(
            f"Uploaded dataset is missing required columns: {', '.join(sorted(missing))}."
        )

    working["date"] = pd.to_datetime(working["date"])
    working["actual_demand"] = pd.to_numeric(working["actual_demand"], errors="coerce")
    working = working.dropna(subset=["kitchen_id", "date", "actual_demand"])
    if working.empty:
        raise ValueError("Uploaded dataset does not contain any valid observation rows.")

    known_kitchens = set(system.repository.list_kitchens()["kitchen_id"].astype(str))
    unknown_kitchens = sorted(set(working["kitchen_id"].astype(str)) - known_kitchens)
    if unknown_kitchens:
        raise ValueError(
            f"Unknown kitchen_id values: {', '.join(unknown_kitchens)}."
        )

    if "menu_type" not in working.columns:
        working["menu_type"] = working["date"].map(default_menu_for_date)
    working = annotate_calendar(working)
    for column, default_value in (
        ("prepared_quantity", np.nan),
        ("waste_quantity", 0.0),
        ("shortage_quantity", 0.0),
        ("attendance_variation", 0.0),
        ("temperature", np.nan),
        ("rainfall", np.nan),
        ("predicted_demand", np.nan),
        ("selected_model", None),
        ("meal_session", "daily_aggregate"),
        ("is_augmented", 0),
    ):
        if column not in working.columns:
            working[column] = default_value
    working["data_source"] = "uploaded"
    return working[
        [
            "kitchen_id",
            "date",
            "actual_demand",
            "prepared_quantity",
            "waste_quantity",
            "shortage_quantity",
            "attendance_variation",
            "menu_type",
            "is_holiday",
            "is_exam_week",
            "is_event_day",
            "event_name",
            "temperature",
            "rainfall",
            "predicted_demand",
            "selected_model",
            "data_source",
            "meal_session",
            "is_augmented",
        ]
    ].sort_values(["date", "kitchen_id"])
