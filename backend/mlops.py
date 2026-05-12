"""MLOps orchestration layer for the Kitchen Demand Forecasting system.

This module provides the top-level entry-points that FastAPI routes (and
scheduled jobs) call to bootstrap the system, trigger retraining, ingest
user feedback and uploaded datasets, and assemble dashboard payloads.

Architecture
------------
* A **singleton** :class:`KitchenForecastSystem` is held in ``_SYSTEM``
  and lazily initialised via :func:`get_system`.  All orchestration
  functions delegate to this instance.
* **Scheduled retraining** is driven by APScheduler with a configurable
  interval (default: 24 h, ``ForecastConfig.retrain_interval_hours``).
* **MLflow integration** is *optional*.  When the ``mlflow`` package is
  importable, training runs are automatically logged (params, metrics,
  and the model bundle artifact).  When it is absent, a no-op fallback
  is used so the rest of the pipeline is unaffected.

Thread Safety
-------------
``_SYSTEM`` is not protected by a lock.  In production, the singleton is
initialised once at startup (``bootstrap_system``) before any concurrent
requests arrive, so this is safe for the single-process Uvicorn default.
If running under a multi-threaded server, wrap :func:`get_system` with a
``threading.Lock``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler

from backend.calendar_utils import annotate_calendar, default_menu_for_date
from backend.config import (
    ForecastConfig,
    MODEL_BUNDLE_FILE,
    ensure_directories,
)
from backend.forecasting import KitchenForecastSystem

# ---------------------------------------------------------------------------
# Optional MLflow integration
# ---------------------------------------------------------------------------

try:
    import mlflow  # type: ignore[import-untyped]

    _MLFLOW_AVAILABLE: bool = True
except ImportError:  # pragma: no cover
    mlflow = None  # type: ignore[assignment]
    _MLFLOW_AVAILABLE = False

logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton system instance
# ---------------------------------------------------------------------------

_SYSTEM: Optional[KitchenForecastSystem] = None

_FINAL_OUTPUT_COLUMNS: List[str] = [
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
"""Canonical column ordering for ingested observation DataFrames."""


# ---------------------------------------------------------------------------
# System lifecycle
# ---------------------------------------------------------------------------


def get_system(config: Optional[ForecastConfig] = None) -> KitchenForecastSystem:
    """Return (or lazily create) the singleton forecast system.

    The first call constructs the system, initialises the SQLite
    database, and wires up the feature builder, optimizer, and
    repository.  Subsequent calls return the cached instance.

    Args:
        config: Forecast hyper-parameters.  Only used on first call;
            ignored if the singleton already exists.

    Returns:
        The shared :class:`KitchenForecastSystem` instance.
    """
    global _SYSTEM
    if _SYSTEM is None:
        _SYSTEM = KitchenForecastSystem(config=config or ForecastConfig())
    return _SYSTEM


def bootstrap_system(config: Optional[ForecastConfig] = None) -> Dict[str, Any]:
    """Cold-start the system: ensure seed data, train if needed.

    Workflow:

    1. Create all required directories.
    2. Initialise the singleton system.
    3. Ensure seed data is present (ingestion panel → synthetic fallback).
    4. If no trained model exists, run a full training cycle.
    5. Optionally log the training run to MLflow.

    Args:
        config: Forecast hyper-parameters (defaults to ``ForecastConfig()``).

    Returns:
        A metrics payload dict if a model was already trained, or the
        full training-result dict from ``train_all_models()``.
    """
    ensure_directories()
    system: KitchenForecastSystem = get_system(config)
    system.ensure_seed_data()
    metrics: Dict[str, Any] = system.get_metrics_payload()
    if not metrics.get("trained_at"):
        result: Dict[str, Any] = system.train_all_models()
        log_training_run_to_mlflow(result, config or ForecastConfig())
        return result
    return metrics


def run_retraining_job(config: Optional[ForecastConfig] = None) -> Dict[str, Any]:
    """Execute a full retraining cycle (called by the scheduler).

    This is the entry-point for the nightly APScheduler job.  It
    delegates to ``KitchenForecastSystem.train_all_models()``, which
    handles tuning, evaluation, champion/challenger promotion, and
    artifact persistence.

    Args:
        config: Optional override config.

    Returns:
        The training-result dict including ``run_id``, ``selected_model``,
        ``promoted``, and ``business_metrics``.
    """
    result: Dict[str, Any] = get_system(config).train_all_models()
    log_training_run_to_mlflow(result, config or ForecastConfig())
    return result


# ---------------------------------------------------------------------------
# MLflow experiment tracking
# ---------------------------------------------------------------------------


def log_training_run_to_mlflow(
    training_result: Dict[str, Any],
    config: ForecastConfig,
    experiment_name: str = "kitchen-demand-forecasting",
) -> None:
    """Log a completed training run to MLflow (no-op if MLflow is absent).

    Creates or reuses an MLflow experiment, starts a run scoped to the
    training ``run_id``, and logs:

    * **Parameters** — selected model name, model version, promotion
      reason, and key ``ForecastConfig`` scalars.
    * **Metrics** — per-candidate RMSE, MAE, interval coverage, and
      aggregate business metrics (waste reduction %, annual savings).
    * **Artifacts** — the serialised production model bundle (joblib).

    All MLflow operations are wrapped in a top-level try/except so that
    a tracking-server outage or misconfiguration never crashes the
    training pipeline.

    Args:
        training_result: The dict returned by
            ``KitchenForecastSystem.train_all_models()``.
        config: The active forecast configuration (logged as params).
        experiment_name: MLflow experiment name (default:
            ``"kitchen-demand-forecasting"``).
    """
    if not _MLFLOW_AVAILABLE or mlflow is None:
        logger.debug("MLflow is not installed — skipping experiment logging.")
        return

    try:
        mlflow.set_experiment(experiment_name)

        run_name: str = (
            f"{training_result.get('selected_model', 'unknown')}"
            f"-{training_result.get('run_id', 'no-id')[:8]}"
        )

        with mlflow.start_run(run_name=run_name) as _run:
            # -- Parameters ------------------------------------------------
            mlflow.log_params(
                {
                    "selected_model": training_result.get("selected_model", ""),
                    "selected_model_version": training_result.get(
                        "selected_model_version", ""
                    ),
                    "promoted": str(training_result.get("promoted", False)),
                    "promotion_reason": str(
                        training_result.get("promotion_reason", "")
                    )[:250],
                    "holdout_days": config.holdout_days,
                    "prediction_interval": config.prediction_interval,
                    "service_level_target": config.service_level_target,
                    "waste_cost": config.waste_cost,
                    "shortage_cost": config.shortage_cost,
                    "synthetic_period_days": config.synthetic_period_days,
                }
            )

            # -- Per-candidate metrics -------------------------------------
            model_metrics: List[Dict[str, Any]] = training_result.get(
                "model_metrics", []
            )
            for candidate in model_metrics:
                prefix: str = candidate.get("model_name", "unknown")
                mlflow.log_metrics(
                    {
                        f"{prefix}_rmse": float(candidate.get("rmse", 0.0)),
                        f"{prefix}_mae": float(candidate.get("mae", 0.0)),
                        f"{prefix}_weekly_rmse": float(
                            candidate.get("weekly_rmse", 0.0)
                        ),
                        f"{prefix}_interval_coverage": float(
                            candidate.get("interval_coverage", 0.0)
                        ),
                    }
                )

            # -- Business metrics ------------------------------------------
            biz: Dict[str, Any] = training_result.get("business_metrics", {})
            mlflow.log_metrics(
                {
                    "waste_reduction_pct": float(
                        biz.get("waste_reduction_pct", 0.0)
                    ),
                    "daily_cost_savings_inr": float(
                        biz.get("daily_cost_savings_inr", 0.0)
                    ),
                    "annual_savings_inr": float(
                        biz.get("annual_savings_inr", 0.0)
                    ),
                    "prediction_interval_coverage": float(
                        biz.get("prediction_interval_coverage", 0.0)
                    ),
                }
            )

            # -- Model bundle artifact -------------------------------------
            bundle_path: str = str(MODEL_BUNDLE_FILE)
            try:
                mlflow.log_artifact(bundle_path, artifact_path="model_bundle")
            except Exception as artifact_exc:
                logger.warning(
                    "MLflow: could not log model bundle artifact: %s",
                    artifact_exc,
                )

        logger.info(
            "MLflow run '%s' logged to experiment '%s'.",
            run_name,
            experiment_name,
        )
    except Exception as exc:
        logger.warning(
            "MLflow logging failed (non-fatal): %s. Training result is unaffected.",
            exc,
        )


# ---------------------------------------------------------------------------
# Feedback & data ingestion
# ---------------------------------------------------------------------------


def ingest_feedback(
    feedback_payload: Dict[str, Any],
    config: Optional[ForecastConfig] = None,
    retrain_on_feedback: bool = False,
) -> Dict[str, Any]:
    """Log a ground-truth feedback observation and optionally retrain.

    The feedback loop closes the predict → observe → learn cycle:

    1. The payload (actual demand, prepared quantity, waste) is persisted
       as a new observation row.
    2. Realized cost metrics are back-filled onto the corresponding
       optimization decision.
    3. If *retrain_on_feedback* is ``True``, a full retraining cycle is
       triggered immediately (useful for online-learning experiments;
       disabled by default in production to avoid mid-day model swaps).

    Args:
        feedback_payload: Dict with keys ``kitchen_id``, ``date``,
            ``actual_demand``, ``prepared_quantity``, ``waste_quantity``,
            ``menu_type``, and optional weather/calendar overrides.
        config: Forecast hyper-parameters.
        retrain_on_feedback: Whether to trigger ``train_all_models()``
            after logging.

    Returns:
        A status dict containing ``realized_shortage`` and
        ``realized_cost``.  If retraining was triggered, includes a
        nested ``retraining`` key with the training result.
    """
    system: KitchenForecastSystem = get_system(config)
    result: Dict[str, Any] = system.log_feedback(feedback_payload)
    if retrain_on_feedback:
        retrain: Dict[str, Any] = system.train_all_models()
        result["retraining"] = retrain
    return result


def ingest_uploaded_dataset(
    uploaded_frame: pd.DataFrame,
    config: Optional[ForecastConfig] = None,
    retrain: bool = True,
) -> Dict[str, Any]:
    """Validate, normalise, and ingest a user-uploaded CSV dataset.

    Validation steps:

    1. Strip whitespace from column headers.
    2. Assert required columns: ``kitchen_id``, ``date``,
       ``actual_demand``.
    3. Coerce types and drop rows with null keys.
    4. Reject unknown ``kitchen_id`` values not in the registry.
    5. Annotate with calendar flags and default menu types.
    6. Back-fill optional columns (weather, waste, etc.) with sensible
       defaults.

    After ingestion, a retraining cycle is triggered by default so the
    model incorporates the new observations.

    Args:
        uploaded_frame: Raw DataFrame from the uploaded CSV.
        config: Forecast hyper-parameters.
        retrain: Whether to trigger retraining after ingestion
            (default ``True``).

    Returns:
        Dict with ``rows_ingested`` count and ``training_result``
        (empty dict if *retrain* is ``False``).

    Raises:
        ValueError: If required columns are missing, all rows are
            invalid, or unknown kitchen IDs are found.
    """
    system: KitchenForecastSystem = get_system(config)
    prepared: pd.DataFrame = _prepare_uploaded_frame(uploaded_frame, system)
    system.repository.upsert_observations(prepared)
    training_result: Dict[str, Any] = system.train_all_models() if retrain else {}
    return {
        "rows_ingested": int(len(prepared)),
        "training_result": training_result,
    }


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


def get_dashboard_payload(config: Optional[ForecastConfig] = None) -> Dict[str, Any]:
    """Assemble the full dashboard payload (metrics + history).

    Combines the summary metrics (model comparison, business KPIs,
    feature importance) with the detailed forecast and training
    history for the frontend dashboard.

    Args:
        config: Forecast hyper-parameters.

    Returns:
        Dict with ``metrics`` and ``history`` sub-dicts.
    """
    system: KitchenForecastSystem = get_system(config)
    return {
        "metrics": system.get_metrics_payload(),
        "history": system.get_dashboard_history(),
    }


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


def start_scheduler(config: Optional[ForecastConfig] = None) -> BackgroundScheduler:
    """Start the APScheduler background retraining job.

    Configures a ``BackgroundScheduler`` in the ``Asia/Kolkata``
    timezone with an interval trigger whose period is set by
    ``ForecastConfig.retrain_interval_hours`` (default 24 h).

    The job ID ``"nightly-kitchen-retraining"`` is deterministic so
    that ``replace_existing=True`` prevents duplicate schedules if the
    server is restarted without shutting down the old scheduler.

    Args:
        config: Forecast hyper-parameters.

    Returns:
        The started ``BackgroundScheduler`` instance (caller should
        keep a reference to call ``scheduler.shutdown()`` on exit).
    """
    config = config or ForecastConfig()
    scheduler: BackgroundScheduler = BackgroundScheduler(
        timezone=ZoneInfo("Asia/Kolkata")
    )
    scheduler.add_job(
        run_retraining_job,
        trigger="interval",
        hours=config.retrain_interval_hours,
        id="nightly-kitchen-retraining",
        replace_existing=True,
    )
    scheduler.start()
    return scheduler


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _prepare_uploaded_frame(
    frame: pd.DataFrame,
    system: KitchenForecastSystem,
) -> pd.DataFrame:
    """Validate, coerce, and normalise an uploaded observation DataFrame.

    This is the gatekeeper function that ensures user-supplied data
    conforms to the schema expected by the repository and feature
    builder.  Any row that survives this pipeline is safe to upsert.

    Args:
        frame: Raw uploaded DataFrame (may have messy headers,
            missing columns, or unknown kitchen IDs).
        system: The active forecast system (used to query the
            kitchen registry for ID validation).

    Returns:
        A cleaned, calendar-annotated DataFrame with columns in the
        canonical order defined by ``_FINAL_OUTPUT_COLUMNS``.

    Raises:
        ValueError: On missing required columns, all-NaN rows after
            coercion, or unknown ``kitchen_id`` values.
    """
    working: pd.DataFrame = frame.copy()
    working.columns = [column.strip() for column in working.columns]

    # --- Required-column check --------------------------------------------
    required: set[str] = {"kitchen_id", "date", "actual_demand"}
    missing: set[str] = required - set(working.columns)
    if missing:
        raise ValueError(
            f"Uploaded dataset is missing required columns: {', '.join(sorted(missing))}."
        )

    # --- Type coercion & null-key drop ------------------------------------
    working["date"] = pd.to_datetime(working["date"])
    working["actual_demand"] = pd.to_numeric(
        working["actual_demand"], errors="coerce"
    )
    working = working.dropna(subset=["kitchen_id", "date", "actual_demand"])
    if working.empty:
        raise ValueError(
            "Uploaded dataset does not contain any valid observation rows."
        )

    # --- Kitchen-ID validation --------------------------------------------
    known_kitchens: set[str] = set(
        system.repository.list_kitchens()["kitchen_id"].astype(str)
    )
    unknown_kitchens: List[str] = sorted(
        set(working["kitchen_id"].astype(str)) - known_kitchens
    )
    if unknown_kitchens:
        raise ValueError(
            f"Unknown kitchen_id values: {', '.join(unknown_kitchens)}."
        )

    # --- Calendar & menu annotation ---------------------------------------
    if "menu_type" not in working.columns:
        working["menu_type"] = working["date"].map(default_menu_for_date)
    working = annotate_calendar(working)

    # --- Back-fill optional columns with sensible defaults ----------------
    _optional_defaults: List[tuple[str, Any]] = [
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
    ]
    for column, default_value in _optional_defaults:
        if column not in working.columns:
            working[column] = default_value

    working["data_source"] = "uploaded"

    return working[_FINAL_OUTPUT_COLUMNS].sort_values(["date", "kitchen_id"])
