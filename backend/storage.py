from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from pandas.errors import EmptyDataError

from backend.config import (
    FORECAST_HISTORY_FILE,
    MODEL_BUNDLE_FILE,
    MODEL_COMPARISON_FILE,
    MONITORING_LOG_FILE,
    SUMMARY_METRICS_FILE,
    ensure_directories,
)


def save_model_bundle(bundle: dict[str, Any]) -> None:
    ensure_directories()
    joblib.dump(bundle, MODEL_BUNDLE_FILE)


def load_model_bundle() -> dict[str, Any] | None:
    if not MODEL_BUNDLE_FILE.exists():
        return None
    return joblib.load(MODEL_BUNDLE_FILE)


def save_json(payload: dict[str, Any], path: Path) -> None:
    ensure_directories()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_summary_metrics() -> dict[str, Any]:
    return load_json(SUMMARY_METRICS_FILE)


def save_training_artifacts(
    summary_metrics: dict[str, Any],
    model_comparison: pd.DataFrame,
    forecast_history: pd.DataFrame,
    monitoring_history: pd.DataFrame,
) -> None:
    save_json(summary_metrics, SUMMARY_METRICS_FILE)
    model_comparison.to_csv(MODEL_COMPARISON_FILE, index=False)
    forecast_history.to_csv(FORECAST_HISTORY_FILE, index=False)
    monitoring_history.to_csv(MONITORING_LOG_FILE, index=False)


def load_dataframe(path: Path, parse_dates: list[str] | None = None) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, parse_dates=parse_dates)
    except EmptyDataError:
        return pd.DataFrame()
