from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
LOGS_DIR = DATA_DIR / "logs"
# Canonical outputs from ``backend.data_ingestion`` (merged public + augmented panel).
PROCESSED_OPERATIONS_FILE = PROCESSED_DATA_DIR / "kitchen_operations_panel.csv"
INGESTION_MANIFEST_FILE = LOGS_DIR / "ingestion_manifest.jsonl"
METRICS_DIR = DATA_DIR / "metrics"
FIGURES_DIR = DATA_DIR / "figures"
MODELS_DIR = BASE_DIR / "models"
ARTIFACTS_DIR = MODELS_DIR / "artifacts"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"

LOCAL_RUNTIME_DIR = Path(os.environ.get("LOCALAPPDATA", str(DATA_DIR))) / "CodexRuntime"
SQLITE_DB_FILE = LOCAL_RUNTIME_DIR / "kitchen_ops.sqlite"
MODEL_BUNDLE_FILE = ARTIFACTS_DIR / "production_bundle.joblib"
SUMMARY_METRICS_FILE = METRICS_DIR / "summary_metrics.json"
MONITORING_LOG_FILE = METRICS_DIR / "monitoring_log.csv"
MODEL_COMPARISON_FILE = METRICS_DIR / "model_comparison.csv"
FORECAST_HISTORY_FILE = METRICS_DIR / "forecast_history.csv"

PLOT_FILENAMES = {
    "demand_vs_actual": "demand_vs_actual.png",
    "model_comparison": "model_comparison.png",
    "waste_comparison": "waste_comparison.png",
    "cost_savings": "cost_savings.png",
    "feature_importance": "feature_importance.png",
    "residual_analysis": "residual_analysis.png",
}

DEFAULT_MENU_TYPES = [
    "regular",
    "protein_rich",
    "regional_special",
    "comfort_food",
    "festive",
    "light_weekend",
]

DEFAULT_KITCHENS = [
    {
        "kitchen_id": "JU_BH1",
        "hostel_name": "Jadavpur Boys Hostel 1",
        "campus_zone": "south",
        "latitude": 22.4984,
        "longitude": 88.3702,
        "capacity": 2100,
        "default_attendance_band": "high",
    },
    {
        "kitchen_id": "JU_GH1",
        "hostel_name": "Jadavpur Girls Hostel",
        "campus_zone": "south",
        "latitude": 22.4972,
        "longitude": 88.3691,
        "capacity": 1650,
        "default_attendance_band": "medium",
    },
    {
        "kitchen_id": "CU_RAJA",
        "hostel_name": "Presidency-Raja Hostel Mess",
        "campus_zone": "central",
        "latitude": 22.5750,
        "longitude": 88.3626,
        "capacity": 1850,
        "default_attendance_band": "high",
    },
    {
        "kitchen_id": "CU_EDEN",
        "hostel_name": "Eden Hostel Central Kitchen",
        "campus_zone": "central",
        "latitude": 22.5727,
        "longitude": 88.3639,
        "capacity": 2400,
        "default_attendance_band": "high",
    },
    {
        "kitchen_id": "IIT_KGP_H1",
        "hostel_name": "Techno Annex Mess",
        "campus_zone": "west",
        "latitude": 22.3143,
        "longitude": 87.3105,
        "capacity": 2800,
        "default_attendance_band": "high",
    },
    {
        "kitchen_id": "IEM_MAIN",
        "hostel_name": "IEM Main Hostel Kitchen",
        "campus_zone": "east",
        "latitude": 22.5778,
        "longitude": 88.4338,
        "capacity": 1750,
        "default_attendance_band": "medium",
    },
    {
        "kitchen_id": "SXC_NORTH",
        "hostel_name": "St Xavier's North Mess",
        "campus_zone": "central",
        "latitude": 22.5644,
        "longitude": 88.3630,
        "capacity": 1350,
        "default_attendance_band": "medium",
    },
    {
        "kitchen_id": "BESC_MAIN",
        "hostel_name": "BESC Main Hostel Kitchen",
        "campus_zone": "south",
        "latitude": 22.5280,
        "longitude": 88.3678,
        "capacity": 1550,
        "default_attendance_band": "medium",
    },
    {
        "kitchen_id": "NIT_DGP_A",
        "hostel_name": "NIT Durgapur Hall A Mess",
        "campus_zone": "northwest",
        "latitude": 23.5460,
        "longitude": 87.2936,
        "capacity": 2300,
        "default_attendance_band": "high",
    },
    {
        "kitchen_id": "IIEST_SHIB",
        "hostel_name": "IIEST Shibpur Main Kitchen",
        "campus_zone": "west",
        "latitude": 22.5550,
        "longitude": 88.3070,
        "capacity": 2050,
        "default_attendance_band": "high",
    },
]

DEFAULT_RECIPES = {
    "regular": [
        {"ingredient_name": "rice", "unit": "kg", "qty_per_100_meals": 18.0},
        {"ingredient_name": "dal", "unit": "kg", "qty_per_100_meals": 7.0},
        {"ingredient_name": "vegetables", "unit": "kg", "qty_per_100_meals": 11.0},
        {"ingredient_name": "oil", "unit": "l", "qty_per_100_meals": 2.8},
    ],
    "protein_rich": [
        {"ingredient_name": "rice", "unit": "kg", "qty_per_100_meals": 17.0},
        {"ingredient_name": "dal", "unit": "kg", "qty_per_100_meals": 8.0},
        {"ingredient_name": "egg_or_chicken", "unit": "kg", "qty_per_100_meals": 14.0},
        {"ingredient_name": "vegetables", "unit": "kg", "qty_per_100_meals": 9.0},
    ],
    "regional_special": [
        {"ingredient_name": "rice", "unit": "kg", "qty_per_100_meals": 19.0},
        {"ingredient_name": "fish", "unit": "kg", "qty_per_100_meals": 15.0},
        {"ingredient_name": "mustard_sauce", "unit": "kg", "qty_per_100_meals": 2.0},
        {"ingredient_name": "vegetables", "unit": "kg", "qty_per_100_meals": 8.0},
    ],
    "comfort_food": [
        {"ingredient_name": "rice", "unit": "kg", "qty_per_100_meals": 16.0},
        {"ingredient_name": "khichdi_mix", "unit": "kg", "qty_per_100_meals": 14.0},
        {"ingredient_name": "potato", "unit": "kg", "qty_per_100_meals": 8.0},
        {"ingredient_name": "oil", "unit": "l", "qty_per_100_meals": 2.2},
    ],
    "festive": [
        {"ingredient_name": "rice", "unit": "kg", "qty_per_100_meals": 18.0},
        {"ingredient_name": "chicken", "unit": "kg", "qty_per_100_meals": 18.0},
        {"ingredient_name": "paneer", "unit": "kg", "qty_per_100_meals": 5.0},
        {"ingredient_name": "dessert_mix", "unit": "kg", "qty_per_100_meals": 6.0},
    ],
    "light_weekend": [
        {"ingredient_name": "rice", "unit": "kg", "qty_per_100_meals": 15.0},
        {"ingredient_name": "dal", "unit": "kg", "qty_per_100_meals": 6.0},
        {"ingredient_name": "vegetables", "unit": "kg", "qty_per_100_meals": 10.0},
        {"ingredient_name": "curd", "unit": "kg", "qty_per_100_meals": 4.5},
    ],
}


@dataclass(frozen=True)
class ForecastConfig:
    random_state: int = 42
    synthetic_period_days: int = 730
    holdout_days: int = 60
    min_training_rows: int = 120
    max_prediction_length: int = 7
    max_encoder_length: int = 30
    prediction_interval: float = 0.90
    service_level_target: float = 0.95
    waste_cost: float = 35.0
    shortage_cost: float = 55.0
    retrain_interval_hours: int = 24
    minimum_sigma: float = 28.0
    promotion_improvement_pct: float = 1.0
    rmse_tie_threshold_pct: float = 2.0
    dashboard_history_days: int = 90
    tft_max_epochs: int = 8
    tft_patience: int = 3
    tft_batch_size: int = 64
    tft_learning_rate: float = 0.03
    tft_hidden_size: int = 24
    tft_attention_heads: int = 4
    tft_dropout: float = 0.15
    tft_hidden_continuous_size: int = 12
    tft_gradient_clip_val: float = 0.1
    enable_cpu_tft: bool = False
    counterfactual_attendance_drop_pct: float = 0.20
    candidate_models: tuple[str, ...] = field(
        default=("random_forest", "xgboost", "lightgbm", "ridge", "ensemble", "tft")
    )


def ensure_directories() -> None:
    for directory in (
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        LOGS_DIR,
        METRICS_DIR,
        FIGURES_DIR,
        MODELS_DIR,
        ARTIFACTS_DIR,
        CHECKPOINTS_DIR,
        LOCAL_RUNTIME_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
