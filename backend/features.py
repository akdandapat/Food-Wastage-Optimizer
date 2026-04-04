from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from backend.calendar_utils import annotate_calendar, default_menu_for_date
from backend.config import ForecastConfig


@dataclass(frozen=True)
class TabularFeatureSpec:
    numeric_features: list[str]
    categorical_features: list[str]

    @property
    def feature_columns(self) -> list[str]:
        return self.numeric_features + self.categorical_features


TABULAR_FEATURE_SPEC = TabularFeatureSpec(
    numeric_features=[
        "month",
        "day_of_month",
        "weekend_flag",
        "temperature",
        "rainfall",
        "is_holiday",
        "is_exam_week",
        "is_event_day",
        "attendance_variation",
        "capacity",
        "latitude",
        "longitude",
        "lag_1",
        "lag_3",
        "lag_7",
        "lag_14",
        "waste_lag_1",
        "waste_lag_7",
        "rolling_mean_7",
        "rolling_std_7",
        "rolling_mean_14",
        "waste_mean_7",
        "time_idx",
    ],
    categorical_features=[
        "kitchen_id",
        "campus_zone",
        "capacity_band",
        "default_attendance_band",
        "day_of_week",
        "season",
        "menu_type",
        "event_name",
    ],
)


class SpatioTemporalFeatureBuilder:
    """Shared feature layer for tabular models and the TFT sequence model."""

    def __init__(self, config: ForecastConfig | None = None) -> None:
        self.config = config or ForecastConfig()
        self.tabular_spec = TABULAR_FEATURE_SPEC

    def _sanitize(self, frame: pd.DataFrame) -> pd.DataFrame:
        working = frame.copy()
        working["date"] = pd.to_datetime(working["date"])
        working = working.dropna(subset=["date", "kitchen_id"]).sort_values(["kitchen_id", "date"])
        working = working.drop_duplicates(subset=["kitchen_id", "date"], keep="last")
        working = annotate_calendar(working)

        if "menu_type" not in working.columns:
            working["menu_type"] = working["date"].map(default_menu_for_date)
        else:
            working["menu_type"] = working["menu_type"].fillna(
                working["date"].map(default_menu_for_date)
            )

        working["event_name"] = working["event_name"].fillna("none")
        working["temperature"] = pd.to_numeric(working.get("temperature"), errors="coerce")
        working["rainfall"] = pd.to_numeric(working.get("rainfall"), errors="coerce")
        working["attendance_variation"] = pd.to_numeric(
            working.get("attendance_variation"), errors="coerce"
        )
        working["waste_quantity"] = pd.to_numeric(working.get("waste_quantity"), errors="coerce")
        working["prepared_quantity"] = pd.to_numeric(
            working.get("prepared_quantity"), errors="coerce"
        )
        working["shortage_quantity"] = pd.to_numeric(
            working.get("shortage_quantity"), errors="coerce"
        )
        working["actual_demand"] = pd.to_numeric(working.get("actual_demand"), errors="coerce")

        monthly_temp = working.groupby(working["date"].dt.month)["temperature"].transform("median")
        working["temperature"] = (
            working["temperature"].interpolate(limit_direction="both").fillna(monthly_temp).fillna(29.0)
        )
        working["rainfall"] = working["rainfall"].fillna(0.0).clip(lower=0.0)
        working["attendance_variation"] = working["attendance_variation"].fillna(0.0).clip(-0.40, 0.30)
        working["waste_quantity"] = working["waste_quantity"].fillna(0.0).clip(lower=0.0)
        working["prepared_quantity"] = working["prepared_quantity"].fillna(working["actual_demand"])
        working["shortage_quantity"] = working["shortage_quantity"].fillna(0.0).clip(lower=0.0)

        working["day_of_week"] = working["date"].dt.day_name()
        working["month"] = working["date"].dt.month
        working["day_of_month"] = working["date"].dt.day
        working["weekend_flag"] = (working["date"].dt.dayofweek >= 5).astype(int)
        working["time_idx"] = (working["date"] - working["date"].min()).dt.days.astype(int)
        return working

    def _add_history_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        working = frame.copy()
        group = working.groupby("kitchen_id", group_keys=False)

        # Shifted historical demand and waste features preserve leak-free training.
        working["lag_1"] = group["actual_demand"].shift(1)
        working["lag_3"] = group["actual_demand"].shift(3)
        working["lag_7"] = group["actual_demand"].shift(7)
        working["lag_14"] = group["actual_demand"].shift(14)
        working["waste_lag_1"] = group["waste_quantity"].shift(1)
        working["waste_lag_7"] = group["waste_quantity"].shift(7)
        working["rolling_mean_7"] = group["actual_demand"].shift(1).rolling(window=7).mean().reset_index(level=0, drop=True)
        working["rolling_std_7"] = (
            group["actual_demand"].shift(1).rolling(window=7).std().reset_index(level=0, drop=True)
        )
        working["rolling_mean_14"] = group["actual_demand"].shift(1).rolling(window=14).mean().reset_index(level=0, drop=True)
        working["waste_mean_7"] = group["waste_quantity"].shift(1).rolling(window=7).mean().reset_index(level=0, drop=True)

        working["rolling_std_7"] = working["rolling_std_7"].fillna(0.0)
        working["waste_mean_7"] = working["waste_mean_7"].fillna(0.0)
        return working

    def build_training_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        working = self._sanitize(frame)
        working = self._add_history_features(working)
        return working.dropna(
            subset=[
                "actual_demand",
                "lag_14",
                "rolling_mean_14",
                "lag_7",
                "rolling_mean_7",
            ]
        ).reset_index(drop=True)

    def build_prediction_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        working = self._sanitize(frame)
        working = self._add_history_features(working)
        return working

    def build_preprocessor(self) -> ColumnTransformer:
        numeric_pipeline = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median"))]
        )
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        return ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, self.tabular_spec.numeric_features),
                ("cat", categorical_pipeline, self.tabular_spec.categorical_features),
            ]
        )

    def build_sequence_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        working = self._sanitize(frame)
        working["target"] = working["actual_demand"].fillna(0.0).astype(float)
        return working
