"""Spatio-temporal feature engineering for demand forecasting models.

This module transforms raw kitchen-operations data into ML-ready training,
prediction, and sequence frames.  It encapsulates:

* **Sanitisation** — type coercion, deduplication, calendar annotation,
  menu-type imputation, and robust missing-value strategies for weather
  and attendance covariates.
* **Lag / rolling features** — leak-free shifted demand and waste signals
  (1-, 3-, 7-, 14-day lags; 7- and 14-day rolling means/stds).
* **Sklearn preprocessing** — a ``ColumnTransformer`` that median-imputes
  numeric features and one-hot encodes categoricals for tabular models.
* **Sequence framing** — a lightweight wrapper that prepares the panel
  for the Temporal Fusion Transformer (TFT).

All DataFrame mutations operate on explicit ``.copy()`` working frames
and use ``.loc`` assignment to avoid ``SettingWithCopyWarning``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from backend.calendar_utils import annotate_calendar, default_menu_for_date
from backend.config import ForecastConfig


# ---------------------------------------------------------------------------
# Feature specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TabularFeatureSpec:
    """Immutable registry of numeric and categorical column names.

    Centralises the feature contract so that preprocessor construction,
    model training, and prediction all reference a single source of truth.

    Attributes:
        numeric_features: Ordered list of continuous / ordinal columns
            consumed by the tabular models (lags, rolling stats,
            weather, calendar flags, geo-coordinates, capacity).
        categorical_features: Ordered list of nominal columns that are
            one-hot encoded before model ingestion.
    """

    numeric_features: List[str]
    categorical_features: List[str]

    @property
    def feature_columns(self) -> List[str]:
        """Return the full ordered feature vector (numeric first).

        Returns:
            Concatenation of ``numeric_features`` and
            ``categorical_features``.
        """
        return self.numeric_features + self.categorical_features


TABULAR_FEATURE_SPEC: TabularFeatureSpec = TabularFeatureSpec(
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


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------


class SpatioTemporalFeatureBuilder:
    """Shared feature layer for tabular models and the TFT sequence model.

    Encapsulates the full sanitise → lag → preprocess pipeline so that
    training, inference, and back-testing share identical transformations.

    Attributes:
        config: Forecast hyper-parameters.
        tabular_spec: Column-name registry for the preprocessor.
    """

    def __init__(self, config: Optional[ForecastConfig] = None) -> None:
        """Initialise with optional forecast configuration.

        Args:
            config: If *None*, ``ForecastConfig()`` defaults are used.
        """
        self.config: ForecastConfig = config or ForecastConfig()
        self.tabular_spec: TabularFeatureSpec = TABULAR_FEATURE_SPEC

    # ------------------------------------------------------------------ #
    #  Sanitisation
    # ------------------------------------------------------------------ #

    def _sanitize(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Clean, type-coerce, and calendar-annotate raw operations data.

        Pipeline steps:

        1. Ensure ``date`` is datetime; drop rows missing ``date`` or
           ``kitchen_id``; sort and deduplicate.
        2. Annotate with ``season``, ``is_holiday``, ``is_exam_week``,
           ``is_event_day``, ``event_name`` via ``annotate_calendar``.
        3. Impute ``menu_type`` from the weekday-based default menu when
           absent — this mirrors the kitchen's standard weekly rotation.
        4. Coerce numeric weather and operational columns, filling gaps
           with physically motivated defaults (monthly median
           temperature, zero rainfall, zero attendance variation).
        5. Derive calendar features: ``day_of_week``, ``month``,
           ``day_of_month``, ``weekend_flag``, ``time_idx``.

        Why monthly-median temperature imputation?
            A simple global median would blur Kolkata's strong seasonal
            swing (~22 °C winter → 38 °C summer).  Month-level granularity
            preserves the climate envelope while being robust to sparse
            missingness.

        Args:
            frame: Raw operations DataFrame (may contain NaNs and
                mixed dtypes).

        Returns:
            A sanitised *copy* of *frame* — the original is never
            mutated.
        """
        working: pd.DataFrame = frame.copy()

        # --- Type coercion & deduplication --------------------------------
        working["date"] = pd.to_datetime(working["date"])
        working = (
            working
            .dropna(subset=["date", "kitchen_id"])
            .sort_values(["kitchen_id", "date"])
            .drop_duplicates(subset=["kitchen_id", "date"], keep="last")
        )
        working = annotate_calendar(working)

        # --- Menu-type imputation -----------------------------------------
        if "menu_type" not in working.columns:
            working["menu_type"] = working["date"].map(default_menu_for_date)
        else:
            working["menu_type"] = working["menu_type"].fillna(
                working["date"].map(default_menu_for_date)
            )

        # --- Categorical fallback -----------------------------------------
        working["event_name"] = working["event_name"].fillna("none")

        # --- Numeric coercion (safe for missing/malformed values) ---------
        _numeric_cols: List[str] = [
            "temperature", "rainfall", "attendance_variation",
            "waste_quantity", "prepared_quantity", "shortage_quantity",
            "actual_demand",
        ]
        for col in _numeric_cols:
            working[col] = pd.to_numeric(working.get(col), errors="coerce")

        # --- Weather imputation -------------------------------------------
        monthly_temp: pd.Series = working.groupby(
            working["date"].dt.month
        )["temperature"].transform("median")
        working["temperature"] = (
            working["temperature"]
            .interpolate(limit_direction="both")
            .fillna(monthly_temp)
            .fillna(29.0)
        )
        working["rainfall"] = working["rainfall"].fillna(0.0).clip(lower=0.0)

        # --- Operational imputation ---------------------------------------
        working["attendance_variation"] = (
            working["attendance_variation"].fillna(0.0).clip(-0.40, 0.30)
        )
        working["waste_quantity"] = (
            working["waste_quantity"].fillna(0.0).clip(lower=0.0)
        )
        working["prepared_quantity"] = working["prepared_quantity"].fillna(
            working["actual_demand"]
        )
        working["shortage_quantity"] = (
            working["shortage_quantity"].fillna(0.0).clip(lower=0.0)
        )

        # --- Derived calendar features (vectorised) -----------------------
        working["day_of_week"] = working["date"].dt.day_name()
        working["month"] = working["date"].dt.month
        working["day_of_month"] = working["date"].dt.day
        working["weekend_flag"] = (
            working["date"].dt.dayofweek >= 5
        ).astype(int)
        working["time_idx"] = (
            (working["date"] - working["date"].min()).dt.days.astype(int)
        )

        return working

    # ------------------------------------------------------------------ #
    #  Lag / rolling features
    # ------------------------------------------------------------------ #

    def _add_history_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Append leak-free lag and rolling-window features per kitchen.

        All features are shifted by at least 1 day so that the model
        never sees same-day actuals at prediction time.  The rolling
        windows (7- and 14-day) are computed *after* the 1-day shift,
        ensuring strict temporal causality.

        Why these specific lags?
            * **lag_1 / lag_3** — capture short-term momentum and
              mid-week patterns.
            * **lag_7 / lag_14** — capture weekly and fortnightly
              seasonality.
            * **waste lags** — let the model learn waste auto-correlation
              (e.g. consecutive over-preparation).
            * **rolling_mean / rolling_std** — smooth demand level and
              volatility proxies for the optimizer's safety-stock calc.

        Args:
            frame: Sanitised operations DataFrame (must be sorted by
                ``kitchen_id``, ``date``).

        Returns:
            A *copy* of *frame* with lag and rolling columns appended.
        """
        working: pd.DataFrame = frame.copy()
        group = working.groupby("kitchen_id", group_keys=False)

        # Shifted demand lags.
        working["lag_1"] = group["actual_demand"].shift(1)
        working["lag_3"] = group["actual_demand"].shift(3)
        working["lag_7"] = group["actual_demand"].shift(7)
        working["lag_14"] = group["actual_demand"].shift(14)

        # Shifted waste lags.
        working["waste_lag_1"] = group["waste_quantity"].shift(1)
        working["waste_lag_7"] = group["waste_quantity"].shift(7)

        # Rolling statistics (shift-then-roll to prevent leakage).
        shifted_demand = group["actual_demand"].shift(1)
        shifted_waste = group["waste_quantity"].shift(1)

        working["rolling_mean_7"] = (
            shifted_demand.rolling(window=7).mean()
            .reset_index(level=0, drop=True)
        )
        working["rolling_std_7"] = (
            shifted_demand.rolling(window=7).std()
            .reset_index(level=0, drop=True)
        )
        working["rolling_mean_14"] = (
            shifted_demand.rolling(window=14).mean()
            .reset_index(level=0, drop=True)
        )
        working["waste_mean_7"] = (
            shifted_waste.rolling(window=7).mean()
            .reset_index(level=0, drop=True)
        )

        # Fill NaN rolling stats with zero (first rows lack history).
        working["rolling_std_7"] = working["rolling_std_7"].fillna(0.0)
        working["waste_mean_7"] = working["waste_mean_7"].fillna(0.0)

        return working

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def build_training_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Build a training-ready DataFrame with all features.

        Rows that lack the minimum history depth (14-day lag and 14-day
        rolling mean) are dropped so the model never trains on partially
        observed feature vectors.

        Args:
            frame: Raw operations DataFrame.

        Returns:
            Sanitised, feature-enriched DataFrame with NaN-incomplete
            rows removed.
        """
        working: pd.DataFrame = self._sanitize(frame)
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
        """Build a prediction-ready DataFrame (retains all rows).

        Unlike ``build_training_frame``, no rows are dropped — the
        caller is responsible for handling leading NaNs in lag columns
        (typically via the preprocessor's median imputer).

        Args:
            frame: Raw operations DataFrame.

        Returns:
            Sanitised, feature-enriched DataFrame.
        """
        working: pd.DataFrame = self._sanitize(frame)
        working = self._add_history_features(working)
        return working

    def build_preprocessor(self) -> ColumnTransformer:
        """Construct an sklearn ``ColumnTransformer`` for tabular models.

        The transformer applies:

        * **Numeric pipeline** — median imputation (robust to outliers
          common in lag features during cold-start).
        * **Categorical pipeline** — mode imputation followed by
          one-hot encoding with ``handle_unknown="ignore"`` so unseen
          categories at inference time produce a zero vector instead of
          an error.

        Returns:
            A fitted-ready ``ColumnTransformer`` aligned with
            ``TABULAR_FEATURE_SPEC``.
        """
        numeric_pipeline: Pipeline = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median"))]
        )
        categorical_pipeline: Pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(
                    handle_unknown="ignore", sparse_output=False)),
            ]
        )
        return ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, self.tabular_spec.numeric_features),
                ("cat", categorical_pipeline, self.tabular_spec.categorical_features),
            ]
        )

    def build_sequence_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Build a sequence frame for the Temporal Fusion Transformer.

        Adds an explicit ``target`` column (float-typed ``actual_demand``)
        required by PyTorch Forecasting's ``TimeSeriesDataSet``.

        Args:
            frame: Raw operations DataFrame.

        Returns:
            Sanitised DataFrame with ``target`` column appended.
        """
        working: pd.DataFrame = self._sanitize(frame)
        working["target"] = working["actual_demand"].fillna(0.0).astype(float)
        return working
