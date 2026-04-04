from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from math import sqrt
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

from backend.calibration import build_interval_calibration_payload
from backend.calendar_utils import annotate_calendar, default_menu_for_date
from backend.config import (
    CHECKPOINTS_DIR,
    FIGURES_DIR,
    PLOT_FILENAMES,
    PROCESSED_OPERATIONS_FILE,
    ForecastConfig,
)
from backend.database import SQLiteRepository, initialize_database
from backend.explainability import build_why_summary, explain_tabular_instance, tft_attention_summary
from backend.features import SpatioTemporalFeatureBuilder
from backend.optimizer import NewsvendorOptimizer
from backend.storage import (
    FORECAST_HISTORY_FILE,
    MODEL_COMPARISON_FILE,
    MONITORING_LOG_FILE,
    load_dataframe,
    load_model_bundle,
    load_summary_metrics,
    save_model_bundle,
    save_training_artifacts,
)
from backend.visualization import generate_plots

try:
    import lightgbm as lgb
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover
    lgb = None
    LGBMRegressor = None

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None

try:
    import torch
    from lightning.pytorch import Trainer, seed_everything
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer
    from pytorch_forecasting.metrics import QuantileLoss
except Exception:  # pragma: no cover
    torch = None
    Trainer = None
    seed_everything = None
    EarlyStopping = None
    ModelCheckpoint = None
    CSVLogger = None
    TemporalFusionTransformer = None
    TimeSeriesDataSet = None
    GroupNormalizer = None
    QuantileLoss = None


@dataclass
class CandidateResult:
    model_name: str
    model_version: str
    rmse: float
    mae: float
    weekly_rmse: float
    weekly_mae: float
    interval_coverage: float
    residual_std: float
    mean_prediction_jump: float
    notes: str = ""
    selected_model: bool = False
    promoted: bool = False
    improvement_pct: float = 0.0

    def to_record(self, run_id: str, trained_at: str) -> dict[str, Any]:
        payload = asdict(self)
        payload["run_id"] = run_id
        payload["trained_at"] = trained_at
        return payload


class KitchenForecastSystem:
    """Training, model selection, forecasting, and feedback loop for kitchen demand."""

    def __init__(
        self,
        repository: SQLiteRepository | None = None,
        config: ForecastConfig | None = None,
    ) -> None:
        self.config = config or ForecastConfig()
        initialize_database()
        self.repository = repository or SQLiteRepository()
        self.feature_builder = SpatioTemporalFeatureBuilder(self.config)
        self.optimizer = NewsvendorOptimizer(
            waste_cost=self.config.waste_cost,
            shortage_cost=self.config.shortage_cost,
            service_level_target=self.config.service_level_target,
        )
        self.z_interval = float(norm.ppf(0.5 + self.config.prediction_interval / 2))
        self.tft_quantiles = [0.1, 0.5, 0.9]
        self.tft_sigma_z = float(norm.ppf(0.9))

    def ensure_seed_data(self) -> None:
        """
        Prefer merged public-ingestion panel (weather + holidays + demand base),
        then fall back to synthetic operations aligned to cached weather if present.
        """
        if self.repository.observation_count() > 0:
            return

        if PROCESSED_OPERATIONS_FILE.exists():
            panel = pd.read_csv(PROCESSED_OPERATIONS_FILE, parse_dates=["date"])
            self.repository.upsert_observations(panel)
            if self.repository.observation_count() > 0:
                return

        try:
            from backend.data_ingestion import run_full_ingestion

            run_full_ingestion(self.config)
            if PROCESSED_OPERATIONS_FILE.exists():
                panel = pd.read_csv(PROCESSED_OPERATIONS_FILE, parse_dates=["date"])
                self.repository.upsert_observations(panel)
                if self.repository.observation_count() > 0:
                    return
        except Exception:
            pass

        kitchens = self.repository.list_kitchens()
        from backend.config import RAW_DATA_DIR
        from backend.data_generation import generate_synthetic_operations

        weather_csv = None
        weather_file = RAW_DATA_DIR / "kolkata_weather_daily.csv"
        if weather_file.exists():
            try:
                weather_csv = pd.read_csv(weather_file, parse_dates=["date"])
            except Exception:
                weather_csv = None

        synthetic = generate_synthetic_operations(
            kitchens, self.config, weather_override=weather_csv
        )
        self.repository.upsert_observations(synthetic)

    def _metric_payload(self, actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
        return (
            float(sqrt(mean_squared_error(actual, predicted))),
            float(mean_absolute_error(actual, predicted)),
        )

    def _date_folds(
        self, frame: pd.DataFrame, validation_days: int = 14
    ) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        unique_dates = pd.Index(sorted(pd.to_datetime(frame["date"]).dt.normalize().unique()))
        if len(unique_dates) < self.config.min_training_rows:
            return []
        max_folds = min(2, max((len(unique_dates) - 120) // validation_days, 1))
        start_idx = len(unique_dates) - max_folds * validation_days
        folds: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        for fold in range(max_folds):
            train_end_idx = start_idx + fold * validation_days
            if train_end_idx <= 90:
                continue
            folds.append(
                (
                    pd.Timestamp(unique_dates[train_end_idx - 1]),
                    pd.Timestamp(unique_dates[min(train_end_idx + validation_days - 1, len(unique_dates) - 1)]),
                )
            )
        return folds

    def _rf_param_grid(self) -> list[dict[str, Any]]:
        return [
            {"n_estimators": 120, "max_depth": 10, "min_samples_leaf": 2, "max_features": 0.8},
            {"n_estimators": 180, "max_depth": 14, "min_samples_leaf": 2, "max_features": 0.9},
        ]

    def _xgb_param_grid(self) -> list[dict[str, Any]]:
        return [
            {
                "n_estimators": 180,
                "learning_rate": 0.05,
                "max_depth": 6,
                "min_child_weight": 3,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "reg_alpha": 0.0,
                "reg_lambda": 1.2,
            },
            {
                "n_estimators": 240,
                "learning_rate": 0.035,
                "max_depth": 7,
                "min_child_weight": 5,
                "subsample": 0.90,
                "colsample_bytree": 0.80,
                "reg_alpha": 0.05,
                "reg_lambda": 1.0,
            },
        ]

    def _lgbm_param_grid(self) -> list[dict[str, Any]]:
        return [
            {
                "n_estimators": 220,
                "learning_rate": 0.05,
                "num_leaves": 50,
                "max_depth": 16,
                "min_child_samples": 25,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
            },
            {
                "n_estimators": 300,
                "learning_rate": 0.035,
                "num_leaves": 64,
                "max_depth": 18,
                "min_child_samples": 30,
                "subsample": 0.90,
                "colsample_bytree": 0.80,
                "reg_alpha": 0.05,
                "reg_lambda": 1.2,
            },
        ]

    def _ridge_param_grid(self) -> list[dict[str, Any]]:
        """Ridge alphas on one-hot expanded features (strong regularization)."""
        return [
            {"alpha": 1.0},
            {"alpha": 4.0},
            {"alpha": 16.0},
            {"alpha": 64.0},
        ]

    def _tft_param_grid(self) -> list[dict[str, Any]]:
        return [
            {
                "hidden_size": self.config.tft_hidden_size,
                "attention_head_size": self.config.tft_attention_heads,
                "dropout": self.config.tft_dropout,
                "hidden_continuous_size": self.config.tft_hidden_continuous_size,
                "learning_rate": self.config.tft_learning_rate,
            },
            {
                "hidden_size": 32,
                "attention_head_size": 4,
                "dropout": 0.20,
                "hidden_continuous_size": 16,
                "learning_rate": 0.02,
            },
        ]

    def _instantiate_tabular_model(self, model_name: str, params: dict[str, Any]) -> Any:
        if model_name == "random_forest":
            return RandomForestRegressor(
                random_state=self.config.random_state,
                n_jobs=-1,
                **params,
            )
        if model_name == "xgboost" and XGBRegressor is not None:
            return XGBRegressor(
                objective="reg:squarederror",
                tree_method="hist",
                random_state=self.config.random_state,
                n_jobs=0,
                **params,
            )
        if model_name == "lightgbm" and LGBMRegressor is not None:
            return LGBMRegressor(
                objective="regression",
                random_state=self.config.random_state,
                n_jobs=-1,
                **params,
            )
        if model_name == "ridge":
            return Ridge(**params)
        raise RuntimeError(f"Unsupported or unavailable model: {model_name}")

    def _fit_tabular_model(
        self,
        model_name: str,
        train_frame: pd.DataFrame,
        params: dict[str, Any],
        validation_frame: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        preprocessor = self.feature_builder.build_preprocessor()
        x_train = train_frame[self.feature_builder.tabular_spec.feature_columns]
        y_train = train_frame["actual_demand"].to_numpy()
        x_train_matrix = preprocessor.fit_transform(x_train)
        model = self._instantiate_tabular_model(model_name, params)

        if model_name == "lightgbm" and validation_frame is not None and lgb is not None:
            x_val = validation_frame[self.feature_builder.tabular_spec.feature_columns]
            y_val = validation_frame["actual_demand"].to_numpy()
            x_val_matrix = preprocessor.transform(x_val)
            model.fit(
                x_train_matrix,
                y_train,
                eval_set=[(x_val_matrix, y_val)],
                eval_metric="l2",
                callbacks=[lgb.early_stopping(30, verbose=False)],
            )
        else:
            model.fit(x_train_matrix, y_train)

        train_predictions = model.predict(x_train_matrix)
        return {
            "model_name": model_name,
            "params": params,
            "preprocessor": preprocessor,
            "model": model,
            "feature_columns": self.feature_builder.tabular_spec.feature_columns,
            "residual_std": float(
                max(np.std(y_train - train_predictions), self.config.minimum_sigma)
            ),
        }

    def _predict_tabular_row(self, artifact: dict[str, Any], row_frame: pd.DataFrame) -> float:
        latest = self.feature_builder.build_prediction_frame(row_frame).tail(1)
        matrix = artifact["preprocessor"].transform(
            latest[self.feature_builder.tabular_spec.feature_columns]
        )
        return float(artifact["model"].predict(matrix)[0])

    def _tune_tabular_candidate(self, model_name: str, frame: pd.DataFrame) -> tuple[dict[str, Any], float]:
        if model_name == "xgboost" and XGBRegressor is None:
            raise RuntimeError("xgboost is not installed.")
        if model_name == "lightgbm" and LGBMRegressor is None:
            raise RuntimeError("lightgbm is not installed.")

        grid = {
            "random_forest": self._rf_param_grid(),
            "xgboost": self._xgb_param_grid(),
            "lightgbm": self._lgbm_param_grid(),
            "ridge": self._ridge_param_grid(),
        }[model_name]

        folds = self._date_folds(frame)
        if not folds:
            return grid[0], 0.0

        best_params = grid[0]
        best_mae = float("inf")
        for params in grid:
            fold_maes: list[float] = []
            for train_end_date, val_end_date in folds:
                train_subset = frame[frame["date"] <= train_end_date]
                validation_subset = frame[
                    (frame["date"] > train_end_date) & (frame["date"] <= val_end_date)
                ]
                if validation_subset.empty:
                    continue
                artifact = self._fit_tabular_model(
                    model_name,
                    train_subset,
                    params,
                    validation_frame=validation_subset,
                )
                predictions = artifact["model"].predict(
                    artifact["preprocessor"].transform(
                        validation_subset[self.feature_builder.tabular_spec.feature_columns]
                    )
                )
                fold_maes.append(
                    float(mean_absolute_error(validation_subset["actual_demand"].to_numpy(), predictions))
                )
            if fold_maes and np.mean(fold_maes) < best_mae:
                best_mae = float(np.mean(fold_maes))
                best_params = params
        return best_params, best_mae

    def _split_holdout(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
        unique_dates = pd.Index(sorted(pd.to_datetime(frame["date"]).dt.normalize().unique()))
        holdout_days = min(self.config.holdout_days, max(len(unique_dates) // 6, 30))
        holdout_start = pd.Timestamp(unique_dates[-holdout_days])
        return frame[frame["date"] < holdout_start].copy(), holdout_start

    def _recursive_tabular_forecasts(
        self,
        artifact: dict[str, Any],
        observations: pd.DataFrame,
        start_date: pd.Timestamp,
        future_context: pd.DataFrame,
    ) -> pd.DataFrame:
        history = observations[observations["date"] < start_date].copy()
        forecasts: list[dict[str, Any]] = []
        for target_date in sorted(pd.to_datetime(future_context["date"]).unique()):
            day_context = future_context[future_context["date"] == target_date].copy()
            for context_row in day_context.to_dict("records"):
                kitchen_history = history[history["kitchen_id"] == context_row["kitchen_id"]]
                future_stub = pd.DataFrame([context_row])
                future_stub["actual_demand"] = np.nan
                future_stub["prepared_quantity"] = np.nan
                future_stub["waste_quantity"] = np.nan
                future_stub["shortage_quantity"] = np.nan
                future_stub["predicted_demand"] = np.nan
                future_stub["selected_model"] = None
                row_frame = pd.concat([kitchen_history, future_stub], ignore_index=True, sort=False)
                prediction = self._predict_tabular_row(artifact, row_frame)
                sigma_scale = float(artifact.get("sigma_scale_factor", 1.0))
                sigma = float(
                    max(artifact["residual_std"] * sigma_scale, self.config.minimum_sigma)
                )
                lower = prediction - self.z_interval * sigma
                upper = prediction + self.z_interval * sigma
                forecasts.append(
                    {
                        "kitchen_id": context_row["kitchen_id"],
                        "date": pd.Timestamp(target_date).normalize(),
                        "predicted_demand": prediction,
                        "lower_bound": lower,
                        "upper_bound": upper,
                        "sigma": sigma,
                        "menu_type": context_row["menu_type"],
                    }
                )
                appended_row = future_stub.copy()
                appended_row["actual_demand"] = prediction
                history = pd.concat([history, appended_row], ignore_index=True, sort=False)
        return pd.DataFrame(forecasts)

    def _evaluate_tabular_candidate(
        self,
        artifact: dict[str, Any],
        full_observations: pd.DataFrame,
        holdout_start: pd.Timestamp,
        model_name: str,
    ) -> tuple[CandidateResult, pd.DataFrame, pd.DataFrame]:
        holdout_context = full_observations[full_observations["date"] >= holdout_start].copy()
        one_step = self._recursive_tabular_forecasts(
            artifact,
            full_observations,
            holdout_start,
            holdout_context,
        )
        actual_holdout = holdout_context[
            [
                "kitchen_id",
                "date",
                "actual_demand",
                "prepared_quantity",
                "waste_quantity",
                "shortage_quantity",
                "menu_type",
            ]
        ].copy()
        actual_holdout["date"] = pd.to_datetime(actual_holdout["date"]).dt.normalize()
        evaluation = one_step.merge(actual_holdout, on=["kitchen_id", "date"], how="left")
        evaluation["menu_type"] = evaluation["menu_type_x"].fillna(evaluation["menu_type_y"])
        evaluation = evaluation.drop(columns=["menu_type_x", "menu_type_y"])
        rmse, mae = self._metric_payload(
            evaluation["actual_demand"].to_numpy(), evaluation["predicted_demand"].to_numpy()
        )
        interval_coverage = float(
            np.mean(
                (evaluation["actual_demand"] >= evaluation["lower_bound"])
                & (evaluation["actual_demand"] <= evaluation["upper_bound"])
            )
        )
        residual_std = float(
            max(np.std(evaluation["actual_demand"] - evaluation["predicted_demand"]), self.config.minimum_sigma)
        )
        jump = (
            evaluation.sort_values(["kitchen_id", "date"])
            .groupby("kitchen_id")["predicted_demand"]
            .diff()
            .abs()
            .fillna(0.0)
            .mean()
        )

        weekly_eval = evaluation.sort_values(["kitchen_id", "date"]).copy()
        weekly_eval["weekly_actual"] = (
            weekly_eval.groupby("kitchen_id")["actual_demand"]
            .rolling(window=self.config.max_prediction_length, min_periods=self.config.max_prediction_length)
            .sum()
            .reset_index(level=0, drop=True)
        )
        weekly_eval["weekly_predicted"] = (
            weekly_eval.groupby("kitchen_id")["predicted_demand"]
            .rolling(window=self.config.max_prediction_length, min_periods=self.config.max_prediction_length)
            .sum()
            .reset_index(level=0, drop=True)
        )
        weekly_eval = weekly_eval.dropna(subset=["weekly_actual", "weekly_predicted"])
        if weekly_eval.empty:
            weekly_rmse, weekly_mae = rmse, mae
        else:
            weekly_rmse, weekly_mae = self._metric_payload(
                weekly_eval["weekly_actual"].to_numpy(),
                weekly_eval["weekly_predicted"].to_numpy(),
            )
        evaluation = self._attach_operational_metrics(evaluation)

        result = CandidateResult(
            model_name=model_name,
            model_version=f"{model_name}-{datetime.utcnow().isoformat()}",
            rmse=rmse,
            mae=mae,
            weekly_rmse=weekly_rmse,
            weekly_mae=weekly_mae,
            interval_coverage=interval_coverage,
            residual_std=residual_std,
            mean_prediction_jump=float(jump),
        )
        return result, evaluation, weekly_eval

    def _combine_ensemble(
        self,
        forecasts_by_model: dict[str, pd.DataFrame],
        weights: dict[str, float],
    ) -> pd.DataFrame:
        merged = None
        for model_name, frame in forecasts_by_model.items():
            renamed = frame.rename(
                columns={
                    "predicted_demand": f"{model_name}_pred",
                    "lower_bound": f"{model_name}_lower",
                    "upper_bound": f"{model_name}_upper",
                    "sigma": f"{model_name}_sigma",
                }
            )
            keep_columns = [
                "kitchen_id",
                "date",
                "menu_type",
                f"{model_name}_pred",
                f"{model_name}_lower",
                f"{model_name}_upper",
                f"{model_name}_sigma",
            ]
            merged = renamed[keep_columns] if merged is None else merged.merge(
                renamed[keep_columns], on=["kitchen_id", "date", "menu_type"], how="inner"
            )

        if merged is None:
            return pd.DataFrame()

        merged["predicted_demand"] = 0.0
        merged["sigma"] = 0.0
        for model_name, weight in weights.items():
            merged["predicted_demand"] += merged[f"{model_name}_pred"] * weight
            merged["sigma"] += merged[f"{model_name}_sigma"] * weight

        merged["lower_bound"] = merged["predicted_demand"] - self.z_interval * merged["sigma"]
        merged["upper_bound"] = merged["predicted_demand"] + self.z_interval * merged["sigma"]
        return merged[["kitchen_id", "date", "menu_type", "predicted_demand", "lower_bound", "upper_bound", "sigma"]]

    def _build_tft_dataset(
        self, frame: pd.DataFrame, predict_mode: bool = False, min_prediction_idx: int | None = None
    ) -> TimeSeriesDataSet:
        dataset_kwargs = {
            "time_idx": "time_idx",
            "target": "target",
            "group_ids": ["kitchen_id"],
            "max_encoder_length": self.config.max_encoder_length,
            "min_encoder_length": self.config.max_encoder_length // 2,
            "max_prediction_length": self.config.max_prediction_length,
            "static_categoricals": ["kitchen_id", "campus_zone", "capacity_band", "default_attendance_band"],
            "static_reals": ["capacity", "latitude", "longitude"],
            "time_varying_known_categoricals": ["day_of_week", "season", "menu_type", "event_name"],
            "time_varying_known_reals": [
                "time_idx",
                "month",
                "day_of_month",
                "weekend_flag",
                "temperature",
                "rainfall",
                "is_holiday",
                "is_exam_week",
                "is_event_day",
                "attendance_variation",
            ],
            "time_varying_unknown_reals": ["target", "waste_quantity"],
            "target_normalizer": GroupNormalizer(groups=["kitchen_id"], transformation="softplus"),
            "add_relative_time_idx": True,
            "add_target_scales": True,
            "add_encoder_length": True,
            "allow_missing_timesteps": False,
            "predict_mode": predict_mode,
        }
        if min_prediction_idx is not None:
            dataset_kwargs["min_prediction_idx"] = min_prediction_idx
        return TimeSeriesDataSet(frame, **dataset_kwargs)

    def _fit_tft_candidate(
        self, sequence_frame: pd.DataFrame, params: dict[str, Any], model_version: str
    ) -> dict[str, Any]:
        if TemporalFusionTransformer is None or torch is None:
            raise RuntimeError("TFT dependencies are not installed.")

        seed_everything(self.config.random_state, workers=True)
        sequence_frame = sequence_frame.sort_values(["kitchen_id", "time_idx"]).reset_index(drop=True)
        unique_dates = pd.Index(sorted(sequence_frame["date"].dt.normalize().unique()))
        validation_days = min(30, max(self.config.max_prediction_length * 3, 14))
        validation_start = pd.Timestamp(unique_dates[-validation_days])
        validation_idx = int(sequence_frame.loc[sequence_frame["date"] == validation_start, "time_idx"].min())

        training_source = sequence_frame[sequence_frame["date"] < validation_start].copy()
        training_dataset = self._build_tft_dataset(training_source, predict_mode=False)
        validation_dataset = TimeSeriesDataSet.from_dataset(
            training_dataset,
            sequence_frame,
            min_prediction_idx=validation_idx,
            stop_randomization=True,
        )

        train_loader = training_dataset.to_dataloader(
            train=True,
            batch_size=self.config.tft_batch_size,
            num_workers=0,
        )
        val_loader = validation_dataset.to_dataloader(
            train=False,
            batch_size=self.config.tft_batch_size,
            num_workers=0,
        )

        checkpoint_dir = CHECKPOINTS_DIR / model_version
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="{epoch}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        )
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=self.config.tft_patience,
            mode="min",
        )
        trainer = Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            max_epochs=self.config.tft_max_epochs,
            gradient_clip_val=self.config.tft_gradient_clip_val,
            logger=CSVLogger("data/logs", name="tft", version=model_version),
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[checkpoint_callback, early_stopping],
        )
        model = TemporalFusionTransformer.from_dataset(
            training_dataset,
            learning_rate=params["learning_rate"],
            hidden_size=params["hidden_size"],
            attention_head_size=params["attention_head_size"],
            dropout=params["dropout"],
            hidden_continuous_size=params["hidden_continuous_size"],
            loss=QuantileLoss(self.tft_quantiles),
            output_size=len(self.tft_quantiles),
            log_interval=-1,
            reduce_on_plateau_patience=2,
        )
        trainer.fit(model, train_loader, val_loader)
        best_model = TemporalFusionTransformer.load_from_checkpoint(checkpoint_callback.best_model_path)
        return {
            "model_name": "tft",
            "model": best_model,
            "checkpoint_path": checkpoint_callback.best_model_path,
            "dataset_parameters": training_dataset.get_parameters(),
            "params": params,
        }

    def _tune_tft(self, sequence_frame: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
        best_params = self._tft_param_grid()[0]
        best_artifact: dict[str, Any] | None = None
        best_score = float("inf")
        for index, params in enumerate(self._tft_param_grid()):
            model_version = f"tft-tune-{index}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
            artifact = self._fit_tft_candidate(sequence_frame, params, model_version)
            prediction_dataset = TimeSeriesDataSet.from_parameters(
                artifact["dataset_parameters"],
                sequence_frame,
                predict=True,
                stop_randomization=True,
            )
            raw_prediction = artifact["model"].predict(
                prediction_dataset,
                mode="quantiles",
                return_index=True,
                trainer_kwargs={
                    "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
                    "devices": 1,
                    "enable_progress_bar": False,
                    "logger": False,
                },
            )
            prediction_frame = self._prediction_object_to_frame(
                raw_prediction,
                sequence_frame,
                model_name="tft",
            )
            merged = prediction_frame.merge(
                sequence_frame[["kitchen_id", "date", "actual_demand"]],
                on=["kitchen_id", "date"],
                how="inner",
            )
            if merged.empty:
                continue
            score = float(mean_absolute_error(merged["actual_demand"], merged["predicted_demand"]))
            if score < best_score:
                best_score = score
                best_params = params
                best_artifact = artifact
        if best_artifact is None:
            raise RuntimeError("TFT tuning did not produce a valid artifact.")
        return best_params, best_artifact

    def _prediction_object_to_frame(
        self, prediction_object: Any, reference_frame: pd.DataFrame, model_name: str
    ) -> pd.DataFrame:
        quantiles = prediction_object.output.detach().cpu().numpy()
        index_frame = prediction_object.index.copy()
        origin = reference_frame["date"].min()
        index_frame["prediction_start"] = pd.to_datetime(index_frame["time_idx"], unit="D", origin=origin)
        records: list[dict[str, Any]] = []
        for row_idx, row in enumerate(index_frame.itertuples(index=False)):
            for horizon_idx in range(min(quantiles.shape[1], self.config.max_prediction_length)):
                forecast_date = (pd.Timestamp(row.prediction_start) + timedelta(days=horizon_idx)).normalize()
                records.append(
                    {
                        "kitchen_id": row.kitchen_id,
                        "date": forecast_date,
                        "predicted_demand": float(quantiles[row_idx, horizon_idx, 1]),
                        "lower_bound": float(quantiles[row_idx, horizon_idx, 0]),
                        "upper_bound": float(quantiles[row_idx, horizon_idx, 2]),
                        "sigma": float(
                            max(
                                (quantiles[row_idx, horizon_idx, 2] - quantiles[row_idx, horizon_idx, 0])
                                / (2 * self.tft_sigma_z),
                                self.config.minimum_sigma,
                            )
                        ),
                        "model_name": model_name,
                    }
                )
        return pd.DataFrame(records)

    def _evaluate_tft_candidate(
        self,
        artifact: dict[str, Any],
        full_observations: pd.DataFrame,
        holdout_start: pd.Timestamp,
    ) -> tuple[CandidateResult, pd.DataFrame]:
        holdout_context = full_observations[full_observations["date"] >= holdout_start].copy()
        anchor_dates = sorted(pd.to_datetime(holdout_context["date"]).dt.normalize().unique())
        weekly_predictions: list[dict[str, Any]] = []
        for anchor_date in anchor_dates:
            window_context = holdout_context[
                (holdout_context["date"] >= anchor_date)
                & (holdout_context["date"] < anchor_date + timedelta(days=self.config.max_prediction_length))
            ].copy()
            if window_context.empty:
                continue
            history = full_observations[full_observations["date"] < anchor_date].copy()
            future_stub = window_context.copy()
            future_stub["actual_demand"] = 0.0
            future_stub["prepared_quantity"] = np.nan
            future_stub["shortage_quantity"] = np.nan
            future_stub["waste_quantity"] = 0.0
            combined = pd.concat([history, future_stub], ignore_index=True, sort=False)
            sequence_frame = self.feature_builder.build_sequence_frame(combined)
            prediction_dataset = TimeSeriesDataSet.from_parameters(
                artifact["dataset_parameters"],
                sequence_frame,
                predict=True,
                stop_randomization=True,
            )
            prediction = artifact["model"].predict(
                prediction_dataset,
                mode="quantiles",
                return_index=True,
                trainer_kwargs={
                    "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
                    "devices": 1,
                    "enable_progress_bar": False,
                    "logger": False,
                },
            )
            prediction_frame = self._prediction_object_to_frame(
                prediction,
                sequence_frame,
                model_name="tft",
            )
            weekly_predictions.extend(prediction_frame.to_dict("records"))

        weekly_frame = pd.DataFrame(weekly_predictions).drop_duplicates(
            subset=["kitchen_id", "date"], keep="last"
        )
        actual_holdout = holdout_context[
            [
                "kitchen_id",
                "date",
                "actual_demand",
                "prepared_quantity",
                "waste_quantity",
                "shortage_quantity",
                "menu_type",
            ]
        ].copy()
        actual_holdout["date"] = pd.to_datetime(actual_holdout["date"]).dt.normalize()
        if not weekly_frame.empty:
            weekly_frame["date"] = pd.to_datetime(weekly_frame["date"]).dt.normalize()
        evaluation = weekly_frame.merge(actual_holdout, on=["kitchen_id", "date"], how="inner")
        next_day_eval = (
            evaluation.sort_values(["kitchen_id", "date"])
            .groupby("kitchen_id")
            .head(len(anchor_dates))
        )
        rmse, mae = self._metric_payload(
            next_day_eval["actual_demand"].to_numpy(),
            next_day_eval["predicted_demand"].to_numpy(),
        )
        weekly_rmse, weekly_mae = self._metric_payload(
            evaluation["actual_demand"].to_numpy(),
            evaluation["predicted_demand"].to_numpy(),
        )
        interval_coverage = float(
            np.mean(
                (evaluation["actual_demand"] >= evaluation["lower_bound"])
                & (evaluation["actual_demand"] <= evaluation["upper_bound"])
            )
        )
        residual_std = float(
            max(np.std(next_day_eval["actual_demand"] - next_day_eval["predicted_demand"]), self.config.minimum_sigma)
        )
        jump = (
            next_day_eval.sort_values(["kitchen_id", "date"])
            .groupby("kitchen_id")["predicted_demand"]
            .diff()
            .abs()
            .fillna(0.0)
            .mean()
        )
        result = CandidateResult(
            model_name="tft",
            model_version=f"tft-{datetime.utcnow().isoformat()}",
            rmse=rmse,
            mae=mae,
            weekly_rmse=weekly_rmse,
            weekly_mae=weekly_mae,
            interval_coverage=interval_coverage,
            residual_std=residual_std,
            mean_prediction_jump=float(jump),
        )
        return result, self._attach_operational_metrics(evaluation)

    def _select_winner(self, results: list[CandidateResult]) -> CandidateResult:
        ranked = sorted(results, key=lambda item: item.rmse)
        best = ranked[0]
        for candidate in ranked[1:]:
            threshold = best.rmse * (1 + self.config.rmse_tie_threshold_pct / 100)
            if candidate.rmse > threshold:
                break
            if candidate.residual_std < best.residual_std:
                best = candidate
            elif (
                np.isclose(candidate.residual_std, best.residual_std)
                and candidate.mean_prediction_jump < best.mean_prediction_jump
            ):
                best = candidate
        return best

    def _promotion_decision(
        self, winner: CandidateResult, incumbent: dict | None
    ) -> tuple[bool, float, str]:
        if incumbent is None:
            return True, 100.0, "No incumbent model was registered."
        incumbent_rmse = float(incumbent["next_day_rmse"])
        improvement_pct = 100.0 * (incumbent_rmse - winner.rmse) / max(incumbent_rmse, 1e-6)
        if improvement_pct >= self.config.promotion_improvement_pct:
            return True, improvement_pct, "Challenger cleared the promotion threshold."
        return False, improvement_pct, "Incumbent retained because improvement threshold was not met."

    def _build_summary_metrics(
        self,
        run_id: str,
        trained_at: str,
        active_model: str,
        active_version: str,
        results: list[CandidateResult],
        promotion_reason: str,
        promoted: bool,
        evaluation_frame: pd.DataFrame,
    ) -> dict[str, Any]:
        baseline_result = next(result for result in results if result.model_name == "random_forest")
        active_result = next(result for result in results if result.model_name == active_model)
        waste_delta = 0.0
        daily_savings = max((baseline_result.rmse - active_result.rmse) * self.config.waste_cost, 0.0)
        optimized_waste_pct = 0.0
        interval_coverage = active_result.interval_coverage
        before_after_table = self._build_before_after_table(evaluation_frame)
        service_level_metrics = self._build_service_level_metrics(evaluation_frame)
        coverage_metrics = self._build_coverage_metrics(active_result)
        if not evaluation_frame.empty and "baseline_waste_realized" in evaluation_frame.columns:
            baseline_waste = evaluation_frame["baseline_waste_realized"].sum()
            optimized_waste = evaluation_frame["optimized_waste_realized"].sum()
            waste_delta = 100.0 * (baseline_waste - optimized_waste) / max(baseline_waste, 1.0)
            daily_savings = float(evaluation_frame["daily_cost_saving_inr"].mean())
            optimized_waste_pct = 100.0 * optimized_waste / max(
                evaluation_frame["optimal_quantity"].sum(),
                1.0,
            )

        return {
            "run_id": run_id,
            "trained_at": trained_at,
            "current_model": active_model,
            "selected_model_version": active_version,
            "promotion_reason": promotion_reason,
            "promoted": promoted,
            "business_metrics": {
                "waste_reduction_pct": float(waste_delta),
                "daily_cost_savings_inr": float(daily_savings),
                "annual_savings_inr": float(daily_savings * 365),
                "optimized_waste_pct": float(optimized_waste_pct),
                "prediction_interval_coverage": float(interval_coverage),
                "before_after_table": before_after_table,
                "coverage_metrics": coverage_metrics,
                "service_level_metrics": service_level_metrics,
            },
        }

    def _safe_pct(self, numerator: float, denominator: float) -> float:
        return float(100.0 * numerator / max(denominator, 1e-6))

    def _build_before_after_table(self, evaluation_frame: pd.DataFrame) -> list[dict[str, Any]]:
        if evaluation_frame.empty or "baseline_quantity" not in evaluation_frame.columns:
            return []

        baseline_rmse, _ = self._metric_payload(
            evaluation_frame["actual_demand"].to_numpy(),
            evaluation_frame["baseline_quantity"].to_numpy(),
        )
        optimized_rmse, _ = self._metric_payload(
            evaluation_frame["actual_demand"].to_numpy(),
            evaluation_frame["predicted_demand"].to_numpy(),
        )
        total_actual = float(evaluation_frame["actual_demand"].sum())
        return [
            {
                "metric": "Avg Waste",
                "before_value": float(evaluation_frame["baseline_waste_realized"].mean()),
                "after_value": float(evaluation_frame["optimized_waste_realized"].mean()),
                "unit": "meals/day",
            },
            {
                "metric": "RMSE",
                "before_value": float(baseline_rmse),
                "after_value": float(optimized_rmse),
                "unit": "meals",
            },
            {
                "metric": "Cost",
                "before_value": float(evaluation_frame["baseline_cost_inr"].mean()),
                "after_value": float(evaluation_frame["optimized_cost_inr"].mean()),
                "unit": "INR/day",
            },
            {
                "metric": "Shortage %",
                "before_value": self._safe_pct(
                    float(evaluation_frame["baseline_shortage_realized"].sum()),
                    total_actual,
                ),
                "after_value": self._safe_pct(
                    float(evaluation_frame["optimized_shortage_realized"].sum()),
                    total_actual,
                ),
                "unit": "%",
            },
        ]

    def _build_coverage_metrics(self, active_result: CandidateResult) -> dict[str, float]:
        expected_coverage_pct = float(self.config.prediction_interval * 100.0)
        actual_coverage_pct = float(active_result.interval_coverage * 100.0)
        return {
            "expected_coverage_pct": expected_coverage_pct,
            "actual_coverage_pct": actual_coverage_pct,
            "calibration_gap_pct": float(actual_coverage_pct - expected_coverage_pct),
        }

    def _build_service_level_metrics(self, evaluation_frame: pd.DataFrame) -> dict[str, float]:
        if evaluation_frame.empty or "shortage_probability" not in evaluation_frame.columns:
            return {
                "target_service_level_pct": float(self.config.service_level_target * 100.0),
                "target_max_shortage_probability_pct": float((1.0 - self.config.service_level_target) * 100.0),
                "planned_shortage_probability_pct": float((1.0 - self.config.service_level_target) * 100.0),
                "realized_shortage_rate_pct": 0.0,
            }

        total_actual = float(evaluation_frame["actual_demand"].sum())
        return {
            "target_service_level_pct": float(self.config.service_level_target * 100.0),
            "target_max_shortage_probability_pct": float((1.0 - self.config.service_level_target) * 100.0),
            "planned_shortage_probability_pct": float(evaluation_frame["shortage_probability"].mean() * 100.0),
            "realized_shortage_rate_pct": self._safe_pct(
                float(evaluation_frame["optimized_shortage_realized"].sum()),
                total_actual,
            ),
        }

    def _attach_operational_metrics(self, evaluation: pd.DataFrame) -> pd.DataFrame:
        if evaluation.empty:
            return evaluation

        enriched = evaluation.copy()
        optimized_records: list[dict[str, float]] = []
        baseline_records: list[dict[str, float]] = []
        for row in enriched.itertuples(index=False):
            sigma = float(max(getattr(row, "sigma", self.config.minimum_sigma), self.config.minimum_sigma))
            recommendation = self.optimizer.recommend(float(row.predicted_demand), sigma)
            baseline_quantity = float(self._heuristic_quantity(float(row.predicted_demand)))
            baseline_expected = self.optimizer.evaluate_quantity(
                float(row.predicted_demand),
                sigma,
                baseline_quantity,
            )
            optimized_realized = self.optimizer.realized_metrics(
                recommendation.optimal_quantity,
                float(row.actual_demand),
            )
            baseline_realized = self.optimizer.realized_metrics(
                baseline_quantity,
                float(row.actual_demand),
            )
            optimized_records.append(
                {
                    "optimal_quantity": recommendation.optimal_quantity,
                    "expected_waste": recommendation.expected_waste,
                    "expected_shortage": recommendation.expected_shortage,
                    "expected_cost_inr": recommendation.expected_cost_inr,
                    "critical_ratio": recommendation.critical_ratio,
                    "shortage_probability": recommendation.shortage_probability,
                    "service_level_target": recommendation.service_level_target,
                    "service_level_satisfied": recommendation.service_level_satisfied,
                    "optimized_waste_realized": optimized_realized["waste"],
                    "optimized_shortage_realized": optimized_realized["shortage"],
                    "optimized_cost_inr": optimized_realized["cost_inr"],
                }
            )
            baseline_records.append(
                {
                    "baseline_quantity": baseline_quantity,
                    "baseline_expected_waste": baseline_expected.expected_waste,
                    "baseline_expected_shortage": baseline_expected.expected_shortage,
                    "baseline_expected_cost_inr": baseline_expected.expected_cost_inr,
                    "baseline_shortage_probability": baseline_expected.shortage_probability,
                    "baseline_waste_realized": baseline_realized["waste"],
                    "baseline_shortage_realized": baseline_realized["shortage"],
                    "baseline_cost_inr": baseline_realized["cost_inr"],
                }
            )

        enriched = pd.concat(
            [
                enriched.reset_index(drop=True),
                pd.DataFrame(optimized_records),
                pd.DataFrame(baseline_records),
            ],
            axis=1,
        )
        enriched["daily_cost_saving_inr"] = (
            enriched["baseline_cost_inr"] - enriched["optimized_cost_inr"]
        )
        return enriched

    def _feature_names_from_artifact(self, artifact: dict[str, Any]) -> list[str]:
        names = artifact["preprocessor"].get_feature_names_out(
            artifact["feature_columns"]
        )
        return [value.split("__", 1)[-1] for value in names]

    def _build_feature_importance_frame(
        self,
        selected_model: str,
        bundle: dict[str, Any],
    ) -> pd.DataFrame:
        if selected_model == "ensemble":
            weighted_frames: list[pd.DataFrame] = []
            for model_name, weight in bundle.get("ensemble_weights", {}).items():
                artifact = bundle.get("tabular_models", {}).get(model_name)
                if artifact is None:
                    continue
                model = artifact["model"]
                names = self._feature_names_from_artifact(artifact)
                if model_name == "lightgbm":
                    booster = getattr(model, "booster_", None)
                    if booster is None:
                        continue
                    importance = booster.feature_importance(importance_type="gain")
                elif hasattr(model, "feature_importances_"):
                    importance = np.asarray(model.feature_importances_)
                else:
                    continue
                frame = pd.DataFrame({"feature": names, "importance": importance * float(weight)})
                weighted_frames.append(frame)

            if weighted_frames:
                return (
                    pd.concat(weighted_frames, ignore_index=True)
                    .groupby("feature", as_index=False)["importance"]
                    .sum()
                    .sort_values("importance", ascending=False)
                    .reset_index(drop=True)
                )

        if selected_model == "ridge" and selected_model in bundle.get("tabular_models", {}):
            artifact = bundle["tabular_models"][selected_model]
            model = artifact["model"]
            names = self._feature_names_from_artifact(artifact)
            coef = np.abs(np.asarray(model.coef_).ravel())
            return (
                pd.DataFrame({"feature": names, "importance": coef})
                .sort_values("importance", ascending=False)
                .reset_index(drop=True)
            )

        if selected_model == "lightgbm" and selected_model in bundle.get("tabular_models", {}):
            artifact = bundle["tabular_models"][selected_model]
            model = artifact["model"]
            booster = getattr(model, "booster_", None)
            if booster is not None:
                importance = booster.feature_importance(importance_type="gain")
                names = self._feature_names_from_artifact(artifact)
                return (
                    pd.DataFrame({"feature": names, "importance": importance})
                    .sort_values("importance", ascending=False)
                    .reset_index(drop=True)
                )

        if selected_model in bundle.get("tabular_models", {}):
            artifact = bundle["tabular_models"][selected_model]
            model = artifact["model"]
            if hasattr(model, "feature_importances_"):
                names = self._feature_names_from_artifact(artifact)
                return (
                    pd.DataFrame(
                        {"feature": names, "importance": np.asarray(model.feature_importances_)}
                    )
                    .sort_values("importance", ascending=False)
                    .reset_index(drop=True)
                )

        return pd.DataFrame({"feature": ["no_feature_importance"], "importance": [1.0]})

    def _group_feature_importance(self, feature_importance: pd.DataFrame) -> pd.DataFrame:
        if feature_importance.empty:
            return pd.DataFrame(columns=["driver", "importance"])

        def driver_group(feature_name: str) -> str:
            if feature_name.startswith("lag_") or feature_name.startswith("rolling_") or feature_name.startswith("waste_lag"):
                return "lag_demand_and_history"
            if feature_name.startswith("day_of_week") or feature_name in {"month", "day_of_month", "weekend_flag", "season"}:
                return "calendar_features"
            if feature_name.startswith("menu_type"):
                return "menu_type"
            if feature_name in {"temperature", "rainfall"}:
                return "weather"
            if feature_name in {"is_holiday", "is_exam_week", "is_event_day", "event_name"}:
                return "academic_and_event_flags"
            if feature_name.startswith("kitchen_id") or feature_name.startswith("campus_zone") or feature_name.startswith("capacity"):
                return "kitchen_profile"
            return "other"

        grouped = feature_importance.copy()
        grouped["driver"] = grouped["feature"].map(driver_group)
        return (
            grouped.groupby("driver", as_index=False)["importance"]
            .sum()
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def _top_drivers_payload(self, feature_importance: pd.DataFrame) -> list[dict[str, Any]]:
        grouped = self._group_feature_importance(feature_importance).head(5)
        return [
            {
                "driver": row.driver,
                "importance": float(row.importance),
            }
            for row in grouped.itertuples(index=False)
        ]

    def _heuristic_quantity(self, predicted_demand: float) -> int:
        return int(np.ceil(max(predicted_demand * 1.12, 0.0)))

    def _decision_strategy_payload(
        self,
        strategy_name: str,
        recommendation: Any,
    ) -> dict[str, Any]:
        return {
            "strategy_name": strategy_name,
            "quantity": int(recommendation.optimal_quantity),
            "expected_waste": float(recommendation.expected_waste),
            "expected_shortage": float(recommendation.expected_shortage),
            "expected_cost": float(recommendation.expected_cost_inr),
            "shortage_probability_pct": float(recommendation.shortage_probability * 100.0),
            "service_level_target_pct": float(recommendation.service_level_target * 100.0),
            "service_level_satisfied": bool(recommendation.service_level_satisfied),
            "critical_ratio": float(recommendation.critical_ratio),
        }

    def _build_counterfactual_scenarios(
        self,
        predicted_demand: float,
        sigma: float,
    ) -> list[dict[str, Any]]:
        attendance_drop = self.config.counterfactual_attendance_drop_pct
        scenarios = [
            ("normal", 1.0),
            ("attendance_down_20pct", 1.0 - attendance_drop),
        ]
        payloads: list[dict[str, Any]] = []
        for scenario_name, multiplier in scenarios:
            scenario_mu = float(predicted_demand * multiplier)
            scenario_sigma = float(max(sigma * multiplier, self.config.minimum_sigma))
            optimized = self.optimizer.recommend(scenario_mu, scenario_sigma)
            heuristic = self.optimizer.evaluate_quantity(
                scenario_mu,
                scenario_sigma,
                self._heuristic_quantity(scenario_mu),
            )
            payloads.append(
                {
                    "scenario_name": scenario_name,
                    "attendance_multiplier": float(multiplier),
                    "predicted_demand": scenario_mu,
                    "optimized_quantity": int(optimized.optimal_quantity),
                    "optimized_waste": float(optimized.expected_waste),
                    "optimized_cost": float(optimized.expected_cost_inr),
                    "heuristic_quantity": int(heuristic.optimal_quantity),
                    "heuristic_waste": float(heuristic.expected_waste),
                    "heuristic_cost": float(heuristic.expected_cost_inr),
                }
            )
        return payloads

    def _plot_urls(self) -> dict[str, str]:
        return {key: f"/artifacts/{filename}" for key, filename in PLOT_FILENAMES.items()}

    def _frame_records(self, frame: pd.DataFrame | None) -> list[dict[str, Any]]:
        if frame is None or frame.empty:
            return []
        return frame.replace({np.nan: None}).to_dict("records")

    def train_all_models(self) -> dict[str, Any]:
        self.ensure_seed_data()
        observations = self.repository.load_observations()
        training_frame = self.feature_builder.build_training_frame(observations)
        if training_frame.empty:
            raise RuntimeError("No valid training data is available.")

        if len(training_frame) < self.config.min_training_rows or training_frame["date"].nunique() < 45:
            params = self._rf_param_grid()[0]
            artifact = self._fit_tabular_model("random_forest", training_frame, params)
            predictions = artifact["model"].predict(
                artifact["preprocessor"].transform(
                    training_frame[self.feature_builder.tabular_spec.feature_columns]
                )
            )
            residual_std = float(
                max(
                    np.std(training_frame["actual_demand"].to_numpy() - predictions),
                    self.config.minimum_sigma,
                )
            )
            artifact["residual_std"] = residual_std
            model_version = f"random_forest-{datetime.utcnow().isoformat()}"
            artifact["model_version"] = model_version
            bundle = {
                "trained_at": datetime.utcnow().isoformat(),
                "selected_model": "random_forest",
                "model_version": model_version,
                "winner_reason": "Low-data fallback retained the random forest baseline.",
                "candidate_metrics": [],
                "ensemble_weights": {},
                "tabular_models": {"random_forest": artifact},
                "tft": None,
            }
            save_model_bundle(bundle)
            evaluation_frame = training_frame.tail(min(30, len(training_frame))).copy()
            evaluation_frame["predicted_demand"] = predictions[-len(evaluation_frame) :]
            evaluation_frame["sigma"] = residual_std
            evaluation_frame["lower_bound"] = (
                evaluation_frame["predicted_demand"] - self.z_interval * residual_std
            )
            evaluation_frame["upper_bound"] = (
                evaluation_frame["predicted_demand"] + self.z_interval * residual_std
            )
            evaluation_frame = self._attach_operational_metrics(evaluation_frame)
            rmse, mae = self._metric_payload(
                evaluation_frame["actual_demand"].to_numpy(),
                evaluation_frame["predicted_demand"].to_numpy(),
            )
            candidate_results = [
                CandidateResult(
                    model_name="random_forest",
                    model_version=model_version,
                    rmse=rmse,
                    mae=mae,
                    weekly_rmse=rmse,
                    weekly_mae=mae,
                    interval_coverage=float(
                        np.mean(
                            (evaluation_frame["actual_demand"] >= evaluation_frame["lower_bound"])
                            & (evaluation_frame["actual_demand"] <= evaluation_frame["upper_bound"])
                        )
                    ),
                    residual_std=residual_std,
                    mean_prediction_jump=float(
                        evaluation_frame["predicted_demand"].diff().abs().fillna(0.0).mean()
                    ),
                    notes="Low-data fallback path.",
                    selected_model=True,
                    promoted=True,
                    improvement_pct=100.0,
                )
            ]
            run_id = str(uuid4())
            trained_at = bundle["trained_at"]
            comparison_frame = pd.DataFrame(
                [candidate_results[0].to_record(run_id, trained_at)]
            )
            summary = self._build_summary_metrics(
                run_id,
                trained_at,
                "random_forest",
                model_version,
                candidate_results,
                bundle["winner_reason"],
                True,
                evaluation_frame,
            )
            monitoring_history = pd.DataFrame(
                [
                    {
                        "timestamp": trained_at,
                        "run_id": run_id,
                        "current_model": "random_forest",
                        "rmse": rmse,
                        "mae": mae,
                        "waste_reduction_pct": summary["business_metrics"]["waste_reduction_pct"],
                        "annual_savings_inr": summary["business_metrics"]["annual_savings_inr"],
                    }
                ]
            )
            save_training_artifacts(summary, comparison_frame, evaluation_frame, monitoring_history)
            self.repository.insert_training_runs(comparison_frame)
            self.repository.replace_model_registry(
                pd.DataFrame(
                    [
                        {
                            "model_name": "random_forest",
                            "model_version": model_version,
                            "artifact_path": str(Path("models/artifacts/production_bundle.joblib")),
                            "metadata_path": None,
                            "is_selected": 1,
                            "trained_at": trained_at,
                            "next_day_rmse": rmse,
                            "weekly_rmse": rmse,
                            "promotion_reason": bundle["winner_reason"],
                        }
                    ]
                )
            )
            feature_importance = self._build_feature_importance_frame("random_forest", bundle)
            generate_plots(evaluation_frame, comparison_frame, summary["business_metrics"], feature_importance)
            return {
                "status": "trained",
                "run_id": run_id,
                "trained_at": trained_at,
                "selected_model": "random_forest",
                "selected_model_version": model_version,
                "promotion_reason": bundle["winner_reason"],
                "promoted": True,
                "model_metrics": [asdict(candidate_results[0])],
                "business_metrics": summary["business_metrics"],
            }

        preholdout_frame, holdout_start = self._split_holdout(training_frame)
        candidate_results: list[CandidateResult] = []
        tabular_artifacts: dict[str, dict[str, Any]] = {}
        holdout_frames: dict[str, pd.DataFrame] = {}
        validation_maes: dict[str, float] = {}

        for model_name in ("random_forest", "xgboost", "lightgbm", "ridge"):
            try:
                params, validation_mae = self._tune_tabular_candidate(model_name, preholdout_frame)
                validation_maes[model_name] = validation_mae
                validation_dates = sorted(preholdout_frame["date"].unique())[-14:]
                validation_frame = preholdout_frame[preholdout_frame["date"].isin(validation_dates)]
                artifact = self._fit_tabular_model(
                    model_name,
                    preholdout_frame,
                    params,
                    validation_frame=validation_frame if not validation_frame.empty else None,
                )
                result, evaluation, _weekly = self._evaluate_tabular_candidate(
                    artifact,
                    observations,
                    holdout_start,
                    model_name,
                )
                artifact["residual_std"] = result.residual_std
                artifact["model_version"] = result.model_version
                tabular_artifacts[model_name] = artifact
                holdout_frames[model_name] = evaluation
                candidate_results.append(result)
            except Exception as exc:  # pragma: no cover
                candidate_results.append(
                    CandidateResult(
                        model_name=model_name,
                        model_version=f"{model_name}-failed",
                        rmse=1e9,
                        mae=1e9,
                        weekly_rmse=1e9,
                        weekly_mae=1e9,
                        interval_coverage=0.0,
                        residual_std=1e9,
                        mean_prediction_jump=1e9,
                        notes=str(exc),
                    )
                )

        ensemble_weights = {}
        valid_base_results = [
            result
            for result in candidate_results
            if result.model_name in {"random_forest", "xgboost", "lightgbm"}
            and result.rmse < 1e8
        ]
        if valid_base_results:
            inverse_mae = {
                result.model_name: 1.0 / max(validation_maes.get(result.model_name, result.mae), 1.0)
                for result in valid_base_results
            }
            total = sum(inverse_mae.values())
            ensemble_weights = {name: value / total for name, value in inverse_mae.items()}
            ensemble_frame = self._combine_ensemble(
                {name: holdout_frames[name] for name in ensemble_weights},
                ensemble_weights,
            )
            actual_holdout = observations[observations["date"] >= holdout_start][
                [
                    "kitchen_id",
                    "date",
                    "actual_demand",
                    "prepared_quantity",
                    "waste_quantity",
                    "shortage_quantity",
                    "menu_type",
                ]
            ].copy()
            actual_holdout["date"] = pd.to_datetime(actual_holdout["date"]).dt.normalize()
            ensemble_eval = ensemble_frame.merge(actual_holdout, on=["kitchen_id", "date"], how="inner")
            ensemble_eval["menu_type"] = ensemble_eval["menu_type_x"].fillna(ensemble_eval["menu_type_y"])
            ensemble_eval = ensemble_eval.drop(columns=["menu_type_x", "menu_type_y"])
            rmse, mae = self._metric_payload(
                ensemble_eval["actual_demand"].to_numpy(),
                ensemble_eval["predicted_demand"].to_numpy(),
            )
            interval_coverage = float(
                np.mean(
                    (ensemble_eval["actual_demand"] >= ensemble_eval["lower_bound"])
                    & (ensemble_eval["actual_demand"] <= ensemble_eval["upper_bound"])
                )
            )
            jump = (
                ensemble_eval.sort_values(["kitchen_id", "date"])
                .groupby("kitchen_id")["predicted_demand"]
                .diff()
                .abs()
                .fillna(0.0)
                .mean()
            )
            candidate_results.append(
                CandidateResult(
                    model_name="ensemble",
                    model_version=f"ensemble-{datetime.utcnow().isoformat()}",
                    rmse=rmse,
                    mae=mae,
                    weekly_rmse=rmse,
                    weekly_mae=mae,
                    interval_coverage=interval_coverage,
                    residual_std=float(max(ensemble_eval["sigma"].mean(), self.config.minimum_sigma)),
                    mean_prediction_jump=float(jump),
                )
            )
            holdout_frames["ensemble"] = self._attach_operational_metrics(ensemble_eval)

        tft_artifact = None
        tft_available = TemporalFusionTransformer is not None and torch is not None
        tft_allowed = tft_available and (
            self.config.enable_cpu_tft or torch.cuda.is_available()
        )
        if tft_allowed and len(preholdout_frame) >= 400:
            try:
                sequence_frame = self.feature_builder.build_sequence_frame(observations)
                _best_params, tft_artifact = self._tune_tft(
                    sequence_frame[sequence_frame["date"] < holdout_start].copy()
                )
                tft_result, tft_eval = self._evaluate_tft_candidate(
                    tft_artifact,
                    observations,
                    holdout_start,
                )
                candidate_results.append(tft_result)
                holdout_frames["tft"] = tft_eval
            except Exception as exc:  # pragma: no cover
                candidate_results.append(
                    CandidateResult(
                        model_name="tft",
                        model_version="tft-failed",
                        rmse=1e9,
                        mae=1e9,
                        weekly_rmse=1e9,
                        weekly_mae=1e9,
                        interval_coverage=0.0,
                        residual_std=1e9,
                        mean_prediction_jump=1e9,
                        notes=str(exc),
                    )
                )
        elif not tft_allowed:
            candidate_results.append(
                CandidateResult(
                    model_name="tft",
                    model_version="tft-skipped",
                    rmse=1e9,
                    mae=1e9,
                    weekly_rmse=1e9,
                    weekly_mae=1e9,
                    interval_coverage=0.0,
                    residual_std=1e9,
                    mean_prediction_jump=1e9,
                    notes="TFT is enabled only when CUDA is available or enable_cpu_tft is set.",
                )
            )

        winner = self._select_winner(candidate_results)
        incumbent = self.repository.get_selected_model_registry()
        promoted, improvement_pct, promotion_reason = self._promotion_decision(winner, incumbent)
        run_id = str(uuid4())
        trained_at = datetime.utcnow().isoformat()

        for result in candidate_results:
            result.selected_model = result.model_name == winner.model_name
            result.promoted = promoted and result.model_name == winner.model_name
            result.improvement_pct = improvement_pct if result.model_name == winner.model_name else 0.0

        if promoted:
            full_frame = self.feature_builder.build_training_frame(observations)
            final_tabular_artifacts = {}
            eval_for_calibration = holdout_frames.get(winner.model_name, pd.DataFrame())
            interval_calibration = build_interval_calibration_payload(
                eval_for_calibration,
                self.z_interval,
                winner.residual_std,
                self.config.prediction_interval,
                winner.model_name,
            )
            tabular_sigma_scale = float(interval_calibration.get("tabular_sigma_scale", 1.0))

            for model_name in ("random_forest", "xgboost", "lightgbm", "ridge"):
                if model_name in tabular_artifacts:
                    final_artifact = self._fit_tabular_model(
                        model_name,
                        full_frame,
                        tabular_artifacts[model_name]["params"],
                    )
                    final_artifact["residual_std"] = next(
                        result.residual_std for result in candidate_results if result.model_name == model_name
                    )
                    final_artifact["model_version"] = next(
                        result.model_version for result in candidate_results if result.model_name == model_name
                    )
                    final_artifact["sigma_scale_factor"] = tabular_sigma_scale
                    final_tabular_artifacts[model_name] = final_artifact

            if tft_artifact is not None:
                tft_artifact = dict(tft_artifact)
                tft_artifact["tft_quantile_scale"] = float(
                    interval_calibration.get("tft_quantile_scale", 1.0)
                )

            bundle = {
                "trained_at": trained_at,
                "selected_model": winner.model_name,
                "model_version": winner.model_version,
                "winner_reason": promotion_reason,
                "candidate_metrics": [asdict(result) for result in candidate_results],
                "ensemble_weights": ensemble_weights,
                "tabular_models": final_tabular_artifacts,
                "tft": tft_artifact,
                "interval_calibration": interval_calibration,
            }
            save_model_bundle(bundle)

            registry = pd.DataFrame(
                [
                    {
                        "model_name": result.model_name,
                        "model_version": result.model_version,
                        "artifact_path": str(Path("models/artifacts/production_bundle.joblib")),
                        "metadata_path": None,
                        "is_selected": int(result.model_name == winner.model_name),
                        "trained_at": trained_at,
                        "next_day_rmse": result.rmse,
                        "weekly_rmse": result.weekly_rmse,
                        "promotion_reason": promotion_reason if result.model_name == winner.model_name else "",
                    }
                    for result in candidate_results
                ]
            )
            self.repository.replace_model_registry(registry)

        selected_registry = self.repository.get_selected_model_registry()
        active_model = selected_registry["model_name"] if selected_registry is not None else winner.model_name
        active_version = (
            selected_registry["model_version"] if selected_registry is not None else winner.model_version
        )

        comparison_frame = pd.DataFrame([result.to_record(run_id, trained_at) for result in candidate_results])
        evaluation_frame = holdout_frames.get(active_model)
        if evaluation_frame is None:
            evaluation_frame = holdout_frames.get(winner.model_name, pd.DataFrame())

        existing_monitoring = load_dataframe(MONITORING_LOG_FILE)
        monitoring_history = existing_monitoring if existing_monitoring is not None else pd.DataFrame()
        monitoring_row = pd.DataFrame(
            [
                {
                    "timestamp": trained_at,
                    "run_id": run_id,
                    "current_model": active_model,
                    "rmse": next(result.rmse for result in candidate_results if result.model_name == active_model),
                    "mae": next(result.mae for result in candidate_results if result.model_name == active_model),
                    "waste_reduction_pct": 0.0,
                    "annual_savings_inr": 0.0,
                }
            ]
        )
        monitoring_history = pd.concat([monitoring_history, monitoring_row], ignore_index=True)

        summary = self._build_summary_metrics(
            run_id,
            trained_at,
            active_model,
            active_version,
            candidate_results,
            promotion_reason,
            promoted,
            evaluation_frame,
        )
        monitoring_history.loc[monitoring_history.index[-1], "waste_reduction_pct"] = summary["business_metrics"]["waste_reduction_pct"]
        monitoring_history.loc[monitoring_history.index[-1], "annual_savings_inr"] = summary["business_metrics"]["annual_savings_inr"]

        save_training_artifacts(summary, comparison_frame, evaluation_frame, monitoring_history)
        self.repository.insert_training_runs(comparison_frame)
        active_bundle = load_model_bundle()
        if active_bundle is not None:
            feature_importance = self._build_feature_importance_frame(active_model, active_bundle)
            generate_plots(
                evaluation_frame,
                comparison_frame,
                summary["business_metrics"],
                feature_importance,
            )
        return {
            "status": "trained",
            "run_id": run_id,
            "trained_at": trained_at,
            "selected_model": active_model,
            "selected_model_version": active_version,
            "promotion_reason": promotion_reason,
            "promoted": promoted,
            "model_metrics": [asdict(result) for result in candidate_results],
            "business_metrics": summary["business_metrics"],
        }

    def _normalize_future_context(
        self,
        kitchen_id: str,
        forecast_start_date: pd.Timestamp,
        horizon_days: int,
        future_context: list[dict[str, Any]],
        history: pd.DataFrame,
    ) -> pd.DataFrame:
        kitchen_history = history[history["kitchen_id"] == kitchen_id].copy()
        if kitchen_history.empty:
            raise ValueError(f"Unknown kitchen_id '{kitchen_id}'.")
        kitchen_meta = kitchen_history.iloc[-1].to_dict()
        date_index = pd.date_range(forecast_start_date, periods=horizon_days, freq="D")
        context_frame = pd.DataFrame({"date": date_index})
        provided = pd.DataFrame(future_context)
        if not provided.empty:
            provided["date"] = pd.to_datetime(provided["date"])
            context_frame = context_frame.merge(provided, on="date", how="left")
        context_frame = annotate_calendar(context_frame)

        monthly_stats = kitchen_history.groupby(kitchen_history["date"].dt.month)[
            ["temperature", "rainfall", "attendance_variation"]
        ].median(numeric_only=True)
        records: list[dict[str, Any]] = []
        for row in context_frame.itertuples(index=False):
            month_defaults = monthly_stats.loc[row.date.month] if row.date.month in monthly_stats.index else None
            records.append(
                {
                    "kitchen_id": kitchen_id,
                    "date": row.date,
                    "menu_type": getattr(row, "menu_type", None) or default_menu_for_date(row.date),
                    "temperature": float(
                        getattr(row, "temperature", np.nan)
                        if pd.notna(getattr(row, "temperature", np.nan))
                        else (month_defaults["temperature"] if month_defaults is not None else 29.0)
                    ),
                    "rainfall": float(
                        getattr(row, "rainfall", np.nan)
                        if pd.notna(getattr(row, "rainfall", np.nan))
                        else (month_defaults["rainfall"] if month_defaults is not None else 0.0)
                    ),
                    "attendance_variation": float(
                        getattr(row, "attendance_variation", np.nan)
                        if pd.notna(getattr(row, "attendance_variation", np.nan))
                        else (month_defaults["attendance_variation"] if month_defaults is not None else 0.0)
                    ),
                    "is_holiday": int(
                        getattr(row, "is_holiday", None)
                        if getattr(row, "is_holiday", None) is not None
                        else row.is_holiday
                    ),
                    "is_exam_week": int(
                        getattr(row, "is_exam_week", None)
                        if getattr(row, "is_exam_week", None) is not None
                        else row.is_exam_week
                    ),
                    "is_event_day": int(
                        getattr(row, "is_event_day", None)
                        if getattr(row, "is_event_day", None) is not None
                        else row.is_event_day
                    ),
                    "event_name": getattr(row, "event_name", None) or row.event_name or "none",
                    "hostel_name": kitchen_meta["hostel_name"],
                    "campus_zone": kitchen_meta["campus_zone"],
                    "latitude": kitchen_meta["latitude"],
                    "longitude": kitchen_meta["longitude"],
                    "capacity": kitchen_meta["capacity"],
                    "capacity_band": kitchen_meta["capacity_band"],
                    "default_attendance_band": kitchen_meta["default_attendance_band"],
                    "prepared_quantity": np.nan,
                    "waste_quantity": np.nan,
                    "shortage_quantity": np.nan,
                    "predicted_demand": np.nan,
                    "selected_model": None,
                    "data_source": "forecast",
                }
            )
        return pd.DataFrame(records)

    def _forecast_with_bundle(
        self,
        bundle: dict[str, Any],
        kitchen_id: str,
        forecast_start_date: pd.Timestamp,
        horizon_days: int,
        future_context: list[dict[str, Any]],
        context_frame: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        observations = self.repository.load_observations()
        history = observations[observations["date"] < forecast_start_date].copy()
        if context_frame is None:
            context_frame = self._normalize_future_context(
                kitchen_id,
                forecast_start_date,
                horizon_days,
                future_context,
                observations,
            )

        if bundle["selected_model"] == "ensemble":
            component_predictions = {}
            for model_name, artifact in bundle["tabular_models"].items():
                component_predictions[model_name] = self._recursive_tabular_forecasts(
                    artifact,
                    history,
                    forecast_start_date,
                    context_frame,
                )
            forecast_frame = self._combine_ensemble(component_predictions, bundle["ensemble_weights"])
        elif bundle["selected_model"] == "tft" and bundle.get("tft") is not None:
            tft_artifact = bundle["tft"]
            combined = pd.concat(
                [history, context_frame.assign(actual_demand=np.nan, waste_quantity=0.0)],
                ignore_index=True,
                sort=False,
            )
            sequence_frame = self.feature_builder.build_sequence_frame(combined)
            prediction_dataset = TimeSeriesDataSet.from_parameters(
                tft_artifact["dataset_parameters"],
                sequence_frame,
                predict=True,
                stop_randomization=True,
            )
            prediction = tft_artifact["model"].predict(
                prediction_dataset,
                mode="quantiles",
                return_index=True,
                trainer_kwargs={
                    "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
                    "devices": 1,
                    "enable_progress_bar": False,
                    "logger": False,
                },
            )
            forecast_frame = self._prediction_object_to_frame(prediction, sequence_frame, model_name="tft")
            forecast_frame = forecast_frame.merge(
                context_frame[["date", "menu_type"]].assign(
                    date=lambda frame: pd.to_datetime(frame["date"]).dt.normalize()
                ),
                on="date",
                how="inner",
            )
            qscale = float(
                tft_artifact.get("tft_quantile_scale")
                or (bundle.get("interval_calibration") or {}).get("tft_quantile_scale")
                or 1.0
            )
            med = forecast_frame["predicted_demand"].to_numpy(dtype=float)
            lo = forecast_frame["lower_bound"].to_numpy(dtype=float)
            hi = forecast_frame["upper_bound"].to_numpy(dtype=float)
            forecast_frame["lower_bound"] = med - (med - lo) * qscale
            forecast_frame["upper_bound"] = med + (hi - med) * qscale
            forecast_frame["sigma"] = np.maximum(
                (forecast_frame["upper_bound"] - forecast_frame["lower_bound"])
                / (2.0 * self.tft_sigma_z),
                self.config.minimum_sigma,
            )
        else:
            selected_artifact = bundle["tabular_models"][bundle["selected_model"]]
            forecast_frame = self._recursive_tabular_forecasts(
                selected_artifact,
                history,
                forecast_start_date,
                context_frame,
            )

        return (
            forecast_frame[
                pd.to_datetime(forecast_frame["date"]) >= pd.Timestamp(forecast_start_date)
            ]
            .sort_values("date")
            .head(horizon_days)
            .reset_index(drop=True)
        )

    def _build_prediction_explanation(
        self,
        bundle: dict[str, Any],
        kitchen_id: str,
        history: pd.DataFrame,
        context_frame: pd.DataFrame,
    ) -> dict[str, Any]:
        """Local SHAP / Ridge attributions plus TFT attention hook for the first horizon day."""
        try:
            first_day = context_frame.iloc[:1].copy()
            row_frame = pd.concat([history[history["kitchen_id"] == kitchen_id], first_day], ignore_index=True)
            row_frame = self.feature_builder.build_prediction_frame(row_frame).tail(1)
            selected = bundle["selected_model"]
            if selected == "ensemble":
                member = max(bundle.get("ensemble_weights", {}).items(), key=lambda item: item[1])[0]
                artifact = bundle["tabular_models"][member]
                local = explain_tabular_instance(artifact, row_frame, member)
                method_note = f"ensemble_primary_component:{member}"
            elif selected in bundle.get("tabular_models", {}):
                artifact = bundle["tabular_models"][selected]
                local = explain_tabular_instance(artifact, row_frame, selected)
                method_note = local["method"]
            else:
                return {
                    "explanation_method": "tft_or_unknown",
                    "local_feature_attributions": [],
                    "why_summary": "TFT champion: see global drivers on /metrics.",
                    "tft_attention": tft_attention_summary(bundle.get("tft")),
                }

            global_frame = self._build_feature_importance_frame(selected, bundle)
            top_global = self._top_drivers_payload(global_frame)
            return {
                "explanation_method": method_note,
                "local_feature_attributions": local["local_top_features"],
                "why_summary": build_why_summary(local["local_top_features"], top_global),
                "tft_attention": tft_attention_summary(bundle.get("tft")),
            }
        except Exception as exc:  # pragma: no cover
            return {"explanation_method": "error", "error": str(exc)}

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        bundle = load_model_bundle()
        if bundle is None:
            raise RuntimeError("No trained production model is available.")

        forecast_start_date = pd.Timestamp(payload["forecast_start_date"])
        observations = self.repository.load_observations()
        history_before = observations[observations["date"] < forecast_start_date].copy()
        context_frame = self._normalize_future_context(
            payload["kitchen_id"],
            forecast_start_date,
            payload["horizon_days"],
            payload["future_context"],
            observations,
        )
        forecast_frame = self._forecast_with_bundle(
            bundle,
            payload["kitchen_id"],
            forecast_start_date,
            payload["horizon_days"],
            payload["future_context"],
            context_frame=context_frame,
        )
        if forecast_frame.empty:
            raise RuntimeError("Unable to generate forecasts for the requested horizon.")

        prediction_id = str(uuid4())
        generated_at = datetime.utcnow().isoformat()
        decision_records: list[dict[str, Any]] = []
        prediction_records: list[dict[str, Any]] = []
        for horizon_idx, row in enumerate(forecast_frame.itertuples(index=False), start=1):
            recommendation = self.optimizer.recommend(row.predicted_demand, row.sigma)
            prediction_records.append(
                {
                    "prediction_id": prediction_id,
                    "kitchen_id": payload["kitchen_id"],
                    "forecast_date": pd.Timestamp(row.date).date().isoformat(),
                    "generated_at": generated_at,
                    "horizon_day": horizon_idx,
                    "model_name": bundle["selected_model"],
                    "model_version": bundle["model_version"],
                    "point_forecast": row.predicted_demand,
                    "lower_bound": row.lower_bound,
                    "upper_bound": row.upper_bound,
                    "selected_flag": 1,
                }
            )
            decision_records.append(
                {
                    "decision_id": str(uuid4()),
                    "prediction_id": prediction_id,
                    "kitchen_id": payload["kitchen_id"],
                    "forecast_date": pd.Timestamp(row.date).date().isoformat(),
                    "optimal_quantity": recommendation.optimal_quantity,
                    "expected_waste": recommendation.expected_waste,
                    "expected_shortage": recommendation.expected_shortage,
                    "expected_cost": recommendation.expected_cost_inr,
                    "realized_waste": np.nan,
                    "realized_shortage": np.nan,
                    "realized_cost": np.nan,
                    "prepared_quantity": np.nan,
                    "menu_type": row.menu_type,
                }
            )

        predictions_df = pd.DataFrame(prediction_records)
        decisions_df = pd.DataFrame(decision_records)
        self.repository.insert_predictions(predictions_df)
        self.repository.insert_optimization_decisions(decisions_df.drop(columns=["menu_type"]))

        recipes = self.repository.list_recipes()
        ingredient_plan = self.optimizer.ingredient_plan(
            decisions_df[["menu_type", "optimal_quantity"]],
            recipes,
        )

        next_day = decisions_df.iloc[0]
        next_day_forecast = forecast_frame.iloc[0]
        optimized_next_day = self.optimizer.recommend(
            float(next_day_forecast.predicted_demand),
            float(next_day_forecast.sigma),
        )
        heuristic_next_day = self.optimizer.evaluate_quantity(
            float(next_day_forecast.predicted_demand),
            float(next_day_forecast.sigma),
            self._heuristic_quantity(float(next_day_forecast.predicted_demand)),
        )
        decision_comparison = {
            "baseline": self._decision_strategy_payload("heuristic_buffer", heuristic_next_day),
            "optimized": self._decision_strategy_payload("uncertainty_aware_optimizer", optimized_next_day),
            "expected_cost_savings": float(
                heuristic_next_day.expected_cost_inr - optimized_next_day.expected_cost_inr
            ),
            "expected_waste_reduction_pct": self._safe_pct(
                heuristic_next_day.expected_waste - optimized_next_day.expected_waste,
                heuristic_next_day.expected_waste,
            ),
        }
        return {
            "prediction_id": prediction_id,
            "kitchen_id": payload["kitchen_id"],
            "selected_model": bundle["selected_model"],
            "model_version": bundle["model_version"],
            "winner_reason": bundle.get("winner_reason", "Incumbent production model."),
            "forecasts": [
                {
                    "date": pd.Timestamp(row.date).date(),
                    "horizon_day": idx + 1,
                    "predicted_demand": float(row.predicted_demand),
                    "lower_bound": float(row.lower_bound),
                    "upper_bound": float(row.upper_bound),
                    "sigma": float(row.sigma),
                    "menu_type": row.menu_type,
                }
                for idx, row in forecast_frame.iterrows()
            ],
            "next_day_optimization": {
                "forecast_date": pd.Timestamp(next_day_forecast.date).date(),
                "predicted_demand": float(next_day_forecast.predicted_demand),
                "optimal_quantity": int(next_day.optimal_quantity),
                "expected_waste": float(next_day.expected_waste),
                "expected_shortage": float(next_day.expected_shortage),
                "expected_cost": float(next_day.expected_cost),
                "critical_ratio": float(self.optimizer.critical_ratio),
                "shortage_probability_pct": float(optimized_next_day.shortage_probability * 100.0),
                "service_level_target_pct": float(optimized_next_day.service_level_target * 100.0),
                "service_level_satisfied": bool(optimized_next_day.service_level_satisfied),
            },
            "decision_comparison": decision_comparison,
            "scenario_analysis": self._build_counterfactual_scenarios(
                float(next_day_forecast.predicted_demand),
                float(next_day_forecast.sigma),
            ),
            "ingredient_plan": ingredient_plan,
            "explanation": self._build_prediction_explanation(
                bundle,
                payload["kitchen_id"],
                history_before,
                context_frame,
            ),
        }

    def log_feedback(self, payload: dict[str, Any]) -> dict[str, Any]:
        shortage = max(float(payload["actual_demand"]) - float(payload["prepared_quantity"]), 0.0)
        realized_cost = (
            float(payload["waste_quantity"]) * self.config.waste_cost
            + shortage * self.config.shortage_cost
        )
        observations = pd.DataFrame(
            [
                {
                    "kitchen_id": payload["kitchen_id"],
                    "date": payload["date"],
                    "actual_demand": payload["actual_demand"],
                    "prepared_quantity": payload["prepared_quantity"],
                    "waste_quantity": payload["waste_quantity"],
                    "shortage_quantity": shortage,
                    "attendance_variation": payload.get("attendance_variation", 0.0),
                    "menu_type": payload["menu_type"],
                    "is_holiday": int(payload.get("is_holiday", False)),
                    "is_exam_week": int(payload.get("is_exam_week", False)),
                    "is_event_day": int(payload.get("is_event_day", False)),
                    "event_name": "none",
                    "temperature": payload.get("temperature"),
                    "rainfall": payload.get("rainfall"),
                    "predicted_demand": None,
                    "selected_model": None,
                    "data_source": "feedback",
                }
            ]
        )
        self.repository.upsert_observations(observations)
        self.repository.update_realized_decision(
            payload["kitchen_id"],
            pd.Timestamp(payload["date"]).date().isoformat(),
            int(payload["prepared_quantity"]),
            float(payload["waste_quantity"]),
            float(shortage),
            float(realized_cost),
        )
        return {
            "status": "feedback_logged",
            "realized_shortage": float(shortage),
            "realized_cost": float(realized_cost),
        }

    def get_metrics_payload(self) -> dict[str, Any]:
        summary = load_summary_metrics()
        comparison = load_dataframe(MODEL_COMPARISON_FILE)
        monitoring = load_dataframe(MONITORING_LOG_FILE)
        current_registry = self.repository.get_selected_model_registry()
        active_bundle = load_model_bundle()
        feature_importance = pd.DataFrame()
        if active_bundle is not None:
            feature_importance = self._build_feature_importance_frame(
                current_registry["model_name"] if current_registry else summary.get("current_model", ""),
                active_bundle,
            )
        business_metrics = summary.get("business_metrics", {})
        coverage_metrics = business_metrics.get(
            "coverage_metrics",
            {
                "expected_coverage_pct": float(self.config.prediction_interval * 100.0),
                "actual_coverage_pct": 0.0,
                "calibration_gap_pct": 0.0,
            },
        )
        service_level_metrics = business_metrics.get(
            "service_level_metrics",
            {
                "target_service_level_pct": float(self.config.service_level_target * 100.0),
                "target_max_shortage_probability_pct": float((1.0 - self.config.service_level_target) * 100.0),
                "planned_shortage_probability_pct": float((1.0 - self.config.service_level_target) * 100.0),
                "realized_shortage_rate_pct": 0.0,
            },
        )
        return {
            "current_model": current_registry["model_name"] if current_registry else summary.get("current_model"),
            "trained_at": summary.get("trained_at"),
            "selected_model_version": current_registry["model_version"] if current_registry else summary.get("selected_model_version"),
            "model_comparison": self._frame_records(comparison),
            "business_metrics": business_metrics,
            "before_after_table": business_metrics.get("before_after_table", []),
            "coverage_metrics": coverage_metrics,
            "service_level_metrics": service_level_metrics,
            "feature_importance": feature_importance.head(10).to_dict("records"),
            "top_drivers": self._top_drivers_payload(feature_importance),
            "monitoring": self._frame_records(monitoring),
            "plot_urls": self._plot_urls(),
        }

    def get_dashboard_history(self) -> dict[str, Any]:
        forecast_history = load_dataframe(FORECAST_HISTORY_FILE)
        comparison = load_dataframe(MODEL_COMPARISON_FILE)
        latest_predictions = self.repository.latest_predictions()
        latest_training = self.repository.latest_training_runs()
        active_bundle = load_model_bundle()
        current_registry = self.repository.get_selected_model_registry()
        feature_importance = pd.DataFrame()
        if active_bundle is not None and current_registry is not None:
            feature_importance = self._build_feature_importance_frame(
                current_registry["model_name"],
                active_bundle,
            )
        return {
            "forecast_history": self._frame_records(forecast_history),
            "model_comparison": self._frame_records(comparison),
            "latest_predictions": self._frame_records(latest_predictions),
            "latest_training_runs": self._frame_records(latest_training),
            "feature_importance": feature_importance.head(10).to_dict("records"),
            "top_drivers": self._top_drivers_payload(feature_importance),
            "plot_urls": self._plot_urls(),
        }
