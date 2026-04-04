from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ApiSchema(BaseModel):
    model_config = ConfigDict(protected_namespaces=())


class FutureContextInput(ApiSchema):
    date: date
    menu_type: str
    temperature: float
    rainfall: float
    is_holiday: bool | None = None
    is_exam_week: bool | None = None
    is_event_day: bool | None = None
    attendance_variation: float | None = None


class PredictRequest(ApiSchema):
    kitchen_id: str
    forecast_start_date: date
    horizon_days: int = Field(..., description="Allowed horizons are 1 or 7.")
    future_context: list[FutureContextInput]

    @field_validator("horizon_days")
    @classmethod
    def validate_horizon_days(cls, value: int) -> int:
        if value not in (1, 7):
            raise ValueError("horizon_days must be 1 or 7.")
        return value


class ForecastDayResponse(ApiSchema):
    date: date
    horizon_day: int
    predicted_demand: float
    lower_bound: float
    upper_bound: float
    sigma: float
    menu_type: str


class IngredientPlanResponse(ApiSchema):
    ingredient_name: str
    unit: str
    total_quantity: float


class DecisionStrategyResponse(ApiSchema):
    strategy_name: str
    quantity: int
    expected_waste: float
    expected_shortage: float
    expected_cost: float
    shortage_probability_pct: float
    service_level_target_pct: float
    service_level_satisfied: bool
    critical_ratio: float


class DecisionComparisonResponse(ApiSchema):
    baseline: DecisionStrategyResponse
    optimized: DecisionStrategyResponse
    expected_cost_savings: float
    expected_waste_reduction_pct: float


class ScenarioResponse(ApiSchema):
    scenario_name: str
    attendance_multiplier: float
    predicted_demand: float
    optimized_quantity: int
    optimized_waste: float
    optimized_cost: float
    heuristic_quantity: int
    heuristic_waste: float
    heuristic_cost: float


class OptimizationResponse(ApiSchema):
    forecast_date: date
    predicted_demand: float
    optimal_quantity: int
    expected_waste: float
    expected_shortage: float
    expected_cost: float
    critical_ratio: float
    shortage_probability_pct: float
    service_level_target_pct: float
    service_level_satisfied: bool


class PredictionResponse(ApiSchema):
    prediction_id: str
    kitchen_id: str
    selected_model: str
    model_version: str
    winner_reason: str
    forecasts: list[ForecastDayResponse]
    next_day_optimization: OptimizationResponse
    decision_comparison: DecisionComparisonResponse
    scenario_analysis: list[ScenarioResponse]
    ingredient_plan: list[IngredientPlanResponse]
    explanation: dict[str, Any] = Field(default_factory=dict)


class ModelMetricResponse(ApiSchema):
    model_name: str
    model_version: str
    rmse: float
    mae: float
    weekly_rmse: float
    weekly_mae: float
    interval_coverage: float
    residual_std: float
    mean_prediction_jump: float
    selected_model: bool
    promoted: bool
    improvement_pct: float
    notes: str | None = None


class MetricComparisonRowResponse(ApiSchema):
    metric: str
    before_value: float
    after_value: float
    unit: str


class CoverageMetricsResponse(ApiSchema):
    expected_coverage_pct: float
    actual_coverage_pct: float
    calibration_gap_pct: float


class ServiceLevelMetricsResponse(ApiSchema):
    target_service_level_pct: float
    target_max_shortage_probability_pct: float
    planned_shortage_probability_pct: float
    realized_shortage_rate_pct: float


class DriverResponse(ApiSchema):
    driver: str
    importance: float


class FeatureImportanceResponse(ApiSchema):
    feature: str
    importance: float


class TrainResponse(ApiSchema):
    status: str
    run_id: str
    trained_at: datetime
    selected_model: str
    selected_model_version: str | None = None
    promoted: bool
    promotion_reason: str
    model_metrics: list[ModelMetricResponse]
    business_metrics: dict[str, Any]


class MetricsResponse(ApiSchema):
    current_model: str | None
    trained_at: datetime
    selected_model_version: str | None
    model_comparison: list[ModelMetricResponse]
    business_metrics: dict[str, Any]
    before_after_table: list[MetricComparisonRowResponse]
    coverage_metrics: CoverageMetricsResponse
    service_level_metrics: ServiceLevelMetricsResponse
    feature_importance: list[FeatureImportanceResponse]
    top_drivers: list[DriverResponse]
    monitoring: list[dict[str, Any]]
    plot_urls: dict[str, str]


class FeedbackRequest(ApiSchema):
    kitchen_id: str
    date: date
    actual_demand: int
    prepared_quantity: int
    waste_quantity: float
    menu_type: str
    temperature: float | None = None
    rainfall: float | None = None
    is_holiday: bool | None = None
    is_exam_week: bool | None = None
    is_event_day: bool | None = None
    attendance_variation: float | None = None


class FeedbackResponse(ApiSchema):
    status: str
    realized_shortage: float
    realized_cost: float


class UploadResponse(ApiSchema):
    status: str
    rows_ingested: int
    trained: bool
    selected_model: str | None = None
    model_version: str | None = None


class KitchenResponse(ApiSchema):
    kitchen_id: str
    hostel_name: str
    campus_zone: str
    latitude: float
    longitude: float
    capacity: int
    capacity_band: str
    default_attendance_band: str
