import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";

import DatasetUpload from "../components/DatasetUpload";
import KpiCard from "../components/KpiCard";
import Layout from "../components/Layout";
import SectionCard from "../components/SectionCard";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

const menuOptions = [
  "regular",
  "protein_rich",
  "regional_special",
  "comfort_food",
  "festive",
  "light_weekend"
] as const;

type Kitchen = {
  kitchen_id: string;
  hostel_name: string;
  campus_zone: string;
  capacity: number;
  capacity_band: string;
};

type ModelMetric = {
  model_name: string;
  model_version: string;
  rmse: number;
  mae: number;
  weekly_rmse: number;
  weekly_mae: number;
  interval_coverage: number;
  residual_std: number;
  mean_prediction_jump: number;
  selected_model: boolean;
  promoted: boolean;
  improvement_pct: number;
  notes?: string;
};

type MetricsResponse = {
  current_model: string | null;
  trained_at: string;
  selected_model_version: string | null;
  model_comparison: ModelMetric[];
  business_metrics: {
    waste_reduction_pct?: number;
    daily_cost_savings_inr?: number;
    annual_savings_inr?: number;
    optimized_waste_pct?: number;
    prediction_interval_coverage?: number;
    before_after_table?: MetricComparisonRow[];
    coverage_metrics?: CoverageMetrics;
    service_level_metrics?: ServiceLevelMetrics;
  };
  before_after_table: MetricComparisonRow[];
  coverage_metrics: CoverageMetrics;
  service_level_metrics: ServiceLevelMetrics;
  feature_importance: FeatureImportancePoint[];
  top_drivers: DriverPoint[];
  monitoring: {
    timestamp: string;
    current_model: string;
    rmse: number;
    mae: number;
    waste_reduction_pct: number;
    annual_savings_inr: number;
  }[];
  plot_urls: Record<string, string>;
};

type MetricComparisonRow = {
  metric: string;
  before_value: number;
  after_value: number;
  unit: string;
};

type CoverageMetrics = {
  expected_coverage_pct: number;
  actual_coverage_pct: number;
  calibration_gap_pct: number;
};

type ServiceLevelMetrics = {
  target_service_level_pct: number;
  target_max_shortage_probability_pct: number;
  planned_shortage_probability_pct: number;
  realized_shortage_rate_pct: number;
};

type FeatureImportancePoint = {
  feature: string;
  importance: number;
};

type DriverPoint = {
  driver: string;
  importance: number;
};

type HistoryPoint = {
  kitchen_id: string;
  date: string;
  actual_demand: number;
  predicted_demand: number;
  baseline_waste_realized: number;
  optimized_waste_realized: number;
  daily_cost_saving_inr: number;
};

type HistoryResponse = {
  forecast_history: HistoryPoint[];
  model_comparison: ModelMetric[];
  latest_predictions: {
    kitchen_id: string;
    forecast_date: string;
    point_forecast: number;
    model_name: string;
  }[];
  latest_training_runs: ModelMetric[];
  feature_importance: FeatureImportancePoint[];
  top_drivers: DriverPoint[];
  plot_urls: Record<string, string>;
};

type PredictionResponse = {
  prediction_id: string;
  kitchen_id: string;
  selected_model: string;
  model_version: string;
  winner_reason: string;
  forecasts: {
    date: string;
    horizon_day: number;
    predicted_demand: number;
    lower_bound: number;
    upper_bound: number;
    sigma: number;
    menu_type: string;
  }[];
  next_day_optimization: {
    forecast_date: string;
    predicted_demand: number;
    optimal_quantity: number;
    expected_waste: number;
    expected_shortage: number;
    expected_cost: number;
    critical_ratio: number;
    shortage_probability_pct: number;
    service_level_target_pct: number;
    service_level_satisfied: boolean;
  };
  decision_comparison: {
    baseline: DecisionStrategy;
    optimized: DecisionStrategy;
    expected_cost_savings: number;
    expected_waste_reduction_pct: number;
  };
  scenario_analysis: ScenarioPoint[];
  ingredient_plan: {
    ingredient_name: string;
    unit: string;
    total_quantity: number;
  }[];
  explanation?: {
    explanation_method?: string;
    why_summary?: string;
    local_feature_attributions?: { feature: string; value: number }[];
    tft_attention?: { available?: boolean; note?: string };
    error?: string;
  };
};

type DecisionStrategy = {
  strategy_name: string;
  quantity: number;
  expected_waste: number;
  expected_shortage: number;
  expected_cost: number;
  shortage_probability_pct: number;
  service_level_target_pct: number;
  service_level_satisfied: boolean;
  critical_ratio: number;
};

type ScenarioPoint = {
  scenario_name: string;
  attendance_multiplier: number;
  predicted_demand: number;
  optimized_quantity: number;
  optimized_waste: number;
  optimized_cost: number;
  heuristic_quantity: number;
  heuristic_waste: number;
  heuristic_cost: number;
};

type ForecastFormState = {
  kitchenId: string;
  forecastStartDate: string;
  horizonDays: "1" | "7";
  menuType: string;
  temperature: string;
  rainfall: string;
  attendanceVariation: string;
  isHoliday: boolean;
  isExamWeek: boolean;
  isEventDay: boolean;
};

type FeedbackFormState = {
  kitchenId: string;
  date: string;
  actualDemand: string;
  preparedQuantity: string;
  wasteQuantity: string;
  menuType: string;
  temperature: string;
  rainfall: string;
};

const today = new Date().toISOString().slice(0, 10);

const defaultForecastForm: ForecastFormState = {
  kitchenId: "",
  forecastStartDate: today,
  horizonDays: "7",
  menuType: "regular",
  temperature: "31",
  rainfall: "8",
  attendanceVariation: "0.00",
  isHoliday: false,
  isExamWeek: false,
  isEventDay: false
};

const defaultFeedbackForm: FeedbackFormState = {
  kitchenId: "",
  date: today,
  actualDemand: "1600",
  preparedQuantity: "1680",
  wasteQuantity: "60",
  menuType: "regular",
  temperature: "31",
  rainfall: "8"
};

function formatShortDate(value: string) {
  return new Date(value).toLocaleDateString("en-IN", {
    day: "2-digit",
    month: "short"
  });
}

function formatCurrency(value: number | undefined) {
  if (value === undefined) {
    return "n/a";
  }
  return `INR ${value.toFixed(0)}`;
}

function formatPercent(value: number | undefined, digits = 1) {
  if (value === undefined) {
    return "n/a";
  }
  return `${value.toFixed(digits)}%`;
}

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, init);
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new Error(payload?.detail ?? "Request failed.");
  }
  return response.json() as Promise<T>;
}

export default function HomePage() {
  const [kitchens, setKitchens] = useState<Kitchen[]>([]);
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [history, setHistory] = useState<HistoryResponse | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [forecastForm, setForecastForm] = useState<ForecastFormState>(
    defaultForecastForm
  );
  const [feedbackForm, setFeedbackForm] = useState<FeedbackFormState>(
    defaultFeedbackForm
  );
  const [loading, setLoading] = useState(true);
  const [predicting, setPredicting] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [training, setTraining] = useState(false);
  const [feedbacking, setFeedbacking] = useState(false);
  const [status, setStatus] = useState("Loading kitchen dashboard...");

  const refreshDashboard = async () => {
    setLoading(true);
    try {
      const kitchensPayload = await fetchJson<Kitchen[]>("/kitchens");
      setKitchens(kitchensPayload);
      if (!forecastForm.kitchenId && kitchensPayload[0]) {
        setForecastForm((current) => ({
          ...current,
          kitchenId: kitchensPayload[0].kitchen_id
        }));
        setFeedbackForm((current) => ({
          ...current,
          kitchenId: kitchensPayload[0].kitchen_id
        }));
      }

      const metricsPromise = fetchJson<MetricsResponse>("/metrics").catch(
        () => null
      );
      const historyPromise = fetchJson<HistoryResponse>("/history").catch(
        () => null
      );
      const [metricsPayload, historyPayload] = await Promise.all([
        metricsPromise,
        historyPromise
      ]);

      setMetrics(metricsPayload);
      setHistory(historyPayload);
      setStatus("Dashboard synchronized with the latest artifacts.");
    } catch (error) {
      setStatus(
        error instanceof Error ? error.message : "Unable to load dashboard."
      );
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void refreshDashboard();
  }, []);

  const demandSeries = useMemo(() => {
    const rows = history?.forecast_history ?? [];
    const grouped = new Map<
      string,
      { date: string; actual: number; predicted: number }
    >();

    rows.slice(-200).forEach((row) => {
      const entry = grouped.get(row.date) ?? {
        date: formatShortDate(row.date),
        actual: 0,
        predicted: 0
      };
      entry.actual += row.actual_demand ?? 0;
      entry.predicted += row.predicted_demand ?? 0;
      grouped.set(row.date, entry);
    });

    return Array.from(grouped.entries())
      .sort(([left], [right]) => left.localeCompare(right))
      .slice(-60)
      .map(([, value]) => value);
  }, [history]);

  const wasteSeries = useMemo(() => {
    const rows = history?.forecast_history ?? [];
    const grouped = new Map<
      string,
      { date: string; baselineWaste: number; optimizedWaste: number }
    >();

    rows.slice(-200).forEach((row) => {
      const entry = grouped.get(row.date) ?? {
        date: formatShortDate(row.date),
        baselineWaste: 0,
        optimizedWaste: 0
      };
      entry.baselineWaste += row.baseline_waste_realized ?? 0;
      entry.optimizedWaste += row.optimized_waste_realized ?? 0;
      grouped.set(row.date, entry);
    });

    return Array.from(grouped.entries())
      .sort(([left], [right]) => left.localeCompare(right))
      .slice(-30)
      .map(([, value]) => value);
  }, [history]);

  const comparisonSeries = useMemo(
    () =>
      (metrics?.model_comparison ?? []).map((row) => ({
        model: row.model_name,
        rmse: Number(row.rmse.toFixed(1)),
        weeklyRmse: Number(row.weekly_rmse.toFixed(1))
      })),
    [metrics]
  );

  const monitoringSeries = useMemo(
    () =>
      (metrics?.monitoring ?? []).slice(-14).map((row) => ({
        date: formatShortDate(row.timestamp),
        rmse: Number(row.rmse.toFixed(1)),
        wasteReduction: Number(row.waste_reduction_pct.toFixed(1))
      })),
    [metrics]
  );

  const buildFutureContext = () => {
    const start = new Date(forecastForm.forecastStartDate);
    return Array.from({ length: Number(forecastForm.horizonDays) }).map(
      (_, index) => {
        const date = new Date(start);
        date.setDate(start.getDate() + index);
        return {
          date: date.toISOString().slice(0, 10),
          menu_type: forecastForm.menuType,
          temperature: Number(forecastForm.temperature),
          rainfall: Number(forecastForm.rainfall),
          attendance_variation: Number(forecastForm.attendanceVariation),
          is_holiday: forecastForm.isHoliday,
          is_exam_week: forecastForm.isExamWeek,
          is_event_day: forecastForm.isEventDay
        };
      }
    );
  };

  const handlePredict = async () => {
    setPredicting(true);
    setStatus("Generating kitchen forecast and newsvendor recommendation...");
    try {
      const result = await fetchJson<PredictionResponse>("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          kitchen_id: forecastForm.kitchenId,
          forecast_start_date: forecastForm.forecastStartDate,
          horizon_days: Number(forecastForm.horizonDays),
          future_context: buildFutureContext()
        })
      });
      setPrediction(result);
      setStatus(
        `Forecast ready. Production model: ${result.selected_model} (${result.model_version}).`
      );
      await refreshDashboard();
    } catch (error) {
      setStatus(
        error instanceof Error ? error.message : "Forecast generation failed."
      );
    } finally {
      setPredicting(false);
    }
  };

  const handleUpload = async (file: File) => {
    setUploading(true);
    setStatus("Uploading dataset and starting retraining...");
    try {
      const formData = new FormData();
      formData.append("file", file);
      await fetchJson("/dataset/upload", {
        method: "POST",
        body: formData
      });
      await refreshDashboard();
      setStatus("Dataset uploaded and retraining completed.");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Dataset upload failed.");
    } finally {
      setUploading(false);
    }
  };

  const handleTrain = async () => {
    setTraining(true);
    setStatus("Running full candidate retraining...");
    try {
      await fetchJson("/train", { method: "POST" });
      await refreshDashboard();
      setStatus("Training completed and dashboard refreshed.");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Training failed.");
    } finally {
      setTraining(false);
    }
  };

  const handleFeedback = async () => {
    setFeedbacking(true);
    setStatus("Logging actual demand and waste feedback...");
    try {
      await fetchJson("/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          kitchen_id: feedbackForm.kitchenId,
          date: feedbackForm.date,
          actual_demand: Number(feedbackForm.actualDemand),
          prepared_quantity: Number(feedbackForm.preparedQuantity),
          waste_quantity: Number(feedbackForm.wasteQuantity),
          menu_type: feedbackForm.menuType,
          temperature: Number(feedbackForm.temperature),
          rainfall: Number(feedbackForm.rainfall)
        })
      });
      setStatus("Feedback logged. Nightly retraining will consume the new actuals.");
      await refreshDashboard();
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Feedback logging failed.");
    } finally {
      setFeedbacking(false);
    }
  };

  return (
    <Layout>
      <div className="full-span">
        <div className="status-banner">{status}</div>
      </div>

      <div className="full-span">
        <div className="kpi-grid">
          <KpiCard
            label="Production Model"
            value={metrics?.current_model ?? (loading ? "..." : "n/a")}
            caption={metrics?.selected_model_version ?? "Current promoted forecaster"}
            tone="warm"
          />
          <KpiCard
            label="Next-day RMSE"
            value={
              metrics?.model_comparison.find((row) => row.selected_model)?.rmse.toFixed(1) ??
              (loading ? "..." : "n/a")
            }
            caption="Primary winner-selection metric."
            tone="cool"
          />
          <KpiCard
            label="Waste Reduction"
            value={
              metrics?.business_metrics.waste_reduction_pct !== undefined
                ? `${metrics.business_metrics.waste_reduction_pct.toFixed(1)}%`
                : loading
                  ? "..."
                  : "n/a"
            }
            caption="Historical prepared quantity vs optimized output."
            tone="earth"
          />
          <KpiCard
            label="Annual Savings"
            value={
              metrics
                ? formatCurrency(metrics.business_metrics.annual_savings_inr)
                : loading
                  ? "..."
                  : "n/a"
            }
            caption="Projected annual savings from optimized cooking."
            tone="warm"
          />
          <KpiCard
            label="Coverage"
            value={
              metrics ? formatPercent(metrics.coverage_metrics.actual_coverage_pct) : loading ? "..." : "n/a"
            }
            caption={
              metrics
                ? `Expected ${formatPercent(metrics.coverage_metrics.expected_coverage_pct, 0)} interval coverage.`
                : "Prediction interval calibration."
            }
            tone="cool"
          />
          <KpiCard
            label="Shortage Risk"
            value={
              metrics ? formatPercent(metrics.service_level_metrics.planned_shortage_probability_pct) : loading ? "..." : "n/a"
            }
            caption={
              metrics
                ? `Target <= ${formatPercent(metrics.service_level_metrics.target_max_shortage_probability_pct)}`
                : "Service-level constraint."
            }
            tone="earth"
          />
        </div>
      </div>

      <SectionCard
        title="Production Forecast"
        subtitle="Next-day and weekly demand prediction using the promoted production model."
      >
        <div className="form-grid">
          <label className="field-label">
            Kitchen
            <select
              value={forecastForm.kitchenId}
              onChange={(event) =>
                setForecastForm((current) => ({
                  ...current,
                  kitchenId: event.target.value
                }))
              }
            >
              {kitchens.map((kitchen) => (
                <option key={kitchen.kitchen_id} value={kitchen.kitchen_id}>
                  {kitchen.hostel_name}
                </option>
              ))}
            </select>
          </label>
          <label className="field-label">
            Forecast Start
            <input
              type="date"
              value={forecastForm.forecastStartDate}
              onChange={(event) =>
                setForecastForm((current) => ({
                  ...current,
                  forecastStartDate: event.target.value
                }))
              }
            />
          </label>
          <label className="field-label">
            Horizon
            <select
              value={forecastForm.horizonDays}
              onChange={(event) =>
                setForecastForm((current) => ({
                  ...current,
                  horizonDays: event.target.value as "1" | "7"
                }))
              }
            >
              <option value="1">Next day</option>
              <option value="7">Weekly</option>
            </select>
          </label>
          <label className="field-label">
            Menu Type
            <select
              value={forecastForm.menuType}
              onChange={(event) =>
                setForecastForm((current) => ({
                  ...current,
                  menuType: event.target.value
                }))
              }
            >
              {menuOptions.map((menu) => (
                <option key={menu} value={menu}>
                  {menu}
                </option>
              ))}
            </select>
          </label>
          <label className="field-label">
            Temperature (C)
            <input
              type="number"
              step="0.1"
              value={forecastForm.temperature}
              onChange={(event) =>
                setForecastForm((current) => ({
                  ...current,
                  temperature: event.target.value
                }))
              }
            />
          </label>
          <label className="field-label">
            Rainfall (mm)
            <input
              type="number"
              step="0.1"
              value={forecastForm.rainfall}
              onChange={(event) =>
                setForecastForm((current) => ({
                  ...current,
                  rainfall: event.target.value
                }))
              }
            />
          </label>
          <label className="field-label">
            Attendance Variation
            <input
              type="number"
              step="0.01"
              value={forecastForm.attendanceVariation}
              onChange={(event) =>
                setForecastForm((current) => ({
                  ...current,
                  attendanceVariation: event.target.value
                }))
              }
            />
          </label>
          <div className="toggle-row">
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={forecastForm.isHoliday}
                onChange={(event) =>
                  setForecastForm((current) => ({
                    ...current,
                    isHoliday: event.target.checked
                  }))
                }
              />
              Holiday
            </label>
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={forecastForm.isExamWeek}
                onChange={(event) =>
                  setForecastForm((current) => ({
                    ...current,
                    isExamWeek: event.target.checked
                  }))
                }
              />
              Exam Week
            </label>
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={forecastForm.isEventDay}
                onChange={(event) =>
                  setForecastForm((current) => ({
                    ...current,
                    isEventDay: event.target.checked
                  }))
                }
              />
              Event Day
            </label>
          </div>
        </div>
        <div className="button-row">
          <button
            className="action-button"
            type="button"
            disabled={predicting || !forecastForm.kitchenId}
            onClick={() => void handlePredict()}
          >
            {predicting ? "Forecasting..." : "Run Production Forecast"}
          </button>
        </div>
        {prediction ? (
          <div className="result-grid">
            <article className="result-card">
              <span className="pill">{prediction.selected_model}</span>
              <h3>Next-day Cooking Decision</h3>
              <p>
                Predicted demand:{" "}
                <strong>
                  {prediction.next_day_optimization.predicted_demand.toFixed(0)}
                </strong>
              </p>
              <p>
                Optimal quantity:{" "}
                <strong>{prediction.next_day_optimization.optimal_quantity}</strong>
              </p>
              <p>
                Expected waste:{" "}
                {prediction.next_day_optimization.expected_waste.toFixed(1)} meals
              </p>
              <p>
                Expected shortage:{" "}
                {prediction.next_day_optimization.expected_shortage.toFixed(1)} meals
              </p>
              <p>
                Expected cost:{" "}
                {formatCurrency(prediction.next_day_optimization.expected_cost)}
              </p>
              <p>
                Shortage probability:{" "}
                {prediction.next_day_optimization.shortage_probability_pct.toFixed(1)}%
              </p>
              <p>
                Service level target:{" "}
                {prediction.next_day_optimization.service_level_target_pct.toFixed(1)}%
              </p>
            </article>
            <article className="result-card">
              <span className="pill">Winner reason</span>
              <h3>Promotion Logic</h3>
              <p>{prediction.winner_reason}</p>
              <p>
                Forecast window: {prediction.forecasts.length} day
                {prediction.forecasts.length > 1 ? "s" : ""}
              </p>
              <p>Prediction ID: {prediction.prediction_id}</p>
              <p>
                Service constraint:{" "}
                {prediction.next_day_optimization.service_level_satisfied
                  ? "Satisfied"
                  : "Violated"}
              </p>
            </article>
            {prediction.explanation ? (
              <article className="result-card" style={{ gridColumn: "1 / -1" }}>
                <span className="pill">Explainability</span>
                <h3>Why this forecast (next day)</h3>
                {prediction.explanation.error ? (
                  <p className="muted">{prediction.explanation.error}</p>
                ) : (
                  <>
                    <p>{prediction.explanation.why_summary}</p>
                    <p className="muted">
                      Method: {prediction.explanation.explanation_method}
                    </p>
                    {prediction.explanation.tft_attention?.note ? (
                      <p className="muted">{prediction.explanation.tft_attention.note}</p>
                    ) : null}
                    <div className="table-wrap">
                      <table className="data-table">
                        <thead>
                          <tr>
                            <th>Feature</th>
                            <th>Local contribution</th>
                          </tr>
                        </thead>
                        <tbody>
                          {(prediction.explanation.local_feature_attributions ?? []).map(
                            (row) => (
                              <tr key={row.feature}>
                                <td>{row.feature}</td>
                                <td>{row.value.toFixed(2)}</td>
                              </tr>
                            )
                          )}
                        </tbody>
                      </table>
                    </div>
                  </>
                )}
              </article>
            ) : null}
          </div>
        ) : null}
      </SectionCard>

      <SectionCard
        title="Model Comparison"
        subtitle="Random Forest, XGBoost, LightGBM, Ridge, validation-weighted ensemble, and TFT challenger."
      >
        <div className="chart-shell">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={comparisonSeries}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="model" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="rmse" fill="#b6532d" name="Next-day RMSE" />
              <Bar dataKey="weeklyRmse" fill="#1f5b50" name="Weekly RMSE" />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>RMSE</th>
                <th>MAE</th>
                <th>Status</th>
                <th>Notes</th>
              </tr>
            </thead>
            <tbody>
              {(metrics?.model_comparison ?? []).map((row) => (
                <tr key={row.model_name}>
                  <td>{row.model_name}</td>
                  <td>{row.rmse.toFixed(1)}</td>
                  <td>{row.mae.toFixed(1)}</td>
                  <td>{row.selected_model ? "Production" : "Challenger"}</td>
                  <td>{row.notes || "-"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard
        title="Pre vs Post Impact"
        subtitle="Structured before-vs-after comparison between heuristic planning and the optimized decision layer."
      >
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Metric</th>
                <th>Before (Heuristic)</th>
                <th>After (Optimized)</th>
                <th>Unit</th>
              </tr>
            </thead>
            <tbody>
              {(metrics?.before_after_table ?? []).map((row) => (
                <tr key={row.metric}>
                  <td>{row.metric}</td>
                  <td>{row.before_value.toFixed(2)}</td>
                  <td>{row.after_value.toFixed(2)}</td>
                  <td>{row.unit}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard
        title="Uncertainty and Service Level"
        subtitle="Coverage calibration and service-level guarantee for shortage risk."
      >
        <div className="result-grid">
            <article className="result-card">
              <span className="pill">Calibration</span>
              <h3>Prediction Interval Coverage</h3>
              <p>
                Expected coverage:{" "}
                <strong>{formatPercent(metrics?.coverage_metrics?.expected_coverage_pct)}</strong>
              </p>
              <p>
                Actual coverage:{" "}
                <strong>{formatPercent(metrics?.coverage_metrics?.actual_coverage_pct)}</strong>
              </p>
              <p>
                Calibration gap:{" "}
                {formatPercent(metrics?.coverage_metrics?.calibration_gap_pct)}
              </p>
            </article>
            <article className="result-card">
              <span className="pill">Constraint</span>
              <h3>Shortage Probability Control</h3>
              <p>
                Target service level:{" "}
                <strong>{formatPercent(metrics?.service_level_metrics?.target_service_level_pct)}</strong>
              </p>
              <p>
                Max shortage probability:{" "}
                <strong>
                  {formatPercent(metrics?.service_level_metrics?.target_max_shortage_probability_pct)}
                </strong>
              </p>
              <p>
                Planned shortage probability:{" "}
                {formatPercent(metrics?.service_level_metrics?.planned_shortage_probability_pct)}
              </p>
              <p>
                Realized shortage rate:{" "}
                {formatPercent(metrics?.service_level_metrics?.realized_shortage_rate_pct)}
              </p>
            </article>
        </div>
      </SectionCard>

      <SectionCard
        title="Demand and Waste"
        subtitle="Recent total demand, predicted demand, and waste delta across the kitchen network."
      >
        <div className="chart-shell">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={demandSeries}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="actual" stroke="#183a37" strokeWidth={2.5} />
              <Line
                type="monotone"
                dataKey="predicted"
                stroke="#b6532d"
                strokeWidth={2.2}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div className="chart-shell compact-chart">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={wasteSeries}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="baselineWaste" fill="#d39c3d" name="Historical waste" />
              <Bar dataKey="optimizedWaste" fill="#1f5b50" name="Optimized waste" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </SectionCard>

      <SectionCard
        title="Monitoring Loop"
        subtitle="Nightly retraining history and error drift across successive runs."
      >
        <div className="chart-shell">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={monitoringSeries}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="rmse" stroke="#b6532d" strokeWidth={2.2} />
              <Line
                type="monotone"
                dataKey="wasteReduction"
                stroke="#1f5b50"
                strokeWidth={2.2}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </SectionCard>

      <SectionCard
        title="Explainability"
        subtitle="Dominant demand drivers from the production model feature importance profile."
      >
        <div className="result-grid">
          <article className="result-card">
            <span className="pill">Top Drivers</span>
            <h3>Grouped Importance</h3>
            <div className="table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Driver</th>
                    <th>Importance</th>
                  </tr>
                </thead>
                <tbody>
                  {(metrics?.top_drivers ?? history?.top_drivers ?? []).map((row) => (
                    <tr key={row.driver}>
                      <td>{row.driver.replaceAll("_", " ")}</td>
                      <td>{row.importance.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </article>
          <article className="result-card">
            <span className="pill">Raw Features</span>
            <h3>Highest Importance Features</h3>
            <div className="table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Feature</th>
                    <th>Importance</th>
                  </tr>
                </thead>
                <tbody>
                  {(metrics?.feature_importance ?? history?.feature_importance ?? []).map((row) => (
                    <tr key={row.feature}>
                      <td>{row.feature}</td>
                      <td>{row.importance.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </article>
        </div>
      </SectionCard>

      <SectionCard
        title="Feedback Loop"
        subtitle="Log actual demand and observed waste so the nightly training run can update the production model."
      >
        <div className="form-grid">
          <label className="field-label">
            Kitchen
            <select
              value={feedbackForm.kitchenId}
              onChange={(event) =>
                setFeedbackForm((current) => ({
                  ...current,
                  kitchenId: event.target.value
                }))
              }
            >
              {kitchens.map((kitchen) => (
                <option key={kitchen.kitchen_id} value={kitchen.kitchen_id}>
                  {kitchen.hostel_name}
                </option>
              ))}
            </select>
          </label>
          <label className="field-label">
            Service Date
            <input
              type="date"
              value={feedbackForm.date}
              onChange={(event) =>
                setFeedbackForm((current) => ({
                  ...current,
                  date: event.target.value
                }))
              }
            />
          </label>
          <label className="field-label">
            Actual Demand
            <input
              type="number"
              value={feedbackForm.actualDemand}
              onChange={(event) =>
                setFeedbackForm((current) => ({
                  ...current,
                  actualDemand: event.target.value
                }))
              }
            />
          </label>
          <label className="field-label">
            Prepared Quantity
            <input
              type="number"
              value={feedbackForm.preparedQuantity}
              onChange={(event) =>
                setFeedbackForm((current) => ({
                  ...current,
                  preparedQuantity: event.target.value
                }))
              }
            />
          </label>
          <label className="field-label">
            Waste Quantity
            <input
              type="number"
              step="0.1"
              value={feedbackForm.wasteQuantity}
              onChange={(event) =>
                setFeedbackForm((current) => ({
                  ...current,
                  wasteQuantity: event.target.value
                }))
              }
            />
          </label>
          <label className="field-label">
            Menu Type
            <select
              value={feedbackForm.menuType}
              onChange={(event) =>
                setFeedbackForm((current) => ({
                  ...current,
                  menuType: event.target.value
                }))
              }
            >
              {menuOptions.map((menu) => (
                <option key={menu} value={menu}>
                  {menu}
                </option>
              ))}
            </select>
          </label>
        </div>
        <div className="button-row">
          <button
            className="action-button"
            type="button"
            disabled={feedbacking || !feedbackForm.kitchenId}
            onClick={() => void handleFeedback()}
          >
            {feedbacking ? "Logging..." : "Log Actual Demand and Waste"}
          </button>
        </div>
      </SectionCard>

      <div className="full-span">
        <SectionCard
          title="Retraining and Artifacts"
          subtitle="Upload historical observations, force a retrain, and inspect generated plots."
        >
          <div className="result-grid">
            <article className="result-card">
              <h3>Dataset Upload</h3>
              <DatasetUpload onUpload={handleUpload} uploading={uploading} />
            </article>
            <article className="result-card">
              <h3>Manual Retraining</h3>
              <p>
                Force candidate comparison, winner selection, and artifact refresh.
              </p>
              <button
                className="action-button"
                type="button"
                disabled={training}
                onClick={() => void handleTrain()}
              >
                {training ? "Retraining..." : "Run /train"}
              </button>
            </article>
          </div>
          <div className="artifact-grid">
            {Object.entries(metrics?.plot_urls ?? history?.plot_urls ?? {}).map(
              ([key, url]) => (
                <figure className="artifact-card" key={key}>
                  <img
                    alt={key}
                    className="artifact-image"
                    src={`${API_BASE_URL}${url}`}
                  />
                  <figcaption>{key.replaceAll("_", " ")}</figcaption>
                </figure>
              )
            )}
          </div>
        </SectionCard>
      </div>

      {prediction ? (
        <div className="full-span">
          <SectionCard
            title="Forecast Horizon and Ingredient Plan"
            subtitle="Operational output passed from demand prediction into quantity and ingredient planning."
          >
            <div className="result-grid">
              <article className="result-card">
                <span className="pill">Before vs After</span>
                <h3>Decision Comparison</h3>
                <div className="table-wrap">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Strategy</th>
                        <th>Quantity</th>
                        <th>Waste</th>
                        <th>Cost</th>
                        <th>Shortage %</th>
                      </tr>
                    </thead>
                    <tbody>
                      {[prediction.decision_comparison.baseline, prediction.decision_comparison.optimized].map((row) => (
                        <tr key={row.strategy_name}>
                          <td>{row.strategy_name.replaceAll("_", " ")}</td>
                          <td>{row.quantity}</td>
                          <td>{row.expected_waste.toFixed(1)}</td>
                          <td>{row.expected_cost.toFixed(0)}</td>
                          <td>{row.shortage_probability_pct.toFixed(1)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p>
                  Expected cost savings:{" "}
                  <strong>{formatCurrency(prediction.decision_comparison.expected_cost_savings)}</strong>
                </p>
                <p>
                  Expected waste reduction:{" "}
                  <strong>{prediction.decision_comparison.expected_waste_reduction_pct.toFixed(1)}%</strong>
                </p>
              </article>
              <article className="result-card">
                <span className="pill">What-if</span>
                <h3>Counterfactual Simulation</h3>
                <div className="table-wrap">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Scenario</th>
                        <th>Waste</th>
                        <th>Cost</th>
                      </tr>
                    </thead>
                    <tbody>
                      {prediction.scenario_analysis.map((row) => (
                        <tr key={row.scenario_name}>
                          <td>{row.scenario_name.replaceAll("_", " ")}</td>
                          <td>{row.optimized_waste.toFixed(1)}</td>
                          <td>{row.optimized_cost.toFixed(0)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </article>
            </div>
            <div className="forecast-strip">
              {prediction.forecasts.map((row) => (
                <article className="forecast-card" key={`${row.date}-${row.horizon_day}`}>
                  <span className="pill">Day {row.horizon_day}</span>
                  <h3>{formatShortDate(row.date)}</h3>
                  <p>{row.menu_type}</p>
                  <p>
                    Demand: <strong>{row.predicted_demand.toFixed(0)}</strong>
                  </p>
                  <p>
                    Interval: {row.lower_bound.toFixed(0)} - {row.upper_bound.toFixed(0)}
                  </p>
                </article>
              ))}
            </div>
            <div className="table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Ingredient</th>
                    <th>Unit</th>
                    <th>Total quantity</th>
                  </tr>
                </thead>
                <tbody>
                  {prediction.ingredient_plan.map((row) => (
                    <tr key={`${row.ingredient_name}-${row.unit}`}>
                      <td>{row.ingredient_name}</td>
                      <td>{row.unit}</td>
                      <td>{row.total_quantity.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </SectionCard>
        </div>
      ) : null}
    </Layout>
  );
}
