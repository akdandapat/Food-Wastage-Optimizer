import { ChangeEvent } from "react";

export type PredictionFormState = {
  date: string;
  temperature: string;
  rainfall: string;
  menu: string;
  attendanceVariation: string;
  isHoliday: boolean;
  isExamWeek: boolean;
};

type PredictionResult = {
  predicted_demand: number;
  lower_bound: number;
  upper_bound: number;
  optimal_quantity: number;
  expected_waste: number;
  expected_shortage: number;
};

type PredictionFormProps = {
  value: PredictionFormState;
  onChange: (nextState: PredictionFormState) => void;
  onSubmit: () => Promise<void>;
  loading: boolean;
  result: PredictionResult | null;
};

const menuOptions = [
  "regular",
  "protein_rich",
  "regional_special",
  "comfort_food",
  "festive",
  "light_weekend"
];

export default function PredictionForm({
  value,
  onChange,
  onSubmit,
  loading,
  result
}: PredictionFormProps) {
  const updateField =
    (field: keyof PredictionFormState) =>
    (event: ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
      const fieldValue =
        event.target instanceof HTMLInputElement &&
        event.target.type === "checkbox"
          ? event.target.checked
          : event.target.value;

      onChange({
        ...value,
        [field]: fieldValue
      } as PredictionFormState);
    };

  return (
    <div className="prediction-grid">
      <div className="form-grid">
        <label className="field-label">
          Forecast Date
          <input type="date" value={value.date} onChange={updateField("date")} />
        </label>
        <label className="field-label">
          Temperature (°C)
          <input
            type="number"
            step="0.1"
            value={value.temperature}
            onChange={updateField("temperature")}
          />
        </label>
        <label className="field-label">
          Rainfall (mm)
          <input
            type="number"
            step="0.1"
            value={value.rainfall}
            onChange={updateField("rainfall")}
          />
        </label>
        <label className="field-label">
          Menu Type
          <select value={value.menu} onChange={updateField("menu")}>
            {menuOptions.map((menu) => (
              <option key={menu} value={menu}>
                {menu}
              </option>
            ))}
          </select>
        </label>
        <label className="field-label">
          Attendance Variation
          <input
            type="number"
            step="0.01"
            value={value.attendanceVariation}
            onChange={updateField("attendanceVariation")}
          />
        </label>
        <label className="checkbox-label">
          <input
            type="checkbox"
            checked={value.isHoliday}
            onChange={updateField("isHoliday")}
          />
          Holiday Override
        </label>
        <label className="checkbox-label">
          <input
            type="checkbox"
            checked={value.isExamWeek}
            onChange={updateField("isExamWeek")}
          />
          Exam Week Override
        </label>
      </div>
      <div className="control-stack">
        <button
          className="action-button"
          disabled={loading}
          onClick={() => void onSubmit()}
          type="button"
        >
          {loading ? "Forecasting..." : "Generate Prediction"}
        </button>
        {result ? (
          <div className="prediction-result">
            <p>
              Predicted demand: <strong>{result.predicted_demand.toFixed(0)}</strong>
            </p>
            <p>
              Interval: {result.lower_bound.toFixed(0)} to{" "}
              {result.upper_bound.toFixed(0)}
            </p>
            <p>
              Optimal quantity: <strong>{result.optimal_quantity}</strong>
            </p>
            <p>
              Expected waste: {result.expected_waste.toFixed(1)} meals
            </p>
            <p>
              Expected shortage: {result.expected_shortage.toFixed(1)} meals
            </p>
          </div>
        ) : null}
      </div>
    </div>
  );
}

