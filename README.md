# Spatiotemporal Demand Forecasting and Food Waste Optimization

Production-oriented reference system for large Kolkata university hostel mess operations. The stack covers synthetic multi-kitchen data generation, shared feature engineering, tabular forecasting baselines, a GPU-first Temporal Fusion Transformer challenger, newsvendor optimization, FastAPI serving, a Next.js dashboard, and an MLOps feedback loop.

## Architecture

Raw Data -> Feature Engineering -> Random Forest / XGBoost / LightGBM -> Weighted Ensemble -> TFT Challenger -> Winner Selection -> Demand Prediction -> Optimization -> Feedback Loop -> Nightly Retraining

## Project Structure

```text
backend/
  main.py
  forecasting.py
  model.py
  features.py
  optimizer.py
  calibration.py
  explainability.py
  data_ingestion.py
  data_generation.py
  database.py
  repository.py
  calendar_utils.py
  mlops.py
  storage.py
  schemas.py
  visualization.py

frontend/
  pages/
  components/
  styles/

data/
  raw/                 # API downloads + hashes (ingestion_manifest.jsonl)
  processed/           # merged kitchen panel (kitchen_operations_panel.csv)
models/
```

## Backend Design

- SQLite-backed operational repository with tables for kitchens, recipes, daily observations, predictions, optimization decisions, training runs, and model registry.
- Runtime SQLite file is stored under `%LOCALAPPDATA%\CodexRuntime\kitchen_ops.sqlite` by default because SQLite file locking was unreliable on the workspace filesystem in this environment.
- Synthetic data generation creates 10 Kolkata hostel kitchens over 2 years of daily demand with:
  - weekly seasonality
  - holiday and exam-week shifts
  - event spikes
  - weather effects
  - menu effects
  - attendance variation
  - realistic prepared quantity and waste fields
- Tabular feature view includes:
  - demand lags `lag_1`, `lag_3`, `lag_7`, `lag_14`
  - rolling statistics over 7 and 14 days
  - calendar features
  - weather features
  - kitchen metadata
  - categorical encoding for menu, season, zone, and kitchen id
- Sequence feature view supports TFT with:
  - `group_ids=["kitchen_id"]`
  - static kitchen metadata
  - known future calendar and weather features
  - unknown historical demand and waste features
- Candidate models:
  - Random Forest
  - XGBoost
  - LightGBM
  - Ridge (linear baseline on engineered features)
  - validation-weighted ensemble (RF + XGB + LGBM only)
  - Temporal Fusion Transformer challenger
- Public ingestion merges **Open-Meteo** Kolkata weather, **India holidays** (Nager.Date or `holidays` fallback), a **food-demand CSV** mirror or `data/raw/mess_demand.csv`, and **cited waste priors** (`data/raw/public_waste_benchmarks.csv`).
- **Calibration** widens or tightens Gaussian intervals and TFT quantile spreads on the holdout so empirical coverage tracks the nominal level; scales are saved in the production bundle.
- **Explainability**: SHAP (trees) or Ridge coefficients locally when available; permutation importance available in code; TFT attention hook documented for extension.
- Winner selection:
  - lowest next-day holdout RMSE
  - residual standard deviation tiebreak
  - mean prediction jump tiebreak
- Optimization layer:
  - newsvendor critical-ratio quantity decision
  - expected waste
  - expected shortage
  - expected cost
  - ingredient plan derived from menu recipes
- MLOps loop:
  - prediction logging
  - actual demand and waste feedback logging
  - retraining scheduler every 24 hours
  - RMSE and savings monitoring history
  - model registry and promotion logic

## API Endpoints

- `GET /health`
- `GET /kitchens`
- `POST /predict`
- `POST /train`
- `POST /feedback`
- `POST /dataset/upload`
- `GET /metrics`
- `GET /history`
- `POST /ingest` — refresh `data/raw` and `data/processed` from public sources (run `POST /train` afterward to reload models from the new panel if needed)

## Frontend Dashboard

The Next.js dashboard includes:

- kitchen selector and forecast form
- next-day and 7-day prediction view
- optimization summary for cooking quantity
- ingredient planning table
- model comparison chart and table
- monitoring chart
- dataset upload
- manual retraining trigger
- feedback logging form for actual demand and waste
- generated artifact gallery for demand, waste, savings, feature importance, and residual plots

## Local Run

### 1. Backend

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Startup behavior:

- creates the runtime SQLite database in local app data
- seeds from `data/processed/kitchen_operations_panel.csv` when present, otherwise runs public ingestion, otherwise falls back to synthetic operations (using cached Kolkata weather CSV when available)
- trains the production bundle if no trained bundle exists yet
- starts the nightly retraining scheduler

### 2. Frontend

```powershell
Set-Location frontend
npm install
$env:NEXT_PUBLIC_API_BASE_URL="http://localhost:8000"
npm run dev
```

Open `http://localhost:3000`.

## Example Predict Request

```json
{
  "kitchen_id": "JU_BH1",
  "forecast_start_date": "2026-04-05",
  "horizon_days": 7,
  "future_context": [
    {
      "date": "2026-04-05",
      "menu_type": "regular",
      "temperature": 31.0,
      "rainfall": 8.0,
      "attendance_variation": 0.0,
      "is_holiday": false,
      "is_exam_week": false,
      "is_event_day": false
    }
  ]
}
```

## Model selection rule (summary)

Production champion is the candidate with the **lowest next-day holdout RMSE**. If another model is within **2% RMSE**, the tie-break is **lower residual standard deviation**, then **lower mean absolute day-to-day forecast change**. A challenger is promoted only if it beats the incumbent by at least **1% RMSE** (or there is no incumbent).

## Optimization rule (summary)

Decisions use a **Newsvendor** cost with waste and shortage penalties. The service-aware critical ratio maps to a Gaussian quantity \(Q \approx \mu + \sigma \Phi^{-1}(\text{critical ratio})\), minimizing expected asymmetric cost while respecting the target service level.

## Explainability outputs (summary)

- **Global**: grouped feature importance from the production model (tree gain / impurity or Ridge \(|\beta|\)).
- **Local**: next-day forecast carries **SHAP** values for tree champions or **linear term products** for Ridge; the dashboard shows a short **why_summary** string.
- **TFT**: attention extraction is stubbed for extension; global drivers still come from the metrics panel.

## Operational Notes

- Install optional SHAP for richer local explanations: `pip install shap`.
- TFT is implemented as the advanced challenger, but it is configured as GPU-first by default. In CPU-only environments it is skipped unless `enable_cpu_tft=True` is set in `ForecastConfig`.
- The default full 2-year retrain is expensive on CPU. In this environment, I validated the orchestration with a reduced-scale smoke configuration and verified prediction plus feedback logging successfully.
- Uploaded datasets must include at least `kitchen_id`, `date`, and `actual_demand`. If `menu_type` is omitted, the backend derives it from the calendar.

## Validation Completed Here

- Backend syntax parse across core modules.
- Reduced-scale training smoke with Random Forest, XGBoost, LightGBM, ensemble, and TFT skip path.
- Reduced-scale prediction and feedback smoke.
- Frontend TypeScript check via `npx tsc --noEmit`.

## Remaining Runtime Caveat

- A full default 2-year nightly retrain was not completed end-to-end in this CPU-only sandbox because of runtime cost. The production code path is present, but full-scale timing should be validated on the target machine, ideally with CUDA available for TFT.
