from __future__ import annotations

from datetime import datetime
from io import BytesIO

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import FIGURES_DIR, ForecastConfig, ensure_directories
from backend.data_ingestion import run_full_ingestion
from backend.mlops import (
    bootstrap_system,
    get_dashboard_payload,
    get_system,
    ingest_feedback,
    ingest_uploaded_dataset,
    run_retraining_job,
    start_scheduler,
)
from backend.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    KitchenResponse,
    MetricsResponse,
    PredictRequest,
    PredictionResponse,
    TrainResponse,
    UploadResponse,
)


app = FastAPI(
    title="Kolkata Hostel Kitchen Forecasting API",
    description="Spatiotemporal demand forecasting, food waste optimization, and MLOps feedback loop.",
    version="2.0.0",
)
config = ForecastConfig()
ensure_directories()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/artifacts", StaticFiles(directory=str(FIGURES_DIR)), name="artifacts")

scheduler = None


@app.on_event("startup")
def on_startup() -> None:
    global scheduler
    bootstrap_system(config)
    scheduler = start_scheduler(config)


@app.on_event("shutdown")
def on_shutdown() -> None:
    global scheduler
    if scheduler is not None:
        scheduler.shutdown(wait=False)
        scheduler = None


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/kitchens", response_model=list[KitchenResponse])
def list_kitchens() -> list[KitchenResponse]:
    kitchens = get_system(config).repository.list_kitchens()
    return [KitchenResponse(**row) for row in kitchens.to_dict("records")]


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictRequest) -> PredictionResponse:
    try:
        result = get_system(config).predict(payload.model_dump(mode="json"))
        return PredictionResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/ingest")
def ingest_public_data() -> dict:
    """
    Refresh ``data/raw`` and ``data/processed`` from public APIs (weather, holidays, demand).
    Does not automatically reload SQLite; call ``POST /train`` after ingesting if you need a new panel in the DB.
    """
    return run_full_ingestion(config)


@app.post("/train", response_model=TrainResponse)
def train() -> TrainResponse:
    try:
        return TrainResponse(**run_retraining_job(config))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(payload: FeedbackRequest) -> FeedbackResponse:
    try:
        result = ingest_feedback(
            payload.model_dump(mode="json"),
            config=config,
            retrain_on_feedback=False,
        )
        return FeedbackResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/dataset/upload", response_model=UploadResponse)
async def upload_dataset(file: UploadFile = File(...)) -> UploadResponse:
    try:
        content = await file.read()
        dataset = pd.read_csv(BytesIO(content))
        outcome = ingest_uploaded_dataset(dataset, config=config, retrain=True)
        training_result = outcome.get("training_result", {})
        return UploadResponse(
            status="uploaded",
            rows_ingested=outcome["rows_ingested"],
            trained=bool(training_result),
            selected_model=training_result.get("selected_model"),
            model_version=training_result.get("selected_model_version"),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    payload = get_system(config).get_metrics_payload()
    if not payload.get("trained_at"):
        raise HTTPException(status_code=404, detail="Metrics are not available yet.")
    return MetricsResponse(**payload)


@app.get("/history")
def history() -> dict:
    return get_dashboard_payload(config)["history"]
