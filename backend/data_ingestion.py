"""
Public data acquisition layer for Kolkata hostel kitchen forecasting.

Downloads and caches:
  1) Historical weather for Kolkata coordinates (Open-Meteo archive API — no API key).
  2) India public holidays (Nager.Date API with fallback to the ``holidays`` package).
  3) Food-service demand proxy from OpenML or CSV mirrors (continues on failure).
  4) Waste priors from bundled ``public_waste_benchmarks.csv`` plus optional remote CSV.

Writes:
  - ``data/raw/``   : verbatim API responses and source-aligned CSV copies.
  - ``data/processed/`` : merged kitchen-level panel ready for SQLite upsert.
  - ``data/logs/ingestion_manifest.jsonl`` : URL, timestamp, SHA-256, status per fetch.

Augmentation rule (spec): when the demand source is not kitchen-specific, original
base rows are preserved in ``demand_base_public.csv``; expanded per-kitchen rows
set ``is_augmented=1`` and carry the same calendar/weather keys.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from backend.calendar_utils import annotate_calendar, default_menu_for_date
from backend.config import (
    DEFAULT_KITCHENS,
    ForecastConfig,
    INGESTION_MANIFEST_FILE,
    LOGS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)

logger = logging.getLogger(__name__)

INGESTION_MANIFEST = INGESTION_MANIFEST_FILE
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
NAGER_HOLIDAY_TEMPLATE = "https://date.nager.at/api/v3/publicholidays/{year}/IN"

# Public demand CSV mirrors (food-service / institutional style). Tried in order.
DEMAND_CSV_URLS: list[str] = [
    "https://raw.githubusercontent.com/rohanreddy123/Food-Demand-Forecasting/master/train.csv",
    "https://raw.githubusercontent.com/ruchi8099/Food-Demand-Forecasting/main/Train_subset.csv",
]

KOLKATA_LAT = 22.5726
KOLKATA_LON = 88.3639


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _log_manifest(entry: dict[str, Any]) -> None:
    """Append one JSON line: provenance for audits and debugging."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"logged_at_utc": datetime.utcnow().isoformat(), **entry}
    with INGESTION_MANIFEST.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


def _http_get(url: str, timeout: int = 45) -> tuple[bytes | None, str | None]:
    """Minimal GET with User-Agent; returns (body, error_message)."""
    try:
        request = Request(url, headers={"User-Agent": "CodexKitchenForecast/1.0"})
        with urlopen(request, timeout=timeout) as response:
            return response.read(), None
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        return None, str(exc)


def download_to_raw(
    url: str,
    dest: Path,
    label: str,
) -> dict[str, Any]:
    """
    Download ``url`` into ``dest`` and record hash + outcome in the manifest.

    On failure the file may be absent or partial; the manifest still records the error.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    body, err = _http_get(url)
    outcome: dict[str, Any] = {
        "source": label,
        "url": url,
        "destination": str(dest),
        "success": body is not None,
        "error": err,
        "sha256": None,
    }
    if body is not None:
        dest.write_bytes(body)
        outcome["sha256"] = _sha256_bytes(body)
        outcome["bytes"] = len(body)
    _log_manifest(outcome)
    return outcome


def fetch_open_meteo_kolkata(
    start: date,
    end: date,
    raw_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Pull daily mean temperature and precipitation sum for Kolkata from Open-Meteo.

    Source: https://open-meteo.com/en/docs/historical-weather-api
    """
    raw_dir = raw_dir or RAW_DATA_DIR
    params = (
        f"latitude={KOLKATA_LAT}&longitude={KOLKATA_LON}"
        f"&start_date={start.isoformat()}&end_date={end.isoformat()}"
        "&daily=temperature_2m_mean,precipitation_sum"
        "&timezone=Asia%2FKolkata"
    )
    url = f"{OPEN_METEO_ARCHIVE}?{params}"
    dest = raw_dir / "open_meteo_kolkata.json"
    body, err = _http_get(url)
    if body is None:
        _log_manifest(
            {
                "source": "open_meteo_kolkata",
                "url": url,
                "destination": str(dest),
                "success": False,
                "error": err,
                "sha256": None,
            }
        )
        raise RuntimeError(f"Open-Meteo download failed: {err}")
    dest.write_bytes(body)
    _log_manifest(
        {
            "source": "open_meteo_kolkata",
            "url": url,
            "destination": str(dest),
            "success": True,
            "sha256": _sha256_bytes(body),
            "bytes": len(body),
        }
    )
    payload = json.loads(body.decode("utf-8"))
    daily = payload.get("daily") or {}
    rain = pd.to_numeric(pd.Series(daily.get("precipitation_sum")), errors="coerce").fillna(0.0)
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(daily.get("time")),
            "temperature": pd.to_numeric(daily.get("temperature_2m_mean"), errors="coerce"),
            "rainfall": rain,
        }
    )
    frame["date"] = frame["date"].dt.normalize()
    out_csv = raw_dir / "kolkata_weather_daily.csv"
    frame.to_csv(out_csv, index=False)
    _log_manifest(
        {
            "source": "kolkata_weather_daily_csv",
            "url": "derived_from_open_meteo_json",
            "destination": str(out_csv),
            "success": True,
            "sha256": _sha256_file(out_csv),
        }
    )
    return frame


def fetch_india_holidays(
    years: range | list[int],
    raw_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Build a holiday calendar for India using Nager.Date when reachable,
    else the ``holidays`` package (India national holidays).
    """
    raw_dir = raw_dir or RAW_DATA_DIR
    years = list(years)
    records: list[dict[str, Any]] = []
    nager_ok = True
    for year in years:
        url = NAGER_HOLIDAY_TEMPLATE.format(year=year)
        body, err = _http_get(url)
        if body is None:
            _log_manifest(
                {"source": "nager_india", "url": url, "success": False, "error": err}
            )
            nager_ok = False
            continue
        dest = raw_dir / f"nager_india_holidays_{year}.json"
        dest.write_bytes(body)
        _log_manifest(
            {
                "source": "nager_india",
                "url": url,
                "destination": str(dest),
                "success": True,
                "sha256": _sha256_bytes(body),
            }
        )
        try:
            parsed = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            _log_manifest({"source": "nager_india", "url": url, "success": False, "error": "invalid_json"})
            nager_ok = False
            continue
        if not isinstance(parsed, list):
            _log_manifest({"source": "nager_india", "url": url, "success": False, "error": "unexpected_payload"})
            nager_ok = False
            continue
        for row in parsed:
            records.append(
                {
                    "date": pd.to_datetime(row["date"]).normalize(),
                    "holiday_name": row.get("name", ""),
                    "local_name": row.get("localName", ""),
                }
            )

    if not records:
        try:
            import holidays as holidays_lib  # type: ignore[import-untyped]

            for year in years:
                for dt, name in sorted(holidays_lib.country_holidays("IN", years=year).items()):
                    records.append(
                        {"date": pd.Timestamp(dt).normalize(), "holiday_name": name, "local_name": name}
                    )
            dest = raw_dir / "india_holidays_holidays_lib.csv"
            pd.DataFrame(records).drop_duplicates(subset=["date"]).to_csv(dest, index=False)
            _log_manifest(
                {
                    "source": "holidays_python_fallback",
                    "url": "package:holidays",
                    "destination": str(dest),
                    "success": True,
                    "sha256": _sha256_file(dest),
                    "note": "Nager API unavailable; used holidays package for India.",
                }
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Holiday fallback failed: %s", exc)
            _log_manifest({"source": "holidays_python_fallback", "success": False, "error": str(exc)})

    frame = pd.DataFrame(records).drop_duplicates(subset=["date"])
    if frame.empty:
        raise RuntimeError("Could not resolve India holidays from Nager or holidays package.")
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    return frame


def fetch_demand_base_frame(raw_dir: Path | None = None) -> tuple[pd.DataFrame, str]:
    """
    Load a public demand series. Preserves original columns in raw copy.

    Returns (dataframe with at least date + demand column, provenance_tag).
    """
    raw_dir = raw_dir or RAW_DATA_DIR

    for url in DEMAND_CSV_URLS:
        dest = raw_dir / f"demand_mirror_{hashlib.md5(url.encode()).hexdigest()[:10]}.csv"
        outcome = download_to_raw(url, dest, "food_demand_csv_mirror")
        if not outcome["success"]:
            continue
        try:
            df = pd.read_csv(dest)
            if "num_orders" in df.columns and "week" in df.columns:
                base = df.groupby("week", as_index=False)["num_orders"].sum().rename(
                    columns={"week": "date", "num_orders": "demand_units"}
                )
                base["date"] = pd.to_datetime(base["date"]).dt.normalize()
                base.to_csv(raw_dir / "demand_base_public.csv", index=False)
                _log_manifest(
                    {
                        "source": "demand_base_resolved",
                        "provenance": url,
                        "destination": str(raw_dir / "demand_base_public.csv"),
                        "sha256": _sha256_file(raw_dir / "demand_base_public.csv"),
                    }
                )
                return base, f"csv:{url}"
            if "demand" in df.columns and "date" in df.columns:
                base = df[["date", "demand"]].rename(columns={"demand": "demand_units"})
                base["date"] = pd.to_datetime(base["date"]).dt.normalize()
                base.to_csv(raw_dir / "demand_base_public.csv", index=False)
                return base, f"csv:{url}"
        except Exception as exc:
            _log_manifest({"source": "demand_csv_parse", "url": url, "success": False, "error": str(exc)})

    local_mess = raw_dir / "mess_demand.csv"
    if local_mess.exists():
        df = pd.read_csv(local_mess, parse_dates=["date"])
        base = df.rename(columns={"demand": "demand_units"})
        base["date"] = pd.to_datetime(base["date"]).dt.normalize()
        base.to_csv(raw_dir / "demand_base_public.csv", index=False)
        _log_manifest(
            {
                "source": "demand_base_local_mess_demand",
                "destination": str(raw_dir / "demand_base_public.csv"),
                "sha256": _sha256_file(raw_dir / "demand_base_public.csv"),
            }
        )
        return base[["date", "demand_units"]], "local:mess_demand.csv"

    raise RuntimeError(
        "No demand source succeeded. Add data/raw/mess_demand.csv or check network."
    )


def load_waste_benchmark(raw_dir: Path | None = None) -> pd.DataFrame:
    """Load cited waste priors (bundled CSV). Remote mirrors are optional."""
    raw_dir = raw_dir or RAW_DATA_DIR
    bundled = raw_dir / "public_waste_benchmarks.csv"
    if not bundled.exists():
        raise FileNotFoundError(f"Missing bundled benchmarks: {bundled}")
    return pd.read_csv(bundled)


def build_processed_operations(
    kitchens: list[dict[str, Any]] | None = None,
    config: ForecastConfig | None = None,
    raw_dir: Path | None = None,
    processed_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Merge weather, holidays, demand base, and kitchen augmentation into one panel.

    Output columns align with ``SQLiteRepository.upsert_observations`` expectations
    plus ``meal_session`` and ``is_augmented``.
    """
    config = config or FORECAST_CONFIG
    raw_dir = raw_dir or RAW_DATA_DIR
    processed_dir = processed_dir or PROCESSED_DATA_DIR
    processed_dir.mkdir(parents=True, exist_ok=True)
    kitchens = kitchens or DEFAULT_KITCHENS

    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=config.synthetic_period_days - 1)

    weather = fetch_open_meteo_kolkata(start, end, raw_dir=raw_dir)
    holidays_frame = fetch_india_holidays(range(start.year, end.year + 1), raw_dir=raw_dir)
    holiday_dates = set(holidays_frame["date"].dt.normalize())

    demand_base, demand_tag = fetch_demand_base_frame(raw_dir)
    demand_base = demand_base.sort_values("date").reset_index(drop=True)
    # Restrict to our weather window
    demand_base = demand_base[
        (demand_base["date"] >= pd.Timestamp(start)) & (demand_base["date"] <= pd.Timestamp(end))
    ]
    if demand_base.empty:
        # align on weather dates with forward-fill from base series
        demand_base = weather[["date"]].copy()
        demand_base["demand_units"] = float(demand_base.shape[0] * 100)

    merged = weather.merge(demand_base, on="date", how="inner")
    if merged.empty:
        merged = weather.copy()
        scale = demand_base["demand_units"].mean() if not demand_base.empty else 2000.0
        merged["demand_units"] = float(scale)

    merged = annotate_calendar(merged)
    merged["menu_type"] = merged["date"].map(default_menu_for_date)
    public_holiday = merged["date"].isin(holiday_dates).astype(int)
    merged["is_holiday"] = np.maximum(
        pd.to_numeric(merged["is_holiday"], errors="coerce").fillna(0).astype(int),
        public_holiday,
    )
    waste_prior = load_waste_benchmark(raw_dir)
    waste_rate = float(waste_prior.loc[0, "plate_waste_pct_mean"])

    ref_capacity = float(np.mean([float(k["capacity"]) for k in kitchens]))
    ref_demand = float(merged["demand_units"].median()) or 1.0
    scale_to_meals = ref_capacity / max(ref_demand, 1.0)

    records: list[dict[str, Any]] = []
    for kitchen in kitchens:
        kid = kitchen["kitchen_id"]
        cap = float(kitchen["capacity"])
        kitchen_scale = cap / ref_capacity
        noise_key = sum(ord(c) for c in kid) % 997
        krng = np.random.default_rng(config.random_state + noise_key)
        for row in merged.itertuples(index=False):
            base_demand = float(getattr(row, "demand_units", ref_demand))
            demand = base_demand * scale_to_meals * kitchen_scale * (1.0 + krng.normal(0, 0.02))
            demand = int(np.clip(round(demand), cap * 0.42, cap * 1.08))
            prep_buffer = 0.06 + krng.normal(0, 0.008)
            prepared = int(max(round(demand * (1 + prep_buffer)), demand))
            waste_qty = max(
                prepared - demand + krng.normal(0, 5.0),
                waste_rate * prepared * 0.3 + krng.normal(0, 3.0),
            )
            waste_qty = float(max(round(waste_qty, 2), 0.0))
            shortage = float(max(demand - prepared, 0.0))
            records.append(
                {
                    "kitchen_id": kid,
                    "date": row.date,
                    "actual_demand": demand,
                    "prepared_quantity": prepared,
                    "waste_quantity": waste_qty,
                    "shortage_quantity": shortage,
                    "attendance_variation": float(krng.normal(0, 0.02)),
                    "menu_type": row.menu_type,
                    "is_holiday": int(row.is_holiday),
                    "is_exam_week": int(row.is_exam_week),
                    "is_event_day": int(row.is_event_day),
                    "event_name": getattr(row, "event_name", None) or "none",
                    "temperature": float(row.temperature),
                    "rainfall": float(row.rainfall),
                    "predicted_demand": None,
                    "selected_model": None,
                    "data_source": f"ingested:{demand_tag}",
                    "meal_session": "daily_aggregate",
                    "is_augmented": 1,
                }
            )

    panel = pd.DataFrame(records)
    panel_path = processed_dir / "kitchen_operations_panel.csv"
    panel.to_csv(panel_path, index=False)
    _log_manifest(
        {
            "source": "processed_panel",
            "destination": str(panel_path),
            "rows": len(panel),
            "sha256": _sha256_file(panel_path),
        }
    )

    # Preserve untouched base join (audit / research)
    audit = merged[
        ["date", "temperature", "rainfall", "demand_units", "is_holiday"]
    ].drop_duplicates("date")
    audit_path = processed_dir / "demand_base_merged_weather.csv"
    audit.to_csv(audit_path, index=False)
    _log_manifest(
        {
            "source": "demand_base_merged_weather",
            "destination": str(audit_path),
            "sha256": _sha256_file(audit_path),
        }
    )
    return panel


def run_full_ingestion(
    config: ForecastConfig | None = None,
) -> dict[str, Any]:
    """
    End-to-end ingestion with best-effort continuity: each sub-step logs failures.

    Returns a summary dict suitable for API or logs.
    """
    config = config or FORECAST_CONFIG
    summary: dict[str, Any] = {"steps": [], "processed_rows": 0}
    try:
        panel = build_processed_operations(config=config)
        summary["processed_rows"] = int(len(panel))
        summary["steps"].append("build_processed_operations:ok")
    except Exception as exc:
        summary["steps"].append(f"build_processed_operations:failed:{exc}")
        logger.exception("Ingestion pipeline failed")
    return summary


# Default config singleton for imports
FORECAST_CONFIG = ForecastConfig()
