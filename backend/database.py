from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

import pandas as pd

from backend.config import (
    DEFAULT_KITCHENS,
    DEFAULT_RECIPES,
    SQLITE_DB_FILE,
    ensure_directories,
)


SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS kitchens (
        kitchen_id TEXT PRIMARY KEY,
        hostel_name TEXT NOT NULL,
        campus_zone TEXT NOT NULL,
        latitude REAL NOT NULL,
        longitude REAL NOT NULL,
        capacity INTEGER NOT NULL,
        capacity_band TEXT NOT NULL,
        default_attendance_band TEXT NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS recipes (
        menu_type TEXT NOT NULL,
        ingredient_name TEXT NOT NULL,
        unit TEXT NOT NULL,
        qty_per_100_meals REAL NOT NULL,
        PRIMARY KEY (menu_type, ingredient_name)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS daily_observations (
        observation_id INTEGER PRIMARY KEY AUTOINCREMENT,
        kitchen_id TEXT NOT NULL,
        date TEXT NOT NULL,
        actual_demand INTEGER NOT NULL,
        prepared_quantity INTEGER,
        waste_quantity REAL,
        shortage_quantity REAL,
        attendance_variation REAL,
        menu_type TEXT NOT NULL,
        is_holiday INTEGER NOT NULL DEFAULT 0,
        is_exam_week INTEGER NOT NULL DEFAULT 0,
        is_event_day INTEGER NOT NULL DEFAULT 0,
        event_name TEXT,
        temperature REAL,
        rainfall REAL,
        predicted_demand REAL,
        selected_model TEXT,
        data_source TEXT NOT NULL DEFAULT 'synthetic',
        meal_session TEXT NOT NULL DEFAULT 'daily_aggregate',
        is_augmented INTEGER NOT NULL DEFAULT 0,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        UNIQUE (kitchen_id, date),
        FOREIGN KEY (kitchen_id) REFERENCES kitchens (kitchen_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS predictions (
        prediction_id TEXT NOT NULL,
        kitchen_id TEXT NOT NULL,
        forecast_date TEXT NOT NULL,
        generated_at TEXT NOT NULL,
        horizon_day INTEGER NOT NULL,
        model_name TEXT NOT NULL,
        model_version TEXT NOT NULL,
        point_forecast REAL NOT NULL,
        lower_bound REAL NOT NULL,
        upper_bound REAL NOT NULL,
        selected_flag INTEGER NOT NULL DEFAULT 1,
        PRIMARY KEY (prediction_id, horizon_day)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS optimization_decisions (
        decision_id TEXT PRIMARY KEY,
        prediction_id TEXT NOT NULL,
        kitchen_id TEXT NOT NULL,
        forecast_date TEXT NOT NULL,
        optimal_quantity INTEGER NOT NULL,
        expected_waste REAL NOT NULL,
        expected_shortage REAL NOT NULL,
        expected_cost REAL NOT NULL,
        realized_waste REAL,
        realized_shortage REAL,
        realized_cost REAL,
        prepared_quantity INTEGER
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS training_runs (
        run_id TEXT NOT NULL,
        trained_at TEXT NOT NULL,
        model_name TEXT NOT NULL,
        model_version TEXT NOT NULL,
        rmse REAL NOT NULL,
        mae REAL NOT NULL,
        weekly_rmse REAL NOT NULL,
        weekly_mae REAL NOT NULL,
        interval_coverage REAL NOT NULL,
        residual_std REAL NOT NULL,
        mean_prediction_jump REAL NOT NULL,
        selected_model INTEGER NOT NULL DEFAULT 0,
        promoted INTEGER NOT NULL DEFAULT 0,
        improvement_pct REAL NOT NULL DEFAULT 0,
        notes TEXT,
        PRIMARY KEY (run_id, model_name)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS model_registry (
        model_name TEXT PRIMARY KEY,
        model_version TEXT NOT NULL,
        artifact_path TEXT NOT NULL,
        metadata_path TEXT,
        is_selected INTEGER NOT NULL DEFAULT 0,
        trained_at TEXT NOT NULL,
        next_day_rmse REAL NOT NULL,
        weekly_rmse REAL NOT NULL,
        promotion_reason TEXT
    )
    """,
]


def capacity_band_for_value(capacity: int) -> str:
    if capacity >= 2400:
        return "xl"
    if capacity >= 2000:
        return "large"
    if capacity >= 1600:
        return "medium"
    return "compact"


def _backup_corrupted_database() -> None:
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    db_candidates = [
        SQLITE_DB_FILE,
        Path(f"{SQLITE_DB_FILE}-journal"),
        Path(f"{SQLITE_DB_FILE}-wal"),
        Path(f"{SQLITE_DB_FILE}-shm"),
    ]
    for candidate in db_candidates:
        if candidate.exists():
            backup_name = candidate.with_name(f"{candidate.stem}.{timestamp}.corrupt{candidate.suffix}")
            candidate.replace(backup_name)


def _connect_with_recovery() -> sqlite3.Connection:
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(SQLITE_DB_FILE, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        connection.execute("SELECT 1").fetchone()
        return connection
    except sqlite3.OperationalError:
        if connection is not None:
            connection.close()
        _backup_corrupted_database()
        connection = sqlite3.connect(SQLITE_DB_FILE, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        return connection


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    ensure_directories()
    connection = _connect_with_recovery()
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def _migrate_daily_observations(connection: sqlite3.Connection) -> None:
    """Add columns introduced after first deployments (SQLite lightweight migration)."""
    columns = {
        row[1] for row in connection.execute("PRAGMA table_info(daily_observations)").fetchall()
    }
    if "meal_session" not in columns:
        connection.execute(
            """
            ALTER TABLE daily_observations
            ADD COLUMN meal_session TEXT NOT NULL DEFAULT 'daily_aggregate'
            """
        )
    if "is_augmented" not in columns:
        connection.execute(
            """
            ALTER TABLE daily_observations
            ADD COLUMN is_augmented INTEGER NOT NULL DEFAULT 0
            """
        )


def initialize_database() -> None:
    ensure_directories()
    with get_connection() as connection:
        for statement in SCHEMA_STATEMENTS:
            connection.execute(statement)
        _migrate_daily_observations(connection)

        existing_kitchens = connection.execute("SELECT COUNT(*) AS count FROM kitchens").fetchone()[
            "count"
        ]
        if existing_kitchens == 0:
            for kitchen in DEFAULT_KITCHENS:
                connection.execute(
                    """
                    INSERT INTO kitchens (
                        kitchen_id, hostel_name, campus_zone, latitude, longitude,
                        capacity, capacity_band, default_attendance_band
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        kitchen["kitchen_id"],
                        kitchen["hostel_name"],
                        kitchen["campus_zone"],
                        kitchen["latitude"],
                        kitchen["longitude"],
                        kitchen["capacity"],
                        capacity_band_for_value(kitchen["capacity"]),
                        kitchen["default_attendance_band"],
                    ),
                )

        existing_recipes = connection.execute("SELECT COUNT(*) AS count FROM recipes").fetchone()[
            "count"
        ]
        if existing_recipes == 0:
            for menu_type, ingredients in DEFAULT_RECIPES.items():
                for ingredient in ingredients:
                    connection.execute(
                        """
                        INSERT INTO recipes (menu_type, ingredient_name, unit, qty_per_100_meals)
                        VALUES (?, ?, ?, ?)
                        """,
                        (
                            menu_type,
                            ingredient["ingredient_name"],
                            ingredient["unit"],
                            ingredient["qty_per_100_meals"],
                        ),
                    )


@dataclass
class SQLiteRepository:
    """Small repository layer around the operational SQLite store."""

    def list_kitchens(self) -> pd.DataFrame:
        with get_connection() as connection:
            return pd.read_sql_query(
                "SELECT * FROM kitchens ORDER BY kitchen_id",
                connection,
            )

    def list_recipes(self) -> pd.DataFrame:
        with get_connection() as connection:
            return pd.read_sql_query(
                "SELECT * FROM recipes ORDER BY menu_type, ingredient_name",
                connection,
            )

    def observation_count(self) -> int:
        with get_connection() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM daily_observations"
            ).fetchone()
            return int(row["count"])

    def upsert_observations(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        records = frame.copy()
        records["date"] = pd.to_datetime(records["date"]).dt.strftime("%Y-%m-%d")
        if "meal_session" not in records.columns:
            records["meal_session"] = "daily_aggregate"
        if "is_augmented" not in records.columns:
            records["is_augmented"] = 0
        payload = [
            (
                row.kitchen_id,
                row.date,
                int(row.actual_demand),
                int(row.prepared_quantity) if pd.notna(row.prepared_quantity) else None,
                float(row.waste_quantity) if pd.notna(row.waste_quantity) else None,
                float(row.shortage_quantity) if pd.notna(row.shortage_quantity) else None,
                float(row.attendance_variation) if pd.notna(row.attendance_variation) else None,
                row.menu_type,
                int(row.is_holiday),
                int(row.is_exam_week),
                int(row.is_event_day),
                row.event_name if pd.notna(row.event_name) else None,
                float(row.temperature) if pd.notna(row.temperature) else None,
                float(row.rainfall) if pd.notna(row.rainfall) else None,
                float(row.predicted_demand) if pd.notna(row.predicted_demand) else None,
                row.selected_model if pd.notna(row.selected_model) else None,
                row.data_source if pd.notna(row.data_source) else "synthetic",
                str(row.meal_session) if pd.notna(row.meal_session) else "daily_aggregate",
                int(row.is_augmented),
            )
            for row in records.itertuples(index=False)
        ]
        with get_connection() as connection:
            connection.executemany(
                """
                INSERT INTO daily_observations (
                    kitchen_id, date, actual_demand, prepared_quantity, waste_quantity,
                    shortage_quantity, attendance_variation, menu_type, is_holiday,
                    is_exam_week, is_event_day, event_name, temperature, rainfall,
                    predicted_demand, selected_model, data_source, meal_session, is_augmented
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(kitchen_id, date) DO UPDATE SET
                    actual_demand=excluded.actual_demand,
                    prepared_quantity=excluded.prepared_quantity,
                    waste_quantity=excluded.waste_quantity,
                    shortage_quantity=excluded.shortage_quantity,
                    attendance_variation=excluded.attendance_variation,
                    menu_type=excluded.menu_type,
                    is_holiday=excluded.is_holiday,
                    is_exam_week=excluded.is_exam_week,
                    is_event_day=excluded.is_event_day,
                    event_name=excluded.event_name,
                    temperature=excluded.temperature,
                    rainfall=excluded.rainfall,
                    predicted_demand=excluded.predicted_demand,
                    selected_model=excluded.selected_model,
                    data_source=excluded.data_source,
                    meal_session=excluded.meal_session,
                    is_augmented=excluded.is_augmented,
                    updated_at=CURRENT_TIMESTAMP
                """,
                payload,
            )

    def load_observations(self) -> pd.DataFrame:
        with get_connection() as connection:
            return pd.read_sql_query(
                """
                SELECT
                    o.kitchen_id,
                    o.date,
                    o.actual_demand,
                    o.prepared_quantity,
                    o.waste_quantity,
                    o.shortage_quantity,
                    o.attendance_variation,
                    o.menu_type,
                    o.is_holiday,
                    o.is_exam_week,
                    o.is_event_day,
                    o.event_name,
                    o.temperature,
                    o.rainfall,
                    o.predicted_demand,
                    o.selected_model,
                    o.data_source,
                    o.meal_session,
                    o.is_augmented,
                    k.hostel_name,
                    k.campus_zone,
                    k.latitude,
                    k.longitude,
                    k.capacity,
                    k.capacity_band,
                    k.default_attendance_band
                FROM daily_observations o
                INNER JOIN kitchens k ON k.kitchen_id = o.kitchen_id
                ORDER BY o.date, o.kitchen_id
                """,
                connection,
                parse_dates=["date"],
            )

    def insert_predictions(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        payload = [
            (
                row.prediction_id,
                row.kitchen_id,
                row.forecast_date,
                row.generated_at,
                int(row.horizon_day),
                row.model_name,
                row.model_version,
                float(row.point_forecast),
                float(row.lower_bound),
                float(row.upper_bound),
                int(row.selected_flag),
            )
            for row in frame.itertuples(index=False)
        ]
        with get_connection() as connection:
            connection.executemany(
                """
                INSERT INTO predictions (
                    prediction_id, kitchen_id, forecast_date, generated_at, horizon_day,
                    model_name, model_version, point_forecast, lower_bound, upper_bound,
                    selected_flag
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )

    def insert_optimization_decisions(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        payload = [
            (
                row.decision_id,
                row.prediction_id,
                row.kitchen_id,
                row.forecast_date,
                int(row.optimal_quantity),
                float(row.expected_waste),
                float(row.expected_shortage),
                float(row.expected_cost),
                float(row.realized_waste) if pd.notna(row.realized_waste) else None,
                float(row.realized_shortage) if pd.notna(row.realized_shortage) else None,
                float(row.realized_cost) if pd.notna(row.realized_cost) else None,
                int(row.prepared_quantity) if pd.notna(row.prepared_quantity) else None,
            )
            for row in frame.itertuples(index=False)
        ]
        with get_connection() as connection:
            connection.executemany(
                """
                INSERT INTO optimization_decisions (
                    decision_id, prediction_id, kitchen_id, forecast_date, optimal_quantity,
                    expected_waste, expected_shortage, expected_cost, realized_waste,
                    realized_shortage, realized_cost, prepared_quantity
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )

    def update_realized_decision(
        self,
        kitchen_id: str,
        forecast_date: str,
        prepared_quantity: int,
        realized_waste: float,
        realized_shortage: float,
        realized_cost: float,
    ) -> None:
        with get_connection() as connection:
            connection.execute(
                """
                UPDATE optimization_decisions
                SET prepared_quantity = ?, realized_waste = ?, realized_shortage = ?, realized_cost = ?
                WHERE kitchen_id = ? AND forecast_date = ?
                """,
                (
                    prepared_quantity,
                    realized_waste,
                    realized_shortage,
                    realized_cost,
                    kitchen_id,
                    forecast_date,
                ),
            )

    def insert_training_runs(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        payload = [
            (
                row.run_id,
                row.trained_at,
                row.model_name,
                row.model_version,
                float(row.rmse),
                float(row.mae),
                float(row.weekly_rmse),
                float(row.weekly_mae),
                float(row.interval_coverage),
                float(row.residual_std),
                float(row.mean_prediction_jump),
                int(row.selected_model),
                int(row.promoted),
                float(row.improvement_pct),
                row.notes,
            )
            for row in frame.itertuples(index=False)
        ]
        with get_connection() as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO training_runs (
                    run_id, trained_at, model_name, model_version, rmse, mae, weekly_rmse,
                    weekly_mae, interval_coverage, residual_std, mean_prediction_jump,
                    selected_model, promoted, improvement_pct, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )

    def replace_model_registry(self, frame: pd.DataFrame) -> None:
        with get_connection() as connection:
            connection.execute("DELETE FROM model_registry")
            if frame.empty:
                return
            payload = [
                (
                    row.model_name,
                    row.model_version,
                    row.artifact_path,
                    row.metadata_path,
                    int(row.is_selected),
                    row.trained_at,
                    float(row.next_day_rmse),
                    float(row.weekly_rmse),
                    row.promotion_reason,
                )
                for row in frame.itertuples(index=False)
            ]
            connection.executemany(
                """
                INSERT INTO model_registry (
                    model_name, model_version, artifact_path, metadata_path, is_selected,
                    trained_at, next_day_rmse, weekly_rmse, promotion_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )

    def get_selected_model_registry(self) -> dict | None:
        with get_connection() as connection:
            row = connection.execute(
                "SELECT * FROM model_registry WHERE is_selected = 1 ORDER BY trained_at DESC LIMIT 1"
            ).fetchone()
            return dict(row) if row else None

    def latest_training_runs(self, limit: int = 20) -> pd.DataFrame:
        with get_connection() as connection:
            return pd.read_sql_query(
                """
                SELECT *
                FROM training_runs
                ORDER BY trained_at DESC, model_name ASC
                LIMIT ?
                """,
                connection,
                params=(limit,),
            )

    def latest_predictions(self, limit: int = 200) -> pd.DataFrame:
        with get_connection() as connection:
            return pd.read_sql_query(
                """
                SELECT *
                FROM predictions
                ORDER BY generated_at DESC, horizon_day ASC
                LIMIT ?
                """,
                connection,
                params=(limit,),
            )
