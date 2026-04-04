from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from backend.config import FIGURES_DIR, PLOT_FILENAMES


sns.set_theme(style="whitegrid")


def _empty_figure(title: str, output_key: str) -> str:
    plt.figure(figsize=(8, 4))
    plt.text(0.5, 0.5, "No data available", ha="center", va="center", fontsize=14)
    plt.title(title)
    plt.axis("off")
    path = FIGURES_DIR / PLOT_FILENAMES[output_key]
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return str(path)


def generate_plots(
    forecast_history: pd.DataFrame,
    model_comparison: pd.DataFrame,
    business_metrics: dict,
    feature_importance: pd.DataFrame | None = None,
) -> dict[str, str]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    generated: dict[str, str] = {}

    history = forecast_history.copy()
    if not history.empty:
        history["date"] = pd.to_datetime(history["date"])
        history = history.sort_values(["date", "kitchen_id"])
        daily = (
            history.groupby("date", as_index=False)[
                [
                    "actual_demand",
                    "predicted_demand",
                    "lower_bound",
                    "upper_bound",
                    "baseline_waste_realized",
                    "optimized_waste_realized",
                    "daily_cost_saving_inr",
                ]
            ]
            .sum(numeric_only=True)
            .sort_values("date")
        )

        plt.figure(figsize=(14, 6))
        plt.plot(daily["date"], daily["actual_demand"], label="Actual", linewidth=2.2, color="#204d45")
        plt.plot(daily["date"], daily["predicted_demand"], label="Predicted", linewidth=2.0, color="#b95c32")
        plt.fill_between(
            daily["date"],
            daily["lower_bound"],
            daily["upper_bound"],
            alpha=0.18,
            color="#d39c3d",
            label="Prediction interval",
        )
        plt.title("Kitchen Network Demand vs Forecast")
        plt.xlabel("Date")
        plt.ylabel("Meals")
        plt.legend()
        plt.tight_layout()
        path = FIGURES_DIR / PLOT_FILENAMES["demand_vs_actual"]
        plt.savefig(path, dpi=180)
        plt.close()
        generated["demand_vs_actual"] = str(path)

        residuals = history["actual_demand"] - history["predicted_demand"]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].scatter(history["predicted_demand"], residuals, color="#204d45", alpha=0.45)
        axes[0].axhline(0.0, linestyle="--", color="#444444")
        axes[0].set_title("Residuals vs Predicted")
        axes[0].set_xlabel("Predicted meals")
        axes[0].set_ylabel("Residual")
        sns.histplot(residuals, bins=24, kde=True, ax=axes[1], color="#b95c32")
        axes[1].set_title("Residual Distribution")
        plt.tight_layout()
        path = FIGURES_DIR / PLOT_FILENAMES["residual_analysis"]
        plt.savefig(path, dpi=180)
        plt.close(fig)
        generated["residual_analysis"] = str(path)

        waste_frame = pd.DataFrame(
            {
                "strategy": ["Historical baseline", "Optimized"],
                "waste": [
                    history["baseline_waste_realized"].sum(),
                    history["optimized_waste_realized"].sum(),
                ],
                "shortage": [
                    history["baseline_shortage_realized"].sum(),
                    history["optimized_shortage_realized"].sum(),
                ],
            }
        )
        waste_frame.plot(
            kind="bar",
            x="strategy",
            y=["waste", "shortage"],
            figsize=(10, 6),
            color=["#d07b4c", "#365c7a"],
        )
        plt.title("Waste and Shortage Comparison")
        plt.ylabel("Meals")
        plt.tight_layout()
        path = FIGURES_DIR / PLOT_FILENAMES["waste_comparison"]
        plt.savefig(path, dpi=180)
        plt.close()
        generated["waste_comparison"] = str(path)

        plt.figure(figsize=(12, 5))
        rolling_savings = daily["daily_cost_saving_inr"].rolling(14, min_periods=1).mean()
        plt.plot(daily["date"], rolling_savings, linewidth=2.4, color="#204d45")
        plt.axhline(0.0, linestyle="--", color="#666666")
        plt.title(
            f"Rolling Cost Savings | Annualized INR {business_metrics.get('annual_savings_inr', 0.0):,.0f}"
        )
        plt.xlabel("Date")
        plt.ylabel("INR")
        plt.tight_layout()
        path = FIGURES_DIR / PLOT_FILENAMES["cost_savings"]
        plt.savefig(path, dpi=180)
        plt.close()
        generated["cost_savings"] = str(path)
    else:
        for key, title in (
            ("demand_vs_actual", "Kitchen Network Demand vs Forecast"),
            ("residual_analysis", "Residual Analysis"),
            ("waste_comparison", "Waste and Shortage Comparison"),
            ("cost_savings", "Cost Savings"),
        ):
            generated[key] = _empty_figure(title, key)

    comparison = model_comparison.copy()
    if not comparison.empty:
        comparison = comparison.sort_values("rmse")
        plt.figure(figsize=(11, 5))
        x = np.arange(len(comparison))
        width = 0.38
        plt.bar(x - width / 2, comparison["rmse"], width=width, label="Next-day RMSE", color="#204d45")
        plt.bar(x + width / 2, comparison["weekly_rmse"], width=width, label="Weekly RMSE", color="#d39c3d")
        plt.xticks(x, comparison["model_name"], rotation=20)
        plt.ylabel("RMSE")
        plt.title("Candidate Model Comparison")
        plt.legend()
        plt.tight_layout()
        path = FIGURES_DIR / PLOT_FILENAMES["model_comparison"]
        plt.savefig(path, dpi=180)
        plt.close()
        generated["model_comparison"] = str(path)
    else:
        generated["model_comparison"] = _empty_figure("Candidate Model Comparison", "model_comparison")

    importance = feature_importance.copy() if feature_importance is not None else pd.DataFrame()
    if not importance.empty:
        importance = importance.head(12).sort_values("importance")
        plt.figure(figsize=(10, 6))
        plt.barh(importance["feature"], importance["importance"], color="#b95c32")
        plt.title("Feature Importance")
        plt.xlabel("Importance")
        plt.tight_layout()
        path = FIGURES_DIR / PLOT_FILENAMES["feature_importance"]
        plt.savefig(path, dpi=180)
        plt.close()
        generated["feature_importance"] = str(path)
    else:
        generated["feature_importance"] = _empty_figure("Feature Importance", "feature_importance")

    return generated
